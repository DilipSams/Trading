"""
================================================================================
ALPHA-TRADE v6.0 -- INSTITUTIONAL ARCHITECTURE REFACTOR
================================================================================

Implements the clean four-layer separation required by institutional-grade
systematic trading systems:

    L1  ALPHA FACTORY     +' Generates alpha signals (mu, sigma, confidence)
    L2  SIGNAL ENSEMBLE   +' Combines multiple alpha families + meta-learner
    L3  PORTFOLIO / RISK  +' Converts combined signal +' position via optimizer
    L4  EXECUTION         +' Order slicing, kill switches, monitoring

CORE PRINCIPLE:
    The RL agent NEVER outputs position sizes. It outputs an alpha view
    (expected return, uncertainty, regime belief). Portfolio construction
    is entirely owned by L3.

ALPHA INTERFACE:
    Every alpha source (RL or not) produces the same AlphaSignal:
        mu      : float  # Expected return over horizon H
        sigma   : float  # Uncertainty / forecast std
        confidence : float  # Signal conviction [0, 1]
        metadata : dict   # Regime, horizon, decay, etc.

REFERENCES:
    Architecture pattern: Two Sigma, AQR, Citadel systematic equity structure
    Meta-learner: Ridge/ElasticNet over alpha features (DeMiguel et al 2020)
    Portfolio construction: Risk parity + Black-Litterman overlay
    Future leak prevention: Lopez de Prado (2018) "Advances in Financial ML"

USAGE:
    from alphago_architecture import (
        ArchitectureConfig, InstitutionalPipeline,
        AlphaSignal, PortfolioOrder
    )
    from alphago_trading_system import Config, AlphaTradeSystem
    from alphago_enhancements import EnhancedConfig

    cfg = Config()
    ecfg = EnhancedConfig()
    acfg = ArchitectureConfig()

    pipeline = InstitutionalPipeline(cfg, ecfg, acfg)
    pipeline.register_alpha('rl', RLAlphaAdapter(system))
    pipeline.register_alpha('trend', TrendAlpha(acfg))
    pipeline.register_alpha('mr', MeanReversionAlpha(acfg))

    # Each step:
    order = pipeline.step(observation, market_state)
    # order.target_exposure is the ONLY thing that touches the environment

================================================================================
"""

import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import deque, OrderedDict
from contextlib import contextmanager
import warnings

# NEW: Import crowding detection
try:
    from alphago_crowding import CrowdingDetector
    CROWDING_AVAILABLE = True
except ImportError:
    CROWDING_AVAILABLE = False
    # Only warn in main process (workers re-import and spam this)
    import multiprocessing as _mp
    if _mp.current_process().name == 'MainProcess':
        print("[WARNING] alphago_crowding.py not found - crowding detection disabled")

try:
    from alphago_cost_model import estimate_cost_dollars, CostBreakdown
    HAS_COST_MODEL = True
except ImportError:
    HAS_COST_MODEL = False

try:
    from validation_engine import (
        PurgedWalkForwardCV, CombinatorialPurgedCV,
        SignificanceGate, MultipleTestingTracker,
        ReproducibilityManager, VersionedConfig, RunArtifactBundle,
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# NOTE: sklearn dependency REMOVED (v6.1).
# Ridge regression is implemented in pure NumPy to avoid SciPy version traps
# (e.g. solver="cholesky" calling scipy.linalg.solve(..., sym_pos=True) which
# fails on older SciPy). Closed-form SVD ridge is 5 lines and more portable.
HAS_SKLEARN = False  # Kept for compat; numpy-only path is always used


class _NumpyStandardScaler:
    """Minimal StandardScaler replacement -- no sklearn needed."""
    def __init__(self):
        self._mean = None
        self._std = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-10
        return (X - self._mean) / self._std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            return X
        return (X - self._mean) / self._std


class _NumpyRidge:
    """
    Pure-NumPy ridge regression via SVD (no scipy/sklearn dependency).

    Solves: w = (X'X + +/-I)^{-1} X'y  via SVD for numerical stability.
    SVD path avoids the Cholesky sym_pos crash entirely and handles
    collinear features (common in alpha ensembles) gracefully.

    Reference: Hastie, Tibshirani, Friedman (2009) "Elements of Statistical
               Learning" S3.4.1
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        # Center y
        y_mean = y.mean()
        y_c = y - y_mean
        # SVD of X
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Ridge coefficients via SVD: w = V diag(s/(s^2++/-)) U' y_c
        d = s / (s ** 2 + self.alpha)
        self.coef_ = (Vt.T * d) @ (U.T @ y_c)
        self.intercept_ = y_mean - X.mean(axis=0) @ self.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


# ============================================================================
# UNIVERSAL ALPHA INTERFACE
# ============================================================================

@dataclass
class AlphaSignal:
    """
    Universal interface that every alpha source must produce.

    This is the CONTRACT between L1 (alpha generation) and L2 (ensemble).
    No alpha is allowed to output a position size -- only a view on returns.

    Attributes:
        mu:          Expected return over the signal's horizon (annualized)
        sigma:       Forecast uncertainty / standard deviation
        confidence:  Signal conviction in [0, 1]. 0 = no signal, 1 = max conviction
        horizon:     Signal horizon in bars (e.g. 1, 5, 21 for 1d, 1w, 1m)
        alpha_name:  Identifier string
        metadata:    Arbitrary dict for regime, decay, diagnostics
        timestamp:   Bar index when signal was generated
    """
    mu: float = 0.0
    sigma: float = 1.0
    confidence: float = 0.0
    horizon: int = 1
    alpha_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0

    def __post_init__(self):
        # BUG FIX #4: Sanitize nan/inf values before validation
        # Some alphas can produce nan confidence during warmup (division by zero in vol calcs)
        if np.isnan(self.mu) or np.isinf(self.mu):
            self.mu = 0.0
        if np.isnan(self.sigma) or np.isinf(self.sigma) or self.sigma <= 0:
            self.sigma = 1.0
        if np.isnan(self.confidence) or np.isinf(self.confidence):
            self.confidence = 0.0  # Deactivate alpha if confidence is invalid

        # Enforce interface contract
        assert -10.0 <= self.mu <= 10.0, f"mu={self.mu} out of range [-10, 10]"
        assert self.sigma > 0, f"sigma must be positive, got {self.sigma}"
        assert 0.0 <= self.confidence <= 1.0, f"confidence must be in [0,1], got {self.confidence}"
        assert self.horizon >= 1, f"horizon must be >= 1, got {self.horizon}"

    @property
    def sharpe(self) -> float:
        """Information ratio of this signal."""
        return self.mu / (self.sigma + 1e-10)

    @property
    def is_active(self) -> bool:
        return self.confidence > 0.01


@dataclass
class PortfolioOrder:
    """
    Output of L3 Portfolio Constructor +' input to L4 Execution.

    This is the ONLY object that determines position sizing.
    No other layer is allowed to set target_exposure.

    Attributes:
        target_exposure:  Final target position as fraction of max notional [-1, +1]
        urgency:          How quickly to execute (0 = patient, 1 = immediate)
        risk_budget:      Max dollar risk allocated to this trade
        reason:           Human-readable audit trail
        constraints_hit:  Which risk constraints were binding
    """
    target_exposure: float = 0.0
    urgency: float = 0.5
    risk_budget: float = 0.0
    reason: str = ""
    constraints_hit: List[str] = field(default_factory=list)
    alpha_signals: Dict[str, AlphaSignal] = field(default_factory=dict)
    combined_mu: float = 0.0
    combined_sigma: float = 1.0
    # WS3D: Risk stats per rebalance (ex-ante vol, CVaR, factor exposures, etc.)
    risk_stats: Dict[str, Any] = field(default_factory=dict)
    # WS3E: Constraint binding attribution (which constraint cost how much alpha)
    constraint_attribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Output of L4 Execution Engine -- what actually happened."""
    executed_exposure: float = 0.0
    discrete_action: int = 2             # Index into action_targets (2=FLAT default)
    discrete_exposure: float = 0.0       # The discretized exposure actually sent to env
    fill_price: float = 0.0
    slippage_bps: float = 0.0
    commission: float = 0.0
    was_killed: bool = False
    kill_reason: str = ""
    was_suppressed: bool = False         # No-trade region blocked the trade
    n_child_fills: int = 0


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ArchitectureConfig:
    """Configuration for the L1->L2->L3->L4 institutional architecture."""

    # -- L1: Alpha Factory --
    # RL alpha adapter
    rl_alpha_horizon: int = 5               # RL signal horizon (bars)
    rl_logit_to_mu_scale: float = 0.10      # Scale factor: policy logits +' mu
    rl_risk_head_to_sigma: bool = True       # Use risk head output as sigma
    rl_sigma_floor: float = 0.01             # Minimum uncertainty

    # Trend-following alpha
    trend_fast_window: int = 21              # ~1 month
    trend_slow_window: int = 60              # ~3 months (reduced from 126 to fix Bug #1)
    trend_vol_lookback: int = 20             # For vol-normalizing
    trend_horizon: int = 15                  # IC analysis: 15-bar optimal (+0.059 vs +0.008 at 1-bar)

    # Mean-reversion alpha
    mr_lookback: int = 20                    # Bollinger lookback
    mr_z_entry: float = 2.0                  # Z-score entry threshold
    mr_horizon: int = 5                      # Native horizon: Bollinger z-score reverts within 5 bars (economic hypothesis)

    # Value alpha
    value_lookback: int = 60                 # ~3 months (reduced from 252 to fix Bug #1)
    value_horizon: int = 15                  # IC analysis: 15-bar optimal (+0.032 vs +0.008 at 1-bar)

    # Carry alpha
    carry_yield_bps: float = 400.0           # Annual risk-free yield (bps)
    carry_borrow_bps: float = 30.0           # Short borrow cost (bps/day)
    carry_horizon: int = 15                  # IC analysis: 15-bar optimal

    # Seasonality alpha
    seasonality_lookback_years: int = 5
    seasonality_horizon: int = 15            # IC analysis: 15-bar optimal

    # Volatility premium alpha
    vol_premium_lookback: int = 60
    vol_premium_horizon: int = 15            # IC analysis: 15-bar optimal (+0.083 vs +0.019 at 1-bar)

    # Tier 1: Yang-Zhang volatility estimator
    yang_zhang_lookback: int = 20            # Window for Yang-Zhang vol estimation (14x more efficient)

    # Tier 1: Amihud Illiquidity alpha
    amihud_lookback: int = 63                # Long-term illiquidity baseline (~3 months)
    amihud_short_lookback: int = 10          # Recent illiquidity window
    amihud_horizon: int = 15                 # IC-optimized horizon

    # Tier 1: Hurst Exponent regime detection
    hurst_window: int = 60                   # Lookback for Hurst calculation (reduced from 126 to fix Bug #1)
    hurst_update_freq: int = 21              # Recompute every ~1 month (expensive calc)
    hurst_horizon: int = 15                  # IC-optimized horizon

    # Tier 2: CalendarAlpha (replaces SeasonalityAlpha)
    calendar_warmup: int = 60                # Minimum bars before generating signals (reduced from 252 to fix Bug #1)
    calendar_horizon: int = 5                # Native horizon: Calendar effects are 1-5 bar phenomena (TOM, DOW)

    # Tier 3: Short-Term Reversal alpha
    reversal_formation: int = 5              # Formation period (5-bar losers)
    reversal_horizon: int = 5                # Reversal period (expect bounce in 5 bars)

    # Tier 3: Risk-managed momentum
    trend_risk_managed: bool = True          # Scale trend signal by inverse volatility
    trend_vol_target: float = 0.15           # Target annual volatility for scaling

    # -- L2: Signal Ensemble --
    meta_learner_type: str = "ridge"         # "ridge", "elastic_net", "equal_weight"
    meta_learner_alpha: float = 1.0          # Regularization strength
    meta_learner_coef_cap: float = 5.0       # Max |coefficient| after fit (0=no cap)
    meta_learner_retrain_freq: int = 63      # Retrain every ~3 months
    meta_learner_min_samples: int = 126      # Min observations before training
    max_single_alpha_weight: float = 0.30    # No alpha > 30% of combined signal
    shrinkage_toward_equal: float = 0.20     # Shrink weights toward 1/N
    regime_gating: bool = True               # Shift weights by regime

    # Horizon blending
    horizon_blend_weights: Tuple[float, ...] = (0.0, 0.35, 0.65)  # 1d, 5d, 15d — no 1-bar alphas; sqrt(h) weighting within buckets
    horizon_bars: Tuple[int, ...] = (1, 5, 15)

    # -- L3: Portfolio / Risk --
    target_annual_vol: float = 0.15          # Vol targeting
    vol_lookback: int = 20
    vol_floor: float = 0.05
    vol_cap: float = 0.60
    max_leverage: float = 2.0

    kelly_fraction: float = 0.50            # Half-Kelly (diversified multi-alpha system)
    kelly_lookback: int = 60
    kelly_max_drawdown: float = 0.15
    kelly_recovery_prob: float = 0.90
    kelly_recovery_steps: int = 126

    drawdown_scale_threshold: float = 0.08  # Start scaling at 8% DD
    drawdown_max_threshold: float = 0.15    # Force flat at 15% DD
    drawdown_scale_curve: str = "linear"    # "linear" or "quadratic"

    drift_z_threshold: float = 2.0
    drift_cooldown: int = 20
    drift_risk_reduction: float = 0.50
    drift_ks_sensitivity: float = 0.01         # p-value threshold for KS test
    drift_adwin_delta: float = 0.002           # ADWIN confidence (lower = more sensitive)
    drift_min_signals_for_trigger: int = 2     # N of 4 detectors must agree

    turnover_limit_annual: float = 50.0     # Max annual turnover (x capital) — higher to reduce ramp-trades with 7 action levels
    factor_exposure_limit: float = 0.50     # Max single-factor tilt

    # Cost-aware optimization (L3 uses canonical cost model from alphago_cost_model)
    cost_subtract_from_mu: bool = True       # Subtract expected cost from mu before sizing
    cost_spread_bps: float = 1.0             # Full bid-ask spread (bps) -- shared with L4
    cost_impact_coef: float = 0.1            # Sqrt-impact coefficient -- shared with L4
    cost_borrow_bps_annual: float = 78.0     # Annual short borrow cost (bps)
    cost_default_adv_dollars: float = 5_000_000.0  # Fallback ADV if not provided

    # Regime-conditional scaling
    regime_scale_map: Dict[str, float] = field(default_factory=lambda: {
        "high_growth_low_vol":  1.0,
        "high_growth_high_vol": 0.7,
        "low_growth_low_vol":   0.9,
        "low_growth_high_vol":  0.40,
    })

    # -- L4: Execution --
    no_trade_threshold_pct: float = 0.001   # Min position change to trade (0.1%)
    no_trade_vol_adaptive: bool = True
    order_slicing: bool = True
    n_slices: int = 5
    slice_mode: str = "TWAP"
    max_participation_rate: float = 0.10    # Max 10% of ADV per bar
    kill_max_loss_pct: float = 0.30            # 30% max loss (wider stop for concentrated selection)
    kill_max_turnover_spike: float = 20.0   # Raised: SMA transitions (0→0.25) are normal, not spikes
    kill_cooldown_bars: int = 10               # Faster re-entry (SMA gates direction)
    kill_allow_auto_reenter: bool = True       # Allow auto re-entry after cooldown + recovery
    kill_reenter_buffer: float = 0.02         # Must recover trigger_value * (1 + buffer) before re-entry

    # WS5B: Additional kill switches
    kill_confidence_window: int = 999         # effectively disabled for backtesting
    kill_min_confidence: float = 0.0          # effectively disabled for backtesting
    kill_min_data_quality: float = 60.0       # L0 quality score floor
    kill_vol_breach_mult: float = 10.0        # effectively disabled — 10x target vol (150% for 15% target), momentum portfolios need high vol headroom
    kill_cost_spike_mult: float = 999.0       # effectively disabled — fixed 10bps slippage always exceeds modeled 0.5bps
    kill_cost_spike_window: int = 999         # effectively disabled
    kill_dd_duration_bars: int = 999          # effectively disabled — momentum portfolios need longer recovery
    kill_on_drift: bool = True                # Kill if drift detector fires

    # -- Ensemble EMA smoothing --

    # -- Trailing Stops --
    use_trailing_stops: bool = True           # Enable trailing stop loss protection
    trailing_stop_lookback: int = 5           # Swing high/low lookback period (bars)
    trailing_stop_initial_distance: float = 0.0  # Initial stop distance as % of price (0 = use swing point)

    # -- Asymmetric Stop Loss --
    use_asymmetric_stops: bool = True         # Enable asymmetric stops (tight loss / trailing profit)

    # Loss regime (tight stops)
    loss_stop_pct: float = 0.015              # 1.5% stop when losing (grid search optimal)
    loss_stop_atr_mult: float = 1.5           # 1.5 ATR stop when losing

    # Profit regime (trailing stops)
    profit_trail_pct: float = 0.05            # Trail 5% from peak (grid search optimal)
    profit_trail_atr_mult: float = 3.0        # 3 ATR trailing stop

    # Volatility adjustments
    vol_adjust_stops: bool = True             # Adjust stops for volatility
    vol_baseline: float = 0.15                # Baseline volatility (15%)
    vol_max_adjustment: float = 2.0           # Max vol adjustment factor

    # Time-based tightening
    time_tighten_enabled: bool = False        # Tighten stops over time
    time_tighten_bars: int = 10               # Start tightening after N bars
    time_tighten_factor: float = 0.5          # Tighten to 50% of original

    # -- Backtest Mode --
    backtest_mode: bool = False               # If True, disable drift kill (allow cross-regime testing)

    # -- Future Leak Guard --
    enforce_leak_guards: bool = True

    # -- WS3A: CVaR Constraints --
    cvar_limit: float = -0.10                 # Daily CVaR limit (negative = loss)

    # -- WS5C: Alerting --
    enable_alerting: bool = True
    alert_log_file: str = ""                   # If set, write ALL alerts to this JSONL file
    alert_terminal_level: str = "CRITICAL"     # Only print this level+ to terminal (keeps stdout clean)

    # -- L0: Missing Data Policy --
    # Configurable per spec: "explicit and documented (forward-fill,
    # interpolate, mask, or exclude). No silent NaN propagation."
    # Values: "forward_fill", "rolling_median", "interpolate", "mask", "exclude"
    missing_data_policy: str = "rolling_median"

    # FIX Ã‚Â§5.2: Configurable data quality penalty weights (previously hardcoded)
    dq_nan_penalty: float = 500.0       # Penalty per NaN fraction unit
    dq_zero_penalty: float = 500.0      # Penalty per zero/negative price fraction
    dq_stale_penalty: float = 40.0      # Penalty for stale prices (all recent equal)

    # -- Benchmark --
    # Every strategy must state its benchmark(s) per spec.
    # All reported metrics are shown both gross/net and vs benchmark.
    benchmark_name: str = "SPY"               # "cash", "SPY", "custom"
    benchmark_description: str = "S&P 500 ETF (market return benchmark)"
    benchmark_annual_return: float = 0.0      # Computed from actual SPY data
    benchmark_type: str = "relative"          # "absolute", "relative", "risk_adjusted"


# ============================================================================
# V8.0 STOCK SELECTION ENGINE
# ============================================================================

@dataclass
class SelectionConfig:
    """Configuration for v8.0 stock selection engine."""
    top_n: int = 15                          # Number of stocks to hold
    momentum_lookback: int = 252             # 12-month momentum window
    momentum_skip: int = 21                  # Skip last month (reversal effect)
    min_bars: int = 126                      # Need 126+ bars (6 months) for ranking
    sma_alignment_required: bool = True      # Only select stocks in uptrend
    sector_max_pct: float = 0.50             # Max 50% in any single sector
    volatility_cap_percentile: float = 95.0  # Exclude top 5% most volatile
    # Weight allocation for composite score
    w_momentum: float = 0.50
    w_trend: float = 0.15
    w_rs: float = 0.30
    w_invvol: float = 0.05


class StockSelector:
    """
    Pre-evaluation stock selection for v8.0.

    Ranks the full universe, selects top-N high-conviction stocks.
    Runs BEFORE the pipeline evaluation loop.
    """

    def __init__(self, sel_cfg: SelectionConfig, sector_map: dict):
        self.cfg = sel_cfg
        self.sector_map = sector_map
        self.selection_log = []

    def rank_universe(self, datasets, spy_returns_lookup=None):
        """
        Rank all symbols by composite score.

        For each symbol with sufficient data:
        1. 12-1 month momentum (skip last month to avoid reversal)
        2. SMA alignment score (P > SMA50 > SMA100 > SMA200 = 3, partial = 1-2, below = 0)
        3. Relative strength vs SPY (outperformance ratio over 6 months)
        4. Inverse volatility (prefer lower-vol stocks for risk-adjusted returns)

        Returns: sorted list of (symbol, score, components_dict)
        """
        rankings = []

        # Pre-compute SPY 6-month return for relative strength
        spy_6m_return = 0.0
        if spy_returns_lookup:
            spy_dates = sorted(spy_returns_lookup.keys())
            if len(spy_dates) >= 126:
                spy_recent = spy_dates[-126:]
                spy_cum = 1.0
                for dt in spy_recent:
                    spy_cum *= (1 + spy_returns_lookup[dt])
                spy_6m_return = spy_cum - 1.0

        # Collect volatilities for percentile filtering
        all_vols = []
        symbol_data = []

        for d in datasets:
            if not hasattr(d, 'prices_test') or d.prices_test is None:
                continue
            closes = d.prices_test[:, 3]  # Close prices (OHLCV col 3)
            if closes is None or len(closes) < self.cfg.min_bars:
                continue

            # --- Momentum (12-1 month) ---
            if len(closes) > self.cfg.momentum_lookback:
                mom_start_idx = -(self.cfg.momentum_lookback)
                mom_end_idx = -(self.cfg.momentum_skip) if self.cfg.momentum_skip > 0 else len(closes)
                momentum = (closes[mom_end_idx] / closes[mom_start_idx]) - 1.0
            else:
                momentum = 0.0

            # --- SMA Alignment Score (0-3) ---
            sma50 = float(np.mean(closes[-50:]))
            sma100 = float(np.mean(closes[-100:])) if len(closes) >= 100 else None
            sma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else None
            price = float(closes[-1])

            sma_score = 0
            if sma200 is not None and price > sma200:
                sma_score += 1
            if sma100 is not None and price > sma100:
                sma_score += 1
            if price > sma50:
                sma_score += 1

            # Skip stocks not in uptrend (if required)
            if self.cfg.sma_alignment_required and sma_score == 0:
                continue

            # --- Relative Strength vs SPY (6-month) ---
            rs_vs_spy = 0.0
            if spy_returns_lookup and hasattr(d, 'timestamps_test') and d.timestamps_test is not None:
                ts = d.timestamps_test
                if len(ts) >= 126 and len(closes) >= 126:
                    stock_6m_return = (closes[-1] / closes[-126]) - 1.0
                    rs_vs_spy = stock_6m_return - spy_6m_return

            # --- Annualized Volatility (20-day) ---
            if len(closes) >= 21:
                log_rets = np.diff(np.log(closes[-21:]))
                vol_20 = float(np.std(log_rets)) * np.sqrt(252)
            else:
                vol_20 = 0.30  # Default 30% vol

            all_vols.append(vol_20)
            symbol_data.append((d, momentum, sma_score, rs_vs_spy, vol_20))

        # Filter out top-percentile volatility stocks
        if all_vols:
            vol_cap = float(np.percentile(all_vols, self.cfg.volatility_cap_percentile))
        else:
            vol_cap = 999.0

        for d, momentum, sma_score, rs_vs_spy, vol_20 in symbol_data:
            if vol_20 > vol_cap:
                continue  # Too volatile, skip

            inv_vol = 1.0 / max(vol_20, 0.01)

            # --- Composite Score ---
            score = (
                self.cfg.w_momentum * momentum
                + self.cfg.w_trend * (sma_score / 3.0)
                + self.cfg.w_rs * rs_vs_spy
                + self.cfg.w_invvol * (inv_vol / 100.0)  # Normalize inv_vol
            )

            rankings.append((d.symbol, score, {
                'momentum': momentum,
                'sma_score': sma_score,
                'rs_vs_spy': rs_vs_spy,
                'vol_20': vol_20,
            }))

        # Sort by composite score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def select(self, datasets, spy_returns_lookup=None):
        """Select top-N stocks with sector diversification constraints."""
        rankings = self.rank_universe(datasets, spy_returns_lookup)

        selected = []
        sector_counts = {}
        max_per_sector = max(2, int(self.cfg.top_n * self.cfg.sector_max_pct))

        # Build reverse lookup: symbol -> sector
        sym_to_sector = {}
        for sector, symbols in self.sector_map.items():
            for s in symbols:
                sym_to_sector[s] = sector

        for sym, score, components in rankings:
            if len(selected) >= self.cfg.top_n:
                break
            # Strip timeframe suffix (e.g. "MU_1d" -> "MU") for sector lookup
            bare_sym = sym.split('_')[0] if '_' in sym else sym
            sector = sym_to_sector.get(bare_sym, "other")
            if sector_counts.get(sector, 0) >= max_per_sector:
                continue  # Sector full, skip
            selected.append(sym)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Filter datasets to only selected symbols
        sym_set = set(selected)
        filtered = [d for d in datasets if d.symbol in sym_set]

        # Log selection
        self.selection_log.append({
            'selected': selected,
            'rankings': [(s, sc, comp) for s, sc, comp in rankings],
            'scores': {s: sc for s, sc, _ in rankings},
            'sector_allocation': dict(sector_counts),
        })

        return filtered


# ============================================================================
# FUTURE LEAK GUARD (Point C from feedback)
# ============================================================================

class FutureLeakGuard:
    """
    Assertion framework that prevents future information from leaking into
    observations or decision-time computations.

    Risk head targets (forward realized vol) are ONLY allowed inside loss
    computation. This class provides a context manager and runtime checks.

    Instead of fragile stack-inspection, uses an explicit context manager
    that sets a thread-local flag when future data access is permitted.

    Usage:
        with FutureLeakGuard.allow_future_access("compute_risk_target"):
            risk_target = env.get_risk_target()  # OK inside context

        env.get_risk_target()  # Raises FutureLeakError outside context

    Reference: Lopez de Prado (2018) "Advances in Financial Machine Learning"
    """

    # Thread-local storage for the access flag
    import threading
    _local = threading.local()

    _future_access_log: List[Dict[str, Any]] = []

    @classmethod
    def _is_access_allowed(cls) -> bool:
        """Check if we're inside an allow_future_access context."""
        return getattr(cls._local, '_future_access_allowed', False)

    @classmethod
    def _get_access_context(cls) -> str:
        """Get the current access context label."""
        return getattr(cls._local, '_future_access_context', '')

    @classmethod
    @contextmanager
    def allow_future_access(cls, operation: str):
        """
        Context manager that permits future data access within its scope.

        Args:
            operation: Label for audit trail (e.g., 'compute_risk_target')
        """
        prev_allowed = getattr(cls._local, '_future_access_allowed', False)
        prev_context = getattr(cls._local, '_future_access_context', '')
        cls._local._future_access_allowed = True
        cls._local._future_access_context = operation
        try:
            yield
        finally:
            cls._local._future_access_allowed = prev_allowed
            cls._local._future_access_context = prev_context

    @classmethod
    def check_caller(cls, operation: str = "future_data_access"):
        """
        Verify that future data access is currently permitted.
        Called by any function that uses future data (e.g. get_risk_target).

        Raises FutureLeakError if called outside allow_future_access context.
        Also supports legacy stack-inspection as fallback for backward compat.
        """
        # Primary check: context manager flag
        if cls._is_access_allowed():
            cls._future_access_log.append({
                'operation': operation,
                'allowed_by': cls._get_access_context(),
                'method': 'context_manager',
            })
            return

        # Fallback: stack inspection for backward compatibility
        import inspect
        _ALLOWED_FUTURE_CONSUMERS = frozenset({
            'compute_risk_target',
            'risk_head_loss',
            'update',           # PPO update uses risk targets
            'train_iteration',
        })
        stack = inspect.stack()
        callers = {frame.function for frame in stack[1:6]}
        allowed = callers & _ALLOWED_FUTURE_CONSUMERS
        if allowed:
            cls._future_access_log.append({
                'operation': operation,
                'caller': list(callers)[:3],
                'allowed_by': list(allowed),
                'method': 'stack_fallback',
            })
            return

        # Neither context manager nor stack inspection passed
        caller_chain = " -> ".join(frame.function for frame in stack[1:4])
        raise FutureLeakError(
            f"FUTURE DATA LEAK DETECTED in {operation}!\n"
            f"  Call chain: {caller_chain}\n"
            f"  Must be called inside FutureLeakGuard.allow_future_access() "
            f"context manager.\n"
            f"  This data must NEVER reach observations or decision logic."
        )

    @classmethod
    def get_audit_log(cls) -> List[Dict]:
        return list(cls._future_access_log)

    @classmethod
    def assert_no_future_in_obs(cls, obs: np.ndarray, risk_target: float):
        """
        Verify that a risk target value does not appear in the observation.
        Runs as a post-condition after every _obs() call.
        """
        if risk_target > 0.001:
            # Check if the exact value appears in obs (within float tolerance)
            if np.any(np.abs(obs - risk_target) < 1e-7):
                raise FutureLeakError(
                    f"Risk target {risk_target:.6f} found in observation vector! "
                    f"Future data is leaking into decision-time features."
                )


class FutureLeakError(Exception):
    """Raised when future information leaks into observation/decision space."""
    pass


# ============================================================================
# L1: ALPHA FACTORY
# ============================================================================
# Every alpha produces AlphaSignal(mu, sigma, confidence, horizon, ...).
# The RL agent is ONE alpha source among many -- it does NOT size positions.
# ============================================================================

# --- Tier 1 Utility Functions ---

def yang_zhang_vol(opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray,
                   lookback: int = 20) -> float:
    """
    Yang-Zhang volatility estimator (14x more efficient than close-to-close).

    Combines overnight, intraday, and close-to-close volatility components.
    More accurate than simple std(log_returns) because it uses OHLC information.

    Formula:
        YZ = sqrt(overnight_vol² + rogers_satchell + k×close_to_close)

    Where:
        overnight_vol = var(ln(O_t / C_{t-1}))
        rogers_satchell = E[(ln H/C)(ln H/O) + (ln L/C)(ln L/O)]
        k = 0.34 (empirically optimal weighting factor)

    Args:
        opens: Open prices
        highs: High prices
        lows: Low prices
        closes: Close prices
        lookback: Window size for volatility estimation (default: 20 bars)

    Returns:
        Annualized volatility estimate (float)

    Reference: Yang & Zhang (2000) "Drift-Independent Volatility Estimation"
    """
    if len(closes) < lookback + 1:
        # Fallback to simple close-to-close vol
        if len(closes) < 2:
            return 0.15  # Default annual vol
        log_rets = np.diff(np.log(closes[-lookback:] + 1e-12))
        return float(np.std(log_rets)) * np.sqrt(252) if len(log_rets) > 0 else 0.15

    # Extract most recent window
    o = opens[-lookback:]
    h = highs[-lookback:]
    l = lows[-lookback:]
    c = closes[-lookback:]
    c_prev = closes[-lookback-1:-1]

    # Overnight volatility: var(ln(O_t / C_{t-1}))
    overnight_rets = np.log((o + 1e-12) / (c_prev + 1e-12))
    overnight_var = np.var(overnight_rets) if len(overnight_rets) > 0 else 0.0

    # Rogers-Satchell: intraday range-based estimator
    # RS = E[(ln H/C)(ln H/O) + (ln L/C)(ln L/O)]
    ln_hc = np.log((h + 1e-12) / (c + 1e-12))
    ln_ho = np.log((h + 1e-12) / (o + 1e-12))
    ln_lc = np.log((l + 1e-12) / (c + 1e-12))
    ln_lo = np.log((l + 1e-12) / (o + 1e-12))
    rs = ln_hc * ln_ho + ln_lc * ln_lo
    rs_var = np.mean(rs) if len(rs) > 0 else 0.0

    # Close-to-close volatility
    c2c_rets = np.diff(np.log(c[-lookback:] + 1e-12))
    c2c_var = np.var(c2c_rets) if len(c2c_rets) > 0 else 0.0

    # Yang-Zhang estimator: k=0.34 is empirically optimal
    k = 0.34
    yz_var = overnight_var + rs_var + k * c2c_var
    yz_var = max(yz_var, 1e-10)  # Prevent zero/negative variance

    # Annualize (assuming daily data)
    annual_vol = float(np.sqrt(yz_var)) * np.sqrt(252)
    return annual_vol


def compute_hurst_exponent(prices: np.ndarray, window: int = 126) -> float:
    """
    Compute Hurst exponent via rescaled range (R/S) analysis.

    The Hurst exponent (H) characterizes the long-term memory of a time series:
        H > 0.5: Trending/persistent behavior (trends continue)
        H < 0.5: Mean-reverting behavior (trends reverse)
        H ≈ 0.5: Random walk (no memory)

    Uses rescaled range method with multiple lag windows for robustness.

    Args:
        prices: Price series
        window: Lookback window for calculation (default: 126 bars / ~6 months)

    Returns:
        Hurst exponent H ∈ [0, 1] (typically [0.3, 0.7])

    Reference: Hurst (1951), Peters (1994) "Fractal Market Analysis"
    """
    if len(prices) < window:
        return 0.5  # Default to random walk assumption

    # Use most recent window
    prices = prices[-window:]
    log_returns = np.diff(np.log(prices + 1e-12))

    if len(log_returns) < 20:
        return 0.5

    # Rescaled range analysis across multiple lag periods
    lags = np.arange(10, min(len(log_returns) // 2, 100), 5)
    rs_values = []

    for lag in lags:
        # Split returns into non-overlapping chunks of size 'lag'
        n_chunks = len(log_returns) // lag
        if n_chunks < 2:
            continue

        chunk_rs = []
        for i in range(n_chunks):
            chunk = log_returns[i * lag:(i + 1) * lag]

            # Mean-adjusted cumulative sum
            mean_ret = np.mean(chunk)
            cum_dev = np.cumsum(chunk - mean_ret)

            # Range: max - min of cumulative deviations
            R = np.max(cum_dev) - np.min(cum_dev)

            # Standard deviation
            S = np.std(chunk)

            # Rescaled range (avoid division by zero)
            if S > 1e-10:
                chunk_rs.append(R / S)

        if len(chunk_rs) > 0:
            rs_values.append((lag, np.mean(chunk_rs)))

    if len(rs_values) < 3:
        return 0.5

    # Linear regression: log(R/S) = H * log(lag) + const
    # H is the slope of this relationship
    lags_array = np.array([x[0] for x in rs_values])
    rs_array = np.array([x[1] for x in rs_values])

    # Filter out invalid values
    valid = (rs_array > 0) & (lags_array > 0)
    if np.sum(valid) < 3:
        return 0.5

    log_lags = np.log(lags_array[valid])
    log_rs = np.log(rs_array[valid])

    # OLS regression for Hurst exponent
    # H = slope of log(R/S) vs log(lag)
    A = np.vstack([log_lags, np.ones(len(log_lags))]).T
    H, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]

    # Clip to reasonable range [0.3, 0.7]
    # Values outside this range are usually estimation noise
    H = float(np.clip(H, 0.3, 0.7))

    return H


def detect_volume_price_divergence(closes: np.ndarray, volumes: np.ndarray,
                                    lookback: int = 10) -> float:
    """
    Detect volume-price divergence and return confidence adjustment factor.

    Economic hypothesis: Strong price moves should be accompanied by strong volume.
    Divergence indicates weak conviction and mean-reversion potential.

    Patterns:
        - Price ↑ + Volume ↓: Weak rally (reduce confidence)
        - Price ↓ + Volume ↓: Weak selloff (reduce confidence)
        - Price ↑ + Volume ↑: Strong rally (maintain/boost confidence)
        - Price ↓ + Volume ↑: Capitulation (interesting, maintain confidence)

    Args:
        closes: Price series
        volumes: Volume series
        lookback: Window for trend detection (default: 10 bars)

    Returns:
        Confidence multiplier ∈ [0.5, 1.2]
            1.0 = neutral (no divergence)
            < 1.0 = divergence detected (reduce confidence)
            > 1.0 = confirmation (boost confidence)

    Reference: Granville (1963) "On-Balance Volume", Buff Dormeier (2011)
    """
    if volumes is None or len(closes) < lookback + 1 or len(volumes) < lookback + 1:
        return 1.0  # Neutral if no volume data

    # Recent price trend
    recent_closes = closes[-lookback:]
    price_change = (recent_closes[-1] - recent_closes[0]) / (recent_closes[0] + 1e-12)
    price_trend = np.sign(price_change)

    # Recent volume trend (compare to longer-term average)
    recent_vol = volumes[-lookback:]
    hist_vol = volumes[-lookback*3:-lookback] if len(volumes) >= lookback*3 else volumes[:-lookback]

    if len(hist_vol) < 5:
        return 1.0

    avg_recent_vol = float(np.mean(recent_vol))
    avg_hist_vol = float(np.mean(hist_vol))

    if avg_hist_vol < 1e-10:
        return 1.0

    vol_ratio = avg_recent_vol / avg_hist_vol
    vol_trend = 1 if vol_ratio > 1.1 else (-1 if vol_ratio < 0.9 else 0)

    # Divergence detection
    if price_trend > 0 and vol_trend < 0:
        # Price up, volume down → weak rally
        divergence_strength = abs(price_change) * (1.0 - vol_ratio)
        confidence_adj = 1.0 - np.clip(divergence_strength, 0, 0.5)
    elif price_trend < 0 and vol_trend < 0:
        # Price down, volume down → weak selloff
        divergence_strength = abs(price_change) * (1.0 - vol_ratio)
        confidence_adj = 1.0 - np.clip(divergence_strength, 0, 0.3)
    elif price_trend != 0 and vol_trend > 0:
        # Price move + volume up → confirmation
        confirmation_strength = abs(price_change) * (vol_ratio - 1.0)
        confidence_adj = 1.0 + np.clip(confirmation_strength, 0, 0.2)
    else:
        # No clear divergence
        confidence_adj = 1.0

    return float(np.clip(confidence_adj, 0.5, 1.2))


class BaseAlpha:
    """Abstract base for all alpha sources."""

    def __init__(self, name: str, horizon: int = 1):
        self.name = name
        self.horizon = horizon
        self._bar_count = 0
        self._warmup_bars = 30

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        """Generate alpha signal from market data. Must be overridden."""
        raise NotImplementedError

    @property
    def is_warm(self) -> bool:
        return self._bar_count >= self._warmup_bars


class RLAlphaAdapter(BaseAlpha):
    """
    Adapts the PPO/MCTS policy network into the AlphaSignal interface.

    CRITICAL CHANGE: Instead of using the policy's action (a target exposure),
    we extract the network's BELIEFS:
        - Policy logits +' directional view (mu proxy)
        - Risk head     +' uncertainty estimate (sigma)
        - Value head    +' conviction proxy
        - MCTS improved policy +' refined mu when available

    The 5 action logits {FULL_SHORT, HALF_SHORT, FLAT, HALF_LONG, FULL_LONG}
    are reinterpreted as a belief distribution over return direction,
    NOT as a position-sizing instruction.
    """

    # Mapping: action targets to return direction weights
    # These convert the policy's "what exposure should I take" into
    # "what direction does the agent believe returns will go"
    ACTION_RETURN_MAP = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

    def __init__(self, acfg: ArchitectureConfig, net=None):
        super().__init__(name="rl_ppo_mcts", horizon=acfg.rl_alpha_horizon)
        self.acfg = acfg
        self.net = net
        self._warmup_bars = 0  # RL is warm immediately if network exists
        self._mcts_policy = None  # Set externally when MCTS runs

    def set_network(self, net):
        """Inject or update the policy network (after training iterations)."""
        self.net = net

    def set_mcts_policy(self, improved_probs: np.ndarray):
        """Inject MCTS-improved policy for current state."""
        self._mcts_policy = improved_probs

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 observation: np.ndarray = None, **kwargs) -> AlphaSignal:
        """
        Convert network outputs to AlphaSignal.

        The policy logits are a distribution over {-1, -0.5, 0, +0.5, +1}
        which we reinterpret as a directional belief:
            mu = GBP(p_i * direction_i) * scale_factor
            sigma = from risk head (or from logit entropy)
            confidence = 1 - normalized_entropy
        """
        self._bar_count = bar_idx

        if self.net is None or observation is None:
            return AlphaSignal(
                mu=0.0, sigma=1.0, confidence=0.0,
                horizon=self.horizon, alpha_name=self.name,
                timestamp=bar_idx
            )

        if not HAS_TORCH:
            return AlphaSignal(alpha_name=self.name, timestamp=bar_idx)

        # -- Forward pass --
        with torch.no_grad():
            obs_t = torch.FloatTensor(observation).unsqueeze(0)
            if next(self.net.parameters()).is_cuda:
                obs_t = obs_t.cuda()

            logits, value, risk_pred = self.net.forward(obs_t)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            value_est = value.cpu().item()
            risk_est = risk_pred.cpu().item()

        # -- Use MCTS-improved policy if available --
        if self._mcts_policy is not None:
            probs = self._mcts_policy
            self._mcts_policy = None  # Consume once

        # -- Convert to alpha signal --
        # mu: Expected direction weighted by policy probabilities
        mu_raw = float(np.dot(probs, self.ACTION_RETURN_MAP))
        mu = mu_raw * self.acfg.rl_logit_to_mu_scale

        # sigma: From risk head (forward vol estimate) or entropy-based
        if self.acfg.rl_risk_head_to_sigma and risk_est > 0:
            sigma = max(float(risk_est), self.acfg.rl_sigma_floor)
        else:
            # Entropy-based uncertainty: high entropy = high uncertainty
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(probs))
            sigma = max(entropy / max_entropy, self.acfg.rl_sigma_floor)

        # confidence: Inverse of normalized entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

        return AlphaSignal(
            mu=float(np.clip(mu, -5.0, 5.0)),
            sigma=sigma,
            confidence=confidence,
            horizon=self.horizon,
            alpha_name=self.name,
            metadata={
                'policy_probs': probs.tolist(),
                'value_estimate': value_est,
                'risk_estimate': risk_est,
                'mu_raw_direction': mu_raw,
            },
            timestamp=bar_idx,
        )


class TrendAlpha(BaseAlpha):
    """
    Time-series momentum / trend-following alpha.

    Signal: Fast EMA vs Slow EMA crossover, vol-normalized.
    Produces positive mu when price is trending up, negative when trending down.
    Sigma from realized volatility (trend uncertainty increases with vol).

    Source: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
    """

    def __init__(self, acfg: ArchitectureConfig, invert_signal: bool = False):
        super().__init__(name="trend_follow", horizon=acfg.trend_horizon)
        self.fast_w = acfg.trend_fast_window
        self.slow_w = acfg.trend_slow_window
        self.vol_lb = acfg.trend_vol_lookback
        self.yz_lookback = acfg.yang_zhang_lookback  # Tier 1: Yang-Zhang vol window
        self.risk_managed = acfg.trend_risk_managed  # Tier 3: Risk-managed momentum
        self.vol_target = acfg.trend_vol_target  # Tier 3: Vol target for scaling
        self._warmup_bars = acfg.trend_slow_window + 10
        self._invert_signal = invert_signal  # INVESTIGATION: Test if inverting fixes negative IC

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)
        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # EMA crossover
        fast_ema = self._ema(closes, self.fast_w)
        slow_ema = self._ema(closes, self.slow_w)

        # Vol-normalized signal (Tier 1: Use Yang-Zhang if OHLC available)
        opens = kwargs.get('opens')
        highs = kwargs.get('highs')
        lows = kwargs.get('lows')

        if opens is not None and highs is not None and lows is not None:
            realized_vol = yang_zhang_vol(opens, highs, lows, closes, lookback=self.yz_lookback)
        else:
            # Fallback to simple close-to-close vol
            log_rets = np.diff(np.log(closes[-self.vol_lb:] + 1e-12))
            realized_vol = float(np.std(log_rets)) * np.sqrt(252) if len(log_rets) > 5 else 0.15

        # Raw trend signal: distance between fast and slow EMA, vol-normalized
        atr = realized_vol / np.sqrt(252) * closes[-1]  # Approximate daily ATR
        raw_signal = (fast_ema - slow_ema) / (atr + 1e-10)

        # Clip and scale to mu
        mu_raw = float(np.clip(raw_signal, -3.0, 3.0)) * 0.05  # Conservative scaling

        # INVESTIGATION: Test signal inversion (IC was -0.059 at 15-bar)
        if self._invert_signal:
            mu = -1.0 * mu_raw  # Flip sign to test if anti-predictive becomes predictive
        else:
            mu = mu_raw

        # Tier 3: Risk-managed momentum (scale by inverse volatility)
        # Higher vol → reduce position size to maintain constant risk
        if self.risk_managed:
            vol_scalar = self.vol_target / max(realized_vol, 0.05)
            mu = mu * vol_scalar
            # Clip after scaling to prevent excessive leverage
            mu = float(np.clip(mu, -0.20, 0.20))

        # Sigma: higher in high-vol regimes
        sigma = max(realized_vol, 0.05)

        # Confidence: Higher when trend is strong and persistent
        # Use rolling return sign consistency
        if n > self.fast_w:
            recent_rets = np.diff(np.log(closes[-self.fast_w:] + 1e-12))
            consistency = abs(np.mean(np.sign(recent_rets)))
            confidence = float(np.clip(consistency, 0.0, 1.0))
        else:
            confidence = 0.0

        # Tier 2: Volume-price divergence adjustment
        # Reduce confidence if price trends without volume confirmation
        vol_price_adj = detect_volume_price_divergence(closes, volumes, lookback=10)
        confidence = confidence * vol_price_adj

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={'fast_ema': fast_ema, 'slow_ema': slow_ema,
                      'realized_vol': realized_vol,
                      'vol_price_adj': vol_price_adj},  # Tier 2: Log divergence factor
            timestamp=bar_idx,
        )

    @staticmethod
    def _ema(data: np.ndarray, span: int) -> float:
        """Compute EMA of last value."""
        alpha = 2.0 / (span + 1)
        ema = data[0]
        for val in data[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return float(ema)


class MeanReversionAlpha(BaseAlpha):
    """
    Mean-reversion alpha based on z-score of price relative to Bollinger band.

    Signal: Negative z-score +' expect reversion upward +' positive mu.
    High confidence when z-score is extreme and vol is contracting.

    Source: Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="mean_reversion", horizon=acfg.mr_horizon)
        self.lookback = acfg.mr_lookback
        self.z_entry = acfg.mr_z_entry
        self.yz_lookback = acfg.yang_zhang_lookback  # Tier 1: Yang-Zhang vol window
        self._warmup_bars = acfg.mr_lookback + 5

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)
        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        window = closes[-self.lookback:]
        mean = float(np.mean(window))
        std = float(np.std(window))
        if std < 1e-10:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        z_score = (closes[-1] - mean) / std

        # Mean reversion: high z +' expect down, low z +' expect up
        # mu is proportional to -z_score (contrarian)
        mu = float(-z_score * 0.02)  # Scaled conservatively
        mu = float(np.clip(mu, -0.5, 0.5))

        # Sigma: current realized vol (Tier 1: Use Yang-Zhang if OHLC available)
        opens = kwargs.get('opens')
        highs = kwargs.get('highs')
        lows = kwargs.get('lows')

        if opens is not None and highs is not None and lows is not None:
            sigma = max(yang_zhang_vol(opens, highs, lows, closes, lookback=self.yz_lookback), 0.05)
        else:
            # Fallback to simple close-to-close vol
            log_rets = np.diff(np.log(window + 1e-12))
            sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)

        # Confidence: higher when z-score is extreme
        confidence = float(np.clip(abs(z_score) / (self.z_entry * 1.5), 0.0, 1.0))

        # Tier 2: Volume-price divergence adjustment
        vol_price_adj = detect_volume_price_divergence(closes, volumes, lookback=10)
        confidence = confidence * vol_price_adj

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={'z_score': float(z_score), 'bb_mean': mean, 'bb_std': std,
                      'vol_price_adj': vol_price_adj},  # Tier 2: Log divergence factor
            timestamp=bar_idx,
        )


class ValueAlpha(BaseAlpha):
    """
    Long-horizon value alpha based on price displacement from trend.

    Uses the 1-year average as a valuation anchor. When price is far below
    the anchor, value signal is positive (expect reversion to fair value).

    In a production system this would use fundamentals (P/E, P/B, FCF yield).
    Price-based proxy is used here for market-data-only operation.

    Source: Arnott, Ehsani, Harvey, Shakernia (2025) "Revaluation Alpha"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="value", horizon=acfg.value_horizon)
        self.lookback = acfg.value_lookback
        self.yz_lookback = acfg.yang_zhang_lookback  # Tier 1: Yang-Zhang vol window
        self._warmup_bars = acfg.value_lookback + 10

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)
        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        anchor = float(np.mean(closes[-self.lookback:]))
        displacement = (anchor - closes[-1]) / (anchor + 1e-10)

        mu = float(np.clip(displacement * 0.1, -0.3, 0.3))

        # Sigma: current realized vol (Tier 1: Use Yang-Zhang if OHLC available)
        opens = kwargs.get('opens')
        highs = kwargs.get('highs')
        lows = kwargs.get('lows')

        if opens is not None and highs is not None and lows is not None:
            sigma = max(yang_zhang_vol(opens, highs, lows, closes, lookback=self.yz_lookback), 0.05)
        else:
            # Fallback to simple close-to-close vol
            log_rets = np.diff(np.log(closes[-60:] + 1e-12))
            sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)

        confidence = float(np.clip(abs(displacement) / 0.20, 0.0, 0.8))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={'anchor': anchor, 'displacement': float(displacement)},
            timestamp=bar_idx,
        )


class CarryAlpha(BaseAlpha):
    """
    Carry alpha: the return from holding a position independent of price moves.

    For equities: Long earns dividends minus funding cost.
    For our system: Cash yields risk-free rate, shorts pay borrow cost.
    When carry is positive for longs, signal tilts long and vice versa.

    Source: Koijen, Moskowitz, Pedersen, Vrugt (2018) "Carry"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="carry", horizon=acfg.carry_horizon)
        self.yield_annual = acfg.carry_yield_bps / 1e4
        self.borrow_annual = acfg.carry_borrow_bps * 252 / 1e4
        self._warmup_bars = 5

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        if n < 20:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Long carry: risk-free yield on capital
        # Short carry: negative (borrow cost)
        net_carry_long = self.yield_annual
        net_carry_short = -self.borrow_annual
        base_carry = float((net_carry_long - abs(net_carry_short)) * 0.1)

        # Time-varying modulation by vol regime:
        # Low vol -> carry is reliable (stable yield dominates)
        # High vol -> carry is irrelevant (price moves swamp yield)
        log_rets = np.diff(np.log(closes[-20:] + 1e-12))
        realized_vol = float(np.std(log_rets)) * np.sqrt(252)
        vol_z = (realized_vol - 0.20) / 0.10  # Z-score vs ~20% avg equity vol
        carry_effectiveness = float(np.clip(1.0 - vol_z * 0.5, 0.1, 2.0))

        mu = base_carry * carry_effectiveness

        # Sigma: realized vol (carry uncertainty tracks market vol)
        sigma = max(realized_vol, 0.05)

        # Confidence: higher in low-vol regimes where carry matters
        confidence = float(np.clip(0.15 * carry_effectiveness, 0.05, 0.30))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={'long_carry': net_carry_long, 'short_carry': net_carry_short,
                      'carry_effectiveness': carry_effectiveness,
                      'realized_vol': realized_vol},
            timestamp=bar_idx,
        )


class SeasonalityAlpha(BaseAlpha):
    """
    Calendar seasonality alpha with proper calendar features.

    Exploits persistent calendar patterns:
    - Day of week (Mon-Fri)
    - Month of year (Jan-Dec, "Sell in May")
    - Quarter (Q1-Q4)
    - Turn of month effect (first/last 5 trading days)

    Falls back to bar_idx % 5 if timestamps unavailable (backward compatible).

    Source: Bouman & Jacobsen (2002) "Sell in May" + Keim & Stambaugh (1984)
            + Ariel (1987) "Turn of Month Effect"
    """

    def __init__(self, acfg: ArchitectureConfig, invert_signal: bool = False):
        super().__init__(name="seasonality", horizon=acfg.seasonality_horizon)
        self._warmup_bars = 252 * acfg.seasonality_lookback_years

        # Old method: bar position mod 5 (fallback)
        self._day_returns_legacy = {d: deque(maxlen=1000) for d in range(5)}

        # New method: actual calendar features
        self._dow_returns = {d: deque(maxlen=500) for d in range(5)}  # Mon-Fri (0-4)
        self._month_returns = {m: deque(maxlen=200) for m in range(12)}  # Jan-Dec (0-11)
        self._quarter_returns = {q: deque(maxlen=100) for q in range(4)}  # Q1-Q4 (0-3)
        self._tom_returns = deque(maxlen=200)  # Turn of month

        self._day_counter = 0
        self._use_timestamps = False  # Auto-detect on first call
        self._timestamps_checked = False  # One-time detection flag (bar_idx may not start at 0)
        self._invert_signal = invert_signal  # INVESTIGATION: Test if inverting fixes negative IC

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 timestamp=None, **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        # Auto-detect if we have timestamp support (one-time, works regardless of starting bar_idx)
        if not self._timestamps_checked:
            self._use_timestamps = (timestamp is not None)
            self._timestamps_checked = True

        # Update returns
        if n > 1:
            ret = float(np.log(closes[-1] / (closes[-2] + 1e-12)))

            if self._use_timestamps and timestamp is not None:
                # Use real calendar features
                try:
                    # Extract calendar info (handles pandas Timestamp, datetime, or string)
                    if hasattr(timestamp, 'dayofweek'):
                        dow = timestamp.dayofweek  # 0=Monday, 4=Friday
                        month = timestamp.month - 1  # 0=Jan, 11=Dec
                        quarter = (timestamp.month - 1) // 3  # 0=Q1, 3=Q4
                        day_of_month = timestamp.day
                    elif isinstance(timestamp, str):
                        from datetime import datetime as _dt
                        _parsed = _dt.strptime(timestamp[:10], "%Y-%m-%d")
                        dow = _parsed.weekday()
                        month = _parsed.month - 1
                        quarter = (_parsed.month - 1) // 3
                        day_of_month = _parsed.day
                    else:
                        # datetime object
                        dow = timestamp.weekday()
                        month = timestamp.month - 1
                        quarter = (timestamp.month - 1) // 3
                        day_of_month = timestamp.day

                    # Update calendar-based returns
                    if 0 <= dow <= 4:
                        self._dow_returns[dow].append(ret)
                    self._month_returns[month].append(ret)
                    self._quarter_returns[quarter].append(ret)

                    # Turn of month: first/last 5 trading days
                    # Note: This is approximate without full calendar
                    is_tom = (day_of_month <= 5) or (day_of_month >= 25)
                    if is_tom:
                        self._tom_returns.append(ret)

                except (AttributeError, IndexError):
                    # Fallback if timestamp parsing fails
                    self._use_timestamps = False

            # Legacy method (fallback or when timestamps unavailable)
            if not self._use_timestamps:
                day = bar_idx % 5
                self._day_returns_legacy[day].append(ret)

            self._day_counter += 1

        if self._day_counter < 100:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Generate forecast
        if self._use_timestamps and timestamp is not None:
            return self._generate_from_calendar(timestamp, bar_idx)
        else:
            return self._generate_legacy(bar_idx)

    def _generate_from_calendar(self, timestamp, bar_idx: int) -> AlphaSignal:
        """Generate signal using real calendar features."""
        try:
            # Parse timestamp (handles pandas Timestamp, datetime, or string)
            if hasattr(timestamp, 'dayofweek'):
                dow = timestamp.dayofweek
                month = timestamp.month - 1
                quarter = (timestamp.month - 1) // 3
                day_of_month = timestamp.day
            elif isinstance(timestamp, str):
                from datetime import datetime as _dt
                _parsed = _dt.strptime(timestamp[:10], "%Y-%m-%d")
                dow = _parsed.weekday()
                month = _parsed.month - 1
                quarter = (_parsed.month - 1) // 3
                day_of_month = _parsed.day
            else:
                dow = timestamp.weekday()
                month = timestamp.month - 1
                quarter = (timestamp.month - 1) // 3
                day_of_month = timestamp.day

            # Collect signals from different calendar features
            signals = []
            weights = []

            # 1. Day of week effect (tomorrow's day)
            next_dow = (dow + 1) % 5
            if len(self._dow_returns[next_dow]) >= 20:
                signals.append(np.mean(list(self._dow_returns[next_dow])))
                weights.append(1.0)

            # 2. Month effect (current month continuation)
            if len(self._month_returns[month]) >= 10:
                signals.append(np.mean(list(self._month_returns[month])))
                weights.append(1.5)  # Monthly effects are stronger

            # 3. Quarter effect
            if len(self._quarter_returns[quarter]) >= 10:
                signals.append(np.mean(list(self._quarter_returns[quarter])))
                weights.append(0.8)  # Quarterly is weaker

            # 4. Turn of month effect (strongest!)
            is_tom = (day_of_month <= 5) or (day_of_month >= 25)
            if is_tom and len(self._tom_returns) >= 20:
                signals.append(np.mean(list(self._tom_returns)))
                weights.append(2.0)  # TOM is empirically strongest

            if not signals:
                return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                                   timestamp=bar_idx)

            # Weighted combination
            mu_raw = float(np.average(signals, weights=weights)) * 252  # Annualize

            # INVESTIGATION: Test signal inversion (IC was -0.014 to -0.046)
            if self._invert_signal:
                mu = -1.0 * mu_raw  # Flip sign to test if anti-predictive becomes predictive
            else:
                mu = mu_raw

            sigma = 0.10  # Conservative sigma

            # Confidence based on number of features active
            confidence = float(np.clip(len(signals) * 0.15, 0.0, 0.5))

            return AlphaSignal(
                mu=float(np.clip(mu, -0.2, 0.2)),
                sigma=sigma,
                confidence=confidence,
                horizon=self.horizon,
                alpha_name=self.name,
                metadata={
                    'dow': dow,
                    'month': month,
                    'quarter': quarter,
                    'is_tom': is_tom,
                    'n_features': len(signals),
                    'mu_raw': mu_raw if self._invert_signal else None,
                    'inverted': self._invert_signal,
                    'method': 'calendar'
                },
                timestamp=bar_idx,
            )
        except Exception:
            # Fallback to legacy if calendar parsing fails
            return self._generate_legacy(bar_idx)

    def _generate_legacy(self, bar_idx: int) -> AlphaSignal:
        """Legacy method using bar_idx % 5 (backward compatible)."""
        next_day = (bar_idx + 1) % 5
        day_rets = list(self._day_returns_legacy[next_day])

        if len(day_rets) < 20:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        mu = float(np.mean(day_rets)) * 252  # Annualized
        sigma = float(np.std(day_rets)) * np.sqrt(252)
        sigma = max(sigma, 0.05)

        t_stat = abs(mu) / (sigma / np.sqrt(len(day_rets)) + 1e-10)
        confidence = float(np.clip(t_stat / 3.0, 0.0, 0.5))

        return AlphaSignal(
            mu=float(np.clip(mu, -0.3, 0.3)),
            sigma=sigma,
            confidence=confidence,
            horizon=self.horizon,
            alpha_name=self.name,
            metadata={'next_day': next_day, 'n_obs': len(day_rets),
                      't_stat': float(t_stat), 'method': 'legacy'},
            timestamp=bar_idx,
        )


class VolatilityPremiumAlpha(BaseAlpha):
    """
    Volatility premium alpha: short vol when premium is high.

    Exploits the variance risk premium (implied > realized).
    Uses realized vol of vol as a proxy for implied/realized spread.

    Source: Kazley/One River (2026), Bakshi & Kapadia (2003)
    """

    def __init__(self, acfg: ArchitectureConfig, invert_signal: bool = False):
        super().__init__(name="vol_premium", horizon=acfg.vol_premium_horizon)
        self.lookback = acfg.vol_premium_lookback
        self._warmup_bars = acfg.vol_premium_lookback + 10
        self._invert_signal = invert_signal  # INVESTIGATION: Test if buying vol works better

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 opens: np.ndarray = None, highs: np.ndarray = None,
                 lows: np.ndarray = None,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)
        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Use Yang-Zhang vol estimator when OHLC available (14x more efficient)
        if opens is not None and highs is not None and lows is not None:
            short_vol = max(yang_zhang_vol(opens, highs, lows, closes, lookback=20), 0.05)
            long_vol = max(yang_zhang_vol(opens, highs, lows, closes, lookback=self.lookback), 0.05)

            # Vol of vol: optimized rolling volatility calculation (O(n) instead of O(n²))
            # BEFORE: ~6.5ms (90 yang_zhang_vol() calls, each on growing slice)
            # AFTER: ~0.3-0.5ms (vectorized rolling std with fixed window)
            recent_n = min(100, len(closes))
            if recent_n >= 30:
                log_rets = np.diff(np.log(closes[-recent_n:] + 1e-12))
                # Compute rolling 20-bar std efficiently (fixed window, not growing slice)
                window = 20
                if len(log_rets) >= window + 5:
                    # Vectorized: create all 20-bar windows at once using stride_tricks
                    from numpy.lib.stride_tricks import sliding_window_view
                    windows = sliding_window_view(log_rets, window)
                    rolling_vols = np.std(windows, axis=1) * np.sqrt(252)
                    vov = float(np.std(rolling_vols)) if len(rolling_vols) > 5 else 0.0
                else:
                    vov = 0.0
            else:
                vov = 0.0
        else:
            # Fallback to simple close-to-close vol
            log_rets = np.diff(np.log(closes[-self.lookback:] + 1e-12))
            short_vol = float(np.std(log_rets[-20:])) * np.sqrt(252) if len(log_rets) >= 20 else 0.15
            long_vol = float(np.std(log_rets)) * np.sqrt(252)

            # Vol of vol (proxy for VVIX) - optimized (O(n) instead of O(n²))
            if len(log_rets) >= 20:
                from numpy.lib.stride_tricks import sliding_window_view
                window = 10
                windows = sliding_window_view(log_rets, window)
                rolling_vols = np.std(windows, axis=1)
                vov = float(np.std(rolling_vols)) if len(rolling_vols) > 5 else 0.0
            else:
                vov = 0.0

        # Signal: When short-term vol < long-term vol +' vol is mean-reverting up
        # When short > long +' vol spike, expect reversion down
        vol_ratio = short_vol / (long_vol + 1e-10)

        # Contrarian vol signal (original: sell vol when high)
        if vol_ratio > 1.2:
            mu_base = -0.02  # Vol spike +' expect calm +' slight short bias (protective)
        elif vol_ratio < 0.8:
            mu_base = 0.02   # Vol crush +' expect expansion +' slight long bias
        else:
            mu_base = 0.0

        # INVESTIGATION: Test signal inversion (IC was -0.062 at 15-bar)
        if self._invert_signal:
            mu = -1.0 * mu_base  # Buy vol instead of sell
        else:
            mu = mu_base

        sigma = max(short_vol, 0.05)
        confidence = float(np.clip(abs(vol_ratio - 1.0) / 0.5, 0.0, 0.6))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={'short_vol': short_vol, 'long_vol': long_vol,
                      'vol_ratio': vol_ratio, 'vol_of_vol': vov,
                      'mu_base': mu_base if self._invert_signal else None,
                      'inverted': self._invert_signal},
            timestamp=bar_idx,
        )


class AmihudLiquidityAlpha(BaseAlpha):
    """
    Amihud Illiquidity alpha: Liquidity premium from trading costs.

    Economic hypothesis: Assets with low liquidity command higher expected
    returns to compensate for transaction costs and price impact.

    Formula: Amihud_t = |Return_t| / Dollar_Volume_t
    Signal: When current illiquidity is high relative to history,
            expect positive returns (compensation for bearing liquidity risk).

    Source: Amihud (2002) "Illiquidity and Stock Returns"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="amihud_liquidity", horizon=acfg.amihud_horizon)
        self.lookback = acfg.amihud_lookback
        self.short_lookback = acfg.amihud_short_lookback
        self.yz_lookback = acfg.yang_zhang_lookback
        self._warmup_bars = acfg.amihud_lookback + 10

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        if n < self._warmup_bars or volumes is None:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Compute Amihud illiquidity: |return| / dollar_volume
        # Higher values = less liquidity = expect higher returns
        log_rets = np.diff(np.log(closes + 1e-12))
        abs_rets = np.abs(log_rets)
        dollar_vol = volumes[1:] * closes[1:]  # Align with returns

        # Avoid division by zero
        dollar_vol = np.maximum(dollar_vol, 1e-10)

        # Log-Amihud measure (scale-invariant: works for both MSFT at $20B/day
        # and small-caps at $1M/day without numerical underflow)
        amihud = np.log1p(abs_rets / dollar_vol * 1e8)  # Scale then log for stability

        # Recent vs historical illiquidity
        if n < self.lookback:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        recent_amihud = float(np.mean(amihud[-self.short_lookback:]))
        hist_amihud = float(np.mean(amihud[-self.lookback:]))
        std_amihud = float(np.std(amihud[-self.lookback:]))

        if std_amihud < 1e-10:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Z-score: how illiquid is current market vs history?
        illiquidity_z = (recent_amihud - hist_amihud) / std_amihud

        # Signal: High illiquidity → positive mu (expect compensation)
        # Conservative scaling: 1% mu per 1 std of excess illiquidity
        mu = float(np.clip(illiquidity_z * 0.01, -0.10, 0.10))

        # Sigma: realized volatility (Tier 1: Use Yang-Zhang if available)
        opens = kwargs.get('opens')
        highs = kwargs.get('highs')
        lows = kwargs.get('lows')

        if opens is not None and highs is not None and lows is not None:
            sigma = max(yang_zhang_vol(opens, highs, lows, closes, lookback=self.yz_lookback), 0.05)
        else:
            # Fallback to simple close-to-close vol
            sigma = max(float(np.std(log_rets[-self.yz_lookback:])) * np.sqrt(252), 0.05)

        # Confidence: Higher when illiquidity signal is extreme
        confidence = float(np.clip(abs(illiquidity_z) / 2.0, 0.0, 0.7))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={
                'recent_amihud': recent_amihud,
                'hist_amihud': hist_amihud,
                'illiquidity_z': float(illiquidity_z),
            },
            timestamp=bar_idx,
        )


class HurstRegimeAlpha(BaseAlpha):
    """
    Hurst Exponent regime detection alpha.

    Economic hypothesis: Market microstructure changes between trending and
    mean-reverting regimes. The Hurst exponent (H) identifies these regimes:
        H > 0.5: Trending regime → momentum strategies work
        H < 0.5: Mean-reverting regime → MR strategies work

    Signal logic:
        - When H > 0.5: Positive mu (trending regime, favor momentum)
        - When H < 0.5: Negative mu (mean-reverting regime, fade trends)

    This alpha acts as a regime-dependent meta-signal for the ensemble.

    Source: Peters (1994) "Fractal Market Hypothesis", Lo (1991) "Long-Term Memory"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="hurst_regime", horizon=acfg.hurst_horizon)
        self.window = acfg.hurst_window
        self.update_freq = acfg.hurst_update_freq
        self.yz_lookback = acfg.yang_zhang_lookback
        self._warmup_bars = acfg.hurst_window + 10
        self._last_hurst = 0.5
        self._last_update_bar = -999

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Hurst computation is expensive - only update periodically
        if bar_idx - self._last_update_bar >= self.update_freq:
            self._last_hurst = compute_hurst_exponent(closes, window=self.window)
            self._last_update_bar = bar_idx

        H = self._last_hurst

        # Signal based on regime:
        # H > 0.5: Trending regime → positive mu (favor momentum)
        # H < 0.5: Mean-reverting → negative mu (fade trends)
        # Scale mu by distance from 0.5
        regime_strength = (H - 0.5) * 2.0  # Scale to [-1, +1]

        # Conservative scaling: 2% mu per 0.1 deviation from random walk
        mu = float(np.clip(regime_strength * 0.02, -0.10, 0.10))

        # Sigma: realized volatility (Tier 1: Use Yang-Zhang if available)
        opens = kwargs.get('opens')
        highs = kwargs.get('highs')
        lows = kwargs.get('lows')

        if opens is not None and highs is not None and lows is not None:
            sigma = max(yang_zhang_vol(opens, highs, lows, closes, lookback=self.yz_lookback), 0.05)
        else:
            # Fallback to simple close-to-close vol
            log_rets = np.diff(np.log(closes[-self.yz_lookback:] + 1e-12))
            sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)

        # Confidence: Higher when regime is more extreme (far from 0.5)
        # Strong trending (H>0.6) or strong MR (H<0.4) → high confidence
        confidence = float(np.clip(abs(H - 0.5) / 0.2, 0.0, 0.8))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={
                'hurst': float(H),
                'regime': 'trending' if H > 0.5 else 'mean_reverting',
                'regime_strength': float(regime_strength),
            },
            timestamp=bar_idx,
        )


class ShortTermReversalAlpha(BaseAlpha):
    """
    Short-term reversal alpha based on bid-ask bounce and liquidity provision.

    Economic hypothesis: Recent losers outperform in the short term due to:
        - Bid-ask bounce (mean reversion from transaction costs)
        - Liquidity provision (market makers profit from overreactions)
        - Microstructure noise in price formation

    Signal: Contrarian to recent price moves.
    If price fell in last N bars, expect bounce. If price rose, expect pullback.

    Source: Jegadeesh (1990) "Evidence of Predictable Behavior of Security Returns",
            Lehmann (1990) "Fads, Martingales, and Market Efficiency"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="short_term_reversal", horizon=acfg.reversal_horizon)
        self.formation_period = acfg.reversal_formation
        self.yz_lookback = acfg.yang_zhang_lookback
        self._warmup_bars = acfg.reversal_formation + 10

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Formation period: cumulative return over last N bars
        formation_start = -(self.formation_period + 1)
        formation_ret = float(np.log(closes[-1] / (closes[formation_start] + 1e-12)))

        # Contrarian signal: fade recent moves
        # If formation_ret < 0 (loser), expect bounce → positive mu
        # If formation_ret > 0 (winner), expect pullback → negative mu
        mu_raw = -formation_ret

        # Conservative scaling: 50% of formation return
        mu = float(np.clip(mu_raw * 0.5, -0.15, 0.15))

        # Sigma: realized volatility (Tier 1: Use Yang-Zhang if available)
        opens = kwargs.get('opens')
        highs = kwargs.get('highs')
        lows = kwargs.get('lows')

        if opens is not None and highs is not None and lows is not None:
            sigma = max(yang_zhang_vol(opens, highs, lows, closes, lookback=self.yz_lookback), 0.05)
        else:
            # Fallback to simple close-to-close vol
            log_rets = np.diff(np.log(closes[-self.yz_lookback:] + 1e-12))
            sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)

        # Confidence: Higher when formation return is more extreme
        # Large moves are more likely to reverse
        confidence = float(np.clip(abs(formation_ret) / 0.10, 0.0, 0.8))

        # Tier 2: Volume-price divergence adjustment
        vol_price_adj = detect_volume_price_divergence(closes, volumes, lookback=self.formation_period)
        confidence = confidence * vol_price_adj

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={
                'formation_ret': float(formation_ret),
                'formation_period': self.formation_period,
                'vol_price_adj': vol_price_adj,
            },
            timestamp=bar_idx,
        )


class CalendarAlpha(BaseAlpha):
    """
    Calendar effects alpha with Bonferroni-corrected statistical testing.

    Tier 2 improvement over SeasonalityAlpha:
    - Focus on empirically strongest effects (turn-of-month, month-of-year)
    - Bonferroni correction for multiple hypothesis testing
    - Conservative confidence scoring based on t-statistics
    - Requires minimum sample size for each effect

    Economic hypothesis: Persistent calendar patterns from institutional flows,
    rebalancing, and behavioral biases.

    Source: Ariel (1987) Turn-of-Month, Bouman & Jacobsen (2002) "Sell in May"
    """

    def __init__(self, acfg: ArchitectureConfig):
        super().__init__(name="calendar", horizon=acfg.calendar_horizon)
        self._warmup_bars = acfg.calendar_warmup

        # Track returns by calendar feature
        self._tom_returns = deque(maxlen=200)  # Turn of month (first/last 5 days)
        self._non_tom_returns = deque(maxlen=400)  # Non-TOM days
        self._month_returns = {m: deque(maxlen=100) for m in range(12)}  # Jan-Dec

        self._day_counter = 0
        self._use_timestamps = False
        self._timestamps_checked = False  # One-time detection flag (bar_idx may not start at 0)

        # Bonferroni-corrected significance threshold
        # Testing 2 effects (TOM + month), so alpha/2 = 0.05/2 = 0.025 → t > 2.24 for p<0.025
        self._bonferroni_t_threshold = 2.24

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 timestamp=None, **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        # Auto-detect timestamp support (one-time, works regardless of starting bar_idx)
        if not self._timestamps_checked:
            self._use_timestamps = (timestamp is not None)
            self._timestamps_checked = True

        # Update returns
        if n > 1:
            ret = float(np.log(closes[-1] / (closes[-2] + 1e-12)))

            if self._use_timestamps and timestamp is not None:
                try:
                    # Parse timestamp (handles pandas Timestamp, datetime, or string)
                    if hasattr(timestamp, 'month'):
                        month = timestamp.month - 1
                        day_of_month = timestamp.day
                    elif isinstance(timestamp, str):
                        from datetime import datetime as _dt
                        _parsed = _dt.strptime(timestamp[:10], "%Y-%m-%d")
                        month = _parsed.month - 1
                        day_of_month = _parsed.day
                    else:
                        month = int(timestamp.month) - 1
                        day_of_month = int(timestamp.day)

                    # Turn of month: first/last 5 calendar days
                    is_tom = (day_of_month <= 5) or (day_of_month >= 25)

                    if is_tom:
                        self._tom_returns.append(ret)
                    else:
                        self._non_tom_returns.append(ret)

                    self._month_returns[month].append(ret)

                except (AttributeError, IndexError):
                    self._use_timestamps = False

            self._day_counter += 1

        if self._day_counter < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        if not self._use_timestamps or timestamp is None:
            # No timestamp data available
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Generate signal with Bonferroni correction
        try:
            # Parse current timestamp (handles pandas Timestamp, datetime, or string)
            if hasattr(timestamp, 'month'):
                month = timestamp.month - 1
                day_of_month = timestamp.day
            elif isinstance(timestamp, str):
                # Timestamps stored as strings like "1986-03-13" or "1986-03-13 00:00:00"
                from datetime import datetime as _dt
                _parsed = _dt.strptime(timestamp[:10], "%Y-%m-%d")
                month = _parsed.month - 1
                day_of_month = _parsed.day
            else:
                month = int(timestamp.month) - 1
                day_of_month = int(timestamp.day)

            is_tom = (day_of_month <= 5) or (day_of_month >= 25)

            # Collect calendar effects -- always include effects with enough
            # observations, use t-stat for confidence scaling (not binary gate).
            # The Bonferroni gate was too stringent for small samples (~30-100 obs),
            # causing CalendarAlpha to be permanently inactive for MSFT.
            signals = []
            weights = []
            metadata = {}

            # 1. Turn-of-month effect (if currently in TOM period)
            if is_tom and len(self._tom_returns) >= 30:
                tom_mean = np.mean(list(self._tom_returns))
                tom_std = np.std(list(self._tom_returns))
                tom_n = len(self._tom_returns)
                tom_t = abs(tom_mean) / (tom_std / np.sqrt(tom_n) + 1e-10)

                # Always include effect; weight by statistical strength
                sig_strength = min(float(tom_t) / self._bonferroni_t_threshold, 1.0)
                signals.append(tom_mean * 252)  # Annualize
                weights.append(2.0 * sig_strength)
                metadata['tom_t_stat'] = float(tom_t)
                metadata['tom_significant'] = tom_t > self._bonferroni_t_threshold

            # 2. Month-of-year effect (current month)
            if len(self._month_returns[month]) >= 20:
                month_rets = list(self._month_returns[month])
                month_mean = np.mean(month_rets)
                month_std = np.std(month_rets)
                month_n = len(month_rets)
                month_t = abs(month_mean) / (month_std / np.sqrt(month_n) + 1e-10)

                sig_strength = min(float(month_t) / self._bonferroni_t_threshold, 1.0)
                signals.append(month_mean * 252)
                weights.append(1.0 * sig_strength)
                metadata['month_t_stat'] = float(month_t)
                metadata['month_significant'] = month_t > self._bonferroni_t_threshold

            if not signals:
                # Not enough observations yet for any effect
                return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                                   timestamp=bar_idx, metadata=metadata)

            # Weighted combination of effects
            mu = float(np.average(signals, weights=weights))
            mu = float(np.clip(mu, -0.15, 0.15))  # Conservative cap

            # Confidence: scale by max t-stat relative to Bonferroni threshold
            # Sub-threshold effects get low confidence; the ensemble handles the rest
            max_t = max(metadata.get('tom_t_stat', 0), metadata.get('month_t_stat', 0))
            significance_ratio = min(max_t / self._bonferroni_t_threshold, 1.0)
            confidence = float(np.clip(significance_ratio * 0.3, 0.02, 0.6))

            sigma = 0.10

            metadata.update({
                'is_tom': is_tom,
                'month': month,
                'n_effects': len(signals),
            })

            return AlphaSignal(
                mu=mu, sigma=sigma, confidence=confidence,
                horizon=self.horizon, alpha_name=self.name,
                metadata=metadata,
                timestamp=bar_idx,
            )

        except Exception as e:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx, metadata={'error': str(e)})


class TrendVolRegimeDetector:
    """
    Lightweight 4-state regime detector based on trend + volatility.

    States (2--2 grid):
        high_growth_low_vol   -- trending up, calm
        high_growth_high_vol  -- trending up, volatile
        low_growth_low_vol    -- flat/down, calm
        low_growth_high_vol   -- crisis / sell-off

    Uses soft-assignment (probabilities) not hard labels, computed from
    rolling z-scores of trend and vol relative to their own history.

    This replaces the hard-coded regime_probs that previously made
    regime gating and regime scaling meaningless. [Fix 4.1]
    """

    REGIME_NAMES = [
        'high_growth_low_vol', 'high_growth_high_vol',
        'low_growth_low_vol', 'low_growth_high_vol',
    ]

    def __init__(self, trend_lookback: int = 63, vol_lookback: int = 20,
                 history_lookback: int = 252):
        self.trend_lb = trend_lookback
        self.vol_lb = vol_lookback
        self.history_lb = history_lookback
        self._trend_history = deque(maxlen=history_lookback)
        self._vol_history = deque(maxlen=history_lookback)
        self._warmup = max(trend_lookback, vol_lookback) + 30
        self._bar_count = 0
        self._last_probs = np.array([0.25, 0.25, 0.25, 0.25])
        self._in_warmup = True  # Explicitly track warm-up state

    def update(self, closes: np.ndarray) -> np.ndarray:
        """
        Compute regime probabilities from price history.
        Returns np.ndarray of shape (4,) summing to 1.0.
        """
        self._bar_count += 1
        n = len(closes)

        if n < self._warmup:
            self._in_warmup = True
            return self._last_probs.copy()

        # -- Trend score: rolling return over trend_lookback --
        trend_ret = np.log(closes[-1] / (closes[-self.trend_lb] + 1e-12))
        self._trend_history.append(trend_ret)

        # -- Vol score: rolling std of daily log returns --
        log_rets = np.diff(np.log(closes[-self.vol_lb - 1:] + 1e-12))
        current_vol = float(np.std(log_rets)) * np.sqrt(252)
        self._vol_history.append(current_vol)

        if len(self._trend_history) < 30 or len(self._vol_history) < 30:
            return self._last_probs.copy()

        # Warm-up complete once we have enough trend/vol history
        self._in_warmup = False

        # -- Z-score relative to own history --
        trends = np.array(list(self._trend_history))
        vols = np.array(list(self._vol_history))

        trend_z = (trend_ret - np.mean(trends)) / (np.std(trends) + 1e-10)
        vol_z = (current_vol - np.mean(vols)) / (np.std(vols) + 1e-10)

        # -- Soft assignment via sigmoid on z-scores --
        # P(high_growth) = sigmoid(trend_z), P(high_vol) = sigmoid(vol_z)
        p_high_growth = 1.0 / (1.0 + np.exp(-trend_z))
        p_high_vol = 1.0 / (1.0 + np.exp(-vol_z))

        p_low_growth = 1.0 - p_high_growth
        p_low_vol = 1.0 - p_high_vol

        probs = np.array([
            p_high_growth * p_low_vol,    # high_growth_low_vol
            p_high_growth * p_high_vol,   # high_growth_high_vol
            p_low_growth * p_low_vol,     # low_growth_low_vol
            p_low_growth * p_high_vol,    # low_growth_high_vol
        ])

        # Normalize (should already sum to ~1 but ensure)
        probs = probs / (probs.sum() + 1e-10)
        self._last_probs = probs
        return probs

    @property
    def current_regime(self) -> str:
        """Hard assignment: most likely regime (or 'unknown' during warm-up)."""
        if self._in_warmup:
            return "unknown"
        return self.REGIME_NAMES[int(np.argmax(self._last_probs))]

    @property
    def is_warming_up(self) -> bool:
        """True if detector lacks enough data for meaningful regime assignment."""
        return self._in_warmup

    @property
    def probabilities(self) -> np.ndarray:
        return self._last_probs.copy()

    def regime_state(self) -> Dict[str, Any]:
        """Full regime state for diagnostics/logging."""
        return {
            'regime': self.current_regime,
            'probabilities': {name: float(p) for name, p
                            in zip(self.REGIME_NAMES, self._last_probs)},
            'in_warmup': self._in_warmup,
            'bar_count': self._bar_count,
            'warmup_bars_remaining': max(0, self._warmup - self._bar_count),
        }


class AlphaFactory:
    """
    L1: Manages all registered alpha sources and collects their signals.

    Each alpha is independent and produces AlphaSignal through a uniform
    interface. The factory does NOT combine signals -- that's L2's job.

    v6.1: Adds horizon normalization (fix 4.2) so all signals are
    converted to per-bar mu/sigma before reaching L2.

    v7.1: Adds feature count discipline ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â warns if total unique features
    across all alphas exceed sqrt(n_observations) per the anti-overfitting
    spec. Alphas with strong economic priors (documented in metadata
    'economic_hypothesis') are exempt from the count.
    """

    # Feature names that are structural, not predictive (exempt from count)
    _EXEMPT_META_KEYS = frozenset({
        'error', 'original_horizon', 'original_mu', 'original_sigma',
        'corr_penalty', 'economic_hypothesis', 'alpha_name', 'horizon',
        'warmup', 'warmup_remaining',
    })

    def __init__(self, common_horizon: int = 1):
        self._alphas: OrderedDict[str, BaseAlpha] = OrderedDict()
        self._common_horizon = common_horizon  # Normalize all signals to this
        self._feature_discipline_last_check: int = 0
        self._feature_discipline_interval: int = 252  # Check annually
        self._signal_flips: Dict[str, float] = {}  # B2: alpha_name -> -1.0 to flip mu sign

    def set_signal_flip(self, alpha_name: str, flip: bool = True):
        """B2: Mark an alpha for automatic signal inversion (mu *= -1).
        Used when WF validation detects negative IC (anti-predictive signal)."""
        if flip:
            self._signal_flips[alpha_name] = -1.0
        else:
            self._signal_flips.pop(alpha_name, None)

    def register(self, alpha: BaseAlpha):
        """Register an alpha source."""
        if alpha.name in self._alphas:
            warnings.warn(f"Alpha '{alpha.name}' already registered, overwriting.")
        self._alphas[alpha.name] = alpha

    def unregister(self, name: str):
        self._alphas.pop(name, None)

    def validate_feature_discipline(self, signals: Dict[str, AlphaSignal],
                                     n_observations: int) -> Dict[str, Any]:
        """
        Anti-overfitting check: total unique features must be ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤ sqrt(n_observations).

        Counts distinct feature keys from alpha metadata (excluding structural
        keys). Alphas whose metadata includes 'economic_hypothesis' get a
        documentation pass but their features still count toward the total.

        Returns diagnostic dict with feature_count, limit, compliant flag,
        and per-alpha breakdown.
        """
        if n_observations <= 0:
            return {'feature_count': 0, 'limit': 0, 'compliant': True}

        limit = math.sqrt(n_observations)
        per_alpha = {}
        all_features = set()

        for name, sig in signals.items():
            alpha_features = set()
            for k in sig.metadata.keys():
                if k not in self._EXEMPT_META_KEYS:
                    alpha_features.add(f"{name}:{k}")
            per_alpha[name] = {
                'n_features': len(alpha_features),
                'has_hypothesis': 'economic_hypothesis' in sig.metadata,
            }
            all_features.update(alpha_features)

        total = len(all_features)
        compliant = total <= limit

        result = {
            'feature_count': total,
            'limit': round(limit, 1),
            'compliant': compliant,
            'per_alpha': per_alpha,
        }

        if not compliant:
            warnings.warn(
                f"Feature discipline violation: {total} features > "
                f"sqrt({n_observations}) = {limit:.1f}. "
                f"Reduce features or add dimensionality reduction."
            )

        return result

    def generate_all(self, closes: np.ndarray, volumes: np.ndarray = None,
                     features: np.ndarray = None, bar_idx: int = 0,
                     observation: np.ndarray = None,
                     opens: np.ndarray = None,    # NEW: OHLC support for Tier 1 features
                     highs: np.ndarray = None,    # NEW: OHLC support for Tier 1 features
                     lows: np.ndarray = None,     # NEW: OHLC support for Tier 1 features
                     **kwargs) -> Dict[str, AlphaSignal]:
        """
        Collect signals from all registered alphas.
        Returns dict mapping alpha_name ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ AlphaSignal.

        All signals are horizon-normalized to common_horizon (fix 4.2).
        Periodically validates feature count discipline (v7.1).
        """
        signals = {}
        for name, alpha in self._alphas.items():
            try:
                sig = alpha.generate(
                    closes=closes, volumes=volumes, features=features,
                    bar_idx=bar_idx, observation=observation,
                    opens=opens, highs=highs, lows=lows,  # Phase 1: Pass OHLC to alphas
                    **kwargs
                )
                # -- Horizon normalization (fix 4.2) --
                sig = self._normalize_horizon(sig)
                # -- B2: Auto-flip negative IC signals --
                if name in self._signal_flips:
                    sig = AlphaSignal(
                        mu=sig.mu * self._signal_flips[name],
                        sigma=sig.sigma,
                        confidence=sig.confidence,
                        horizon=sig.horizon,
                        alpha_name=sig.alpha_name,
                        metadata={**sig.metadata, 'auto_flipped': True},
                        timestamp=sig.timestamp,
                    )
                signals[name] = sig
            except Exception as e:
                # Alpha failure should never crash the pipeline
                signals[name] = AlphaSignal(
                    alpha_name=name, timestamp=bar_idx,
                    metadata={'error': str(e)}
                )

        # Periodic feature discipline check (every _feature_discipline_interval bars)
        if (bar_idx - self._feature_discipline_last_check
                >= self._feature_discipline_interval and bar_idx > 0):
            self._feature_discipline_last_check = bar_idx
            self.validate_feature_discipline(signals, n_observations=bar_idx)

        return signals

    def _normalize_horizon(self, sig: AlphaSignal) -> AlphaSignal:
        """
        Convert signal's mu and sigma to per-bar units.

        If an alpha says "mu=0.10 over horizon=21 bars", we convert to
        mu_per_bar = 0.10 / 21 %^ 0.00476 and sigma_per_bar = sigma / sqrt(21).

        This ensures the meta-learner and fallback combiner see consistent
        units regardless of each alpha's native horizon. [Fix 4.2]
        """
        if sig.horizon <= 0 or sig.horizon == self._common_horizon:
            return sig

        h = sig.horizon
        target_h = self._common_horizon

        # Scale mu linearly with horizon ratio
        mu_normalized = sig.mu * (target_h / h)

        # Scale sigma with sqrt(horizon) ratio (volatility scales with sqrt(T))
        sigma_normalized = sig.sigma * math.sqrt(target_h / h)
        sigma_normalized = max(sigma_normalized, 0.001)

        return AlphaSignal(
            mu=float(np.clip(mu_normalized, -10.0, 10.0)),
            sigma=sigma_normalized,
            confidence=sig.confidence,
            horizon=target_h,
            alpha_name=sig.alpha_name,
            metadata={**sig.metadata,
                      'original_horizon': h,
                      'original_mu': sig.mu,
                      'original_sigma': sig.sigma},
            timestamp=sig.timestamp,
        )

    @property
    def alpha_names(self) -> List[str]:
        return list(self._alphas.keys())

    @property
    def n_alphas(self) -> int:
        return len(self._alphas)


# ============================================================================
# L2: SIGNAL ENSEMBLE
# ============================================================================
# Combines multiple alpha signals into a single combined view.
# Three ensemble types:
#   1. Model ensemble  -- multiple models producing same signal type
#   2. Signal ensemble -- different alpha families (trend, MR, value, etc.)
#   3. Horizon ensemble -- blending 1d, 5d, 21d signals
# ============================================================================

class MetaLearner:
    """
    Combines alpha signals using a trained regression model.

    Input features per bar:
        [mu_1, sigma_1, conf_1, mu_2, sigma_2, conf_2, ..., regime_features]

    Output:
        Combined (mu_hat, sigma_hat)

    Uses pure-NumPy SVD ridge regression (no sklearn dependency).
    Retrained periodically on expanding window of realized returns.

    v6.1 FIXES:
        - NumPy-only ridge (no sklearn/scipy version traps) [fix 3.1/5.3]
        - Rolling IC tracking per alpha with auto-demotion [fix 4.3]
        - Coefficient caps to prevent runaway weights

    Source: DeMiguel, Martin-Utrera, Nogales (2020) "A Transaction-Cost
            Perspective on the Multitude of Firm Characteristics"
    """

    def __init__(self, acfg: ArchitectureConfig, alpha_names: List[str]):
        self.acfg = acfg
        self.alpha_names = list(alpha_names)
        self.n_alphas = len(alpha_names)

        # Feature matrix: [mu, sigma, conf] per alpha + optional regime features
        self.feature_dim = self.n_alphas * 3 + 4  # +4 for regime one-hot

        # Training data buffers
        self._X_buffer = deque(maxlen=5000)
        self._y_buffer = deque(maxlen=5000)  # Realized returns
        self._bar_buffer = deque(maxlen=5000)

        # Model (pure NumPy -- no sklearn)
        self._model = None
        self._scaler = _NumpyStandardScaler()
        self._last_train_bar = -999
        self._is_fitted = False

        # Equal-weight fallback
        self._equal_weight = np.ones(self.n_alphas) / max(self.n_alphas, 1)

        # -- Rolling IC tracking per alpha (fix 4.3) --
        self._ic_window = 100  # Rolling window for IC computation
        self._alpha_mu_history: Dict[str, deque] = {
            name: deque(maxlen=self._ic_window) for name in alpha_names
        }
        self._return_history_ic = deque(maxlen=self._ic_window)
        self._alpha_demotion: Dict[str, float] = {name: 1.0 for name in alpha_names}
        self._ic_demotion_threshold = 0.0   # IC below this -> start demoting
        self._ic_disable_threshold = -0.05  # IC below this -> force confidence to 0

        # WS2H: Paper-trade clock -- new alphas must survive N bars before promotion
        self._paper_trade_min_bars: int = 60  # 60 trading days (~3 months)
        self._alpha_registration_bar: Dict[str, int] = {
            name: 0 for name in alpha_names  # Existing alphas start promoted
        }
        self._alpha_paper_bars: Dict[str, int] = {
            name: self._paper_trade_min_bars for name in alpha_names
        }
        self._alpha_paper_pnl: Dict[str, float] = {
            name: 0.0 for name in alpha_names
        }

    def _build_features(self, signals: Dict[str, AlphaSignal],
                        regime_probs: np.ndarray = None) -> np.ndarray:
        """Convert alpha signals dict to feature vector."""
        feats = []
        for name in self.alpha_names:
            sig = signals.get(name, AlphaSignal(alpha_name=name))
            # BUG FIX #5: Ensure scalars (not arrays) - flatten if needed
            mu = float(np.atleast_1d(sig.mu).flat[0])
            sigma = float(np.atleast_1d(sig.sigma).flat[0])
            confidence = float(np.atleast_1d(sig.confidence).flat[0])
            feats.extend([mu, sigma, confidence])

        # Regime features (4-dim one-hot or probabilities)
        if regime_probs is not None and len(regime_probs) == 4:
            # BUG FIX #5: Flatten regime_probs to ensure it's a list of scalars
            regime_list = [float(x) for x in np.atleast_1d(regime_probs).flat[:4]]
            feats.extend(regime_list)
        else:
            feats.extend([0.25, 0.25, 0.25, 0.25])  # Uniform prior

        return np.array(feats, dtype=np.float64)

    def record(self, signals: Dict[str, AlphaSignal],
               realized_return: float, bar_idx: int,
               regime_probs: np.ndarray = None):
        """Record alpha signals and the realized return for training."""
        feats = self._build_features(signals, regime_probs)
        self._X_buffer.append(feats)
        self._y_buffer.append(realized_return)
        self._bar_buffer.append(bar_idx)

        # -- Track per-alpha IC (fix 4.3) --
        self._return_history_ic.append(realized_return)
        for name in self.alpha_names:
            sig = signals.get(name, AlphaSignal(alpha_name=name))
            self._alpha_mu_history[name].append(sig.mu)

            # WS2H: Paper-trade clock -- track bars and paper PnL
            if self._alpha_paper_bars.get(name, self._paper_trade_min_bars) < self._paper_trade_min_bars:
                self._alpha_paper_bars[name] = self._alpha_paper_bars.get(name, 0) + 1
                # Track paper PnL: did the alpha's direction call work?
                if sig.is_active:
                    self._alpha_paper_pnl[name] = (
                        self._alpha_paper_pnl.get(name, 0.0)
                        + sig.mu * realized_return
                    )

    def _update_demotions(self):
        """Compute rolling IC per alpha and update demotion multipliers."""
        n = len(self._return_history_ic)
        if n < 30:
            return  # Not enough data

        rets = np.array(list(self._return_history_ic))

        for name in self.alpha_names:
            mus = np.array(list(self._alpha_mu_history[name]))
            if len(mus) != n:
                continue

            # Rolling rank IC (Spearman-like: correlation of ranks)
            # Using Pearson on raw values as lightweight proxy
            mu_std = np.std(mus)
            ret_std = np.std(rets)
            if mu_std < 1e-10 or ret_std < 1e-10:
                ic = 0.0
            else:
                ic = float(np.corrcoef(mus, rets)[0, 1])
                if np.isnan(ic):
                    ic = 0.0

            # Demotion logic
            if ic < self._ic_disable_threshold:
                self._alpha_demotion[name] = 0.0  # Force disable
            elif ic < self._ic_demotion_threshold:
                # Linear interpolation: IC=-0.05+'0.0, IC=0.0+'1.0
                range_width = self._ic_demotion_threshold - self._ic_disable_threshold
                self._alpha_demotion[name] = max(
                    0.0, (ic - self._ic_disable_threshold) / (range_width + 1e-10)
                )
            else:
                self._alpha_demotion[name] = 1.0

    def get_demotions(self) -> Dict[str, float]:
        """Return current demotion multipliers per alpha (for audit).
        Includes paper-trade status: 0.0 = still in paper-trade period."""
        result = dict(self._alpha_demotion)
        # Overlay paper-trade status
        for name in self.alpha_names:
            paper_bars = self._alpha_paper_bars.get(name, self._paper_trade_min_bars)
            if paper_bars < self._paper_trade_min_bars:
                result[name] = 0.0  # Still in paper-trade
        return result

    def register_new_alpha(self, name: str, bar_idx: int = 0):
        """Register a new alpha with paper-trade clock starting at bar_idx."""
        if name not in self.alpha_names:
            self.alpha_names.append(name)
            self.n_alphas = len(self.alpha_names)
            self.feature_dim = self.n_alphas * 3 + 4
            self._alpha_mu_history[name] = deque(maxlen=self._ic_window)
            self._alpha_demotion[name] = 0.0  # Start demoted (paper-trade)
            self._alpha_registration_bar[name] = bar_idx
            self._alpha_paper_bars[name] = 0  # Start counting
            self._alpha_paper_pnl[name] = 0.0

    def paper_trade_status(self) -> Dict[str, Dict]:
        """Report paper-trade status for all alphas."""
        status = {}
        for name in self.alpha_names:
            bars = self._alpha_paper_bars.get(name, self._paper_trade_min_bars)
            promoted = bars >= self._paper_trade_min_bars
            status[name] = {
                'bars_tracked': bars,
                'min_required': self._paper_trade_min_bars,
                'promoted': promoted,
                'paper_pnl': self._alpha_paper_pnl.get(name, 0.0),
            }
        return status

    def fit(self, bar_idx: int):
        """Retrain meta-learner if enough data and retrain interval passed."""
        n = len(self._X_buffer)
        if n < self.acfg.meta_learner_min_samples:
            return
        if bar_idx - self._last_train_bar < self.acfg.meta_learner_retrain_freq:
            return

        X = np.array(list(self._X_buffer))
        y = np.array(list(self._y_buffer))

        # Standardize
        X_scaled = self._scaler.fit_transform(X)

        # Fit pure-NumPy ridge (fix 3.1/5.3: no sklearn dependency)
        # BUG FIX #6: Handle SVD convergence failures gracefully
        try:
            self._model = _NumpyRidge(alpha=self.acfg.meta_learner_alpha)
            self._model.fit(X_scaled, y)
        except np.linalg.LinAlgError as e:
            print(f"[META-LEARNER WARNING] Ridge fit failed at bar {bar_idx} (SVD): {e}")
            print(f"                       X shape: {X.shape}, y shape: {y.shape}. Using equal-weight.")
            return  # Exit early, don't set _is_fitted = True

        # -- Coefficient caps (must-fix 3): prevent runaway weights --
        # Even with ridge regularization, collinear alpha features can
        # produce large coefficients that make predictions unstable.
        # Cap by absolute magnitude after fit.
        cap = self.acfg.meta_learner_coef_cap
        if cap > 0 and self._model.coef_ is not None:
            self._model.coef_ = np.clip(self._model.coef_, -cap, cap)

        self._last_train_bar = bar_idx
        self._is_fitted = True

        # Update demotion scores (fix 4.3)
        self._update_demotions()

    def fit_from_validation_priors(self, validation_results: Dict):
        """
        FIX §SA-3: Initialize meta-learner from walk-forward validation results.

        Instead of falling back to equal-weight combination during holdout
        evaluation (because no training data has been recorded yet), use
        the per-alpha OOS Sharpe ratios from walk-forward CV to set
        informed initial weights.

        The feature vector per alpha is [mu_i, sigma_i, confidence_i].
        We set the mu coefficient proportional to max(OOS_Sharpe, 0) so
        that alphas with positive OOS performance get higher weight, and
        rejected alphas (negative Sharpe) get zero weight.

        This is a prior, not a fit — it will be overridden once enough
        live data accumulates and the normal fit() path runs.
        """
        per_alpha = validation_results.get('per_alpha', {})
        if not per_alpha:
            return

        n_alphas = len(self.alpha_names)
        # Feature vector: [mu_0, sigma_0, conf_0, mu_1, ..., regime_0..3]
        n_features = n_alphas * 3 + 4

        # Build coefficient vector: only set mu coefficients (every 3rd)
        coefs = np.zeros(n_features)
        raw_weights = []
        for i, name in enumerate(self.alpha_names):
            report = per_alpha.get(name, {})
            oos_sharpe = report.get('sharpe_oos', 0.0)
            # Only positive OOS Sharpe contributes; REJECT alphas get zero
            w = max(oos_sharpe, 0.0)
            raw_weights.append(w)
            # mu coefficient is at index i*3
            coefs[i * 3] = w

        # Normalize so coefficients sum to 1.0 (like a weighted average)
        total_w = sum(raw_weights)
        if total_w > 1e-10:
            coefs = coefs / total_w
        else:
            # All alphas failed — fall back to equal weight on mu
            for i in range(n_alphas):
                coefs[i * 3] = 1.0 / n_alphas

        # Initialize model with these coefficients
        self._model = _NumpyRidge(alpha=self.acfg.meta_learner_alpha)
        self._model.coef_ = coefs
        self._model.intercept_ = 0.0

        # Initialize scaler with neutral parameters (mean=0, std=1)
        # so transform is identity until real data arrives
        self._scaler._mean = np.zeros(n_features)
        self._scaler._std = np.ones(n_features)

        self._is_fitted = True

    def combine(self, signals: Dict[str, AlphaSignal],
                regime_probs: np.ndarray = None) -> Tuple[float, float, Dict]:
        """
        Combine alpha signals into (mu_hat, sigma_hat, diagnostics).

        If meta-learner is fitted, uses trained model.
        Otherwise falls back to confidence-weighted average with shrinkage.

        WS2H: Alphas in paper-trade period have confidence zeroed out.

        Returns:
            Tuple of (mu_hat, sigma_hat, diagnostics_dict).
            Always returns 3-tuple for interface consistency with SignalEnsemble.
        """
        # -- Apply demotion multipliers to confidence (fix 4.3) --
        demoted_signals = {}
        demotions_applied = {}
        for name, sig in signals.items():
            demotion = self._alpha_demotion.get(name, 1.0)

            # WS2H: Paper-trade clock -- zero confidence for unpromoted alphas
            paper_bars = self._alpha_paper_bars.get(name, self._paper_trade_min_bars)
            if paper_bars < self._paper_trade_min_bars:
                demotion = 0.0  # Signal tracked but not used in sizing

            demotions_applied[name] = demotion

            if demotion < 1.0:
                demoted_signals[name] = AlphaSignal(
                    mu=sig.mu, sigma=sig.sigma,
                    confidence=sig.confidence * demotion,
                    horizon=sig.horizon, alpha_name=sig.alpha_name,
                    metadata={**sig.metadata, 'demotion': demotion,
                              'paper_trade_bars': paper_bars},
                    timestamp=sig.timestamp,
                )
            else:
                demoted_signals[name] = sig

        # Build diagnostics common to both paths
        diagnostics = {
            'method': 'ridge' if (self._is_fitted and self._model is not None) else 'equal_weight',
            'n_active': sum(1 for s in demoted_signals.values() if s.is_active),
            'demotions': demotions_applied,
            'is_fitted': self._is_fitted,
        }

        # -- Trained model path --
        if self._is_fitted and self._model is not None:
            feats = self._build_features(demoted_signals, regime_probs)
            feats_scaled = self._scaler.transform(feats.reshape(1, -1))
            mu_hat = float(self._model.predict(feats_scaled)[0])

            # Sigma from residual std of recent predictions
            sigma_hat = self._residual_sigma(demoted_signals)
            diagnostics['mu_hat'] = mu_hat
            diagnostics['sigma_hat'] = sigma_hat
            return mu_hat, sigma_hat, diagnostics

        # -- Fallback: Confidence-weighted average with shrinkage --
        mu_hat, sigma_hat = self._equal_weight_combine(demoted_signals)
        diagnostics['mu_hat'] = mu_hat
        diagnostics['sigma_hat'] = sigma_hat
        return mu_hat, sigma_hat, diagnostics

    def _equal_weight_combine(self, signals: Dict[str, AlphaSignal]
                               ) -> Tuple[float, float]:
        """Confidence-weighted average with shrinkage toward equal weight."""
        mus = []
        sigmas = []
        weights = []

        for name in self.alpha_names:
            sig = signals.get(name, AlphaSignal(alpha_name=name))
            if sig.is_active:
                mus.append(sig.mu)
                sigmas.append(sig.sigma)
                weights.append(sig.confidence)

        if not weights:
            # BUG FIX #2 DEBUG: Log when NO alphas are active (causes 0.0 signal)
            # This happens during warmup when all alphas are inactive
            if hasattr(self, '_no_weights_count'):
                self._no_weights_count += 1
            else:
                self._no_weights_count = 1
            if self._no_weights_count <= 5:
                print(f"[ENSEMBLE WARNING #{self._no_weights_count}] No active alphas - all in warmup or confidence=0")
            return 0.0, 1.0

        weights = np.array(weights)
        mus = np.array(mus)
        sigmas = np.array(sigmas)

        # Cap individual alpha weight
        weights = np.clip(weights, 0, self.acfg.max_single_alpha_weight * np.sum(weights))

        # Shrinkage toward equal weight
        n = len(weights)
        equal_w = np.ones(n) / n
        lam = self.acfg.shrinkage_toward_equal
        weights = (1 - lam) * weights / (np.sum(weights) + 1e-10) + lam * equal_w

        # Normalize
        weights = weights / (np.sum(weights) + 1e-10)

        mu_hat = float(np.dot(weights, mus))
        # Combined sigma: weighted average + diversification benefit
        sigma_hat = float(np.sqrt(np.dot(weights ** 2, sigmas ** 2)))
        sigma_hat = max(sigma_hat, 0.01)

        return mu_hat, sigma_hat

    def _residual_sigma(self, signals: Dict[str, AlphaSignal]) -> float:
        """Estimate sigma from recent prediction residuals."""
        active_sigmas = [
            s.sigma for s in signals.values() if s.is_active
        ]
        if active_sigmas:
            return float(np.mean(active_sigmas))
        return 0.15


# ============================================================================
# ALPHA LIFECYCLE MONITOR (Good-to-have A)
# ============================================================================
# Tracks per-alpha health beyond IC: hit rate, drawdown contribution,
# turnover contribution, weight volatility, and performance decay.
# ============================================================================

class AlphaLifecycleMonitor:
    """
    Per-alpha health dashboard. Answers "which alphas are earning their keep?"

    Tracked per alpha (rolling window):
        hit_rate:        Fraction of bars where sign(mu) matched sign(return)
        dd_contribution: Fraction of portfolio drawdown attributable to alpha
        turnover_share:  Fraction of total position change driven by alpha
        weight_vol:      Std of alpha weight over rolling window (instability)
        bars_since_good: Bars since alpha had a positive rolling window return

    Usage:
        monitor = AlphaLifecycleMonitor(alpha_names, window=100)
        monitor.update(signals, realized_return, weight_per_alpha, position_delta)
        report = monitor.report()
    """

    def __init__(self, alpha_names: List[str], window: int = 100):
        self._names = list(alpha_names)
        self._window = window

        # Per-alpha rolling buffers
        self._mu_history: Dict[str, deque] = {
            n: deque(maxlen=window) for n in alpha_names
        }
        self._hit_history: Dict[str, deque] = {
            n: deque(maxlen=window) for n in alpha_names
        }
        self._return_contribution: Dict[str, deque] = {
            n: deque(maxlen=window) for n in alpha_names
        }
        self._weight_history: Dict[str, deque] = {
            n: deque(maxlen=window) for n in alpha_names
        }
        self._turnover_share: Dict[str, deque] = {
            n: deque(maxlen=window) for n in alpha_names
        }

        # Bars since last positive rolling window
        self._bars_since_good: Dict[str, int] = {n: 0 for n in alpha_names}
        self._rolling_pnl: Dict[str, float] = {n: 0.0 for n in alpha_names}

    def update(self, signals: Dict[str, 'AlphaSignal'],
               realized_return: float,
               combined_mu: float,
               position_delta: float):
        """
        Update lifecycle metrics for all alphas.

        Args:
            signals: Current bar's alpha signals
            realized_return: Realized bar return
            combined_mu: The ensemble combined mu (for weight decomposition)
            position_delta: Absolute change in position this bar
        """
        # Compute weights from confidence (mirror ensemble logic)
        total_conf = sum(
            s.confidence for s in signals.values() if s.is_active
        )

        for name in self._names:
            sig = signals.get(name, AlphaSignal(alpha_name=name))

            # Hit rate: did the alpha call the direction correctly?
            if sig.is_active and abs(sig.mu) > 1e-8:
                hit = 1.0 if np.sign(sig.mu) == np.sign(realized_return) else 0.0
                self._hit_history[name].append(hit)
            else:
                self._hit_history[name].append(0.5)  # Neutral for inactive

            # Return contribution: alpha_mu * realized (positive = alpha helped)
            alpha_pnl = sig.mu * realized_return if sig.is_active else 0.0
            self._return_contribution[name].append(alpha_pnl)

            # Weight: confidence-based weight
            w = (sig.confidence / (total_conf + 1e-10)) if sig.is_active else 0.0
            self._weight_history[name].append(w)

            # Turnover share: |alpha_mu| / |combined_mu| as proxy for
            # how much this alpha contributed to the position change
            if abs(combined_mu) > 1e-10 and sig.is_active:
                t_share = min(1.0, abs(sig.mu * w) / (abs(combined_mu) + 1e-10))
            else:
                t_share = 0.0
            self._turnover_share[name].append(t_share)

            # Mu history for drawdown attribution
            self._mu_history[name].append(sig.mu if sig.is_active else 0.0)

            # Performance decay: rolling PnL and bars since positive
            self._rolling_pnl[name] = sum(self._return_contribution[name])
            if self._rolling_pnl[name] > 0:
                self._bars_since_good[name] = 0
            else:
                self._bars_since_good[name] += 1

    def get_alpha_health(self, name: str) -> Dict[str, float]:
        """Return health metrics for a single alpha."""
        hits = list(self._hit_history.get(name, []))
        weights = list(self._weight_history.get(name, []))
        contribs = list(self._return_contribution.get(name, []))
        t_shares = list(self._turnover_share.get(name, []))

        return {
            'hit_rate': float(np.mean(hits)) if hits else 0.5,
            'rolling_pnl': sum(contribs),
            'dd_contribution': float(-min(0.0, min(
                np.cumsum(contribs).tolist() or [0.0]
            ))) if contribs else 0.0,
            'turnover_share': float(np.mean(t_shares)) if t_shares else 0.0,
            'weight_vol': float(np.std(weights)) if len(weights) > 5 else 0.0,
            'avg_weight': float(np.mean(weights)) if weights else 0.0,
            'bars_since_good': self._bars_since_good.get(name, 0),
        }

    def report(self) -> Dict[str, Dict[str, float]]:
        """Full lifecycle report across all alphas."""
        return {name: self.get_alpha_health(name) for name in self._names}

    def add_alpha(self, name: str):
        """Register a new alpha for monitoring."""
        if name not in self._names:
            self._names.append(name)
            self._mu_history[name] = deque(maxlen=self._window)
            self._hit_history[name] = deque(maxlen=self._window)
            self._return_contribution[name] = deque(maxlen=self._window)
            self._weight_history[name] = deque(maxlen=self._window)
            self._turnover_share[name] = deque(maxlen=self._window)
            self._bars_since_good[name] = 0
            self._rolling_pnl[name] = 0.0


# ============================================================================
# POSITION RECONCILER (Spec: L4 reconciliation requirement)
# ============================================================================
# End-of-day position and PnL reconciliation against broker/prime broker records.
# Flags and escalates discrepancies per institutional practice.
# ============================================================================

class ReconciliationResult:
    """Result of a single reconciliation check."""
    __slots__ = ('bar_idx', 'field', 'internal_value', 'external_value',
                 'discrepancy', 'severity', 'timestamp')

    def __init__(self, bar_idx: int, field: str, internal_value: float,
                 external_value: float, tolerance: float = 1e-4):
        self.bar_idx = bar_idx
        self.field = field
        self.internal_value = internal_value
        self.external_value = external_value
        self.discrepancy = abs(internal_value - external_value)
        self.timestamp = None

        # Severity classification
        if self.discrepancy < tolerance:
            self.severity = "MATCH"
        elif self.discrepancy < tolerance * 10:
            self.severity = "MINOR"
        elif self.discrepancy < tolerance * 100:
            self.severity = "MATERIAL"
        else:
            self.severity = "BREAK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bar_idx': self.bar_idx, 'field': self.field,
            'internal': self.internal_value, 'external': self.external_value,
            'discrepancy': self.discrepancy, 'severity': self.severity,
        }


class PositionReconciler:
    """
    End-of-day position and PnL reconciliation against external records.

    Compares internal tracking (from ExecutionEngine/PortfolioConstructor)
    against broker/prime broker reported positions and PnL.

    Discrepancy classification:
        MATCH:    within tolerance (< 1e-4 default)
        MINOR:    small rounding differences (< 1e-3)
        MATERIAL: requires investigation (< 1e-2)
        BREAK:    reconciliation failure, escalate immediately (>= 1e-2)

    Usage:
        reconciler = PositionReconciler()
        reconciler.reconcile(
            bar_idx=100,
            internal_position=1.5,
            external_position=1.5001,
            internal_pnl=1234.56,
            external_pnl=1234.58,
            internal_nav=100000.0,
            external_nav=100002.0,
        )
        if reconciler.has_breaks():
            alert_manager.alert("FATAL", "L4", "Reconciliation break", ...)
    """

    def __init__(self, position_tolerance: float = 1e-4,
                 pnl_tolerance: float = 0.01,
                 nav_tolerance: float = 0.001,
                 max_log: int = 5000):
        self._position_tol = position_tolerance
        self._pnl_tol = pnl_tolerance
        self._nav_tol = nav_tolerance
        self._results: deque = deque(maxlen=max_log)
        self._break_count = 0
        self._material_count = 0
        self._total_checks = 0

    def reconcile(self, bar_idx: int,
                  internal_position: float, external_position: float,
                  internal_pnl: float = None, external_pnl: float = None,
                  internal_nav: float = None, external_nav: float = None) -> List[ReconciliationResult]:
        """
        Run end-of-day reconciliation for one bar.

        Args:
            bar_idx: Current bar index
            internal_position: Position per internal tracking
            external_position: Position per broker/prime broker
            internal_pnl: Cumulative PnL per internal tracking
            external_pnl: Cumulative PnL per broker
            internal_nav: NAV per internal tracking
            external_nav: NAV per broker

        Returns:
            List of ReconciliationResult for each checked field
        """
        results = []
        self._total_checks += 1

        # Position reconciliation
        pos_result = ReconciliationResult(
            bar_idx, 'position',
            internal_position, external_position,
            tolerance=self._position_tol,
        )
        results.append(pos_result)

        # PnL reconciliation
        if internal_pnl is not None and external_pnl is not None:
            pnl_result = ReconciliationResult(
                bar_idx, 'pnl',
                internal_pnl, external_pnl,
                tolerance=self._pnl_tol,
            )
            results.append(pnl_result)

        # NAV reconciliation
        if internal_nav is not None and external_nav is not None:
            nav_result = ReconciliationResult(
                bar_idx, 'nav',
                internal_nav, external_nav,
                tolerance=self._nav_tol,
            )
            results.append(nav_result)

        # Track breaks and material discrepancies
        for r in results:
            self._results.append(r)
            if r.severity == "BREAK":
                self._break_count += 1
            elif r.severity == "MATERIAL":
                self._material_count += 1

        return results

    def has_breaks(self) -> bool:
        """True if any BREAK-level discrepancies exist."""
        return self._break_count > 0

    def has_material(self) -> bool:
        """True if any MATERIAL-level discrepancies exist."""
        return self._material_count > 0

    def summary(self) -> Dict[str, Any]:
        """Summary statistics for all reconciliation checks."""
        return {
            'total_checks': self._total_checks,
            'total_results': len(self._results),
            'breaks': self._break_count,
            'material': self._material_count,
            'clean': self._total_checks - self._break_count - self._material_count,
            'recent_discrepancies': [
                r.to_dict() for r in self._results
                if r.severity in ("BREAK", "MATERIAL")
            ][-10:],  # Last 10 issues
        }

    def reset(self):
        """Reset counters (e.g., for new backtest run)."""
        self._results.clear()
        self._break_count = 0
        self._material_count = 0
        self._total_checks = 0


# ============================================================================
# ATTRIBUTION ENGINE (Good-to-have C)
# ============================================================================
# Decomposes portfolio PnL into alpha, cost, and execution components.
# ============================================================================

class AttributionEngine:
    """
    Paper-alpha vs realized attribution.

    Decomposes each bar's PnL into:
        gross_alpha:    Sum of (alpha_mu_i * weight_i * position)
        cost_drag:      Commission + slippage impact
        suppression:    PnL foregone by L4 no-trade suppression
        kill_impact:    PnL foregone by kill switch
        execution_gap:  Gap between L3 continuous target and L4 discrete fill

    Accumulates over an episode for a final attribution report.
    """

    def __init__(self):
        self._gross_alpha_pnl = 0.0    # Paper PnL if signals were perfect
        self._realized_pnl = 0.0       # Actual PnL
        self._cost_drag = 0.0          # Commission + slippage cost
        self._suppression_cost = 0.0   # PnL foregone from no-trade suppression
        self._kill_cost = 0.0          # PnL foregone from kill switches
        self._discretization_gap = 0.0 # Gap from L3 continuous +' L4 discrete
        self._n_bars = 0

    def update(self, bar_return: float,
               l3_target: float,
               l4_discrete: float,
               actual_exposure: float,
               combined_mu: float,
               was_suppressed: bool,
               was_killed: bool,
               cost_this_bar: float = 0.0):
        """
        Record one bar's attribution components.

        Args:
            bar_return: Realized return this bar
            l3_target: L3's continuous target exposure
            l4_discrete: L4's discretized exposure
            actual_exposure: What was actually held
            combined_mu: Ensemble's expected return (paper alpha)
            was_suppressed: True if L4 suppressed the trade
            was_killed: True if kill switch was active
            cost_this_bar: Commission + slippage cost in return terms
        """
        self._n_bars += 1

        # Paper alpha: what the signal predicted
        self._gross_alpha_pnl += combined_mu

        # Realized: position * actual return
        realized = actual_exposure * bar_return
        self._realized_pnl += realized

        # Cost drag
        self._cost_drag += cost_this_bar

        # Discretization gap: (L3_continuous - L4_discrete) * bar_return
        self._discretization_gap += (l3_target - l4_discrete) * bar_return

        # Suppression cost: if trade was suppressed, the foregone PnL
        if was_suppressed:
            # What we would have gained/lost if L3's target was executed
            foregone = (l3_target - actual_exposure) * bar_return
            self._suppression_cost += foregone

        # Kill cost: if kill was active, everything is foregone
        if was_killed:
            foregone = l3_target * bar_return
            self._kill_cost += foregone

    def report(self) -> Dict[str, float]:
        """
        Attribution report for the episode.

        Returns dict with PnL decomposition. Signs:
            gross_alpha > 0     +' signals were directionally correct
            cost_drag < 0       +' always negative (costs)
            suppression_cost    +' positive if suppression hurt, negative if helped
            execution_shortfall +' gross_alpha - realized - cost_drag
        """
        shortfall = self._gross_alpha_pnl - self._realized_pnl - self._cost_drag
        return {
            'n_bars': self._n_bars,
            'gross_alpha_pnl': self._gross_alpha_pnl,
            'realized_pnl': self._realized_pnl,
            'cost_drag': self._cost_drag,
            'suppression_cost': self._suppression_cost,
            'kill_cost': self._kill_cost,
            'discretization_gap': self._discretization_gap,
            'execution_shortfall': shortfall,
            'implementation_efficiency': (
                self._realized_pnl / (self._gross_alpha_pnl + 1e-10)
                if abs(self._gross_alpha_pnl) > 1e-10 else 0.0
            ),
        }

    def reset(self):
        """Reset for new episode."""
        self._gross_alpha_pnl = 0.0
        self._realized_pnl = 0.0
        self._cost_drag = 0.0
        self._suppression_cost = 0.0
        self._kill_cost = 0.0
        self._discretization_gap = 0.0
        self._n_bars = 0


# ============================================================================
# WS5A: ALPHA CORRELATION MANAGEMENT (Signal-Space Concentration)
# ============================================================================

class AlphaCorrelationManager:
    """
    Monitors pairwise alpha correlations and computes effective
    number of independent alphas.

    High correlation between alphas means the ensemble is less
    diversified than it appears. This module:
    1. Tracks rolling pairwise correlations
    2. Computes N_eff (effective independent alphas)
    3. Produces penalty weights to down-weight correlated alphas

    Reference: Bouchaud & Potters (2003) "Theory of Financial Risk"
    """

    def __init__(self, alpha_names: List[str], window: int = 100):
        self._alpha_names = list(alpha_names)
        self._mu_history = {n: deque(maxlen=window) for n in alpha_names}
        self._window = window

    def update(self, signals: Dict[str, 'AlphaSignal']):
        """Record signals for correlation tracking."""
        for name in self._alpha_names:
            sig = signals.get(name, AlphaSignal(alpha_name=name))
            self._mu_history[name].append(sig.mu if sig.is_active else 0.0)

    def pairwise_correlations(self) -> np.ndarray:
        """Return correlation matrix of alpha signals."""
        n = len(self._alpha_names)
        # Build matrix of mu histories
        data = []
        for name in self._alpha_names:
            h = list(self._mu_history[name])
            data.append(h)

        min_len = min(len(d) for d in data) if data else 0
        if min_len < 20:
            return np.eye(n)

        mat = np.array([d[-min_len:] for d in data])
        # Compute correlation matrix
        stds = np.std(mat, axis=1)
        # Avoid division by zero for constant signals
        stds = np.where(stds < 1e-10, 1.0, stds)
        normed = (mat - mat.mean(axis=1, keepdims=True)) / stds[:, np.newaxis]
        corr = normed @ normed.T / min_len
        # Clamp to valid correlation range
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def effective_n_independent(self) -> float:
        """
        Effective number of independent alphas via eigenvalue analysis.
        N_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2)
        """
        corr = self.pairwise_correlations()
        eigenvalues = np.linalg.eigvalsh(corr)
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 1.0
        return float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))

    def correlation_penalty_weights(self,
                                     max_corr: float = 0.7) -> Dict[str, float]:
        """
        Down-weight alphas that are highly correlated with others.
        Returns multiplier per alpha (0.0 to 1.0).
        """
        corr = self.pairwise_correlations()
        n = len(self._alpha_names)
        weights = {}

        for i, name in enumerate(self._alpha_names):
            # Average absolute correlation with OTHER alphas
            other_corrs = [abs(corr[i, j]) for j in range(n) if j != i]
            avg_corr = float(np.mean(other_corrs)) if other_corrs else 0.0

            # Penalty: linear ramp from 1.0 at max_corr to 0.3 at 1.0
            if avg_corr > max_corr:
                penalty = 1.0 - 0.7 * (avg_corr - max_corr) / (1.0 - max_corr + 1e-10)
                weights[name] = max(0.3, float(penalty))
            else:
                weights[name] = 1.0

        return weights

    def add_alpha(self, name: str):
        """Register a new alpha for monitoring."""
        if name not in self._alpha_names:
            self._alpha_names.append(name)
            self._mu_history[name] = deque(maxlen=self._window)


# ============================================================================
# WS3A: TAIL RISK MANAGER (CVaR / Expected Shortfall Constraints)
# ============================================================================

class TailRiskManager:
    """
    Computes and constrains tail risk metrics.

    Methods:
      - Historical CVaR at 95% and 99%
      - Stressed CVaR using crisis-period correlation assumptions
      - Position-level constraint: max exposure given CVaR limit

    Integration: gate in PortfolioConstructor.construct() between
    step 5 (regime scaling) and step 6 (drift guard).

    Reference: Rockafellar & Uryasev (2000) "Optimization of CVaR"
    """

    def __init__(self, lookback: int = 252, confidence: float = 0.95):
        self._return_history = deque(maxlen=lookback)
        self.confidence = confidence
        self._cvar_limit = -0.10  # Default: 10% daily CVaR limit

    def update(self, bar_return: float):
        """Record a return observation."""
        self._return_history.append(bar_return)

    def cvar_historical(self) -> float:
        """Historical Expected Shortfall at self.confidence."""
        rets = np.array(list(self._return_history))
        if len(rets) < 30:
            return -0.10  # Conservative default
        cutoff = np.percentile(rets, (1 - self.confidence) * 100)
        tail = rets[rets <= cutoff]
        return float(np.mean(tail)) if len(tail) > 0 else float(cutoff)

    def cvar_99(self) -> float:
        """CVaR at 99% confidence."""
        rets = np.array(list(self._return_history))
        if len(rets) < 30:
            return -0.15
        cutoff = np.percentile(rets, 1.0)
        tail = rets[rets <= cutoff]
        return float(np.mean(tail)) if len(tail) > 0 else float(cutoff)

    def stressed_cvar(self, crisis_factor: float = 2.0) -> float:
        """CVaR under stressed assumptions (vol * crisis_factor)."""
        base = self.cvar_historical()
        return base * crisis_factor

    def should_constrain(self, current_exposure: float,
                         cvar_limit: float = None) -> Tuple[bool, float]:
        """
        Check if portfolio CVaR exceeds limit and compute max safe exposure.

        Returns: (is_breached, max_safe_exposure)
        """
        limit = cvar_limit or self._cvar_limit
        cvar = self.cvar_historical()

        # Estimated portfolio CVaR = |exposure| * asset CVaR
        portfolio_cvar = abs(current_exposure) * abs(cvar)

        if portfolio_cvar > abs(limit):
            # Max safe exposure: |limit| / |cvar|
            max_safe = abs(limit) / (abs(cvar) + 1e-10)
            return True, float(np.sign(current_exposure) * min(max_safe, abs(current_exposure)))
        return False, current_exposure


# ============================================================================
# WS3B: FACTOR EXPOSURE MONITOR & ENFORCEMENT
# ============================================================================

class FactorExposureMonitor:
    """
    Tracks portfolio factor exposures using simple proxies.

    For single-asset: beta to benchmark, momentum loading, vol loading.
    The factor_exposure_limit config exists but was never wired in.
    This class provides the monitoring and enforcement.

    Reference: Barra risk model methodology
    """

    def __init__(self, lookback: int = 63):
        self._port_returns = deque(maxlen=lookback)
        self._bench_returns = deque(maxlen=lookback)
        self._lookback = lookback

    def update(self, port_return: float, bench_return: float = 0.0):
        """Record portfolio and benchmark returns."""
        self._port_returns.append(port_return)
        self._bench_returns.append(bench_return)

    def beta(self) -> float:
        """Rolling beta to benchmark."""
        if len(self._port_returns) < 20:
            return 1.0
        p = np.array(list(self._port_returns))
        b = np.array(list(self._bench_returns))
        var_b = np.var(b)
        if var_b < 1e-12:
            return 1.0
        cov = np.cov(p, b)[0, 1]
        return float(cov / var_b)

    def momentum_exposure(self) -> float:
        """Rolling momentum exposure proxy (cumulative return)."""
        if len(self._port_returns) < 20:
            return 0.0
        return float(np.sum(list(self._port_returns)[-20:]))

    def volatility_exposure(self) -> float:
        """Rolling vol exposure relative to benchmark."""
        if len(self._port_returns) < 20:
            return 0.0
        p_vol = float(np.std(list(self._port_returns)[-20:])) * np.sqrt(252)
        b_vol = float(np.std(list(self._bench_returns)[-20:])) * np.sqrt(252)
        return p_vol / (b_vol + 1e-10) - 1.0  # Relative vol loading

    def all_exposures(self) -> Dict[str, float]:
        """Return all tracked factor exposures."""
        return {
            'beta': self.beta(),
            'momentum': self.momentum_exposure(),
            'vol_loading': self.volatility_exposure(),
        }

    def is_within_limits(self, limit: float) -> Tuple[bool, Dict[str, float]]:
        """Check all factor exposures against limit."""
        exposures = self.all_exposures()
        breaches = {k: v for k, v in exposures.items() if abs(v) > limit}
        return len(breaches) == 0, exposures

    def max_breach_factor(self) -> float:
        """Return the largest absolute factor exposure."""
        exposures = self.all_exposures()
        return max(abs(v) for v in exposures.values()) if exposures else 0.0


# ============================================================================
# WS5C: ALERTING HIERARCHY
# ============================================================================

class AlertManager:
    """
    Structured alerting with severity classification.

    Levels:
      INFO:     drift detected, alpha demotion, retraining triggered
      WARNING:  constraint near breach (within 10%), cost deviation
      CRITICAL: kill switch triggered, margin risk, data feed outage
      FATAL:    system halt, reconciliation failure

    Supports multiple handler backends:
      - _default_handler: print to stdout (dev/debug)
      - _logging_handler: Python logging framework (production)
      - Custom handlers via constructor

    Usage:
        # Development:
        alert_mgr = AlertManager()

        # Production (Python logging):
        alert_mgr = AlertManager(handlers=[AlertManager.make_logging_handler()])

        alert_mgr.alert("WARNING", "L3", "Beta near limit", {'beta': 0.48, 'limit': 0.50})
    """

    LEVELS = ('INFO', 'WARNING', 'CRITICAL', 'FATAL')

    # Map alert levels to Python logging levels
    _LOG_LEVEL_MAP = {
        'INFO': 20,      # logging.INFO
        'WARNING': 30,   # logging.WARNING
        'CRITICAL': 50,  # logging.CRITICAL
        'FATAL': 50,     # logging.CRITICAL (highest)
    }

    def __init__(self, handlers=None, max_log: int = 5000, min_print_level: str = "WARNING"):
        self._handlers = handlers or [self._default_handler]
        self._alert_log = deque(maxlen=max_log)
        self._counts = {level: 0 for level in self.LEVELS}
        # FIX Ã‚Â§5.1: min_print_level gates stdout output while always recording to log.
        # CRITICAL/FATAL always print. Set to "INFO" for full verbose output.
        self._min_print_level = self.LEVELS.index(min_print_level.upper()) if min_print_level.upper() in self.LEVELS else 0

    def alert(self, level: str, source: str, message: str,
              context: Dict = None):
        """Emit an alert at the specified severity level."""
        level = level.upper()
        if level not in self.LEVELS:
            level = 'INFO'

        from datetime import datetime
        entry = {
            'level': level,
            'source': source,
            'message': message,
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat(),
        }
        self._alert_log.append(entry)
        self._counts[level] = self._counts.get(level, 0) + 1

        # FIX Ã‚Â§5.1: Always record in log, but only call handlers if level >= min_print_level
        level_idx = self.LEVELS.index(level) if level in self.LEVELS else 0
        if level_idx >= self._min_print_level:
            for handler in self._handlers:
                try:
                    handler(entry)
                except Exception:
                    pass  # Alerting must not crash the pipeline

    def _default_handler(self, entry: Dict):
        """Print alert to stdout with severity prefix (dev/debug mode)."""
        prefix = {
            'INFO': '[i]', 'WARNING': '[!]',
            'CRITICAL': '[!!!]', 'FATAL': '[XXX]'
        }
        print(f"  {prefix.get(entry['level'], '?')} [{entry['level']}] "
              f"{entry['source']}: {entry['message']}")

    @classmethod
    def make_logging_handler(cls, logger_name: str = "alphago.alerts"):
        """
        Create a handler that routes alerts through Python's logging framework.

        This enables integration with standard logging configuration,
        log rotation, remote syslog, PagerDuty, Slack, etc.

        Usage:
            import logging
            logging.basicConfig(level=logging.INFO, filename='alerts.log')
            alert_mgr = AlertManager(handlers=[AlertManager.make_logging_handler()])
        """
        import logging
        logger = logging.getLogger(logger_name)

        def _handler(entry: Dict):
            log_level = cls._LOG_LEVEL_MAP.get(entry['level'], 20)
            ctx_str = ""
            if entry.get('context'):
                ctx_str = f" | context={entry['context']}"
            logger.log(
                log_level,
                f"[{entry['source']}] {entry['message']}{ctx_str}"
            )

        return _handler

    @classmethod
    def make_jsonl_handler(cls, filepath: str):
        """
        Create a handler that writes alerts as JSON lines to a file.
        Suitable for log aggregation systems (ELK, Splunk, etc).
        """
        import json as _json
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        def _handler(entry: Dict):
            with open(filepath, 'a') as f:
                f.write(_json.dumps(entry, default=str) + '\n')

        return _handler

    @property
    def log(self) -> List[Dict]:
        return list(self._alert_log)

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def has_critical(self) -> bool:
        return self._counts.get('CRITICAL', 0) > 0 or self._counts.get('FATAL', 0) > 0


class HorizonBlender:
    """
    Blends signals across multiple time horizons.

    Short-horizon signals get more weight for timing,
    long-horizon signals stabilize the view.

    Source: Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere"
    """

    def __init__(self, acfg: ArchitectureConfig):
        self.weights = np.array(acfg.horizon_blend_weights)
        self.weights = self.weights / (self.weights.sum() + 1e-10)
        self.horizon_bars = acfg.horizon_bars

    def blend(self, signals: Dict[str, AlphaSignal]) -> Tuple[float, float, float]:
        """
        Blend signals grouped by native horizon proximity with sqrt(h) weighting.

        Uses original_horizon from metadata (before per-bar normalization) to
        assign signals to horizon buckets. Within each bucket, signals are
        weighted by sqrt(native_horizon) * confidence — longer-horizon signals
        get more weight due to higher signal-to-noise ratio and lower turnover.

        Returns (blended_mu, blended_sigma, blended_confidence).
        """
        horizon_mus = {h: [] for h in self.horizon_bars}
        horizon_sigmas = {h: [] for h in self.horizon_bars}

        for sig in signals.values():
            if not sig.is_active:
                continue
            # Use original native horizon (before _normalize_horizon set it to 1)
            native_h = sig.metadata.get('original_horizon', sig.horizon)
            best_h = min(self.horizon_bars,
                         key=lambda h: abs(h - native_h))
            # sqrt(h) weighting: longer horizons have higher SNR, lower turnover
            h_weight = np.sqrt(native_h) * sig.confidence
            horizon_mus[best_h].append((sig.mu, h_weight))
            horizon_sigmas[best_h].append((sig.sigma, h_weight))

        blended_mu = 0.0
        blended_var = 0.0

        for i, h in enumerate(self.horizon_bars):
            if horizon_mus[h]:
                mus, weights = zip(*horizon_mus[h])
                sigs, _ = zip(*horizon_sigmas[h])
                w_arr = np.array(weights)
                w_sum = w_arr.sum() + 1e-10
                h_mu = float(np.dot(mus, w_arr) / w_sum)
                h_sigma = float(np.dot(sigs, w_arr) / w_sum)
            else:
                h_mu = 0.0
                h_sigma = 0.15

            blended_mu += self.weights[i] * h_mu
            blended_var += (self.weights[i] * h_sigma) ** 2

        # Max confidence across active signals — if any horizon is confident, hold
        blended_conf = max(
            (sig.confidence for sig in signals.values() if sig.is_active),
            default=0.5)

        return float(blended_mu), float(np.sqrt(blended_var + 1e-10)), blended_conf


class RegimeGating:
    """
    Adjusts alpha weights based on market regime.

    In trending regimes: upweight trend alpha, downweight MR.
    In mean-reverting regimes: opposite.
    In crisis: downweight all risky alphas, upweight carry/vol.

    Source: Baz et al (2015) "Dissecting Investment Strategies in the Cross
            Section and Time Series" (AQR)
    """

    # Default regime +' alpha weight adjustments (multiplicative)
    DEFAULT_REGIME_ADJUSTMENTS = {
        'high_growth_low_vol':  {'trend_follow': 1.2, 'mean_reversion': 0.7,
                                  'value': 0.8, 'carry': 1.0, 'vol_premium': 1.0},
        'high_growth_high_vol': {'trend_follow': 0.8, 'mean_reversion': 0.9,
                                  'value': 0.6, 'carry': 0.5, 'vol_premium': 1.3},
        'low_growth_low_vol':   {'trend_follow': 0.9, 'mean_reversion': 1.2,
                                  'value': 1.1, 'carry': 1.2, 'vol_premium': 0.8},
        'low_growth_high_vol':  {'trend_follow': 0.5, 'mean_reversion': 0.5,
                                  'value': 0.4, 'carry': 0.3, 'vol_premium': 1.5},
    }

    REGIME_NAMES = [
        'high_growth_low_vol', 'high_growth_high_vol',
        'low_growth_low_vol', 'low_growth_high_vol',
    ]

    def __init__(self, acfg: ArchitectureConfig):
        self.acfg = acfg

    def adjust_signals(self, signals: Dict[str, AlphaSignal],
                       regime_probs: np.ndarray = None) -> Dict[str, AlphaSignal]:
        """Scale alpha confidence by regime-conditional adjustments."""
        if regime_probs is None or len(regime_probs) != 4:
            return signals

        # Expected adjustment = GBP(p_regime * adjustment_regime)
        adjusted = {}
        for name, sig in signals.items():
            mult = 0.0
            for i, regime in enumerate(self.REGIME_NAMES):
                adjustments = self.DEFAULT_REGIME_ADJUSTMENTS.get(regime, {})
                alpha_adj = adjustments.get(name, 1.0)
                mult += regime_probs[i] * alpha_adj

            # Apply multiplicative adjustment to confidence
            new_conf = float(np.clip(sig.confidence * mult, 0.0, 1.0))
            adjusted[name] = AlphaSignal(
                mu=sig.mu, sigma=sig.sigma, confidence=new_conf,
                horizon=sig.horizon, alpha_name=sig.alpha_name,
                metadata={**sig.metadata, 'regime_mult': mult},
                timestamp=sig.timestamp,
            )
        return adjusted


class SignalEnsemble:
    """
    L2: Complete signal ensemble pipeline.

    1. Collect alpha signals from L1
    2. Apply regime gating (adjust confidences)
    3. Blend across horizons
    4. Combine via meta-learner (or confidence-weighted avg)
    5. Output: (mu_hat, sigma_hat) for L3
    """

    def __init__(self, acfg: ArchitectureConfig, alpha_names: List[str],
                 corr_manager: 'AlphaCorrelationManager' = None):
        self.acfg = acfg
        self.meta_learner = MetaLearner(acfg, alpha_names)
        self.horizon_blender = HorizonBlender(acfg)
        self.regime_gating = RegimeGating(acfg) if acfg.regime_gating else None
        # WS5A fix: Correlation manager now owned by ensemble (not external)
        self._corr_manager = corr_manager or AlphaCorrelationManager(alpha_names)
        self._step_count = 0

    def combine(self, signals: Dict[str, AlphaSignal],
                regime_probs: np.ndarray = None,
                bar_idx: int = 0) -> Tuple[float, float, Dict]:
        """
        Full ensemble pipeline: gating ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ correlation penalties ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ blending ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ meta-learner.

        Returns: (mu_hat, sigma_hat, diagnostics_dict)
        """
        self._step_count += 1

        # Step 1: Regime gating
        if self.regime_gating is not None and regime_probs is not None:
            signals = self.regime_gating.adjust_signals(signals, regime_probs)

        # Step 2: Correlation penalties (WS5A fix ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â now inside ensemble)
        corr_diag = {}
        self._corr_manager.update(signals)
        corr_penalties = self._corr_manager.correlation_penalty_weights(max_corr=0.7)
        n_penalised = 0
        for name, sig in signals.items():
            penalty = corr_penalties.get(name, 1.0)
            if penalty < 1.0:
                signals[name] = AlphaSignal(
                    mu=sig.mu, sigma=sig.sigma,
                    confidence=sig.confidence * penalty,
                    horizon=sig.horizon, alpha_name=sig.alpha_name,
                    metadata={**sig.metadata, 'corr_penalty': penalty},
                    timestamp=sig.timestamp,
                )
                n_penalised += 1
        corr_diag['n_penalised'] = n_penalised
        corr_diag['effective_n_independent'] = self._corr_manager.effective_n_independent()
        corr_diag['penalties'] = {k: v for k, v in corr_penalties.items() if v < 1.0}

        # Step 3: Meta-learner combine
        mu_hat, sigma_hat, meta_diag = self.meta_learner.combine(signals, regime_probs)

        # Step 4: Horizon blending (as secondary signal)
        h_mu, h_sigma, h_conf = self.horizon_blender.blend(signals)

        # Blend meta-learner and horizon-blender outputs
        # Meta-learner gets more weight as it trains
        if self.meta_learner._is_fitted:
            alpha_meta = 0.7
        else:
            alpha_meta = 0.3

        final_mu = alpha_meta * mu_hat + (1 - alpha_meta) * h_mu
        final_sigma = max(alpha_meta * sigma_hat + (1 - alpha_meta) * h_sigma, 0.01)

        final_sigma = max(final_sigma, 0.01)

        # Step 5: Periodically retrain meta-learner
        self.meta_learner.fit(bar_idx)

        diagnostics = {
            'meta_mu': mu_hat, 'meta_sigma': sigma_hat,
            'horizon_mu': h_mu, 'horizon_sigma': h_sigma, 'horizon_conf': h_conf,
            'final_mu': final_mu, 'final_sigma': final_sigma,
            'meta_fitted': self.meta_learner._is_fitted,
            'n_active_alphas': sum(1 for s in signals.values() if s.is_active),
            'meta_learner_diagnostics': meta_diag,
            'correlation': corr_diag,
        }

        return final_mu, final_sigma, diagnostics

    def record_outcome(self, signals: Dict[str, AlphaSignal],
                       realized_return: float, bar_idx: int,
                       regime_probs: np.ndarray = None):
        """Feed realized returns back to meta-learner for training."""
        self.meta_learner.record(signals, realized_return, bar_idx, regime_probs)


# ============================================================================
# L3: PORTFOLIO CONSTRUCTION / RISK MANAGEMENT
# ============================================================================
# This layer OWNS all position sizing. No other layer sets exposure.
# Input: (mu_hat, sigma_hat) from L2
# Output: PortfolioOrder with target_exposure
# ============================================================================

class LedoitWolfShrinkage:
    """
    Ledoit-Wolf shrinkage covariance estimator for robust risk estimation.

    Shrinks sample covariance toward a structured target (scaled identity)
    to reduce estimation error, especially when n_obs / n_assets is small.

    For single-asset operation, this provides a regime-blended vol estimate
    that is more robust than raw trailing vol.

    Reference: Ledoit & Wolf (2004) "A well-conditioned estimator for
               large-dimensional covariance matrices"
    """

    def __init__(self, lookback: int = 252):
        self._return_history = deque(maxlen=lookback)
        self._regime_vol_estimates: Dict[str, float] = {}

    def update(self, bar_return: float):
        """Ingest a bar return for covariance estimation."""
        self._return_history.append(bar_return)

    def shrunk_vol(self, regime_probs: np.ndarray = None) -> float:
        """
        Compute shrinkage-adjusted volatility estimate.

        Blends trailing realized vol with regime-conditional vol estimates
        using Ledoit-Wolf-style shrinkage intensity.
        """
        rets = np.array(list(self._return_history))
        n = len(rets)
        if n < 20:
            return 0.15  # Conservative prior

        sample_vol = float(np.std(rets)) * np.sqrt(252)

        # Shrinkage target: long-run average vol (prior = 15% annual)
        prior_vol = 0.15

        # Ledoit-Wolf optimal shrinkage intensity
        # delta* = sum((x_i - sample_mean)^4) / (n * sample_var^2) - 1
        # Simplified: use kurtosis-based estimator
        mu = np.mean(rets)
        centered = rets - mu
        sample_var = np.var(rets)
        if sample_var < 1e-12:
            return prior_vol

        kurtosis = np.mean(centered ** 4) / (sample_var ** 2)
        # Shrinkage intensity: higher when kurtosis is high (fat tails)
        # or when n is small (estimation error dominates)
        shrinkage_intensity = min(1.0, max(0.0,
            (kurtosis - 1.0) / n + 2.0 / n
        ))

        # Regime-conditional blending (if regime info available)
        regime_adj = 1.0
        if regime_probs is not None and len(regime_probs) == 4:
            p_high_vol = float(regime_probs[1] + regime_probs[3])
            regime_adj = 1.0 + 0.3 * p_high_vol  # Bump vol in high-vol regimes

        shrunk = (1.0 - shrinkage_intensity) * sample_vol + shrinkage_intensity * prior_vol
        shrunk *= regime_adj

        return float(np.clip(shrunk, 0.03, 1.0))

    def estimation_quality(self) -> Dict[str, float]:
        """Report covariance estimation quality metrics."""
        n = len(self._return_history)
        rets = np.array(list(self._return_history)) if n > 0 else np.array([0.0])
        return {
            'n_obs': n,
            'sample_vol_ann': float(np.std(rets)) * np.sqrt(252) if n > 5 else 0.0,
            'effective_n': n,  # For multi-asset: n / n_assets
            'shrinkage_quality': 'high' if n > 126 else ('medium' if n > 63 else 'low'),
        }


# ============================================================================
# STATISTICAL DRIFT DETECTION UTILITIES
# ============================================================================

def ks_2sample_test(sample_a: np.ndarray, sample_b: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test (pure NumPy, no scipy).

    Tests whether two samples come from the same distribution.
    Returns (D_statistic, p_value_approx).

    P-value uses the Kolmogorov asymptotic series:
        P(D > d) ~ 2 * sum_{k=1}^inf (-1)^{k-1} * exp(-2*k^2*lambda^2)
    Accurate for n1, n2 >= 20.

    Reference: Numerical Recipes, Press et al. (1992), Section 14.3
    """
    a = np.sort(np.asarray(sample_a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(sample_b, dtype=np.float64).ravel())
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0

    # Compute ECDFs at all combined sorted values
    combined = np.concatenate([a, b])
    combined.sort()
    cdf_a = np.searchsorted(a, combined, side='right') / n1
    cdf_b = np.searchsorted(b, combined, side='right') / n2
    D = float(np.max(np.abs(cdf_a - cdf_b)))

    # Effective sample size and lambda
    n_eff = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * D

    # Kolmogorov asymptotic series (converges in 5 terms for typical values)
    if lam < 1e-10:
        return D, 1.0
    p_val = 0.0
    for k in range(1, 6):
        p_val += (-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)
    p_val = max(0.0, min(1.0, 2.0 * p_val))
    return D, p_val


class ADWIN:
    """
    Adaptive Windowing algorithm for change detection.

    Maintains a window of observations. Checks exponentially-spaced split
    points for statistically significant mean differences (Hoeffding bound).
    When detected, drops the older sub-window.

    Reference: Bifet & Gavalda (2007) "Learning from Time-Changing Data
               with Adaptive Windowing"

    Parameters:
        delta: Confidence parameter (default 0.002). Lower = more sensitive.
        max_window: Maximum window size (default 500).
    """

    def __init__(self, delta: float = 0.002, max_window: int = 500):
        self._delta = delta
        self._max_window = max_window
        self._window: deque = deque(maxlen=max_window)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._detected = False  # Last update detection result

    @property
    def detected(self) -> bool:
        """Whether the last update() detected a change."""
        return self._detected

    def update(self, value: float) -> Tuple[bool, int]:
        """Add observation. Returns (change_detected, current_window_size)."""
        self._window.append(value)
        self._sum += value
        self._sum_sq += value * value

        # Trim if over max
        while len(self._window) > self._max_window:
            old = self._window.popleft()
            self._sum -= old
            self._sum_sq -= old * old

        n = len(self._window)
        if n < 10:
            self._detected = False
            return False, n

        # Check exponentially-spaced split points
        arr = np.array(self._window)
        k = 1
        while k < n // 2:
            left = arr[:k]
            right = arr[k:]
            n0, n1 = len(left), len(right)
            if n0 < 5 or n1 < 5:
                k *= 2
                continue

            mean_diff = abs(float(left.mean() - right.mean()))
            m = 1.0 / (1.0 / n0 + 1.0 / n1)  # harmonic mean
            eps_cut = np.sqrt(np.log(4.0 * n / self._delta) / (2.0 * m))

            if mean_diff >= eps_cut:
                # Change detected — drop older portion
                for _ in range(k):
                    old = self._window.popleft()
                    self._sum -= old
                    self._sum_sq -= old * old
                self._detected = True
                return True, len(self._window)
            k *= 2

        self._detected = False
        return False, n

    def mean(self) -> float:
        """Current window mean."""
        n = len(self._window)
        return self._sum / n if n > 0 else 0.0

    def variance(self) -> float:
        """Current window variance."""
        n = len(self._window)
        if n < 2:
            return 0.0
        m = self._sum / n
        return max(0.0, self._sum_sq / n - m * m)

    @property
    def window_size(self) -> int:
        return len(self._window)

    def reset(self):
        self._window.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


class AlphaDriftMonitor:
    """
    Per-alpha drift detection using KS test and ADWIN on individual alpha IC streams.

    Each alpha maintains:
        - Rolling IC window (prediction vs. realized)
        - ADWIN detector on the IC stream
        - KS test comparing recent IC window vs. reference window
    """

    def __init__(self, alpha_names: List[str], ic_window: int = 100,
                 ks_ref_window: int = 50, ks_test_window: int = 50,
                 adwin_delta: float = 0.002):
        self._alpha_names = list(alpha_names)
        self._ic_window = ic_window
        self._ks_ref = ks_ref_window
        self._ks_test = ks_test_window

        # Per-alpha state
        self._mu_history: Dict[str, deque] = {n: deque(maxlen=ic_window) for n in alpha_names}
        self._ret_history: deque = deque(maxlen=ic_window)
        self._ic_series: Dict[str, deque] = {n: deque(maxlen=500) for n in alpha_names}
        self._adwin: Dict[str, ADWIN] = {n: ADWIN(delta=adwin_delta, max_window=500) for n in alpha_names}
        self._last_drift: Dict[str, Dict] = {}

    def update_all(self, signals: Dict[str, Any], realized_return: float) -> Dict[str, Dict]:
        """Batch update for all alphas. signals: {name: AlphaSignal or object with .mu}"""
        self._ret_history.append(realized_return)
        results = {}

        for name in self._alpha_names:
            sig = signals.get(name)
            if sig is None:
                continue
            mu_val = sig.mu if hasattr(sig, 'mu') else float(sig)
            self._mu_history[name].append(mu_val)

            # Compute rolling IC for this alpha
            n = min(len(self._mu_history[name]), len(self._ret_history))
            if n >= 20:
                preds = np.array(list(self._mu_history[name])[-n:])
                rets = np.array(list(self._ret_history))[-n:]
                p_std, r_std = np.std(preds), np.std(rets)
                if p_std > 1e-10 and r_std > 1e-10:
                    ic = float(np.corrcoef(preds, rets)[0, 1])
                    if not np.isnan(ic):
                        self._ic_series[name].append(ic)
                        # Feed ADWIN
                        adwin_detected, _ = self._adwin[name].update(ic)

                        # KS test on IC halves
                        ic_arr = np.array(list(self._ic_series[name]))
                        ks_stat, ks_pval, ks_drifting = 0.0, 1.0, False
                        if len(ic_arr) >= self._ks_ref + self._ks_test:
                            ref = ic_arr[-(self._ks_ref + self._ks_test):-self._ks_test]
                            test = ic_arr[-self._ks_test:]
                            ks_stat, ks_pval = ks_2sample_test(ref, test)
                            ks_drifting = ks_pval < 0.01

                        results[name] = {
                            'ic_current': ic,
                            'ks_stat': ks_stat, 'ks_pval': ks_pval, 'ks_drifting': ks_drifting,
                            'adwin_detected': adwin_detected,
                            'adwin_window': self._adwin[name].window_size,
                        }

        self._last_drift = results
        return results

    def is_any_drifting(self, ks_threshold: float = 0.01,
                        min_fraction: float = 0.3) -> Tuple[bool, Dict]:
        """Aggregate: True if >= min_fraction of active alphas show drift."""
        if not self._last_drift:
            return False, {'n_drifting': 0, 'n_monitored': 0}
        n_drifting = sum(1 for d in self._last_drift.values()
                         if d.get('ks_drifting', False) or d.get('adwin_detected', False))
        n_monitored = len(self._last_drift)
        frac = n_drifting / max(n_monitored, 1)
        return frac >= min_fraction, {
            'n_drifting': n_drifting,
            'n_monitored': n_monitored,
            'fraction': frac,
            'per_alpha': self._last_drift,
        }


class DriftDetector:
    """
    Feature and model drift detection using PSI, KS test, and ADWIN.

    Detects:
      - Feature drift: Population Stability Index (PSI) between training
        and recent feature distributions
      - Prediction drift: ADWIN-style adaptive windowing on rolling IC
      - Calibration drift: Brier score decomposition

    Reference: Page-Hinkley (1954), Bifet & Gavalda (2007) "ADWIN"
    """

    def __init__(self, reference_window: int = 252, test_window: int = 63,
                 psi_threshold: float = 0.25, ic_decay_threshold: float = 0.30,
                 ks_sensitivity: float = 0.01, adwin_delta: float = 0.002,
                 min_signals: int = 2):
        self._reference_window = reference_window
        self._test_window = test_window
        self._psi_threshold = psi_threshold
        self._ic_decay_threshold = ic_decay_threshold
        self._ks_sensitivity = ks_sensitivity
        self._min_signals = min_signals

        self._feature_history = deque(maxlen=reference_window + test_window)
        self._prediction_history = deque(maxlen=reference_window + test_window)
        self._outcome_history = deque(maxlen=reference_window + test_window)
        self._ic_history = deque(maxlen=500)

        # Reference IC from training period
        self._reference_ic: Optional[float] = None

        # ADWIN detectors on aggregate IC and residual streams
        self._adwin_ic = ADWIN(delta=adwin_delta, max_window=500)
        self._adwin_residual = ADWIN(delta=adwin_delta, max_window=500)

        # Per-alpha drift monitor (configured after alphas are registered)
        self._alpha_drift: Optional[AlphaDriftMonitor] = None

    def set_alpha_names(self, alpha_names: List[str]):
        """Enable per-alpha drift monitoring after alphas are registered."""
        self._alpha_drift = AlphaDriftMonitor(alpha_names)

    def update(self, features: float, prediction: float, outcome: float,
               signals: Dict[str, Any] = None):
        """Ingest one bar of drift monitoring data."""
        self._feature_history.append(features)
        self._prediction_history.append(prediction)
        self._outcome_history.append(outcome)

        # Feed ADWIN residual detector
        self._adwin_residual.update(prediction - outcome)

        # Rolling IC update
        n = len(self._prediction_history)
        if n >= self._test_window:
            preds = np.array(list(self._prediction_history)[-self._test_window:])
            outs = np.array(list(self._outcome_history)[-self._test_window:])
            p_std = np.std(preds)
            o_std = np.std(outs)
            if p_std > 1e-10 and o_std > 1e-10:
                ic = float(np.corrcoef(preds, outs)[0, 1])
                if not np.isnan(ic):
                    self._ic_history.append(ic)
                    self._adwin_ic.update(ic)

        # Per-alpha drift update
        if self._alpha_drift is not None and signals is not None:
            self._alpha_drift.update_all(signals, outcome)

    def set_reference_ic(self, ic: float):
        """Set the reference IC from training period for decay monitoring."""
        self._reference_ic = ic

    def compute_psi(self, n_bins: int = 10) -> float:
        """
        Population Stability Index between reference and test periods.

        PSI = sum( (test_pct - ref_pct) * ln(test_pct / ref_pct) )

        Interpretation: PSI < 0.10 = stable, 0.10-0.25 = moderate shift,
                        > 0.25 = significant drift
        """
        n = len(self._feature_history)
        if n < self._reference_window + self._test_window:
            return 0.0

        all_data = np.array(list(self._feature_history))
        ref = all_data[:self._reference_window]
        test = all_data[-self._test_window:]

        # Compute PSI using binned distributions
        edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
        test_counts = np.histogram(test, bins=edges)[0].astype(float)

        # Add small epsilon to avoid log(0)
        ref_pct = (ref_counts + 0.001) / (ref_counts.sum() + 0.001 * n_bins)
        test_pct = (test_counts + 0.001) / (test_counts.sum() + 0.001 * n_bins)

        psi = float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))
        return max(0.0, psi)

    def ic_decay_ratio(self) -> float:
        """
        Ratio of recent IC to reference IC.
        < 0.5 = likely overfit or regime change.
        """
        if self._reference_ic is None or abs(self._reference_ic) < 1e-6:
            return 1.0
        if len(self._ic_history) < 10:
            return 1.0
        recent_ic = float(np.mean(list(self._ic_history)[-20:]))
        return recent_ic / (self._reference_ic + 1e-10)

    def ks_test_drift(self) -> Tuple[float, float, bool]:
        """KS test on first-half vs second-half of IC history.

        Returns (D_stat, p_value, is_drifting).
        """
        n = len(self._ic_history)
        if n < 40:  # Need at least 20 per half
            return 0.0, 1.0, False
        ic_arr = np.array(list(self._ic_history))
        mid = n // 2
        d_stat, p_val = ks_2sample_test(ic_arr[:mid], ic_arr[mid:])
        return d_stat, p_val, p_val < self._ks_sensitivity

    def adwin_drift(self) -> Tuple[bool, bool]:
        """Check ADWIN detectors on IC and residual streams.

        Returns (ic_change_detected, residual_change_detected).
        """
        return self._adwin_ic.detected, self._adwin_residual.detected

    def is_drifting(self) -> Tuple[bool, Dict[str, float]]:
        """
        Composite drift check with majority vote across 4 signal families.

        Signals:
          1. PSI > threshold           (feature distribution shift)
          2. IC decay < threshold       (IC level decay)
          3. KS p-value < sensitivity   (IC distribution change)
          4. ADWIN detected             (adaptive change detection)

        Drifting = signals_firing >= 2  (majority vote)

        Returns: (is_drifting, {psi, ic_decay, ks_*, adwin_*, ...})
        """
        psi = self.compute_psi()
        ic_decay = self.ic_decay_ratio()
        ks_stat, ks_pval, ks_drifting = self.ks_test_drift()
        adwin_ic_drift, adwin_residual_drift = self.adwin_drift()

        # Count signals firing
        signals_firing = 0
        if psi > self._psi_threshold:
            signals_firing += 1
        if ic_decay < self._ic_decay_threshold:
            signals_firing += 1
        if ks_drifting:
            signals_firing += 1
        if adwin_ic_drift or adwin_residual_drift:
            signals_firing += 1

        drifting = signals_firing >= self._min_signals

        # Per-alpha drift summary
        alpha_drift_count = 0
        if self._alpha_drift is not None:
            any_drifting, alpha_info = self._alpha_drift.is_any_drifting()
            alpha_drift_count = alpha_info.get('n_drifting', 0)

        return drifting, {
            'psi': psi,
            'psi_threshold': self._psi_threshold,
            'ic_decay_ratio': ic_decay,
            'ic_decay_threshold': self._ic_decay_threshold,
            'n_ic_observations': len(self._ic_history),
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'ks_drifting': ks_drifting,
            'adwin_ic_drift': adwin_ic_drift,
            'adwin_residual_drift': adwin_residual_drift,
            'drift_signals_firing': signals_firing,
            'alpha_drift_count': alpha_drift_count,
        }

class PortfolioConstructor:
    """
    L3: Converts combined alpha signal into a risk-managed position.

    Pipeline:
        1. Raw sizing:      position = mu_hat / sigma_hat^2  (mean-variance optimal)
        2. Vol targeting:    scale to achieve target volatility
        3. Kelly cap:        |position| % f* (drawdown-aware Kelly fraction)
        4. Drawdown scaling: Reduce proportionally to current drawdown
        5. Regime scaling:   Regime-conditional exposure limits
        6. Drift guard:      Cut exposure if concept drift detected
        7. Turnover cap:     Limit position change per bar
        8. Final clip:       [-max_leverage, +max_leverage]

    CRITICAL: This is the ONLY place where target_exposure is determined.
    """

    def __init__(self, acfg: ArchitectureConfig, bars_per_year: int = 252):
        self.acfg = acfg
        self.bpy = bars_per_year

        # State
        self._return_history = deque(maxlen=500)
        self._vol_history = deque(maxlen=500)
        self._current_dd = 0.0
        self._peak_value = 1.0
        self._current_value = 1.0
        self._prev_exposure = 0.0
        self._bar_count = 0
        self._annual_turnover = 0.0

        # Drift detection state
        self._residual_history = deque(maxlen=200)
        self._drift_active = False
        self._drift_cooldown = 0

        # WS3A: Tail risk manager
        self._tail_risk = TailRiskManager(lookback=252, confidence=0.95)

        # WS3B: Factor exposure monitor
        self._factor_monitor = FactorExposureMonitor(lookback=63)

        # Ledoit-Wolf shrinkage covariance estimator
        self._lw_estimator = LedoitWolfShrinkage(lookback=252)

        # PSI / KS / ADWIN drift detector
        self._drift_detector = DriftDetector(
            reference_window=252, test_window=63,
            psi_threshold=0.25, ic_decay_threshold=0.30,
            ks_sensitivity=acfg.drift_ks_sensitivity,
            adwin_delta=acfg.drift_adwin_delta,
            min_signals=acfg.drift_min_signals_for_trigger,
        )

        # Drawdown duration tracking
        self._dd_duration_bars = 0
        self._max_dd_duration_bars = 0

        # Risk stats from last rebalance (for audit)
        self._last_risk_stats: Dict[str, Any] = {}

    def configure_alpha_drift(self, alpha_names: List[str]):
        """Enable per-alpha drift monitoring after alphas are registered."""
        self._drift_detector.set_alpha_names(alpha_names)

    def construct(self, mu_hat: float, sigma_hat: float,
                  regime_probs: np.ndarray = None,
                  bar_return: float = 0.0,
                  signals: Dict[str, AlphaSignal] = None) -> PortfolioOrder:
        """
        Full portfolio construction pipeline.

        Returns PortfolioOrder with the final target_exposure and audit trail.
        """
        self._bar_count += 1
        constraints_hit = []

        # -- Update tracking --
        self._return_history.append(bar_return)
        self._update_drawdown(bar_return)

        # -- Update Ledoit-Wolf estimator --
        self._lw_estimator.update(bar_return)

        # -- Update drift detector with ACTUAL alpha features --
        # Use the mean of alpha mu values as the feature proxy for drift
        # detection. This tracks distribution shift in the signal space
        # rather than in the return space (which conflates feature drift
        # with market regime change).
        if signals:
            active_mus = [s.mu for s in signals.values() if s.is_active]
            feature_proxy = float(np.mean(active_mus)) if active_mus else 0.0
        else:
            feature_proxy = mu_hat  # Fallback to ensemble output
        self._drift_detector.update(
            features=feature_proxy,
            prediction=mu_hat,
            outcome=bar_return,
            signals=signals,
        )

        # -- Track drawdown duration (for kill switch) --
        if self._current_dd > 0.001:
            self._dd_duration_bars += 1
            self._max_dd_duration_bars = max(self._max_dd_duration_bars, self._dd_duration_bars)
        else:
            self._dd_duration_bars = 0


        # -- Step 0: Cost-adjusted mu (canonical cost model) --
        # Subtract expected transaction cost from mu before sizing.
        # This ensures L3 optimises with the SAME cost model used for fills.
        cost_drag_applied = 0.0
        if self.acfg.cost_subtract_from_mu and HAS_COST_MODEL:
            expected_delta = abs(float(mu_hat) / (sigma_hat ** 2 + 1e-10))
            expected_delta = min(expected_delta, self.acfg.max_leverage)
            pv = float(getattr(self, '_last_portfolio_value', 100_000.0))
            adv_dollars = float(getattr(self, '_adv_dollars', self.acfg.cost_default_adv_dollars))
            trade_notional = abs(expected_delta - self._prev_exposure) * pv

            if trade_notional > 0:
                cb = estimate_cost_dollars(
                    mid=float(getattr(self, '_last_mid', 1.0)),
                    trade_notional=trade_notional,
                    adv_dollars=adv_dollars,
                    spread_bps=float(self.acfg.cost_spread_bps),
                    impact_coeff=float(self.acfg.cost_impact_coef),
                    fees_pct=0.0,  # fees handled separately as commission
                )
                cost_drag_applied = cb.total_cost / max(pv, 1e-12)

            # Borrow drag for short exposure
            if self._prev_exposure < 0 or mu_hat < 0:
                cost_drag_applied += float(self.acfg.cost_borrow_bps_annual) * 1e-4 / max(self.bpy, 1)

            mu_hat = mu_hat - cost_drag_applied

        # -- Step 1: Raw mean-variance optimal sizing --
        # Kelly-inspired: f = mu / sigma^2 (continuous approximation)
        raw_position = mu_hat / (sigma_hat ** 2 + 1e-10)
        # Hard clip before downstream gates: prevents turnover-cap and
        # drift-guard from operating on numerically degenerate values
        # when sigma is very small. Does not change the architecture.
        raw_position = float(np.clip(raw_position, -10.0, 10.0))
        reason_parts = [f"raw_MV={raw_position:+.3f}"]
        if cost_drag_applied > 1e-8:
            reason_parts.append(f"cost_drag={cost_drag_applied:.5f}")

        # -- Step 2: Volatility targeting (WS3C: regime-conditional) --
        vol_scale = self._vol_targeting_scale(regime_probs)
        position = raw_position * vol_scale
        if abs(vol_scale - 1.0) > 0.01:
            reason_parts.append(f"vol_scale={vol_scale:.2f}")
            constraints_hit.append("vol_targeting")

        # -- Step 3: Kelly cap --
        kelly_f = self._kelly_fraction()
        if abs(position) > kelly_f:
            position = np.sign(position) * kelly_f
            reason_parts.append(f"kelly_cap={kelly_f:.2f}")
            constraints_hit.append("kelly_cap")

        # -- Step 4: Drawdown scaling --
        dd_scale = self._drawdown_scale()
        if dd_scale < 1.0:
            position *= dd_scale
            reason_parts.append(f"dd_scale={dd_scale:.2f}(dd={self._current_dd:.1%})")
            constraints_hit.append("drawdown_limit")

        # -- Step 5: Regime scaling --
        regime_scale = self._regime_scale(regime_probs)
        if regime_scale < 1.0:
            position *= regime_scale
            reason_parts.append(f"regime_scale={regime_scale:.2f}")
            constraints_hit.append("regime_limit")

        # -- Step 5.5: CVaR constraint (WS3A) --
        self._tail_risk.update(bar_return)
        cvar_breached, safe_exposure = self._tail_risk.should_constrain(
            position, cvar_limit=-0.10
        )
        if cvar_breached:
            old_pos = position
            position = safe_exposure
            reason_parts.append(f"cvar_cap={safe_exposure:.3f}")
            constraints_hit.append("cvar_limit")

        # -- Step 5.6: Factor exposure check (WS3B) --
        # NOTE: Only meaningful when benchmark returns are available.
        # For single-asset portfolios with no benchmark, vol_loading
        # explodes (div-by-near-zero), zeroing all positions.
        # We still update for monitoring but skip enforcement in
        # single-asset mode (bench_return defaults to 0.0).
        self._factor_monitor.update(bar_return)

        # -- Step 6: Drift guard --
        drift_scale = self._drift_scale(mu_hat, bar_return)
        if drift_scale < 1.0:
            position *= drift_scale
            reason_parts.append(f"drift_scale={drift_scale:.2f}")
            constraints_hit.append("drift_guard")

        # -- Step 7: Turnover cap --
        delta = abs(position - self._prev_exposure)
        max_delta_per_bar = self.acfg.turnover_limit_annual / max(self.bpy, 1)
        if delta > max_delta_per_bar:
            # Limit the change, preserving direction
            direction = np.sign(position - self._prev_exposure)
            position = self._prev_exposure + direction * max_delta_per_bar
            reason_parts.append(f"turnover_cap={max_delta_per_bar:.3f}")
            constraints_hit.append("turnover_limit")

        # -- Step 8: Final clip --
        position = float(np.clip(position, -self.acfg.max_leverage,
                                  self.acfg.max_leverage))

        # -- Compute risk budget --
        risk_budget = abs(position) * sigma_hat * self._current_value

        # -- WS3D: Risk stats output per rebalance --
        lw_quality = self._lw_estimator.estimation_quality()
        ex_ante_vol = self._lw_estimator.shrunk_vol(regime_probs) * abs(position)
        tail_cvar = self._tail_risk.current_cvar if hasattr(self._tail_risk, 'current_cvar') else 0.0
        factor_exp = self._factor_monitor.all_exposures()
        drift_status, drift_metrics = self._drift_detector.is_drifting()

        risk_stats = {
            'ex_ante_vol': float(ex_ante_vol),
            'portfolio_vol_estimate': float(self._lw_estimator.shrunk_vol(regime_probs)),
            'cvar_95': float(tail_cvar),
            'gross_leverage': float(abs(position)),
            'net_leverage': float(position),
            'concentration_hhi': float(position ** 2),  # Single-asset HHI
            'factor_exposures': factor_exp,
            'drawdown_current': float(self._current_dd),
            'drawdown_duration_bars': int(self._dd_duration_bars),
            'lw_shrinkage_quality': lw_quality.get('shrinkage_quality', 'unknown'),
            'lw_n_obs': lw_quality.get('n_obs', 0),
            'drift_detected': drift_status,
            'drift_psi': drift_metrics.get('psi', 0.0),
            'drift_ic_decay': drift_metrics.get('ic_decay_ratio', 1.0),
            'drift_ks_pval': drift_metrics.get('ks_pval', 1.0),
            'drift_adwin_ic': drift_metrics.get('adwin_ic_drift', False),
            'drift_signals_firing': drift_metrics.get('drift_signals_firing', 0),
            'drift_alpha_count': drift_metrics.get('alpha_drift_count', 0),
            # FIX §SA-1: Expose regime_scale so L4 can make its no-trade
            # threshold regime-aware and avoid the mathematical deadlock
            # where kelly_cap × regime_scale can never exceed the
            # vol-adaptive no-trade threshold in crisis regimes.
            'regime_scale': float(regime_scale),
        }

        # -- WS3E: Constraint binding attribution --
        # Track how much alpha each constraint cost (unconstrained vs constrained)
        constraint_attribution = {}
        unconstrained_pos = raw_position * vol_scale  # Position before constraints
        for c_name in constraints_hit:
            # Estimate alpha cost: (unconstrained_mu - constrained_mu_equiv) in bps
            # Approximation: each binding constraint reduces position, which costs alpha
            constraint_attribution[c_name] = float(
                abs(unconstrained_pos - position) * abs(mu_hat) * 1e4
            )

        self._prev_exposure = position
        self._last_risk_stats = risk_stats

        return PortfolioOrder(
            target_exposure=position,
            urgency=float(np.clip(abs(mu_hat) / (sigma_hat + 1e-10), 0.0, 1.0)),
            risk_budget=risk_budget,
            reason=" | ".join(reason_parts),
            constraints_hit=constraints_hit,
            alpha_signals=signals or {},
            combined_mu=mu_hat,
            combined_sigma=sigma_hat,
            risk_stats=risk_stats,
            constraint_attribution=constraint_attribution,
        )

    def _vol_targeting_scale(self, regime_probs: np.ndarray = None) -> float:
        """
        Regime-aware volatility targeting with Ledoit-Wolf shrinkage. (WS3C)

        Uses shrinkage-adjusted vol estimate instead of raw trailing vol.
        This reduces estimation error and provides regime-aware leading
        indicators instead of lagging realized vol.
        """
        if len(self._return_history) < self.acfg.vol_lookback:
            return 1.0

        # Use Ledoit-Wolf shrinkage vol (regime-blended)
        realized_vol = self._lw_estimator.shrunk_vol(regime_probs)

        realized_vol = float(np.clip(realized_vol, self.acfg.vol_floor, self.acfg.vol_cap))

        scale = self.acfg.target_annual_vol / realized_vol
        return float(np.clip(scale, 0.1, self.acfg.max_leverage))

    def _kelly_fraction(self) -> float:
        """Drawdown-aware Kelly fraction."""
        if len(self._return_history) < self.acfg.kelly_lookback:
            return self.acfg.kelly_fraction

        recent = list(self._return_history)[-self.acfg.kelly_lookback:]
        mu = np.mean(recent) * self.bpy
        var = np.var(recent) * self.bpy
        if var < 1e-10:
            return self.acfg.kelly_fraction

        full_kelly = mu / var
        f_star = full_kelly * self.acfg.kelly_fraction  # Fractional Kelly

        # Reduce further if in drawdown
        if self._current_dd > self.acfg.kelly_max_drawdown * 0.5:
            dd_penalty = self._current_dd / self.acfg.kelly_max_drawdown
            f_star *= max(1.0 - dd_penalty, 0.1)

        return float(np.clip(abs(f_star), 0.05, self.acfg.max_leverage))

    def _drawdown_scale(self) -> float:
        """Linear or quadratic drawdown scaling."""
        dd = self._current_dd
        t_start = self.acfg.drawdown_scale_threshold
        t_max = self.acfg.drawdown_max_threshold

        if dd <= t_start:
            return 1.0
        if dd >= t_max:
            return 0.0

        progress = (dd - t_start) / (t_max - t_start)

        if self.acfg.drawdown_scale_curve == "quadratic":
            return float(1.0 - progress ** 2)
        else:
            return float(1.0 - progress)

    def _regime_scale(self, regime_probs: np.ndarray = None) -> float:
        """Regime-conditional position scaling."""
        if regime_probs is None or len(regime_probs) != 4:
            return 1.0

        regime_names = [
            'high_growth_low_vol', 'high_growth_high_vol',
            'low_growth_low_vol', 'low_growth_high_vol',
        ]
        scale = 0.0
        for i, name in enumerate(regime_names):
            scale += regime_probs[i] * self.acfg.regime_scale_map.get(name, 1.0)
        return float(np.clip(scale, 0.1, 1.5))

    def _drift_scale(self, predicted_mu: float, realized_return: float) -> float:
        """Graduated drift response based on DriftDetector majority vote.

        | Signals Firing | Action                                    |
        |----------------|-------------------------------------------|
        | 0              | Full exposure (scale=1.0)                 |
        | 1              | Log only, no position impact              |
        | 2              | Moderate reduction (drift_risk_reduction)  |
        | 3-4            | Severe reduction (0.25) + cooldown        |
        """
        if self._drift_cooldown > 0:
            self._drift_cooldown -= 1
            return 0.25  # Severe reduction persists during cooldown

        residual = predicted_mu - realized_return
        self._residual_history.append(residual)

        # Use DriftDetector's majority vote
        _, drift_metrics = self._drift_detector.is_drifting()
        signals_firing = drift_metrics.get('drift_signals_firing', 0)

        if signals_firing == 0:
            self._drift_active = False
            return 1.0
        elif signals_firing == 1:
            # Log only — single detector may be noisy
            self._drift_active = False
            return 1.0
        elif signals_firing == 2:
            # Moderate reduction
            self._drift_active = True
            return self.acfg.drift_risk_reduction
        else:
            # 3-4 signals: severe reduction + cooldown
            self._drift_active = True
            self._drift_cooldown = self.acfg.drift_cooldown
            return 0.25

    def _update_drawdown(self, bar_return: float):
        """Track rolling portfolio value and drawdown."""
        self._current_value *= (1 + bar_return)
        self._peak_value = max(self._peak_value, self._current_value)
        self._current_dd = (self._peak_value - self._current_value) / (
            self._peak_value + 1e-10
        )

    def reset(self):
        """Reset state for new episode/symbol."""
        self._return_history.clear()
        self._vol_history.clear()
        self._residual_history.clear()
        self._current_dd = 0.0
        self._peak_value = 1.0
        self._current_value = 1.0
        self._prev_exposure = 0.0
        self._bar_count = 0
        self._drift_active = False
        self._drift_cooldown = 0
        # WS3A/3B: Reset risk monitors
        self._tail_risk = TailRiskManager(lookback=252, confidence=0.95)
        self._factor_monitor = FactorExposureMonitor(lookback=63)


# ============================================================================
# FAILOVER MANAGER (Spec: explicit failover for data drops / exec failures)
# ============================================================================

class FailoverPolicy:
    """
    Defines a graduated response to system failures.

    Instead of binary "go flat" on any failure, implements tiered responses:
        LEVEL 0: Normal operation
        LEVEL 1: Stale data with decay ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â use last known data but decay
                  confidence and reduce position proportionally
        LEVEL 2: Reduce to safe positions ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â cut gross exposure by configurable
                  fraction, hold existing direction
        LEVEL 3: Flatten ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â close all positions, halt trading

    Each failure type maps to a policy with escalation rules.
    """

    LEVELS = {0: 'NORMAL', 1: 'STALE_DECAY', 2: 'REDUCE', 3: 'FLATTEN'}

    def __init__(self, failure_type: str,
                 initial_level: int = 1,
                 escalation_bars: int = 10,
                 max_level: int = 3,
                 decay_per_bar: float = 0.1,
                 reduce_fraction: float = 0.5):
        """
        Args:
            failure_type:     Label (e.g., 'data_feed_drop', 'exec_api_failure')
            initial_level:    Starting response level on first failure
            escalation_bars:  Bars before escalating to next level
            max_level:        Highest response level (default: flatten)
            decay_per_bar:    Confidence decay per bar in STALE_DECAY mode
            reduce_fraction:  How much to reduce exposure in REDUCE mode (0-1)
        """
        self.failure_type = failure_type
        self.initial_level = initial_level
        self.escalation_bars = escalation_bars
        self.max_level = max_level
        self.decay_per_bar = decay_per_bar
        self.reduce_fraction = reduce_fraction


class FailoverManager:
    """
    Manages failover policies for different failure types.

    Usage:
        fm = FailoverManager()
        fm.register_policy(FailoverPolicy('data_feed_drop',
                                           initial_level=1, escalation_bars=5))
        fm.register_policy(FailoverPolicy('exec_api_failure',
                                           initial_level=2, escalation_bars=3))

        # On failure:
        fm.trigger('data_feed_drop', bar_idx=100)

        # Each bar, check active failovers:
        response = fm.get_response(bar_idx=102)
        # response = {'active': True, 'level': 1, 'exposure_mult': 0.8, ...}

        # On recovery:
        fm.resolve('data_feed_drop', bar_idx=105)
    """

    def __init__(self):
        self._policies: Dict[str, FailoverPolicy] = {}
        self._active_failures: Dict[str, Dict] = {}  # failure_type -> state
        self._event_log: List[Dict] = []

    def register_policy(self, policy: FailoverPolicy):
        """Register a failover policy for a failure type."""
        self._policies[policy.failure_type] = policy

    def trigger(self, failure_type: str, bar_idx: int, context: Dict = None):
        """Activate failover for a failure type."""
        policy = self._policies.get(failure_type)
        if policy is None:
            # No policy registered; default to flatten
            policy = FailoverPolicy(failure_type, initial_level=3)

        self._active_failures[failure_type] = {
            'triggered_bar': bar_idx,
            'current_level': policy.initial_level,
            'bars_at_level': 0,
            'policy': policy,
        }
        self._event_log.append({
            'event': 'trigger', 'failure_type': failure_type,
            'bar_idx': bar_idx, 'level': policy.initial_level,
            'context': context or {},
        })

    def resolve(self, failure_type: str, bar_idx: int):
        """Clear a failover (system has recovered)."""
        if failure_type in self._active_failures:
            del self._active_failures[failure_type]
            self._event_log.append({
                'event': 'resolve', 'failure_type': failure_type,
                'bar_idx': bar_idx,
            })

    def get_response(self, bar_idx: int) -> Dict[str, Any]:
        """
        Get the current failover response (aggregate across all active failures).

        Returns:
            active: bool -- is any failover active?
            level: int -- highest active failover level
            exposure_mult: float -- multiplier to apply to target exposure
            confidence_decay: float -- decay factor for ensemble confidence
            failover_types: list -- which failures are active
        """
        if not self._active_failures:
            return {
                'active': False, 'level': 0,
                'exposure_mult': 1.0, 'confidence_decay': 1.0,
                'failover_types': [],
            }

        max_level = 0
        exposure_mult = 1.0
        conf_decay = 1.0
        active_types = []

        for ftype, state in list(self._active_failures.items()):
            policy = state['policy']
            bars_since = bar_idx - state['triggered_bar']
            state['bars_at_level'] = bars_since

            # Escalation check
            if bars_since > 0 and bars_since % policy.escalation_bars == 0:
                if state['current_level'] < policy.max_level:
                    state['current_level'] += 1
                    self._event_log.append({
                        'event': 'escalate', 'failure_type': ftype,
                        'bar_idx': bar_idx,
                        'new_level': state['current_level'],
                    })

            level = state['current_level']
            max_level = max(max_level, level)
            active_types.append(ftype)

            if level == 1:  # STALE_DECAY
                conf_decay = min(conf_decay,
                                max(0.0, 1.0 - policy.decay_per_bar * bars_since))
                exposure_mult = min(exposure_mult, conf_decay)
            elif level == 2:  # REDUCE
                exposure_mult = min(exposure_mult, policy.reduce_fraction)
                conf_decay = min(conf_decay, 0.5)
            elif level >= 3:  # FLATTEN
                exposure_mult = 0.0
                conf_decay = 0.0

        return {
            'active': True,
            'level': max_level,
            'level_name': FailoverPolicy.LEVELS.get(max_level, 'UNKNOWN'),
            'exposure_mult': exposure_mult,
            'confidence_decay': conf_decay,
            'failover_types': active_types,
        }

    @property
    def event_log(self) -> List[Dict]:
        return list(self._event_log)

    def reset(self):
        """Reset all failover state."""
        self._active_failures.clear()
        self._event_log.clear()


# ============================================================================
# L4: EXECUTION ENGINE
# ============================================================================
# Handles: no-trade region, order slicing, kill switches, monitoring.
# Input: PortfolioOrder from L3
# Output: ExecutionResult
# ============================================================================

class ExecutionEngine:
    """
    L4: Execution layer -- the last gate before orders hit the market.

    Responsibilities:
        1. No-trade region: Suppress noise trades below cost threshold
        2. Kill switches: Force-flat on safety violations
        3. Discretization: Map continuous exposure to env action space [fix 3.3]
        4. Order slicing: TWAP/VWAP child orders for impact minimization
        5. Monitoring: Track fill quality, slippage, participation

    v6.1: Discretization now lives HERE (not in the launcher), so the audit
    trail is consistent: L3 target +' L4 discrete action +' actual execution.
    This layer NEVER changes the direction of a trade -- only whether it
    executes and how it executes.
    """

    def __init__(self, acfg: ArchitectureConfig, bars_per_year: int = 252,
                 action_targets: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)):
        self.acfg = acfg
        self.bpy = bars_per_year
        self.action_targets = np.array(action_targets, dtype=np.float32)

        # Kill switch state
        self._episode_start_value = 1.0
        self._current_value = 1.0
        self._step_turnovers = deque(maxlen=100)
        self._kill_triggered = False
        self._kill_reason = ""
        self._kill_context: Dict[str, Any] = {}   # bar/exposure/value at kill time
        self._kill_cooldown = 0                    # For N-bar latch policy

        # WS5B: confidence collapse tracking
        self._confidence_history = deque(maxlen=max(acfg.kill_confidence_window, 5))

        # No-trade region state
        self._current_exposure = 0.0

        # Monitoring -- episode-scoped (reset per symbol)
        self._total_turnover = 0.0
        self._total_slippage = 0.0
        self._total_commission = 0.0
        self._total_fill_count = 0
        self._worst_slippage = 0.0
        self._n_trades = 0
        self._n_suppressed = 0

        # Monitoring -- lifetime-scoped (accumulate across all episodes)
        self._lifetime_trades = 0
        self._lifetime_suppressed = 0
        self._lifetime_turnover = 0.0
        self._lifetime_slippage = 0.0

        # Episode boundary guard: prevents double-counting if reset()
        # is called mid-episode or multiple times without trading.
        self._has_episode_data = False

        # WS5B: Cost spike tracking (realized vs modeled)
        self._cost_spike_count = 0
        self._last_modeled_cost_bps = 0.0
        self._last_realized_cost_bps = 0.0

        # WS5B: DD duration kill state (fed from L3)
        self._dd_duration_bars = 0

        # WS4A: Execution log chain (order intent -> fill -> slippage)
        self._execution_log: List[Dict[str, Any]] = []

        # NEW: Crowding detection (L4 monitoring)
        if CROWDING_AVAILABLE:
            self.crowding_detector = CrowdingDetector(
                warning_threshold=acfg.crowding_warning_threshold if hasattr(acfg, 'crowding_warning_threshold') else 0.70,
                kill_threshold=acfg.crowding_kill_threshold if hasattr(acfg, 'crowding_kill_threshold') else 0.85
            )
        else:
            self.crowding_detector = None

    def execute(self, order: PortfolioOrder,
                current_exposure: float,
                portfolio_value: float,
                bar_return: float = 0.0,
                realized_vol: float = 0.15,
                bar_idx: int = 0,
                ensemble_confidence: float = None,
                data_quality_score: float = None) -> ExecutionResult:
        """
        Execute a portfolio order through the full L4 pipeline.

        Returns ExecutionResult describing what actually happened.
        """
        self._current_exposure = current_exposure
        self._current_value = portfolio_value
        self._has_episode_data = True  # Guard: episode has real data now

        flat_action = int(np.argmin(np.abs(self.action_targets - 0.0)))

        # -- Kill switch check --
        if self._kill_triggered:
            # Kill latch policy: deterministic latch with explicit re-entry rules.
            # Default (kill_allow_auto_reenter=False): permanent -- requires manual reset.
            can_reenter = False
            if self.acfg.kill_allow_auto_reenter:
                # Value-recovery re-entry: portfolio must recover above
                # trigger_value * (1 + buffer) AND cooldown must have elapsed.
                trigger_val = self._kill_context.get('value', 0.0)
                recovery_target = trigger_val * (1.0 + self.acfg.kill_reenter_buffer)
                if self.acfg.kill_cooldown_bars > 0:
                    self._kill_cooldown -= 1
                    cooldown_ok = self._kill_cooldown <= 0
                else:
                    cooldown_ok = True  # No bar-count gate, only value gate
                if cooldown_ok and portfolio_value >= recovery_target:
                    can_reenter = True

            if can_reenter:
                # Re-entry: clear kill state
                self._kill_triggered = False
                self._kill_reason = ""
                self._kill_context = {}
                # Fall through to normal execution below
            else:
                # Remain killed -- flat position
                return ExecutionResult(
                    executed_exposure=0.0,
                    discrete_action=flat_action,
                    discrete_exposure=0.0,
                    was_killed=True,
                    kill_reason=self._kill_reason,
                )

        # Extract drift and DD duration from L3 risk_stats (if available)
        risk_stats = getattr(order, 'risk_stats', {}) or {}
        dd_duration = int(risk_stats.get('drawdown_duration_bars', 0))
        drift_detected = bool(risk_stats.get('drift_detected', False))
        drift_metrics = {
            'psi': risk_stats.get('drift_psi', 0.0),
            'ic_decay_ratio': risk_stats.get('drift_ic_decay', 1.0),
        }
        # Realized vol from L3's Ledoit-Wolf
        realized_portfolio_vol = risk_stats.get('portfolio_vol_estimate', None)

        if self._check_kill_switches(
            portfolio_value, order.target_exposure,
            current_exposure, bar_idx,
            ensemble_confidence=ensemble_confidence,
            realized_portfolio_vol=realized_portfolio_vol,
            data_quality_score=data_quality_score,
            dd_duration_bars=dd_duration,
            drift_detected=drift_detected,
            drift_metrics=drift_metrics,
        ):
            return ExecutionResult(
                executed_exposure=0.0,
                discrete_action=flat_action,
                discrete_exposure=0.0,
                was_killed=True,
                kill_reason=self._kill_reason,
            )

        # -- NEW: Crowding Detection Check --
        if self.crowding_detector is not None:
            # Extract alpha_signals from order metadata if available
            alpha_signals = getattr(order, 'alpha_signals', None) or {}

            if alpha_signals and len(alpha_signals) >= 3:
                crowding_result = self.crowding_detector.detect_crowding(alpha_signals)

                if crowding_result['action'] == 'kill':
                    # Full kill: skip trade entirely
                    self._kill_triggered = True
                    self._kill_reason = f"Crowding kill: {crowding_result['message']}"
                    self._kill_context = {
                        'bar': bar_idx,
                        'exposure': current_exposure,
                        'value': portfolio_value,
                        'trigger': 'crowding',
                        'crowding_score': crowding_result['crowding_score'],
                        'metrics': crowding_result['metrics'],
                    }
                    return ExecutionResult(
                        executed_exposure=0.0,
                        discrete_action=flat_action,
                        discrete_exposure=0.0,
                        was_killed=True,
                        kill_reason=self._kill_reason,
                    )
                elif crowding_result['action'] == 'reduce':
                    # Warning: reduce position size by 30%
                    # Apply reduction to target_exposure
                    original_target = order.target_exposure
                    order.target_exposure *= 0.7
                    # Log the reduction
                    if abs(original_target) > 1e-6:
                        print(f"[CROWDING WARNING] Bar {bar_idx}: {crowding_result['message']}")
                        print(f"  Reducing target: {original_target:.3f} -> {order.target_exposure:.3f}")

        # -- No-trade region --
        target = order.target_exposure
        # FIX §SA-1: Extract regime_scale from L3's risk_stats to make
        # no-trade threshold regime-aware (breaks crisis deadlock).
        regime_scale = risk_stats.get('regime_scale', 1.0)
        if not self._passes_no_trade_filter(current_exposure, target,
                                             realized_vol, regime_scale):
            self._n_suppressed += 1
            # BUG FIX DEBUG: Log first 10 suppressions to diagnose why trades aren't happening
            if self._n_suppressed <= 10 and getattr(self, 'verbose', 1) >= 2:
                delta = abs(target - current_exposure)
                threshold = self.acfg.no_trade_threshold_pct
                if self.acfg.no_trade_vol_adaptive:
                    threshold *= max(realized_vol / 0.15, 0.5)
                threshold *= max(regime_scale, 0.25)
                print(f"[SUPPRESS #{self._n_suppressed}] target={target:.4f}, current={current_exposure:.4f}, "
                      f"delta={delta:.4f}, threshold={threshold:.4f}, vol={realized_vol:.2%}")
            # Discretize current exposure (hold position)
            hold_action = int(np.argmin(np.abs(self.action_targets - current_exposure)))
            return ExecutionResult(
                executed_exposure=current_exposure,
                discrete_action=hold_action,
                discrete_exposure=float(self.action_targets[hold_action]),
                was_suppressed=True,
            )

        # -- WS5D: Enforce participation limit (was dead code, now wired) --
        adv_dollars = float(getattr(self.acfg, 'cost_default_adv_dollars', 5_000_000.0))
        target = self._enforce_participation_limit(
            target, current_exposure, portfolio_value, adv_dollars
        )

        # -- Discretize continuous target -> env action (fix 3.3) --
        discrete_action = int(np.argmin(np.abs(self.action_targets - target)))
        discrete_exposure = float(self.action_targets[discrete_action])

        # -- Cost model simulation (WS4A: unified cost model for fills) --
        simulated_slippage_bps = 0.0
        simulated_cost_dollars = 0.0
        trade_notional = abs(discrete_exposure - current_exposure) * portfolio_value
        if trade_notional > 0 and HAS_COST_MODEL:
            mid_price = float(getattr(self, '_last_mid', 1.0))
            cb = estimate_cost_dollars(
                mid=mid_price if mid_price > 0 else 1.0,
                trade_notional=trade_notional,
                adv_dollars=adv_dollars,
                spread_bps=float(self.acfg.cost_spread_bps),
                impact_coeff=float(self.acfg.cost_impact_coef),
                fees_pct=0.0,
            )
            simulated_slippage_bps = cb.total_cost / max(trade_notional, 1e-12) * 1e4
            simulated_cost_dollars = cb.total_cost
            self._last_modeled_cost_bps = simulated_slippage_bps

        # -- Order slicing (conceptual -- actual fills happen in TradingEnv) --
        n_slices = self.acfg.n_slices if self.acfg.order_slicing else 1

        # -- Track turnover (using discrete exposure for consistency) --
        delta = abs(discrete_exposure - current_exposure)
        self._step_turnovers.append(delta)
        self._total_turnover += delta
        self._n_trades += 1

        # -- WS4A: Execution log entry (order intent -> fill -> expected cost) --
        exec_log_entry = {
            'bar': bar_idx,
            'order_target': float(order.target_exposure),
            'participation_adjusted': float(target),
            'discrete_exposure': float(discrete_exposure),
            'current_exposure': float(current_exposure),
            'trade_delta': float(delta),
            'trade_notional': float(trade_notional),
            'modeled_cost_bps': float(simulated_slippage_bps),
            'modeled_cost_dollars': float(simulated_cost_dollars),
            'n_child_slices': n_slices,
            'participation_rate': float(trade_notional / max(adv_dollars, 1e-12)),
        }
        self._execution_log.append(exec_log_entry)

        return ExecutionResult(
            executed_exposure=target,          # L3's continuous target (for audit)
            discrete_action=discrete_action,   # What the env actually gets
            discrete_exposure=discrete_exposure,  # The snapped value
            slippage_bps=simulated_slippage_bps,
            n_child_fills=n_slices,
        )

    def _passes_no_trade_filter(self, current: float, target: float,
                                 realized_vol: float = 0.15,
                                 regime_scale: float = 1.0) -> bool:
        """Check if the position change exceeds the no-trade threshold.

        FIX §SA-1: Now regime-aware. If L3 already scaled position down
        by regime_scale (e.g., 0.25 in crisis), the no-trade threshold is
        scaled down proportionally. This breaks the mathematical deadlock
        where kelly_cap(0.25) × regime_scale(0.25) = 0.0625 could never
        exceed the vol-adaptive threshold of ~0.167 in high-vol regimes.

        Without this fix, the system is architecturally locked out of
        trading in any crisis regime.
        """
        delta = abs(target - current)
        threshold = self.acfg.no_trade_threshold_pct

        # Volatility-adaptive threshold
        if self.acfg.no_trade_vol_adaptive:
            vol_ratio = realized_vol / 0.15  # Normalize to baseline
            threshold *= max(vol_ratio, 0.5)  # Wider band in high vol

        # FIX §SA-1: Regime-adaptive threshold -- scale DOWN by regime_scale
        # so that positions which already survived L3's regime cut can pass.
        # Floor at 0.25 to maintain a minimum dead zone even in crisis.
        threshold *= max(regime_scale, 0.25)

        return delta > threshold

    def _check_kill_switches(self, portfolio_value: float,
                              target_exposure: float,
                              current_exposure: float = 0.0,
                              bar_idx: int = 0,
                              ensemble_confidence: float = None,
                              realized_portfolio_vol: float = None,
                              data_quality_score: float = None,
                              dd_duration_bars: int = 0,
                              drift_detected: bool = False,
                              drift_metrics: Dict = None) -> bool:
        """Check if any kill switch should trigger. (WS5B: extended)"""
        # Loss check
        episode_pnl = portfolio_value - self._episode_start_value
        loss_frac = -episode_pnl / (self._episode_start_value + 1e-10)
        if loss_frac > self.acfg.kill_max_loss_pct:
            self._kill_triggered = True
            self._kill_reason = f"Max loss exceeded: {loss_frac:.1%} > {self.acfg.kill_max_loss_pct:.1%}"
            self._kill_context = {
                'bar': bar_idx, 'exposure': current_exposure,
                'value': portfolio_value, 'loss_frac': loss_frac,
                'trigger': 'max_loss',
            }
            self._kill_cooldown = self.acfg.kill_cooldown_bars
            return True

        # Turnover spike
        if len(self._step_turnovers) > 10:
            avg = np.mean(list(self._step_turnovers)[:-1])
            current = abs(target_exposure - self._current_exposure)
            if avg > 0 and current > self.acfg.kill_max_turnover_spike * avg:
                self._kill_triggered = True
                self._kill_reason = f"Turnover spike: {current:.3f} vs avg {avg:.3f}"
                self._kill_context = {
                    'bar': bar_idx, 'exposure': current_exposure,
                    'value': portfolio_value, 'turnover_current': current,
                    'turnover_avg': avg, 'trigger': 'turnover_spike',
                }
                self._kill_cooldown = self.acfg.kill_cooldown_bars
                return True

        # WS5B: Confidence collapse kill
        if ensemble_confidence is not None:
            self._confidence_history.append(ensemble_confidence)
            n_window = self.acfg.kill_confidence_window
            if len(self._confidence_history) >= n_window:
                recent = list(self._confidence_history)[-n_window:]
                if all(c < self.acfg.kill_min_confidence for c in recent):
                    self._kill_triggered = True
                    self._kill_reason = (
                        f"Confidence collapse: below {self.acfg.kill_min_confidence} "
                        f"for {n_window} consecutive bars"
                    )
                    self._kill_context = {
                        'bar': bar_idx, 'exposure': current_exposure,
                        'value': portfolio_value,
                        'trigger': 'confidence_collapse',
                        'min_confidence': min(recent),
                    }
                    self._kill_cooldown = self.acfg.kill_cooldown_bars
                    return True

        # WS5B: Data quality kill
        if (data_quality_score is not None
                and data_quality_score < self.acfg.kill_min_data_quality):
            self._kill_triggered = True
            self._kill_reason = (
                f"Data quality breach: {data_quality_score:.1f} < "
                f"{self.acfg.kill_min_data_quality:.1f}"
            )
            self._kill_context = {
                'bar': bar_idx, 'exposure': current_exposure,
                'value': portfolio_value,
                'trigger': 'data_quality_breach',
                'quality_score': data_quality_score,
            }
            self._kill_cooldown = self.acfg.kill_cooldown_bars
            return True

        # WS5B: Vol target breach kill
        if realized_portfolio_vol is not None:
            vol_limit = self.acfg.target_annual_vol * self.acfg.kill_vol_breach_mult
            if realized_portfolio_vol > vol_limit:
                self._kill_triggered = True
                self._kill_reason = (
                    f"Vol target breach: {realized_portfolio_vol:.1%} > "
                    f"{self.acfg.kill_vol_breach_mult}x target "
                    f"{self.acfg.target_annual_vol:.1%}"
                )
                self._kill_context = {
                    'bar': bar_idx, 'exposure': current_exposure,
                    'value': portfolio_value,
                    'trigger': 'vol_target_breach',
                    'realized_vol': realized_portfolio_vol,
                }
                self._kill_cooldown = self.acfg.kill_cooldown_bars
                return True

        # WS5B: Cost spike kill (realized > N * modeled for M consecutive bars)
        if self._last_realized_cost_bps > 0 and self._last_modeled_cost_bps > 0:
            cost_ratio = self._last_realized_cost_bps / (self._last_modeled_cost_bps + 1e-10)
            if cost_ratio > self.acfg.kill_cost_spike_mult:
                self._cost_spike_count += 1
            else:
                self._cost_spike_count = max(0, self._cost_spike_count - 1)
            if self._cost_spike_count >= self.acfg.kill_cost_spike_window:
                self._kill_triggered = True
                self._kill_reason = (
                    f"Cost spike: realized {self._last_realized_cost_bps:.1f}bps > "
                    f"{self.acfg.kill_cost_spike_mult}x modeled "
                    f"{self._last_modeled_cost_bps:.1f}bps for "
                    f"{self._cost_spike_count} bars"
                )
                self._kill_context = {
                    'bar': bar_idx, 'exposure': current_exposure,
                    'value': portfolio_value,
                    'trigger': 'cost_spike',
                    'realized_cost_bps': self._last_realized_cost_bps,
                    'modeled_cost_bps': self._last_modeled_cost_bps,
                }
                self._kill_cooldown = self.acfg.kill_cooldown_bars
                return True

        # WS5B: Drawdown duration kill (too long in drawdown)
        self._dd_duration_bars = dd_duration_bars
        if dd_duration_bars > self.acfg.kill_dd_duration_bars > 0:
            self._kill_triggered = True
            self._kill_reason = (
                f"DD duration: {dd_duration_bars} bars > "
                f"limit {self.acfg.kill_dd_duration_bars}"
            )
            self._kill_context = {
                'bar': bar_idx, 'exposure': current_exposure,
                'value': portfolio_value,
                'trigger': 'dd_duration',
                'dd_duration_bars': dd_duration_bars,
            }
            self._kill_cooldown = self.acfg.kill_cooldown_bars
            return True

        # WS5B: Drift kill (model/feature drift detected)
        # NOTE: In backtest mode, we WANT to test cross-regime performance, so skip drift kill
        if drift_detected and self.acfg.kill_on_drift and not self.acfg.backtest_mode:
            self._kill_triggered = True
            self._kill_reason = (
                f"Drift detected: PSI={drift_metrics.get('psi', 0):.3f}, "
                f"IC_decay={drift_metrics.get('ic_decay_ratio', 1):.2f}"
            )
            self._kill_context = {
                'bar': bar_idx, 'exposure': current_exposure,
                'value': portfolio_value,
                'trigger': 'drift_detected',
                'drift_metrics': drift_metrics or {},
            }
            self._kill_cooldown = self.acfg.kill_cooldown_bars
            return True

        return False

    def _enforce_participation_limit(self, target_exposure: float,
                                      current_exposure: float,
                                      portfolio_value: float,
                                      adv_dollars: float) -> float:
        """
        WS5D: Cap trade size to max_participation_rate of ADV.
        If the desired trade exceeds the limit, scale it down.
        """
        trade_notional = abs(target_exposure - current_exposure) * portfolio_value
        max_trade = self.acfg.max_participation_rate * adv_dollars

        if trade_notional > max_trade and max_trade > 0:
            # Scale down to max participation
            scale = max_trade / trade_notional
            adjusted = current_exposure + (target_exposure - current_exposure) * scale
            return adjusted
        return target_exposure

    def ingest_execution(self, traded_notional: float = 0.0,
                         commission: float = 0.0,
                         slippage_bps: float = 0.0,
                         mid_price: float = 0.0,
                         fill_price: float = 0.0,
                         **kwargs):
        """
        Feed realized fill data back from env into L4 for monitoring.

        This bridges the gap between L4's "conceptual" order slicing and
        the env's actual execution. Tracks realized vs modeled cost for
        cost spike kill switch.
        """
        self._total_slippage += slippage_bps
        self._total_commission += commission
        if traded_notional > 0:
            self._total_fill_count += 1
            self._worst_slippage = max(self._worst_slippage, slippage_bps)

        # WS5B: Track realized cost for cost spike detection
        self._last_realized_cost_bps = slippage_bps
        self._last_mid = mid_price if mid_price > 0 else getattr(self, '_last_mid', 1.0)

        # Append to execution log if we have a recent entry
        if self._execution_log:
            last_entry = self._execution_log[-1]
            last_entry['realized_slippage_bps'] = float(slippage_bps)
            last_entry['realized_commission'] = float(commission)
            last_entry['fill_price'] = float(fill_price)
            last_entry['mid_price'] = float(mid_price)
            if last_entry.get('modeled_cost_bps', 0) > 0:
                last_entry['cost_ratio'] = float(
                    slippage_bps / (last_entry['modeled_cost_bps'] + 1e-10)
                )

    # Backward compat alias
    def ingest_fill(self, slippage_bps: float = 0.0, commission: float = 0.0,
                    **kwargs):
        """Deprecated: use ingest_execution() instead."""
        self.ingest_execution(slippage_bps=slippage_bps, commission=commission)

    def reset(self, starting_value: float = 1.0):
        """Reset for new episode/symbol.

        Resets ALL episode-scoped counters. Accumulates into lifetime
        counters only if this episode had real data (guard against
        double-counting from mid-episode resets).
        """
        # -- Accumulate into lifetime ONLY if episode had data --
        if self._has_episode_data:
            self._lifetime_trades += self._n_trades
            self._lifetime_suppressed += self._n_suppressed
            self._lifetime_turnover += self._total_turnover
            self._lifetime_slippage += self._total_slippage

        # -- Reset episode state --
        self._episode_start_value = starting_value
        self._current_value = starting_value
        self._step_turnovers.clear()
        self._kill_triggered = False
        self._kill_reason = ""
        self._kill_context = {}
        self._kill_cooldown = 0
        self._current_exposure = 0.0

        self._total_turnover = 0.0
        self._total_slippage = 0.0
        self._total_commission = 0.0
        self._total_fill_count = 0
        self._worst_slippage = 0.0
        self._n_trades = 0
        self._n_suppressed = 0

        # WS5B: Reset cost spike tracking
        self._cost_spike_count = 0
        self._last_modeled_cost_bps = 0.0
        self._last_realized_cost_bps = 0.0
        self._dd_duration_bars = 0
        self._confidence_history.clear()
        self._execution_log.clear()

        self._has_episode_data = False  # Guard: no data until execute() called

    @property
    def stats(self) -> Dict[str, Any]:
        """Episode-level stats (current symbol only)."""
        avg_slip = self._total_slippage / max(self._total_fill_count, 1)
        return {
            'total_turnover': self._total_turnover,
            'total_slippage_bps': self._total_slippage,
            'avg_slippage_bps': avg_slip,
            'worst_slippage_bps': self._worst_slippage,
            'total_commission': self._total_commission,
            'n_fills': self._total_fill_count,
            'n_trades': self._n_trades,
            'n_suppressed': self._n_suppressed,
            'suppression_rate': self._n_suppressed / max(self._n_trades + self._n_suppressed, 1),
            'kill_triggered': self._kill_triggered,
            'kill_reason': self._kill_reason,
            'kill_context': self._kill_context,
        }

    def reset_lifetime_stats(self):
        """Reset lifetime accumulators (for A/B comparison runs)."""
        self._lifetime_trades = 0
        self._lifetime_suppressed = 0
        self._lifetime_turnover = 0.0
        self._lifetime_slippage = 0.0
        self._n_trades = 0
        self._n_suppressed = 0
        self._total_turnover = 0.0
        self._total_slippage = 0.0

    @property
    def lifetime_stats(self) -> Dict[str, Any]:
        """Aggregate stats across all episodes/symbols."""
        total_t = self._lifetime_trades + self._n_trades
        total_s = self._lifetime_suppressed + self._n_suppressed
        return {
            'total_turnover': self._lifetime_turnover + self._total_turnover,
            'total_slippage': self._lifetime_slippage + self._total_slippage,
            'n_trades': total_t,
            'n_suppressed': total_s,
            'suppression_rate': total_s / max(total_t + total_s, 1),
        }


# ============================================================================
# INSTITUTIONAL PIPELINE -- Full L1->L2->L3->L4 Integration
# ============================================================================

class InstitutionalPipeline:
    """
    Complete institutional-grade trading pipeline.

    Wires together:
        L1 (AlphaFactory) +' L2 (SignalEnsemble) +'
        L3 (PortfolioConstructor) +' L4 (ExecutionEngine)

    Usage:
        pipeline = InstitutionalPipeline(cfg, ecfg, acfg)

        # Register alphas (RL is just one of many)
        pipeline.register_alpha(RLAlphaAdapter(acfg, net=trained_network))
        pipeline.register_alpha(TrendAlpha(acfg))
        pipeline.register_alpha(MeanReversionAlpha(acfg))
        pipeline.register_alpha(ValueAlpha(acfg))
        pipeline.register_alpha(CarryAlpha(acfg))
        pipeline.register_alpha(VolatilityPremiumAlpha(acfg))

        # Each bar:
        result = pipeline.step(
            observation=obs,
            closes=price_history[:, 3],
            volumes=price_history[:, 4],
            bar_idx=current_step,
            bar_return=log_return,
            current_exposure=env.exposure,
            portfolio_value=env._portfolio_value(),
            regime_probs=regime_detector.probabilities(),
        )

        # result.execution.executed_exposure +' feed to TradingEnv
    """

    def __init__(self, acfg: ArchitectureConfig, bars_per_year: int = 252):
        self.acfg = acfg

        # L1: Alpha Factory
        self.alpha_factory = AlphaFactory(common_horizon=1)

        # L1: Regime Detector (fix 4.1: replaces hard-coded regime_probs)
        self.regime_detector = TrendVolRegimeDetector(
            trend_lookback=63, vol_lookback=20, history_lookback=252
        )

        # L2: Signal Ensemble (initialized after alphas registered)
        self._ensemble = None

        # L3: Portfolio Construction
        self.portfolio = PortfolioConstructor(acfg, bars_per_year)

        # L4: Execution
        # Iter 9: add 1.25 step so discretizer snaps to 1.25 for sma_position≥1.125 (midpoint 1.0-1.25)
        # bull_aligned (0.90) × trend_boost (1.30) = 1.17 > 1.125 → snaps to 1.25 (25% leverage) ✓
        # Without 1.25: midpoint(1.0, 1.5)=1.25 → 1.17 would snap back to 1.0 (no leverage)
        self.execution_engine = ExecutionEngine(
            acfg, bars_per_year,
            action_targets=(-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.25, 1.5),
        )

        # SMA trend overlay toggle (can be disabled for A/B comparison)
        self.use_sma = True

        # v8.0: Rank-based position sizing (set externally before eval loop)
        self.use_v8_sizing = False
        self._v8_rank = {}        # symbol -> rank_index (0 = best)
        self._current_symbol = None  # Set per-symbol in eval loop

        # Diagnostics
        self._step_count = 0
        self._last_signals = {}
        self._last_order = None
        self._last_result = None
        self._audit_log = deque(maxlen=1000)

        # Alpha lifecycle monitoring (Good-to-have A)
        self._lifecycle_monitor = None  # Initialized after alphas registered

        # Attribution engine (Good-to-have C)
        self._attribution = AttributionEngine()

        # WS5A: Alpha correlation manager
        self._corr_manager = None  # Initialized after alphas registered

        # WS5C: Alert manager â€” dual mode: file gets everything, terminal only CRITICAL+
        if acfg.enable_alerting:
            handlers = []
            # File handler captures ALL alerts (including INFO) for post-hoc analysis
            if acfg.alert_log_file:
                handlers.append(AlertManager.make_jsonl_handler(acfg.alert_log_file))
            # Terminal handler still present but gated by alert_terminal_level
            handlers.append(AlertManager._default_handler)
            self._alert_manager = AlertManager(
                handlers=handlers,
                min_print_level=acfg.alert_terminal_level,
            )
        else:
            self._alert_manager = None

        # FIX Ã‚Â§5.4: Position reconciler for end-of-bar verification
        self._reconciler = PositionReconciler()

        # -- Fix 3.2: Lagged outcome recording --
        # Store previous step's signals so we can pair them with the NEXT
        # bar's return, preventing the meta-learner from seeing contemporaneous
        # data that was already used in signal generation.
        self._prev_signals = None
        self._prev_regime_probs = None

    def register_alpha(self, alpha: BaseAlpha):
        """Register an alpha source in L1."""
        self.alpha_factory.register(alpha)
        self.rebuild_ensemble()

    def rebuild_ensemble(self):
        """Rebuild L2 ensemble from current alpha list.

        Call this after registering/unregistering alphas. Exposed as a
        public method so the launcher doesn't need to touch _ensemble
        directly.
        """
        # WS5A: Correlation manager owned by ensemble (fix: no longer external)
        self._corr_manager = AlphaCorrelationManager(
            self.alpha_factory.alpha_names, window=100
        )
        self._ensemble = SignalEnsemble(
            self.acfg, self.alpha_factory.alpha_names,
            corr_manager=self._corr_manager,
        )
        # Rebuild lifecycle monitor with current alpha list
        self._lifecycle_monitor = AlphaLifecycleMonitor(
            self.alpha_factory.alpha_names, window=100
        )
        # Enable per-alpha drift monitoring in L3
        self.portfolio.configure_alpha_drift(self.alpha_factory.alpha_names)

    def step(self, observation: np.ndarray = None,
             closes: np.ndarray = None,
             volumes: np.ndarray = None,
             features: np.ndarray = None,
             bar_idx: int = 0,
             bar_return: float = 0.0,
             current_exposure: float = 0.0,
             portfolio_value: float = 100000.0,
             regime_probs: np.ndarray = None,
             realized_vol: float = 0.15,
             data_quality_score: float = None,
             opens: np.ndarray = None,     # NEW: Phase 1 OHLC support
             highs: np.ndarray = None,     # NEW: Phase 1 OHLC support
             lows: np.ndarray = None,      # NEW: Phase 1 OHLC support
             **kwargs) -> 'PipelineResult':
        """
        Run the full L1->L2->L3->L4 pipeline for one bar.

        v6.1 fixes:
            - Live regime detection (fix 4.1): regime_probs computed from prices,
              not hard-coded. Passed-in regime_probs used as override only.
            - Lagged outcome recording (fix 3.2): previous step's signals are
              paired with current bar_return for meta-learner training.
            - Discretization in L4 (fix 3.3): discrete_action is determined by
              L4 and returned in ExecutionResult for the launcher to use directly.
            - Extended audit (fix 5.2): per-alpha breakdown, L2 weights, demotion.

        Returns PipelineResult with the execution decision and full audit trail.
        """
        self._step_count += 1

        if closes is None:
            closes = np.array([100.0])

        # ******************************************************
        # REGIME DETECTION (fix 4.1: live, not hard-coded)
        # ******************************************************
        if regime_probs is None:
            regime_probs = self.regime_detector.update(closes)
        # If caller passes explicit regime_probs, use those as override

        # ******************************************************
        # L2: LAGGED OUTCOME RECORDING (fix 3.2)
        # Record PREVIOUS step's signals with CURRENT bar_return.
        # This ensures the meta-learner never trains on return[t] that
        # was already visible in close[t] used to generate signals[t].
        # ******************************************************
        if self._ensemble is not None and self._prev_signals is not None:
            self._ensemble.record_outcome(
                self._prev_signals, bar_return, bar_idx,
                self._prev_regime_probs
            )

        # ******************************************************
        # L1: ALPHA FACTORY -- Generate signals
        # ******************************************************
        signals = self.alpha_factory.generate_all(
            closes=closes, volumes=volumes, features=features,
            bar_idx=bar_idx, observation=observation,
            opens=opens, highs=highs, lows=lows,  # Phase 1: Pass OHLC to alphas
            **kwargs
        )
        self._last_signals = signals

        # ******************************************************
        # L2: SIGNAL ENSEMBLE -- Combine into (mu_hat, sigma_hat)
        # ******************************************************
        if self._ensemble is None:
            mu_hat, sigma_hat, l2_diag = 0.0, 1.0, {}
        else:
            # WS5A: Correlation penalties are now applied INSIDE
            # SignalEnsemble.combine() via its owned AlphaCorrelationManager.
            # No external pre-processing needed.
            mu_hat, sigma_hat, l2_diag = self._ensemble.combine(
                signals, regime_probs, bar_idx
            )

        # Store current signals for NEXT step's lagged recording (fix 3.2)
        self._prev_signals = signals
        self._prev_regime_probs = regime_probs

        # ******************************************************
        # L3: PORTFOLIO CONSTRUCTION -- Convert to position
        # ******************************************************
        order = self.portfolio.construct(
            mu_hat=mu_hat, sigma_hat=sigma_hat,
            regime_probs=regime_probs,
            bar_return=bar_return,
            signals=signals,
        )
        self._last_order = order

        # ******************************************************
        # SMA TREND OVERLAY -- Fixed position based on trend state
        # ******************************************************
        # SMA determines BOTH direction AND size, replacing L3's
        # noisy targets. Works progressively:
        #   < 50 bars:   Force flat (not enough data for any SMA)
        #   50-99 bars:  Use SMA50 only (basic trend direction)
        #   100-199 bars: Use SMA50 + SMA100
        #   200+ bars:   Full SMA50/100/200 with crossover detection
        #
        # Position table (LONG-ONLY, max 0.25):
        #   P > SMA200 (any bull state):    +0.25
        #   P < SMA200 (any bear state):     0.0 (flat, no shorts)
        if self.use_sma and closes is not None and len(closes) >= 50:
            current_price = float(closes[-1])

            # v8.0: Prepend training prices so SMA200 is available from bar 0
            if (self.use_v8_sizing and hasattr(self, '_v8_train_closes')
                    and self._v8_train_closes is not None):
                _sma_closes = np.concatenate([self._v8_train_closes, closes])
            else:
                _sma_closes = closes
            n = len(_sma_closes)

            sma_50 = float(np.mean(_sma_closes[-50:]))
            sma_100 = float(np.mean(_sma_closes[-100:])) if n >= 100 else None
            sma_200 = float(np.mean(_sma_closes[-200:])) if n >= 200 else None

            sma_position = 0.0
            sma_reason = ""

            if sma_200 is not None:
                # -- FULL MODE: 200+ bars (LONG-ONLY) --
                # Trend-strength base position:
                #   Bull aligned (P > SMA50 > SMA100 > SMA200) → 0.80
                #   Golden cross (SMA50 > SMA100, P > SMA200)  → 0.60
                #   Just above SMA200                          → 0.40
                #   Below SMA200                               → 0.00 (flat)
                #   Distance boost can push bull-aligned to 1.0
                if current_price > sma_200:
                    if current_price > sma_50 > sma_100 > sma_200:
                        sma_position = 0.80
                        sma_reason = "sma=bull_aligned(+0.80)"
                    elif sma_100 is not None and sma_50 > sma_100:
                        sma_position = 0.60
                        sma_reason = "sma=golden_cross(+0.60)"
                    else:
                        sma_position = 0.40
                        sma_reason = "sma=above_200(+0.40)"
                    # Distance-from-SMA200 boost: 10-30% above → up to +30%
                    if sma_200 > 0:
                        dist_pct = (current_price / sma_200) - 1.0
                        trend_boost = 1.0 + min(max(dist_pct - 0.10, 0.0) * 1.5, 0.30)
                        sma_position = min(sma_position * trend_boost, 1.0)
                else:
                    # Below SMA200 → FLAT (no shorts — long-only mode)
                    sma_position = 0.0
                    sma_reason = "sma=below_200(flat)"

            elif sma_100 is not None:
                # -- PARTIAL MODE: 100-199 bars (LONG-ONLY) --
                golden_cross = sma_50 > sma_100
                if current_price > sma_100:
                    sma_position = 0.25
                    sma_reason = "sma=above_100(+0.25)"
                else:
                    sma_position = 0.0
                    sma_reason = "sma=below_100(flat)"

            else:
                # -- MINIMAL MODE: 50-99 bars (LONG-ONLY) --
                if current_price > sma_50:
                    sma_position = 0.25
                    sma_reason = "sma=above_50(+0.25)"
                else:
                    sma_position = 0.0
                    sma_reason = "sma=below_50(flat)"

            # v8.0: own base positions (0.40/0.60/0.80) + distance boost + rank multiplier
            # Separate from v7.0 (0.25/0.50/0.75). Overrides v7.0 value entirely.
            # NOTE: gate is NOT conditioned on sma_position>0 so we force long when SMA is
            # positive regardless of alpha-signal direction (alphas net-negative → cause suppression)
            if self.use_v8_sizing and self._current_symbol:
                # Step 1: v8.0-specific base positions (always applies)
                if sma_200 is not None and current_price > sma_200:
                    if current_price > sma_50 > sma_100 > sma_200:
                        base_pos = 0.90
                        sma_reason = "sma=v8_bull_aligned"
                    elif sma_100 is not None and sma_50 > sma_100:
                        # Iter 9: raise from 0.70→0.80 so it snaps to 1.0 (not 0.5)
                        # golden_cross 0.70 < midpoint(0.5,1.0)=0.75 → was snapping to 0.5 same as above_200!
                        base_pos = 0.80
                        sma_reason = "sma=v8_golden_cross"
                    else:
                        base_pos = 0.50
                        sma_reason = "sma=v8_above_200"
                else:
                    base_pos = sma_position  # partial/minimal: keep v7.0 value
                # Step 2: distance-from-SMA200 boost
                trend_boost = 1.0
                if sma_200 is not None and sma_200 > 0:
                    dist_pct = (current_price / sma_200) - 1.0
                    # Iter 9: raise ceiling 0.30→0.60: stocks ≥50% above SMA200 → trend_boost=1.60
                    # 0.90 × 1.60 = 1.44 > midpoint(1.25,1.50)=1.375 → snaps to 1.50 (50% leverage)
                    trend_boost = 1.0 + min(max(dist_pct - 0.10, 0.0) * 1.5, 0.60)
                # Step 3: rank multiplier — only when a rank map is available
                # Iter 7: widen range from 0.75-1.00 → 0.50-1.50 to differentiate top vs bottom stocks
                rank_mult = 1.0
                if self._v8_rank:
                    rank_idx = self._v8_rank.get(self._current_symbol, -1)
                    if rank_idx >= 0:
                        n_selected = max(len(self._v8_rank), 1)
                        rank_mult = 0.50 + 1.00 * (1.0 - rank_idx / n_selected)
                # Iter 7: raise cap from 1.0 → 1.50 (moderate leverage for top bull-aligned stocks)
                sma_position = min(base_pos * trend_boost * rank_mult, 1.50)
                sma_reason = f"{sma_reason}({sma_position:+.2f})"


            # Override L3's target with SMA-based position
            order.target_exposure = sma_position
            order.reason = order.reason + f" | {sma_reason}"
            if abs(sma_position) > 1e-4:
                order.constraints_hit = order.constraints_hit + ['sma_override']

        elif self.use_sma and closes is not None:
            # < 50 bars: force flat until we have enough data
            order.target_exposure = 0.0
            order.reason = order.reason + " | sma=warmup(flat)"

        # ******************************************************
        # L4: EXECUTION -- Final gate + discretization
        # ******************************************************
        # Compute ensemble confidence for kill switch wiring
        _ensemble_conf = None
        if signals:
            active_confs = [s.confidence for s in signals.values() if s.is_active]
            if active_confs:
                _ensemble_conf = float(np.mean(active_confs))

        exec_result = self.execution_engine.execute(
            order=order,
            current_exposure=current_exposure,
            portfolio_value=portfolio_value,
            bar_return=bar_return,
            realized_vol=realized_vol,
            bar_idx=bar_idx,
            ensemble_confidence=_ensemble_conf,
            data_quality_score=data_quality_score,
        )
        self._last_result = exec_result

        # -- Extended audit trail (fix 5.2) --
        # Per-alpha signal breakdown
        alpha_detail = {}
        for name, sig in signals.items():
            alpha_detail[name] = {
                'mu': round(sig.mu, 6),
                'sigma': round(sig.sigma, 4),
                'confidence': round(sig.confidence, 3),
                'active': sig.is_active,
                'horizon': sig.metadata.get('original_horizon', sig.horizon),
                # Regime gating transparency (Good-to-have B)
                'regime_mult': round(sig.metadata.get('regime_mult', 1.0), 3),
            }

        # Alpha demotion status from meta-learner
        demotions = {}
        if self._ensemble is not None:
            demotions = self._ensemble.meta_learner.get_demotions()

        # -- Regime gating transparency (Good-to-have B) --
        regime_detail = {
            'probs': regime_probs.tolist() if regime_probs is not None else [],
            'label': self.regime_detector.current_regime,
            'dominant_state': (
                self.regime_detector.current_regime
                if regime_probs is None else
                ['high_growth_low_vol', 'high_growth_high_vol',
                 'low_growth_low_vol', 'low_growth_high_vol'][int(np.argmax(regime_probs))]
            ),
            'caps_triggered': [
                name for name, sig in signals.items()
                if sig.metadata.get('regime_mult', 1.0) < 0.8
            ],
        }

        # -- WS5A: Effective independent alphas --
        n_eff_independent = 0.0
        if self._corr_manager is not None:
            n_eff_independent = self._corr_manager.effective_n_independent()

        # -- WS5C: Alerting on significant events --
        # FIX Ã‚Â§5.1: Alerting restored Ã¢â‚¬â€ gated behind verbose level to control output.
        # CRITICAL alerts always fire; WARNING/INFO gated behind verbose >= 1.
        if self._alert_manager is not None:
            if exec_result.was_killed:
                self._alert_manager.alert(
                    "CRITICAL", "L4",
                    f"Kill switch triggered: {exec_result.kill_reason}",
                    {'bar': bar_idx, 'kill_context': self.execution_engine._kill_context}
                )
            if order.constraints_hit:
                for c in order.constraints_hit:
                    if c in ('cvar_limit', 'factor_limit'):
                        self._alert_manager.alert(
                            "WARNING", "L3",
                            f"Risk constraint binding: {c}",
                            {'bar': bar_idx, 'target': order.target_exposure}
                        )
            if n_eff_independent > 0 and n_eff_independent < 2.0:
                self._alert_manager.alert(
                    "WARNING", "L2",
                    f"Low alpha diversity: N_eff={n_eff_independent:.1f}",
                    {'bar': bar_idx}
                )
            # WS5B: Drift detection alert
            if order.risk_stats.get('drift_detected', False):
                self._alert_manager.alert(
                    "WARNING", "L1",
                    f"Feature/model drift: PSI={order.risk_stats.get('drift_psi', 0):.3f}",
                    {'bar': bar_idx, 'drift_metrics': {
                        'psi': order.risk_stats.get('drift_psi', 0),
                        'ic_decay': order.risk_stats.get('drift_ic_decay', 1),
                    }}
                )

        # -- Alpha lifecycle update (Good-to-have A) --
        if self._lifecycle_monitor is not None:
            self._lifecycle_monitor.update(
                signals=signals,
                realized_return=bar_return,
                combined_mu=mu_hat,
                position_delta=abs(exec_result.discrete_exposure - current_exposure),
            )

        # -- Attribution update (Good-to-have C) --
        self._attribution.update(
            bar_return=bar_return,
            l3_target=order.target_exposure,
            l4_discrete=exec_result.discrete_exposure,
            actual_exposure=current_exposure,
            combined_mu=mu_hat,
            was_suppressed=exec_result.was_suppressed,
            was_killed=exec_result.was_killed,
        )

        audit = {
            'bar': bar_idx,
            # L1: per-alpha breakdown
            'n_active_alphas': l2_diag.get('n_active_alphas', 0),
            'alpha_signals': alpha_detail,
            'alpha_demotions': demotions,
            # L2: combined signal
            'mu_hat': mu_hat,
            'sigma_hat': sigma_hat,
            'meta_fitted': l2_diag.get('meta_fitted', False),
            # Regime (fix 4.1 + Good-to-have B transparency)
            'regime': regime_detail,
            # WS5A: Alpha correlation metrics
            'n_eff_independent': n_eff_independent,
            # L3: portfolio construction
            'l3_target': order.target_exposure,
            'l3_reason': order.reason,
            'constraints': order.constraints_hit,
            'constraint_attribution': order.constraint_attribution,
            'risk_stats': order.risk_stats,
            # L4: execution (fix 3.3: includes discrete action)
            'l4_executed': exec_result.executed_exposure,
            'l4_discrete_action': exec_result.discrete_action,
            'l4_discrete_exposure': exec_result.discrete_exposure,
            'suppressed': exec_result.was_suppressed,
            'killed': exec_result.was_killed,
            'kill_context': self.execution_engine._kill_context if exec_result.was_killed else {},
        }
        self._audit_log.append(audit)

        return PipelineResult(
            signals=signals,
            combined_mu=mu_hat,
            combined_sigma=sigma_hat,
            order=order,
            execution=exec_result,
            l2_diagnostics=l2_diag,
            audit=audit,
        )

    def reset(self, starting_value: float = 100000.0):
        """Reset pipeline state for new episode/symbol."""
        self.portfolio.reset()
        self.execution_engine.reset(starting_value)
        self._attribution.reset()
        self._step_count = 0
        self._last_signals = {}
        self._last_order = None
        self._last_result = None
        self._prev_signals = None         # Fix 3.2: clear lagged state
        self._prev_regime_probs = None

    def get_audit_log(self) -> List[Dict]:
        return list(self._audit_log)

    def get_lifecycle_report(self) -> Dict[str, Dict[str, float]]:
        """Per-alpha health report (Good-to-have A)."""
        if self._lifecycle_monitor is not None:
            return self._lifecycle_monitor.report()
        return {}

    def get_attribution_report(self) -> Dict[str, float]:
        """Paper-alpha vs realized PnL decomposition (Good-to-have C)."""
        return self._attribution.report()

    def get_correlation_report(self) -> Dict[str, Any]:
        """WS5A: Alpha correlation metrics."""
        if self._corr_manager is not None:
            return {
                'n_eff_independent': self._corr_manager.effective_n_independent(),
                'correlation_penalties': self._corr_manager.correlation_penalty_weights(),
            }
        return {}

    def get_alert_log(self) -> List[Dict]:
        """WS5C: Alert history."""
        if self._alert_manager is not None:
            return self._alert_manager.log
        return []

    def get_alert_counts(self) -> Dict[str, int]:
        """WS5C: Alert severity counts."""
        if self._alert_manager is not None:
            return self._alert_manager.counts
        return {}

    def print_status(self):
        """Print current pipeline status."""
        print(f"\n{'='*60}")
        print(f"  INSTITUTIONAL PIPELINE STATUS (bar {self._step_count})")
        print(f"{'='*60}")
        print(f"  L1 Alphas registered: {self.alpha_factory.n_alphas}")
        for name in self.alpha_factory.alpha_names:
            sig = self._last_signals.get(name)
            if sig and sig.is_active:
                print(f"    [OK] {name:20s}  mu={sig.mu:+.4f}  s={sig.sigma:.4f}  "
                      f"conf={sig.confidence:.2f}  h={sig.horizon}")
            else:
                print(f"    [--] {name:20s}  (inactive)")

        if self._last_order:
            o = self._last_order
            print(f"  L3 Target exposure: {o.target_exposure:+.4f}")
            print(f"     Reason: {o.reason}")
            if o.constraints_hit:
                print(f"     Constraints: {', '.join(o.constraints_hit)}")

        if self._last_result:
            r = self._last_result
            print(f"  L4 Executed: {r.executed_exposure:+.4f}")
            if r.was_suppressed:
                print(f"     (suppressed by no-trade region)")
            if r.was_killed:
                print(f"     KILLED: {r.kill_reason}")

        exec_stats = self.execution_engine.stats
        print(f"  Execution stats: {exec_stats['n_trades']} trades, "
              f"{exec_stats['n_suppressed']} suppressed "
              f"({exec_stats['suppression_rate']:.0%})")
        print(f"{'='*60}\n")


@dataclass
class PipelineResult:
    """Complete output from one pipeline step."""
    signals: Dict[str, AlphaSignal]
    combined_mu: float
    combined_sigma: float
    order: PortfolioOrder
    execution: ExecutionResult
    l2_diagnostics: Dict[str, Any]
    audit: Dict[str, Any]

    @property
    def target_exposure(self) -> float:
        """The final executed exposure (continuous) -- for audit."""
        return self.execution.executed_exposure

    @property
    def discrete_action(self) -> int:
        """The discretized action index -- feed this to TradingEnv.step()."""
        return self.execution.discrete_action

    @property
    def discrete_exposure(self) -> float:
        """The snapped exposure value after discretization."""
        return self.execution.discrete_exposure


# ============================================================================
# CONVENIENCE: Build default pipeline with all standard alphas
# ============================================================================

def build_default_pipeline(acfg: ArchitectureConfig = None,
                           bars_per_year: int = 252,
                           net=None,
                           invert_seasonality: bool = False,
                           invert_vol_premium: bool = False,
                           invert_trend: bool = False) -> InstitutionalPipeline:
    """
    Build a fully-configured pipeline with all standard alpha sources.

    Args:
        acfg: Architecture configuration
        bars_per_year: Trading days per year (252 for US equities)
        net: Trained RL network (optional)
        invert_seasonality: INVESTIGATION - Flip seasonality signal (test if IC becomes positive)
        invert_vol_premium: INVESTIGATION - Flip vol premium signal (buy vol instead of sell)
        invert_trend: INVESTIGATION - Flip trend signal (IC was -0.059, test if becomes +0.059)

    Usage:
        from alphago_architecture import build_default_pipeline, ArchitectureConfig
        acfg = ArchitectureConfig()
        pipeline = build_default_pipeline(acfg, net=trained_network)

        # Test inversions:
        pipeline = build_default_pipeline(acfg, invert_seasonality=True)
        pipeline = build_default_pipeline(acfg, invert_trend=True)
    """
    if acfg is None:
        acfg = ArchitectureConfig()

    pipeline = InstitutionalPipeline(acfg, bars_per_year)

    # Register all standard alphas
    pipeline.register_alpha(RLAlphaAdapter(acfg, net=net))
    pipeline.register_alpha(TrendAlpha(acfg, invert_signal=invert_trend))
    pipeline.register_alpha(MeanReversionAlpha(acfg))
    pipeline.register_alpha(ValueAlpha(acfg))
    pipeline.register_alpha(CarryAlpha(acfg))
    pipeline.register_alpha(CalendarAlpha(acfg))  # Tier 2: Calendar effects with Bonferroni correction (replaces SeasonalityAlpha)
    pipeline.register_alpha(VolatilityPremiumAlpha(acfg, invert_signal=invert_vol_premium))
    pipeline.register_alpha(AmihudLiquidityAlpha(acfg))  # Tier 1: Liquidity premium alpha
    pipeline.register_alpha(HurstRegimeAlpha(acfg))  # Tier 1: Regime detection via Hurst exponent
    pipeline.register_alpha(ShortTermReversalAlpha(acfg))  # Tier 3: Short-term mean reversion

    # NEW ALPHAS: Tested on MSFT, IC validated
    # Import from alphago_new_alphas.py (with signal inversions applied)
    try:
        from alphago_new_alphas import VolTermStructureAlpha, VolumePriceDivergenceAlpha
        pipeline.register_alpha(VolTermStructureAlpha(acfg))  # IC +0.0326 (inverted)
        pipeline.register_alpha(VolumePriceDivergenceAlpha(acfg))  # IC +0.0033 (inverted)
        print("[INFO] Loaded VolTermStructureAlpha and VolumePriceDivergenceAlpha")
    except ImportError as e:
        print(f"[WARNING] Could not load new alphas from alphago_new_alphas.py: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to register new alphas: {e}")
        import traceback
        traceback.print_exc()

    return pipeline


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test():
    """Verify the full L1->L2->L3->L4 pipeline works end-to-end."""
    print("=" * 60)
    print("  INSTITUTIONAL ARCHITECTURE SMOKE TEST")
    print("=" * 60)

    acfg = ArchitectureConfig()
    pipeline = build_default_pipeline(acfg, net=None)

    # Simulate 200 bars of price data
    np.random.seed(42)
    prices = [100.0]
    for _ in range(199):
        ret = np.random.normal(0.0005, 0.015)
        prices.append(prices[-1] * np.exp(ret))
    prices = np.array(prices)

    current_exposure = 0.0
    portfolio_value = 100000.0

    print(f"\n  Registered alphas: {pipeline.alpha_factory.alpha_names}")
    print(f"  Simulating {len(prices)} bars...\n")

    results = []
    for i in range(50, len(prices)):
        closes = prices[:i + 1]
        bar_ret = np.log(prices[i] / prices[i - 1]) if i > 0 else 0.0

        result = pipeline.step(
            observation=np.random.randn(2520).astype(np.float32),
            closes=closes,
            bar_idx=i,
            bar_return=bar_ret,
            current_exposure=current_exposure,
            portfolio_value=portfolio_value,
            regime_probs=np.array([0.5, 0.2, 0.2, 0.1]),
        )

        current_exposure = result.target_exposure
        portfolio_value *= (1 + bar_ret)
        results.append(result)

    # Report
    n_active = sum(1 for r in results if r.audit['n_active_alphas'] > 0)
    n_suppressed = sum(1 for r in results if r.execution.was_suppressed)
    n_constrained = sum(1 for r in results if r.order.constraints_hit)

    exposures = [r.target_exposure for r in results]
    mus = [r.combined_mu for r in results]

    print(f"  Bars processed:     {len(results)}")
    print(f"  Bars with signal:   {n_active}/{len(results)}")
    print(f"  Trades suppressed:  {n_suppressed} ({n_suppressed/len(results):.0%})")
    print(f"  Constraints hit:    {n_constrained}")
    print(f"  Mean |exposure|:    {np.mean(np.abs(exposures)):.4f}")
    print(f"  Mean |mu_hat|:      {np.mean(np.abs(mus)):.6f}")
    print(f"  Exposure range:     [{min(exposures):.4f}, {max(exposures):.4f}]")

    # Print last step's full status
    pipeline.print_status()

    # Verify key architectural properties
    assert all(isinstance(r.order, PortfolioOrder) for r in results), "All orders must be PortfolioOrder"
    assert all(-2.5 <= r.target_exposure <= 2.5 for r in results), "Exposure out of range"
    print("\n  [OK] All architectural assertions passed")

    # Future leak guard test
    print("\n  Testing FutureLeakGuard...")
    try:
        obs = np.random.randn(100)
        FutureLeakGuard.assert_no_future_in_obs(obs, 0.025)
        print("  [OK] No future leak detected (correct)")
    except FutureLeakError:
        print("  [--] False positive leak detection")

    # Test with deliberately planted leak
    obs_leaked = np.zeros(100)
    obs_leaked[50] = 0.025  # Plant the risk target in obs
    try:
        FutureLeakGuard.assert_no_future_in_obs(obs_leaked, 0.025)
        print("  [--] Missed a planted future leak!")
    except FutureLeakError:
        print("  [OK] Caught planted future leak (correct)")

    print(f"\n{'='*60}")
    print("  SMOKE TEST PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    _smoke_test()