# Deep Research: Additional Alpha Signals for AlphaGo Trading System

## Current System Inventory

The system currently has **7 alpha sources** in L1:

| Alpha | Horizon | Type | Strength |
|-------|---------|------|----------|
| RL (PPO+MCTS) | 5 bars | Learned patterns | Primary |
| Trend | 21 bars | Price momentum | Strong |
| Mean Reversion | 5 bars | Short-term overshoots | Moderate |
| Value | 63 bars | Price vs. moving avg | Weak (long horizon) |
| Carry | 21 bars | Yield advantage | Tiebreaker |
| Seasonality | 5 bars | Day-of-week | Very weak |
| Volatility Premium | 21 bars | Vol mean reversion | Moderate |

The `alphago_enhancements.py` file documents 20 additional enhancement layers (v4.0 + v5.0), but these are overlays/wrappers rather than new standalone alphas feeding into L2.

**Data available per bar**: OHLCV (Open, High, Low, Close, Volume) + DatetimeIndex on daily bars. No fundamentals, no order book, no options data, no alternative data feeds. This constrains what we can implement.

**Alpha interface contract**: Every alpha produces `AlphaSignal(mu, sigma, confidence, horizon, alpha_name, metadata, timestamp)`. Alphas receive `closes`, `volumes`, `features`, `bar_idx`, `observation`, and `**kwargs`.

**Architecture note**: OHLC data exists in the synthetic generator (`SyntheticMarketGenerator` produces Open/High/Low/Close DataFrames) and in feature engineering, but currently only `closes` and `volumes` are passed as explicit params to `generate()`. New alphas needing raw OHLC will receive them via `**kwargs` (e.g., `opens=`, `highs=`, `lows=`), requiring a minor pipeline plumbing change in `generate_all()`.

---

## PREREQUISITE: Plumbing Change for OHLC Access

Before implementing OHLC-based alphas, the pipeline must pass raw OHLC arrays:

```python
# In AlphaFactory.generate_all() — add opens, highs, lows to the signature:

def generate_all(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 observation: np.ndarray = None,
                 opens: np.ndarray = None,    # NEW
                 highs: np.ndarray = None,    # NEW
                 lows: np.ndarray = None,     # NEW
                 **kwargs) -> Dict[str, AlphaSignal]:
    signals = {}
    for name, alpha in self._alphas.items():
        try:
            sig = alpha.generate(
                closes=closes, volumes=volumes, features=features,
                bar_idx=bar_idx, observation=observation,
                opens=opens, highs=highs, lows=lows,  # pass through
                **kwargs
            )
            sig = self._normalize_horizon(sig)
            signals[name] = sig
        except Exception as e:
            signals[name] = AlphaSignal(
                alpha_name=name, timestamp=bar_idx,
                metadata={'error': str(e)}
            )
    return signals
```

```python
# In InstitutionalPipeline.step() — pass OHLC from the market data:

signals = self.alpha_factory.generate_all(
    closes=closes, volumes=volumes, features=features,
    bar_idx=bar_idx, observation=observation,
    opens=kwargs.get('opens'),     # NEW
    highs=kwargs.get('highs'),     # NEW
    lows=kwargs.get('lows'),       # NEW
    **{k: v for k, v in kwargs.items() if k not in ('opens', 'highs', 'lows')}
)
```

---

## CATEGORY 1: VOLATILITY-BASED ALPHAS

### 1A. Range-Based Volatility Estimators (HIGH PRIORITY)

**What**: Replace close-to-close volatility with OHLC-based estimators that extract 5-8x more information per bar.

**Estimators ranked by efficiency**:

| Estimator | Inputs | Efficiency vs Close-to-Close | Handles Drift? |
|-----------|--------|------------------------------|----------------|
| **Yang-Zhang** | OHLC | ~14x | Yes |
| **Rogers-Satchell** | OHLC | ~8x | Yes |
| **Garman-Klass** | OHLC | ~8x | No |
| **Parkinson** | HL | ~5x | No |
| Close-to-close | C | 1x (baseline) | Yes |

**Formulas** (all use log prices):

- **Parkinson**: σ² = (1/4ln2) × (ln H/L)²
- **Garman-Klass**: σ² = 0.5×(ln H/L)² - (2ln2 - 1)×(ln C/C₋₁)²
- **Rogers-Satchell**: σ² = (ln H/C)(ln H/O) + (ln L/C)(ln L/O)
- **Yang-Zhang**: Combines overnight (O vs prev C) + Rogers-Satchell + close-to-close. Unbiased and independent of drift and opening gaps.

**Economic hypothesis**: Better volatility measurement → better regime detection → better alpha weighting → better risk management. Not an alpha per se, but a foundational improvement.

**Implementation**: Utility module used by all volatility consumers. NOT a new alpha class — replaces `np.std(returns)` everywhere.

**Source**: Garman & Klass (1980), Rogers & Satchell (1991), Yang & Zhang (2000)

```python
# ============================================================================
# OHLC VOLATILITY ESTIMATORS — Utility module (L0→L1 boundary)
# ============================================================================
# File: ohlcv_vol_estimators.py (or add to alphago_architecture.py utilities)
# These replace np.std(returns)*sqrt(252) wherever it appears in the codebase.

import numpy as np
from typing import Optional


def parkinson_vol(highs: np.ndarray, lows: np.ndarray,
                  annualize: bool = True) -> float:
    """
    Parkinson (1980) range-based volatility estimator.
    ~5x more efficient than close-to-close.
    Does NOT handle drift or overnight gaps.

    Args:
        highs: Array of high prices (length N)
        lows: Array of low prices (length N)
        annualize: If True, multiply by sqrt(252)

    Returns:
        Annualized (or daily) volatility estimate
    """
    n = len(highs)
    if n < 2:
        return 0.15  # fallback

    log_hl = np.log(highs / (lows + 1e-12))
    var = np.mean(log_hl ** 2) / (4.0 * np.log(2.0))

    vol = np.sqrt(max(var, 1e-12))
    return float(vol * np.sqrt(252)) if annualize else float(vol)


def garman_klass_vol(opens: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, closes: np.ndarray,
                     annualize: bool = True) -> float:
    """
    Garman-Klass (1980) OHLC volatility estimator.
    ~8x more efficient than close-to-close.
    Does NOT handle drift well.

    Args:
        opens, highs, lows, closes: Price arrays (same length)
        annualize: If True, multiply by sqrt(252)

    Returns:
        Annualized (or daily) volatility estimate
    """
    n = len(closes)
    if n < 2:
        return 0.15

    log_hl = np.log(highs / (lows + 1e-12))
    log_cc = np.log(closes[1:] / (closes[:-1] + 1e-12))

    # GK formula: 0.5 * (ln H/L)^2 - (2*ln2 - 1) * (ln C/C_prev)^2
    var = np.mean(0.5 * log_hl[1:] ** 2 - (2.0 * np.log(2.0) - 1.0) * log_cc ** 2)
    var = max(var, 1e-12)

    vol = np.sqrt(var)
    return float(vol * np.sqrt(252)) if annualize else float(vol)


def rogers_satchell_vol(opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, closes: np.ndarray,
                        annualize: bool = True) -> float:
    """
    Rogers-Satchell (1991) volatility estimator.
    ~8x more efficient, HANDLES DRIFT (non-zero mean returns).
    Preferred over Garman-Klass for trending markets.

    Formula: var = (ln H/C)(ln H/O) + (ln L/C)(ln L/O)

    Args:
        opens, highs, lows, closes: Price arrays (same length)
        annualize: If True, multiply by sqrt(252)

    Returns:
        Annualized (or daily) volatility estimate
    """
    n = len(closes)
    if n < 2:
        return 0.15

    log_hc = np.log(highs / (closes + 1e-12))
    log_ho = np.log(highs / (opens + 1e-12))
    log_lc = np.log(lows / (closes + 1e-12))
    log_lo = np.log(lows / (opens + 1e-12))

    var = np.mean(log_hc * log_ho + log_lc * log_lo)
    var = max(var, 1e-12)

    vol = np.sqrt(var)
    return float(vol * np.sqrt(252)) if annualize else float(vol)


def yang_zhang_vol(opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray,
                   annualize: bool = True) -> float:
    """
    Yang-Zhang (2000) volatility estimator.
    ~14x more efficient than close-to-close.
    HANDLES DRIFT AND OPENING GAPS. Best all-around estimator.

    Combines overnight vol + Rogers-Satchell + close-to-close.

    Args:
        opens, highs, lows, closes: Price arrays (same length)
        annualize: If True, multiply by sqrt(252)

    Returns:
        Annualized (or daily) volatility estimate
    """
    n = len(closes)
    if n < 3:
        return 0.15

    # Overnight returns: open vs previous close
    log_oc = np.log(opens[1:] / (closes[:-1] + 1e-12))
    # Close-to-close returns
    log_cc = np.log(closes[1:] / (closes[:-1] + 1e-12))

    # Overnight variance
    overnight_var = np.var(log_oc, ddof=1)

    # Close-to-close variance
    cc_var = np.var(log_cc, ddof=1)

    # Rogers-Satchell variance (on overlapping segment)
    log_hc = np.log(highs[1:] / (closes[1:] + 1e-12))
    log_ho = np.log(highs[1:] / (opens[1:] + 1e-12))
    log_lc = np.log(lows[1:] / (closes[1:] + 1e-12))
    log_lo = np.log(lows[1:] / (opens[1:] + 1e-12))
    rs_var = np.mean(log_hc * log_ho + log_lc * log_lo)
    rs_var = max(rs_var, 0.0)

    # Optimal weighting (Yang-Zhang 2000, Eq. 12)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = overnight_var + k * cc_var + (1 - k) * rs_var
    var = max(var, 1e-12)

    vol = np.sqrt(var)
    return float(vol * np.sqrt(252)) if annualize else float(vol)


# ============================================================================
# USAGE: Drop-in replacement throughout codebase
# ============================================================================
# BEFORE (in TrendAlpha.generate, MeanReversionAlpha.generate, etc.):
#     log_rets = np.diff(np.log(closes[-self.vol_lb:] + 1e-12))
#     realized_vol = float(np.std(log_rets)) * np.sqrt(252)
#
# AFTER:
#     realized_vol = yang_zhang_vol(
#         opens[-self.vol_lb:], highs[-self.vol_lb:],
#         lows[-self.vol_lb:], closes[-self.vol_lb:]
#     )
```

---

### 1B. Volatility-of-Volatility (Vol-of-Vol) Signal

**What**: Track the second derivative of volatility — how much volatility itself is fluctuating.

**Signal logic**: When vol-of-vol is high, market is uncertain about its own uncertainty → widen risk limits, reduce position sizes. When vol-of-vol is low, market is complacent → conditions for tail events.

**Proxy with OHLCV**: Rolling standard deviation of rolling Garman-Klass vol estimates over 21-bar windows.

**Economic hypothesis**: Vol-of-vol proxies the VVIX (volatility of VIX). High VVIX environments have historically preceded large market dislocations. This signal helps with timing volatility premium harvesting.

**Implementation**: Feature into RL observation vector + conditioning variable for VolatilityPremiumAlpha.

```python
# ============================================================================
# VOL-OF-VOL FEATURE — not a standalone alpha, a feature for L1 and L2
# ============================================================================
# Integrate into VolatilityPremiumAlpha.generate() or as observation feature

class VolOfVolFeature:
    """
    Tracks the volatility of volatility using rolling GK estimates.
    Produces a normalized vol-of-vol score for use as:
      - Feature in RL observation vector
      - Conditioning variable for vol premium harvesting
      - Risk scaling input for L3

    High vol-of-vol → uncertainty about uncertainty → reduce risk.
    Low vol-of-vol → complacency → conditions for tail event.
    """

    def __init__(self, vol_window: int = 21, vov_window: int = 21):
        self.vol_window = vol_window
        self.vov_window = vov_window
        self._vol_history = deque(maxlen=vov_window + 10)
        self._warmup = vol_window + vov_window

    def update(self, opens: np.ndarray, highs: np.ndarray,
               lows: np.ndarray, closes: np.ndarray,
               bar_idx: int) -> dict:
        """
        Call once per bar. Returns dict with vol-of-vol features.

        Returns:
            {
                'vol_of_vol': float,        # Raw vol-of-vol (annualized)
                'vol_of_vol_zscore': float,  # Z-score relative to history
                'vol_regime': str,           # 'low_vov' / 'normal' / 'high_vov'
            }
        """
        n = len(closes)
        if n < self.vol_window + 2:
            return {'vol_of_vol': 0.0, 'vol_of_vol_zscore': 0.0,
                    'vol_regime': 'normal'}

        # Current GK vol estimate over last vol_window bars
        current_vol = garman_klass_vol(
            opens[-self.vol_window:], highs[-self.vol_window:],
            lows[-self.vol_window:], closes[-self.vol_window:],
            annualize=True
        )
        self._vol_history.append(current_vol)

        if len(self._vol_history) < self.vov_window:
            return {'vol_of_vol': 0.0, 'vol_of_vol_zscore': 0.0,
                    'vol_regime': 'normal'}

        # Vol-of-vol = standard deviation of rolling vol estimates
        vol_array = np.array(list(self._vol_history))
        vov = float(np.std(vol_array[-self.vov_window:]))
        vov_mean = float(np.mean(vol_array))
        vov_std = float(np.std(vol_array))

        # Z-score: how extreme is current vol-of-vol?
        vov_z = (vov - vov_mean) / (vov_std + 1e-10)

        # Regime classification
        if vov_z > 1.5:
            regime = 'high_vov'
        elif vov_z < -0.5:
            regime = 'low_vov'
        else:
            regime = 'normal'

        return {
            'vol_of_vol': vov,
            'vol_of_vol_zscore': float(np.clip(vov_z, -3.0, 3.0)),
            'vol_regime': regime,
        }
```

---

### 1C. Realized Volatility Term Structure Slope

**What**: Compare short-term realized vol (5-bar) to long-term realized vol (63-bar). The ratio tells you if vol is in contango (short < long = normal) or backwardation (short > long = stress).

**Signal logic**:
- Ratio < 1.0: Normal environment, slight long bias (vol premium harvesting)
- Ratio > 1.2: Stress, reduce exposure
- Ratio > 2.0: Crisis, activate defensive positioning

**Economic hypothesis**: Vol term structure slope is one of the strongest predictors of forward equity returns. When short-term vol spikes above long-term, mean reversion of vol = positive expected equity return.

```python
# ============================================================================
# VOL TERM STRUCTURE SLOPE — Feature for L2 gating and L3 risk
# ============================================================================

class VolTermStructure:
    """
    Compares short-term vs long-term realized vol.
    Uses range-based estimators for efficiency.

    vol_slope = short_vol / long_vol
      < 1.0 → contango (normal, calm)
      > 1.0 → backwardation (stress, vol spike)
      > 2.0 → crisis
    """

    def __init__(self, short_window: int = 5, long_window: int = 63):
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, opens: np.ndarray, highs: np.ndarray,
                lows: np.ndarray, closes: np.ndarray) -> dict:
        n = len(closes)
        if n < self.long_window + 2:
            return {'vol_slope': 1.0, 'short_vol': 0.15, 'long_vol': 0.15,
                    'vol_term_regime': 'normal'}

        short_vol = rogers_satchell_vol(
            opens[-self.short_window:], highs[-self.short_window:],
            lows[-self.short_window:], closes[-self.short_window:]
        )
        long_vol = rogers_satchell_vol(
            opens[-self.long_window:], highs[-self.long_window:],
            lows[-self.long_window:], closes[-self.long_window:]
        )

        slope = short_vol / (long_vol + 1e-10)

        if slope > 2.0:
            regime = 'crisis'
        elif slope > 1.2:
            regime = 'stress'
        elif slope < 0.8:
            regime = 'calm'
        else:
            regime = 'normal'

        return {
            'vol_slope': float(np.clip(slope, 0.1, 5.0)),
            'short_vol': short_vol,
            'long_vol': long_vol,
            'vol_term_regime': regime,
        }
```

---

## CATEGORY 2: LIQUIDITY / MICROSTRUCTURE ALPHAS

### 2A. Amihud Illiquidity Ratio (HIGH PRIORITY)

**What**: ILLIQ = |Return| / Dollar Volume. Measures price impact per dollar traded.

**Why it matters**: The Amihud measure (cited 12,000+ times) captures the illiquidity premium — less liquid assets earn higher expected returns. Changes in illiquidity predict returns both cross-sectionally and over time.

**Signal logic**:
- Rising ILLIQ: Deteriorating liquidity → reduce position size, widen cost assumptions
- Falling ILLIQ: Improving liquidity → more aggressive positioning possible
- ILLIQ level: Higher ILLIQ assets require higher expected return to hold
- Unexpected illiquidity shocks lower contemporaneous prices → mean reversion opportunity

**Implementation**: Dual use: alpha signal + cost model input. Computable from daily OHLCV data.

**Source**: Amihud (2002) "Illiquidity and stock returns: cross-section and time-series effects"

```python
# ============================================================================
# AMIHUD ILLIQUIDITY ALPHA — L1 Alpha + L3 Cost Model Input
# ============================================================================

class AmihudLiquidityAlpha(BaseAlpha):
    """
    Illiquidity-based alpha using Amihud (2002) ILLIQ measure.

    ILLIQ_t = |r_t| / DollarVolume_t

    Two signals:
      1. Level: Higher ILLIQ → higher expected return (illiquidity premium)
      2. Change: Rising ILLIQ → negative shock, creates MR opportunity

    Also exports liquidity features for L3 cost model adjustment.

    Source: Amihud (2002), cited 12,000+ times.
    Economic hypothesis: Investors demand compensation for holding illiquid
    assets. Unexpected illiquidity shocks lower prices, creating reversion.

    Horizon: 21 bars (illiquidity premium is medium-term)
    """

    def __init__(self, acfg: 'ArchitectureConfig'):
        super().__init__(name="amihud_liquidity", horizon=21)
        self.lookback = getattr(acfg, 'amihud_lookback', 63)
        self.short_lookback = getattr(acfg, 'amihud_short_lookback', 10)
        self._warmup_bars = self.lookback + 10
        self._illiq_history = deque(maxlen=252)

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        if n < self._warmup_bars or volumes is None or len(volumes) < n:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # ---- Compute Amihud ILLIQ over lookback ----
        log_rets = np.diff(np.log(closes[-self.lookback:] + 1e-12))
        abs_rets = np.abs(log_rets)
        dollar_vol = closes[-self.lookback + 1:] * volumes[-self.lookback + 1:]
        dollar_vol = np.maximum(dollar_vol, 1e-10)  # prevent div-by-zero

        # ILLIQ = mean(|r_t| / DollarVolume_t)
        illiq_daily = abs_rets / dollar_vol
        illiq = float(np.mean(illiq_daily))

        # Log-transform to reduce skewness (standard in literature)
        log_illiq = float(np.log(illiq + 1e-15))
        self._illiq_history.append(log_illiq)

        if len(self._illiq_history) < 30:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # ---- Signal 1: Illiquidity change (short-term shock) ----
        illiq_arr = np.array(list(self._illiq_history))
        illiq_mean = np.mean(illiq_arr)
        illiq_std = np.std(illiq_arr) + 1e-10
        illiq_z = (log_illiq - illiq_mean) / illiq_std

        # Rising illiquidity → negative price shock → expect reversion UP
        # Falling illiquidity → positive environment → slight long bias
        mu_shock = float(-illiq_z * 0.01)  # Conservative: MR after illiq spike

        # ---- Signal 2: Illiquidity level premium ----
        # Higher baseline illiquidity → higher expected return to compensate
        short_illiq = np.mean(illiq_daily[-self.short_lookback:])
        long_illiq = np.mean(illiq_daily)
        illiq_ratio = short_illiq / (long_illiq + 1e-15)

        # If recent illiquidity elevated → expect premium
        mu_premium = float(np.clip((illiq_ratio - 1.0) * 0.005, -0.05, 0.05))

        # ---- Combine ----
        mu = float(np.clip(mu_shock + mu_premium, -0.3, 0.3))

        # Sigma from realized vol
        sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)

        # Confidence: higher when illiquidity signal is extreme
        confidence = float(np.clip(abs(illiq_z) / 3.0, 0.0, 0.6))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={
                'log_illiq': log_illiq,
                'illiq_z': float(illiq_z),
                'illiq_ratio': float(illiq_ratio),
                'illiq_raw': illiq,
                # Export for L3 cost model
                'cost_model_illiq_multiplier': float(np.clip(illiq_ratio, 0.5, 3.0)),
            },
            timestamp=bar_idx,
        )
```

---

### 2B. Volume-Price Confirmation/Divergence

**What**: Track whether volume confirms or diverges from price moves.

**Signal logic**:
- Price up + Volume up = confirmed trend → momentum signal strengthened
- Price up + Volume down = divergence → trend weakening, reversal likely
- Price down + Volume up = confirmed selling → trend continuation
- Price down + Volume down = divergence → selling pressure exhausting

**Implementation**: Feature that modifies confidence of Trend and MR alphas. NOT a standalone alpha — it's a signal quality filter.

**Source**: Blume, Easley, O'Hara (1994) "Market Statistics and Technical Analysis"

```python
# ============================================================================
# VOLUME-PRICE DIVERGENCE — Feature / confidence modifier (L1)
# ============================================================================

class VolumePriceDivergence:
    """
    Detects divergence between price moves and volume.
    Used to MODIFY confidence of trend and mean-reversion alphas,
    not as a standalone signal.

    Confirmed moves (price + volume agree) → trust trend signal.
    Divergent moves (price + volume disagree) → trust MR signal.

    Source: Blume, Easley, O'Hara (1994)
    """

    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def compute(self, closes: np.ndarray, volumes: np.ndarray) -> dict:
        """
        Compute volume-price agreement metrics.

        Returns:
            {
                'vp_correlation': float,    # Rolling corr(returns, vol_changes)
                'vp_divergence_score': float, # -1 (full divergence) to +1 (confirmation)
                'trend_confidence_mult': float,  # 0.5 to 1.5
                'mr_confidence_mult': float,     # 0.5 to 1.5
            }
        """
        n = min(len(closes), len(volumes))
        if n < self.lookback + 2:
            return {
                'vp_correlation': 0.0,
                'vp_divergence_score': 0.0,
                'trend_confidence_mult': 1.0,
                'mr_confidence_mult': 1.0,
            }

        # Returns and volume changes over lookback
        rets = np.diff(np.log(closes[-self.lookback - 1:] + 1e-12))
        vol_changes = np.diff(np.log(volumes[-self.lookback - 1:] + 1.0))

        # Rolling correlation
        if len(rets) >= 10 and np.std(rets) > 1e-10 and np.std(vol_changes) > 1e-10:
            corr = float(np.corrcoef(rets, vol_changes)[0, 1])
        else:
            corr = 0.0

        # Guard against NaN
        if np.isnan(corr):
            corr = 0.0

        # Divergence score: positive = confirmation, negative = divergence
        div_score = float(np.clip(corr, -1.0, 1.0))

        # Confidence multipliers for other alphas
        # Confirmed moves → boost trend, penalize MR
        # Divergent moves → penalize trend, boost MR
        trend_mult = float(np.clip(1.0 + div_score * 0.3, 0.5, 1.5))
        mr_mult = float(np.clip(1.0 - div_score * 0.3, 0.5, 1.5))

        return {
            'vp_correlation': corr,
            'vp_divergence_score': div_score,
            'trend_confidence_mult': trend_mult,
            'mr_confidence_mult': mr_mult,
        }

# USAGE in TrendAlpha.generate():
#   vpd = VolumePriceDivergence()
#   vp_metrics = vpd.compute(closes, volumes)
#   confidence *= vp_metrics['trend_confidence_mult']
```

---

### 2C. Volume Anomaly / High-Volume Return Premium

**What**: Unusually high volume predicts positive forward returns. Stocks with volume 2+ standard deviations above average tend to outperform over the next month.

**Economic hypothesis**: High volume = attention/information arrival = prices moving toward fair value. If a stock gets attention, follow-on buying occurs.

**Source**: Gervais, Kaniel, Mingelgrin (2001) "The High-Volume Return Premium"

```python
# ============================================================================
# VOLUME ANOMALY FEATURE — observation vector feature (L1)
# ============================================================================

class VolumeAnomalyFeature:
    """
    Computes volume z-score and produces a feature for the
    observation vector. High-volume bars predict positive
    forward returns.

    Source: Gervais, Kaniel, Mingelgrin (2001)
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def compute(self, volumes: np.ndarray) -> dict:
        n = len(volumes)
        if n < self.lookback + 5:
            return {'volume_zscore': 0.0, 'volume_anomaly': False}

        recent_vol = volumes[-1]
        hist_mean = np.mean(volumes[-self.lookback:])
        hist_std = np.std(volumes[-self.lookback:]) + 1e-10

        z = (recent_vol - hist_mean) / hist_std

        return {
            'volume_zscore': float(np.clip(z, -3.0, 5.0)),
            'volume_anomaly': bool(z > 2.0),  # Unusually high
        }
```

---

## CATEGORY 3: MOMENTUM REFINEMENTS

### 3A. Idiosyncratic Momentum (HIGH PRIORITY for multi-asset)

**What**: Momentum in residual returns after removing market component. Doesn't crash like raw momentum. Doesn't reverse like raw momentum.

**Key finding**: Idiosyncratic momentum "forecasts high short and long-term excess returns" while "conventional momentum forecasts high short-term, and negative long-term excess returns."

**Source**: Gutierrez & Pirinsky (2007), Blitz, Huij & Martens (2011)

```python
# ============================================================================
# IDIOSYNCRATIC MOMENTUM ALPHA — L1 Alpha
# ============================================================================

class IdiosyncraticMomentumAlpha(BaseAlpha):
    """
    Momentum computed on market-residualized returns.

    Standard momentum has crash risk (1932: -91%, 2009: -73%).
    Idiosyncratic momentum strips out market beta, leaving a
    purer signal that doesn't reverse at long horizons.

    For single-stock mode: uses universe mean return as market proxy.
    For multi-asset mode: regresses each stock on market factor.

    Source: Gutierrez & Pirinsky (2007), Blitz, Huij & Martens (2011)
    Economic hypothesis: Underreaction to firm-specific information.
    Horizon: 21 bars (monthly)
    """

    def __init__(self, acfg: 'ArchitectureConfig'):
        super().__init__(name="idio_momentum", horizon=21)
        self.formation_period = getattr(acfg, 'idio_mom_formation', 126)
        self.skip = getattr(acfg, 'idio_mom_skip', 5)  # skip most recent week
        self._warmup_bars = self.formation_period + self.skip + 10
        self._market_returns = deque(maxlen=300)

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        # Get market proxy returns from kwargs (universe mean)
        market_returns = kwargs.get('market_returns', None)

        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Stock returns over formation period (skip recent 'skip' bars)
        end = n - self.skip
        start = end - self.formation_period
        if start < 1 or end < start + 20:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        stock_rets = np.diff(np.log(closes[start:end] + 1e-12))

        if market_returns is not None and len(market_returns) >= len(stock_rets):
            mkt_rets = market_returns[-(len(stock_rets)):]
        else:
            # Fallback: estimate market as rolling mean return
            # (degenerate in single-stock mode, but valid placeholder)
            mkt_rets = np.full_like(stock_rets, np.mean(stock_rets))

        # ---- Regress stock returns on market ----
        # r_stock = alpha + beta * r_market + epsilon
        # Idiosyncratic momentum = cumulative epsilon
        if np.std(mkt_rets) < 1e-10:
            residuals = stock_rets
        else:
            beta = np.cov(stock_rets, mkt_rets)[0, 1] / (np.var(mkt_rets) + 1e-10)
            alpha_intercept = np.mean(stock_rets) - beta * np.mean(mkt_rets)
            residuals = stock_rets - (alpha_intercept + beta * mkt_rets)

        # Cumulative residual return = idiosyncratic momentum signal
        idio_cum = float(np.sum(residuals))

        # Vol-normalize
        residual_vol = float(np.std(residuals)) * np.sqrt(252)
        residual_vol = max(residual_vol, 0.05)

        # Mu: positive idio_cum → expect continuation (underreaction)
        mu = float(np.clip(idio_cum * 0.5, -0.5, 0.5))  # Annualized scale

        # Confidence based on signal strength and consistency
        if len(residuals) > 10:
            # What fraction of residual returns agree with cumulative direction?
            consistency = np.mean(np.sign(residuals) == np.sign(idio_cum))
            confidence = float(np.clip(consistency * 0.8, 0.0, 0.7))
        else:
            confidence = 0.0

        return AlphaSignal(
            mu=mu, sigma=residual_vol, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={
                'idio_cum_return': idio_cum,
                'beta_to_market': float(beta) if np.std(mkt_rets) > 1e-10 else 0.0,
                'residual_vol': residual_vol,
                'n_formation_bars': len(residuals),
            },
            timestamp=bar_idx,
        )
```

---

### 3B. Risk-Managed Momentum

**What**: Scale momentum signal inversely by recent momentum volatility. Momentum crashes happen in high-vol environments. Scaling by inverse vol naturally reduces exposure before crashes.

**Source**: Barroso & Santa-Clara (2015), Daniel & Moskowitz (2016)

```python
# ============================================================================
# RISK-MANAGED MOMENTUM — Modification to existing TrendAlpha
# ============================================================================
# This is NOT a new alpha class — it's a patch to TrendAlpha.generate()

# ADD to TrendAlpha.generate(), right before constructing the AlphaSignal:

def _risk_manage_momentum(self, raw_mu: float, closes: np.ndarray,
                          lookback: int = 126) -> float:
    """
    Scale momentum signal by inverse of recent momentum volatility.

    When momentum strategy vol is high → crashes are likely → scale down.
    When momentum strategy vol is low → safe to be aggressive.

    Source: Barroso & Santa-Clara (2015)
    """
    n = len(closes)
    if n < lookback + 21:
        return raw_mu

    # Compute rolling momentum returns over last 'lookback' bars
    mom_rets = []
    for i in range(lookback - 21, lookback):
        if n - lookback + i + 21 <= n:
            window_end = n - lookback + i + 21
            window_start = n - lookback + i
            mom_ret = np.log(closes[min(window_end, n-1)] /
                           (closes[window_start] + 1e-12))
            mom_rets.append(mom_ret)

    if len(mom_rets) < 10:
        return raw_mu

    mom_vol = float(np.std(mom_rets))
    target_vol = 0.12 / np.sqrt(252)  # 12% annual target for mom strategy

    # Scale factor: target_vol / realized_vol, capped
    if mom_vol > 1e-10:
        scale = min(target_vol / mom_vol, 2.0)
    else:
        scale = 1.0

    return raw_mu * scale

# INTEGRATION into TrendAlpha.generate():
# Replace: mu = float(np.clip(raw_signal, -3.0, 3.0)) * 0.05
# With:    mu = self._risk_manage_momentum(
#              float(np.clip(raw_signal, -3.0, 3.0)) * 0.05, closes)
```

---

### 3C. Short-Term Reversal (1-5 bars)

**What**: Very short-term losers outperform winners over the next week. Distinct from our MeanReversion alpha which uses Bollinger band z-scores.

**Caution**: Very capacity-constrained, high turnover, heavily crowded by HFT.

```python
# ============================================================================
# SHORT-TERM REVERSAL — Potential upgrade to MeanReversionAlpha
# ============================================================================
# Pure return-reversal signal (simpler than z-score approach)
# Can be blended with existing MR alpha or replace it

class ShortTermReversalAlpha(BaseAlpha):
    """
    Pure short-term reversal: past-week losers outperform next week.

    Unlike MeanReversionAlpha (z-score of price vs. Bollinger band),
    this uses raw return reversal — simpler and higher frequency.

    Economic hypothesis: Microstructure effects (bid-ask bounce,
    inventory rebalancing, overreaction to noise).
    Capacity warning: Very high turnover, competes with HFT.
    Horizon: 5 bars
    """

    def __init__(self, acfg: 'ArchitectureConfig'):
        super().__init__(name="short_reversal", horizon=5)
        self.formation = getattr(acfg, 'reversal_formation', 5)
        self._warmup_bars = 30

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)
        if n < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # Past-week return
        past_ret = np.log(closes[-1] / (closes[-1 - self.formation] + 1e-12))

        # Reversal: expect opposite direction
        mu = float(-past_ret * 0.3)  # Conservative scale
        mu = float(np.clip(mu, -0.3, 0.3))

        # Vol
        log_rets = np.diff(np.log(closes[-20:] + 1e-12))
        sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)

        # Confidence: stronger when past return is extreme
        ret_z = abs(past_ret) / (float(np.std(np.diff(np.log(
            closes[-60:] + 1e-12)))) * np.sqrt(self.formation) + 1e-10)
        confidence = float(np.clip(ret_z / 3.0, 0.0, 0.5))

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={'past_week_ret': float(past_ret), 'ret_zscore': float(ret_z)},
            timestamp=bar_idx,
        )
```

---

## CATEGORY 4: REGIME / STATE DETECTION

### 4A. Hurst Exponent (Adaptive Regime) (HIGH PRIORITY)

**What**: The Hurst exponent (H) directly measures whether a time series is trending (H > 0.5), random walking (H ≈ 0.5), or mean-reverting (H < 0.5).

**Signal logic**:
- H > 0.55: Market is trending → increase trend alpha weight, decrease MR weight
- H ≈ 0.50: Random walk → reduce all alpha confidence
- H < 0.45: Market is mean-reverting → increase MR alpha weight, decrease trend weight

**Why it matters**: This directly tells you which of your alphas should be active. It's the theoretical foundation for L2 regime gating.

**Source**: Hurst (1951), Mandelbrot (1971), Kroha & Skoula (technical indicator application)

```python
# ============================================================================
# ROLLING HURST EXPONENT — L2 Ensemble Gating Feature
# ============================================================================

class RollingHurstExponent:
    """
    Rolling Hurst exponent via Rescaled Range (R/S) analysis.

    H > 0.5 → persistent (trending) → trend alpha should dominate
    H ≈ 0.5 → random walk → all alphas low confidence
    H < 0.5 → anti-persistent (mean-reverting) → MR alpha should dominate

    This is THE key L2 gating feature: it directly answers
    "should I trust trend or mean-reversion right now?"

    Source: Hurst (1951), Mandelbrot (1971)

    Caution: Hurst estimation is noisy with small samples.
    Need at least 100+ bars for reliable estimates.
    Use as slow-moving regime indicator (update every ~21 bars),
    not bar-by-bar.
    """

    def __init__(self, window: int = 126, min_window: int = 63,
                 update_freq: int = 21):
        self.window = window
        self.min_window = min_window
        self.update_freq = update_freq
        self._last_hurst = 0.5
        self._last_update = -999
        self._hurst_history = deque(maxlen=50)

    def compute(self, closes: np.ndarray, bar_idx: int = 0) -> dict:
        """
        Compute rolling Hurst exponent.

        Returns dict with:
            'hurst': float (0.0 to 1.0)
            'hurst_regime': str ('trending' / 'random_walk' / 'mean_reverting')
            'trend_gate': float (0.0 to 1.0, weight multiplier for trend alpha)
            'mr_gate': float (0.0 to 1.0, weight multiplier for MR alpha)
        """
        n = len(closes)

        # Only update periodically (expensive computation)
        if bar_idx - self._last_update < self.update_freq and self._last_update > 0:
            return self._format_result(self._last_hurst)

        if n < self.min_window:
            return self._format_result(0.5)

        # Use most recent 'window' bars
        series = np.diff(np.log(closes[-min(self.window, n):] + 1e-12))
        h = self._hurst_rs(series)
        self._last_hurst = h
        self._last_update = bar_idx
        self._hurst_history.append(h)

        return self._format_result(h)

    @staticmethod
    def _hurst_rs(series: np.ndarray) -> float:
        """
        Estimate Hurst exponent via Rescaled Range (R/S) analysis.

        Algorithm:
          1. Divide series into chunks of increasing size
          2. For each chunk size, compute R/S statistic
          3. H = slope of log(R/S) vs log(chunk_size)
        """
        n = len(series)
        if n < 20:
            return 0.5

        # Chunk sizes: powers of 2 up to half the series length
        max_k = int(np.log2(n)) - 1
        if max_k < 2:
            return 0.5

        chunk_sizes = [2 ** i for i in range(2, max_k + 1)]
        rs_values = []

        for size in chunk_sizes:
            n_chunks = n // size
            if n_chunks < 1:
                continue

            rs_list = []
            for i in range(n_chunks):
                chunk = series[i * size:(i + 1) * size]
                mean_chunk = np.mean(chunk)
                # Cumulative deviation from mean
                cum_dev = np.cumsum(chunk - mean_chunk)
                r = np.max(cum_dev) - np.min(cum_dev)  # Range
                s = np.std(chunk, ddof=1)  # Std dev
                if s > 1e-12:
                    rs_list.append(r / s)

            if len(rs_list) > 0:
                rs_values.append((size, np.mean(rs_list)))

        if len(rs_values) < 2:
            return 0.5

        # Linear regression on log-log scale
        log_sizes = np.log([v[0] for v in rs_values])
        log_rs = np.log([v[1] for v in rs_values])

        # H = slope of log(R/S) vs log(n)
        slope = np.polyfit(log_sizes, log_rs, 1)[0]
        return float(np.clip(slope, 0.01, 0.99))

    def _format_result(self, h: float) -> dict:
        """Convert raw Hurst to usable regime signals."""

        if h > 0.55:
            regime = 'trending'
            trend_gate = float(np.clip((h - 0.5) * 4.0, 0.5, 1.0))
            mr_gate = float(np.clip(1.0 - (h - 0.5) * 4.0, 0.2, 0.8))
        elif h < 0.45:
            regime = 'mean_reverting'
            trend_gate = float(np.clip(1.0 - (0.5 - h) * 4.0, 0.2, 0.8))
            mr_gate = float(np.clip((0.5 - h) * 4.0, 0.5, 1.0))
        else:
            regime = 'random_walk'
            trend_gate = 0.5
            mr_gate = 0.5

        return {
            'hurst': h,
            'hurst_regime': regime,
            'trend_gate': trend_gate,
            'mr_gate': mr_gate,
        }

# INTEGRATION into L2 SignalEnsemble.combine():
#   hurst_info = self._hurst.compute(closes, bar_idx)
#   # Modify alpha weights before combining:
#   for name, sig in signals.items():
#       if 'trend' in name:
#           sig.confidence *= hurst_info['trend_gate']
#       elif 'mean_rev' in name or 'reversal' in name:
#           sig.confidence *= hurst_info['mr_gate']
```

---

### 4B. Correlation Regime Detection

**What**: Track rolling correlation between the stock and a market proxy (universe mean return). Rising correlations → systematic risk dominates. Falling correlations → idiosyncratic alpha opportunity.

```python
# ============================================================================
# CORRELATION REGIME — L2/L3 Feature
# ============================================================================

class CorrelationRegimeDetector:
    """
    Tracks rolling correlation between stock and market.

    High correlation → systematic risk → hedge more, reduce gross
    Low correlation → idiosyncratic moves → alpha opportunity, increase gross

    Also useful for detecting crowding (our signals correlating
    with broad market = we're just trading beta).
    """

    def __init__(self, window: int = 63):
        self.window = window
        self._stock_rets = deque(maxlen=window + 5)
        self._market_rets = deque(maxlen=window + 5)

    def update(self, stock_return: float, market_return: float) -> dict:
        self._stock_rets.append(stock_return)
        self._market_rets.append(market_return)

        if len(self._stock_rets) < 20:
            return {'rolling_corr': 0.0, 'corr_regime': 'normal',
                    'gross_exposure_mult': 1.0}

        sr = np.array(list(self._stock_rets))
        mr = np.array(list(self._market_rets))

        if np.std(sr) < 1e-10 or np.std(mr) < 1e-10:
            corr = 0.0
        else:
            corr = float(np.corrcoef(sr, mr)[0, 1])
            if np.isnan(corr):
                corr = 0.0

        # High corr → reduce gross (just trading beta)
        # Low corr → increase gross (genuine alpha opportunity)
        abs_corr = abs(corr)
        if abs_corr > 0.7:
            regime = 'high_corr'
            mult = 0.7  # Reduce exposure — we're just beta
        elif abs_corr < 0.3:
            regime = 'low_corr'
            mult = 1.2  # Alpha opportunity — increase
        else:
            regime = 'normal'
            mult = 1.0

        return {
            'rolling_corr': corr,
            'corr_regime': regime,
            'gross_exposure_mult': mult,
        }
```

---

## CATEGORY 5: CALENDAR / SEASONALITY

### 5A–5C: CalendarAlpha (Unified Upgrade)

Combines Turn-of-Month (TOM), monthly seasonality (Halloween), and fixed day-of-week into a single upgraded alpha, replacing the current weak `SeasonalityAlpha`.

```python
# ============================================================================
# CALENDAR ALPHA — Replaces SeasonalityAlpha (L1)
# ============================================================================

class CalendarAlpha(BaseAlpha):
    """
    Unified calendar effects alpha. Replaces the weak day-of-week-only
    SeasonalityAlpha with three combined effects:

    1. Turn-of-Month (TOM): Last trading day through 3rd of next month.
       Strongest calendar anomaly. 4 days account for ALL positive DJIA
       returns historically. Structural cause: institutional cash flows.
       Source: Lakonishok & Smidt (1988)

    2. Monthly Seasonality (Halloween): Nov-Apr > May-Oct.
       Most pervasive calendar effect (20+ countries, multiple decades).
       Source: Bouman & Jacobsen (2002)

    3. Day-of-Week (upgraded): Uses actual DatetimeIndex instead of
       bar_idx % 5. Tracks historical returns by real weekday.
       Source: French (1980), though largely disappeared post-1987.

    All effects produce weak signals. Confidence capped at 0.4.
    This is a tiebreaker alpha, not a primary signal.

    Horizon: 5 bars
    """

    def __init__(self, acfg: 'ArchitectureConfig'):
        super().__init__(name="calendar", horizon=5)
        self._warmup_bars = getattr(acfg, 'calendar_warmup', 252)
        self._day_returns = {d: deque(maxlen=500) for d in range(5)}
        self._month_returns = {m: deque(maxlen=100) for m in range(1, 13)}
        self._tom_returns = {'tom': deque(maxlen=200),
                             'non_tom': deque(maxlen=500)}
        self._bar_count = 0

    def generate(self, closes: np.ndarray, volumes: np.ndarray = None,
                 features: np.ndarray = None, bar_idx: int = 0,
                 **kwargs) -> AlphaSignal:
        self._bar_count = bar_idx
        n = len(closes)

        # Get datetime info from kwargs (pipeline must pass this)
        dates = kwargs.get('dates', None)  # pd.DatetimeIndex or list of dates
        current_date = kwargs.get('current_date', None)  # datetime for current bar

        if n < 2:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        ret = float(np.log(closes[-1] / (closes[-2] + 1e-12)))

        # ---- Track returns by category ----
        if current_date is not None:
            weekday = current_date.weekday()  # 0=Mon, 4=Fri
            month = current_date.month
            day_of_month = current_date.day

            self._day_returns[weekday].append(ret)
            self._month_returns[month].append(ret)

            # TOM: last day of month (day >= 28) or first 3 days (day <= 3)
            is_tom = day_of_month >= 28 or day_of_month <= 3
            if is_tom:
                self._tom_returns['tom'].append(ret)
            else:
                self._tom_returns['non_tom'].append(ret)
        else:
            # Fallback to modular arithmetic (legacy behavior)
            weekday = bar_idx % 5
            self._day_returns[weekday].append(ret)
            month = None
            current_date = None

        if self._bar_count < self._warmup_bars:
            return AlphaSignal(alpha_name=self.name, horizon=self.horizon,
                               timestamp=bar_idx)

        # ---- Signal 1: Turn-of-Month ----
        tom_mu = 0.0
        tom_conf = 0.0
        if current_date is not None:
            next_day = current_date.day + 1  # rough approximation
            next_is_tom = next_day >= 28 or next_day <= 3

            tom_rets = list(self._tom_returns['tom'])
            non_tom_rets = list(self._tom_returns['non_tom'])

            if len(tom_rets) >= 20 and len(non_tom_rets) >= 20:
                tom_avg = np.mean(tom_rets) * 252
                non_tom_avg = np.mean(non_tom_rets) * 252
                tom_edge = tom_avg - non_tom_avg

                if next_is_tom:
                    tom_mu = float(np.clip(tom_edge * 0.3, -0.1, 0.1))
                    tom_t = abs(tom_edge) / (np.std(tom_rets) * np.sqrt(252) /
                                              np.sqrt(len(tom_rets)) + 1e-10)
                    tom_conf = float(np.clip(tom_t / 3.0, 0.0, 0.3))

        # ---- Signal 2: Halloween / Monthly Seasonality ----
        monthly_mu = 0.0
        monthly_conf = 0.0
        if current_date is not None:
            # Nov-Apr = risk-on (positive tilt), May-Oct = risk-off
            if current_date.month in [11, 12, 1, 2, 3, 4]:
                monthly_mu = 0.02  # Slight positive tilt
                monthly_conf = 0.1
            else:
                monthly_mu = -0.01  # Slight negative tilt
                monthly_conf = 0.1

        # ---- Signal 3: Day-of-Week ----
        dow_mu = 0.0
        dow_conf = 0.0
        next_day_idx = (weekday + 1) % 5 if current_date is not None else (bar_idx + 1) % 5
        day_rets = list(self._day_returns[next_day_idx])
        if len(day_rets) >= 30:
            dow_mu = float(np.mean(day_rets)) * 252
            dow_sigma = float(np.std(day_rets)) * np.sqrt(252) + 1e-10
            dow_t = abs(dow_mu) / (dow_sigma / np.sqrt(len(day_rets)))
            dow_conf = float(np.clip(dow_t / 3.0, 0.0, 0.2))

        # ---- Combine (weighted average) ----
        total_conf = tom_conf + monthly_conf + dow_conf + 1e-10
        mu = (tom_mu * tom_conf + monthly_mu * monthly_conf +
              dow_mu * dow_conf) / total_conf
        mu = float(np.clip(mu, -0.2, 0.2))

        # Overall confidence — capped low (these are weak signals)
        confidence = float(np.clip(total_conf, 0.0, 0.4))

        # Sigma from recent vol
        if n > 20:
            log_rets = np.diff(np.log(closes[-20:] + 1e-12))
            sigma = max(float(np.std(log_rets)) * np.sqrt(252), 0.05)
        else:
            sigma = 0.15

        return AlphaSignal(
            mu=mu, sigma=sigma, confidence=confidence,
            horizon=self.horizon, alpha_name=self.name,
            metadata={
                'tom_mu': tom_mu, 'tom_conf': tom_conf,
                'monthly_mu': monthly_mu, 'monthly_conf': monthly_conf,
                'dow_mu': dow_mu, 'dow_conf': dow_conf,
                'is_tom_window': bool(current_date and
                    (current_date.day >= 28 or current_date.day <= 3)),
                'is_favorable_month': bool(current_date and
                    current_date.month in [11, 12, 1, 2, 3, 4]),
            },
            timestamp=bar_idx,
        )
```

---

## CATEGORY 6: CROSS-ASSET / FACTOR SIGNALS

### 6A. Betting Against Beta (BAB) Signal

**What**: Low-beta stocks earn higher risk-adjusted returns. For single-stock: track rolling beta and tilt against it.

**Evidence spans 90+ years globally.** Also confirmed in pre-1925 data.

**Source**: Frazzini & Pedersen (2014) "Betting Against Beta"

```python
# ============================================================================
# BAB (BETTING AGAINST BETA) — L1 Feature / L2 Signal Modifier
# ============================================================================

class BettingAgainstBetaFeature:
    """
    Tracks rolling beta of stock vs. market. When beta is high,
    expected risk-adjusted return is LOWER (low-vol anomaly).

    For single-stock mode: provides a slight contrarian tilt.
    For multi-asset mode: tilts portfolio toward low-beta stocks.

    Source: Frazzini & Pedersen (2014), confirmed over 90+ years.
    """

    def __init__(self, window: int = 126):
        self.window = window
        self._stock_rets = deque(maxlen=window + 5)
        self._market_rets = deque(maxlen=window + 5)

    def update(self, stock_return: float, market_return: float) -> dict:
        self._stock_rets.append(stock_return)
        self._market_rets.append(market_return)

        if len(self._stock_rets) < 30:
            return {'rolling_beta': 1.0, 'bab_signal': 0.0, 'bab_confidence': 0.0}

        sr = np.array(list(self._stock_rets))
        mr = np.array(list(self._market_rets))

        mkt_var = np.var(mr) + 1e-10
        beta = float(np.cov(sr, mr)[0, 1] / mkt_var)

        # BAB signal: negative mu when beta is high, positive when low
        # Low beta stocks earn higher risk-adjusted returns
        bab_signal = float(np.clip(-(beta - 1.0) * 0.02, -0.1, 0.1))

        # Confidence proportional to how extreme beta is
        bab_conf = float(np.clip(abs(beta - 1.0) / 1.0, 0.0, 0.3))

        return {
            'rolling_beta': beta,
            'bab_signal': bab_signal,
            'bab_confidence': bab_conf,
        }
```

---

### 6B. Cross-Sectional Dispersion Signal

**What**: When return dispersion across the universe is high, stock-picking alphas work better.

```python
# ============================================================================
# DISPERSION SIGNAL — L2 Confidence Modifier
# ============================================================================

class DispersionSignal:
    """
    Cross-sectional return dispersion across the asset universe.

    High dispersion → alpha opportunity → increase overall confidence
    Low dispersion → crowded, correlated → reduce confidence

    Only useful in multi-asset mode with 10+ symbols.
    """

    def __init__(self, window: int = 21):
        self.window = window
        self._dispersion_history = deque(maxlen=252)

    def update(self, cross_sectional_returns: np.ndarray) -> dict:
        """
        Args:
            cross_sectional_returns: Array of returns across all stocks
                                     in the universe for this bar.
        """
        if len(cross_sectional_returns) < 5:
            return {'dispersion': 0.0, 'dispersion_zscore': 0.0,
                    'alpha_environment': 'normal'}

        disp = float(np.std(cross_sectional_returns))
        self._dispersion_history.append(disp)

        if len(self._dispersion_history) < 20:
            return {'dispersion': disp, 'dispersion_zscore': 0.0,
                    'alpha_environment': 'normal'}

        hist = np.array(list(self._dispersion_history))
        z = (disp - np.mean(hist)) / (np.std(hist) + 1e-10)

        if z > 1.0:
            env = 'rich'       # Good alpha environment
        elif z < -1.0:
            env = 'crowded'    # Poor alpha environment
        else:
            env = 'normal'

        return {
            'dispersion': disp,
            'dispersion_zscore': float(np.clip(z, -3.0, 3.0)),
            'alpha_environment': env,
        }
```

---

## CATEGORY 7: ADVANCED TECHNICAL / STRUCTURAL

### 7A. Intrabar Range Ratio (Close Position Within Range)

**What**: Where does the close price fall within the day's high-low range? One of the highest-IC short-term features in quantitative research.

```python
# ============================================================================
# INTRABAR RANGE RATIO — Observation vector feature (L1)
# ============================================================================

class IntrabarRangeFeature:
    """
    Close position within the bar's range:
        range_position = (Close - Low) / (High - Low)

    RP > 0.8 → strong close (buyers in control) → bullish
    RP < 0.2 → weak close (sellers in control) → bearish
    RP ≈ 0.5 → neutral

    Economic hypothesis: Institutional accumulation/distribution
    pushes closes toward extremes. Smart money buys into close.

    This is a single feature, not a standalone alpha.
    Feed into observation vector and/or condition trend alpha.
    """

    def __init__(self, lookback: int = 21):
        self.lookback = lookback
        self._rp_history = deque(maxlen=lookback)

    def compute(self, opens: np.ndarray, highs: np.ndarray,
                lows: np.ndarray, closes: np.ndarray) -> dict:
        """
        Returns features for the current bar.
        """
        if len(closes) < 1:
            return {'range_position': 0.5, 'range_position_ema': 0.5,
                    'range_position_signal': 0.0}

        h = highs[-1]
        l = lows[-1]
        c = closes[-1]
        bar_range = h - l

        if bar_range < 1e-10:
            rp = 0.5
        else:
            rp = (c - l) / bar_range

        rp = float(np.clip(rp, 0.0, 1.0))
        self._rp_history.append(rp)

        # EMA of range position (smoothed signal)
        if len(self._rp_history) >= 5:
            rp_arr = np.array(list(self._rp_history))
            alpha = 2.0 / (min(len(rp_arr), self.lookback) + 1)
            ema = rp_arr[0]
            for v in rp_arr[1:]:
                ema = alpha * v + (1 - alpha) * ema
            rp_ema = float(ema)
        else:
            rp_ema = rp

        # Signal: centered at 0 (-0.5 to +0.5)
        signal = float(rp_ema - 0.5)

        return {
            'range_position': rp,
            'range_position_ema': rp_ema,
            'range_position_signal': signal,  # Feed to trend alpha
        }

# USAGE: add to observation vector building:
#   rp_features = intrabar_range.compute(opens, highs, lows, closes)
#   obs_vector.append(rp_features['range_position'])
```

---

### 7B. Gap Analysis (Overnight Returns)

**What**: Track overnight gaps and their tendency to fill or continue.

```python
# ============================================================================
# GAP ANALYSIS — Overnight return decomposition (L1 feature)
# ============================================================================

class GapAnalysis:
    """
    Decomposes daily return into overnight (close-to-open) and
    intraday (open-to-close) components.

    Key findings:
    - Nearly ALL equity returns accrue overnight, not intraday.
    - Large gap ups that hold = continuation (breakout)
    - Large gap ups that fill = mean reversion (exhaustion)
    - Track gap fill rate as a regime indicator.

    Source: French & Roll (1986), Elm Wealth (2025)
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback
        self._overnight_rets = deque(maxlen=lookback)
        self._intraday_rets = deque(maxlen=lookback)
        self._gap_fill_count = 0
        self._gap_count = 0

    def compute(self, opens: np.ndarray, highs: np.ndarray,
                lows: np.ndarray, closes: np.ndarray) -> dict:
        n = len(closes)
        if n < 3 or len(opens) < n:
            return {'overnight_ret': 0.0, 'intraday_ret': 0.0,
                    'gap_pct': 0.0, 'overnight_dominance': 0.5,
                    'gap_fill_rate': 0.5}

        # Overnight return: open today vs close yesterday
        overnight_ret = float(np.log(opens[-1] / (closes[-2] + 1e-12)))
        # Intraday return: close today vs open today
        intraday_ret = float(np.log(closes[-1] / (opens[-1] + 1e-12)))

        self._overnight_rets.append(overnight_ret)
        self._intraday_rets.append(intraday_ret)

        # Gap detection and fill tracking
        gap_threshold = 0.005  # 0.5% gap
        if abs(overnight_ret) > gap_threshold:
            self._gap_count += 1
            # Did intraday reverse the gap? (gap fill)
            if np.sign(overnight_ret) != np.sign(intraday_ret):
                self._gap_fill_count += 1

        # Overnight dominance: what fraction of total return is overnight?
        if len(self._overnight_rets) >= 10:
            on = np.array(list(self._overnight_rets))
            intra = np.array(list(self._intraday_rets))
            total_abs = np.sum(np.abs(on)) + np.sum(np.abs(intra)) + 1e-10
            overnight_share = np.sum(np.abs(on)) / total_abs
        else:
            overnight_share = 0.5

        gap_fill_rate = (self._gap_fill_count /
                         max(self._gap_count, 1))

        return {
            'overnight_ret': overnight_ret,
            'intraday_ret': intraday_ret,
            'gap_pct': overnight_ret,  # Same as overnight ret
            'overnight_dominance': float(overnight_share),
            'gap_fill_rate': float(gap_fill_rate),
        }
```

---

### 7C. Average True Range (ATR) as Regime Detector

**What**: ATR captures the "true" range including overnight gaps. ATR/Price ratio as a normalized volatility regime indicator.

```python
# ============================================================================
# ATR REGIME DETECTOR — L1/L3 Feature
# ============================================================================

class ATRRegimeDetector:
    """
    Average True Range normalized by price. Provides a regime
    indicator that captures overnight gaps (unlike range-based vol).

    ATR/Price high → volatile regime → tighten risk
    ATR/Price low → calm regime → conditions for breakout

    Also useful for position sizing (tighter stops in high ATR).
    """

    def __init__(self, window: int = 14):
        self.window = window
        self._atr_history = deque(maxlen=252)

    def compute(self, highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray) -> dict:
        n = len(closes)
        if n < self.window + 2:
            return {'atr': 0.0, 'atr_pct': 0.0, 'atr_regime': 'normal',
                    'atr_zscore': 0.0}

        # True Range: max(H-L, |H-prevC|, |L-prevC|)
        tr = np.maximum(
            highs[-self.window:] - lows[-self.window:],
            np.maximum(
                np.abs(highs[-self.window:] - closes[-self.window - 1:-1]),
                np.abs(lows[-self.window:] - closes[-self.window - 1:-1])
            )
        )

        atr = float(np.mean(tr))
        atr_pct = atr / (closes[-1] + 1e-10)  # Normalized by price
        self._atr_history.append(atr_pct)

        # Z-score vs. history
        if len(self._atr_history) >= 30:
            arr = np.array(list(self._atr_history))
            z = (atr_pct - np.mean(arr)) / (np.std(arr) + 1e-10)
        else:
            z = 0.0

        if z > 1.5:
            regime = 'high_atr'
        elif z < -1.0:
            regime = 'low_atr'
        else:
            regime = 'normal'

        return {
            'atr': atr,
            'atr_pct': float(atr_pct),
            'atr_regime': regime,
            'atr_zscore': float(np.clip(z, -3.0, 3.0)),
        }
```

---

## CATEGORY 8: CROWDING / FLOW-BASED

### 8A. Crowding Detection via Factor Return Correlation

**What**: Monitor whether our composite signal correlates with simple factor returns. High correlation = our "alpha" is just factor beta.

```python
# ============================================================================
# CROWDING DETECTOR — L4 Monitoring
# ============================================================================

class CrowdingDetector:
    """
    Monitors correlation of our composite signal with simple
    factor returns. When correlation is high, our "alpha" is
    just disguised factor beta → reduce confidence.

    Factors tracked (computable from OHLCV):
      - Market beta (return)
      - Momentum (12-1 month return)
      - Short-term reversal (1-week return, inverted)
      - Volatility (inverse of realized vol)

    Source: Resonanz Capital (2025): "Crowding is the natural
    by-product of shared inputs and incentives."
    """

    def __init__(self, window: int = 63):
        self.window = window
        self._our_signals = deque(maxlen=window)
        self._market_rets = deque(maxlen=window)
        self._mom_signals = deque(maxlen=window)
        self._rev_signals = deque(maxlen=window)

    def update(self, our_mu: float, market_ret: float,
               closes: np.ndarray) -> dict:
        """
        Call each bar with our composite mu and market return.
        """
        self._our_signals.append(our_mu)
        self._market_rets.append(market_ret)

        n = len(closes)
        if n > 252:
            # Momentum: 12-month return skipping last month
            mom = np.log(closes[-21] / (closes[-252] + 1e-12))
            self._mom_signals.append(mom)
        if n > 5:
            # Reversal: negative of past-week return
            rev = -np.log(closes[-1] / (closes[-5] + 1e-12))
            self._rev_signals.append(rev)

        if len(self._our_signals) < 30:
            return {'crowding_score': 0.0, 'most_correlated_factor': 'none',
                    'crowding_alert': False}

        our = np.array(list(self._our_signals))

        # Correlate with each factor
        corrs = {}
        for name, factor_data in [
            ('market', list(self._market_rets)),
            ('momentum', list(self._mom_signals)),
            ('reversal', list(self._rev_signals)),
        ]:
            if len(factor_data) >= len(our):
                fdata = np.array(factor_data[-len(our):])
                if np.std(our) > 1e-10 and np.std(fdata) > 1e-10:
                    c = float(np.corrcoef(our, fdata)[0, 1])
                    if not np.isnan(c):
                        corrs[name] = abs(c)

        if not corrs:
            return {'crowding_score': 0.0, 'most_correlated_factor': 'none',
                    'crowding_alert': False}

        max_factor = max(corrs, key=corrs.get)
        max_corr = corrs[max_factor]

        return {
            'crowding_score': max_corr,
            'factor_correlations': corrs,
            'most_correlated_factor': max_factor,
            'crowding_alert': max_corr > 0.7,
        }
```

---

## PRIORITY RANKING: FINAL RECOMMENDATIONS

### Tier 1: Implement Now (Foundational improvements, high impact/effort ratio)

| # | Alpha/Feature | Category | Code Provided | Type |
|---|--------------|----------|---------------|------|
| 1 | **Yang-Zhang vol estimator** | Volatility | `yang_zhang_vol()` | Utility (replaces np.std everywhere) |
| 2 | **Amihud Illiquidity** | Liquidity | `AmihudLiquidityAlpha` | New L1 alpha |
| 3 | **Rolling Hurst Exponent** | Regime | `RollingHurstExponent` | L2 gating feature |
| 4 | **Intrabar Range Ratio** | Technical | `IntrabarRangeFeature` | Observation feature |

### Tier 2: Implement Soon (New alpha sources)

| # | Alpha/Feature | Category | Code Provided | Type |
|---|--------------|----------|---------------|------|
| 5 | **Volume-Price Divergence** | Microstructure | `VolumePriceDivergence` | Confidence modifier |
| 6 | **CalendarAlpha** | Seasonality | `CalendarAlpha` | Replaces SeasonalityAlpha |
| 7 | **Gap Analysis** | Microstructure | `GapAnalysis` | Observation features |
| 8 | **Idiosyncratic Momentum** | Momentum | `IdiosyncraticMomentumAlpha` | New L1 alpha (multi-asset) |

### Tier 3: Implement Later (Valuable but more complex)

| # | Alpha/Feature | Category | Code Provided | Type |
|---|--------------|----------|---------------|------|
| 9 | **Vol-of-Vol** | Volatility | `VolOfVolFeature` | Feature |
| 10 | **Vol Term Structure** | Volatility | `VolTermStructure` | Feature |
| 11 | **Dispersion** | Factor | `DispersionSignal` | L2 modifier (multi-asset) |
| 12 | **BAB signal** | Factor | `BettingAgainstBetaFeature` | Feature |
| 13 | **Crowding Detection** | Monitoring | `CrowdingDetector` | L4 monitoring |
| 14 | **ATR Regime** | Technical | `ATRRegimeDetector` | Feature |
| 15 | **Correlation Regime** | Regime | `CorrelationRegimeDetector` | L2/L3 feature |
| 16 | **Short-Term Reversal** | Momentum | `ShortTermReversalAlpha` | New L1 alpha |
| 17 | **Risk-Managed Momentum** | Momentum | `_risk_manage_momentum()` | TrendAlpha patch |

### Not Recommended (requires data we don't have)

| Alpha | Why Not |
|-------|---------|
| VPIN / Order Flow Imbalance | Requires tick/order book data |
| Put-Call Ratio / Options Skew | Requires options data |
| Short Interest | Requires borrow/lending data |
| Earnings Announcement effects | Requires earnings calendar |
| FOMC drift | Requires event calendar |
| Sentiment / NLP | Requires text data feeds |
| Fundamental factors (P/E, quality) | Requires fundamental data |

---

## IMPLEMENTATION NOTES

### ArchitectureConfig Additions

```python
# Add to ArchitectureConfig dataclass:

# Amihud Liquidity Alpha
amihud_lookback: int = 63
amihud_short_lookback: int = 10

# Calendar Alpha
calendar_warmup: int = 252

# Idiosyncratic Momentum
idio_mom_formation: int = 126
idio_mom_skip: int = 5

# Short-Term Reversal
reversal_formation: int = 5

# Hurst Exponent
hurst_window: int = 126
hurst_update_freq: int = 21
```

### Registration in Pipeline

```python
# In pipeline initialization, register new alphas:

pipeline.register_alpha('amihud_liquidity',
    AmihudLiquidityAlpha(acfg))
pipeline.register_alpha('calendar',
    CalendarAlpha(acfg))
# Remove old: pipeline.register_alpha('seasonality', SeasonalityAlpha(acfg))
```

### Observation Vector Impact

Current obs dim = 42 features (base). Adding all Tier 1 features:
- Yang-Zhang vol: replaces existing vol (no dim change)
- Amihud ratio: +2 features (log_illiq, illiq_z)
- Rolling Hurst: +2 features (hurst, trend_gate)
- Range position: +1 feature (range_position)

Total: 47 features. Well within sqrt(n_observations) limit for 300+ bar training windows.

### Anti-Overfitting Discipline

Per system spec, any new alpha must:
- Have stated economic hypothesis (documented above for all)
- Pass t-stat > 3.0 threshold
- Paper-trade minimum 60 days before live allocation
- Specify expected half-life and capacity
- Be subject to same demotion/retirement policies