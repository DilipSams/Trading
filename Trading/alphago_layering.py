"""
================================================================================
ALPHA-TRADE v6.0 -- INSTITUTIONAL ARCHITECTURE LAUNCHER
================================================================================

Integrates the base trading system (v3.0) with the institutional L1->L4
pipeline from alphago_architecture.py.

REQUIRED FILES (same directory):
    alphago_trading_system.py   -- Base PPO/MCTS trading system
    alphago_architecture.py     -- L1->L2->L3->L4 institutional pipeline

OPTIONAL:
    alphago_enhancements.py     -- v5.0 enhancement layers (observation augmentation)

USAGE:
    # -- Quick test with synthetic data --
    python alphago_layering.py --synthetic

    # -- Fewer iterations for fast testing --
    python alphago_layering.py --synthetic --iterations 3

    # -- Live market data --
    python alphago_layering.py --symbols AAPL,MSFT,NVDA

    # -- Load CSV/Parquet files --
    python alphago_layering.py --data-dir ./my_data

    # -- Skip training, just test the pipeline on synthetic data --
    python alphago_layering.py --synthetic --eval-only

    # -- Disable specific alphas --
    python alphago_layering.py --synthetic --disable-alphas carry,seasonality

    # -- Show pipeline status after each eval step --
    python alphago_layering.py --synthetic --verbose 2

ARCHITECTURE:
    L1  Alpha Factory    +' RL + Trend + MR + Value + Carry + Seasonality + VolPremium + Amihud + Hurst + STReversal
    L2  Signal Ensemble  +' Regime gating + meta-learner + horizon blending
    L3  Portfolio / Risk +' Vol-target +' Kelly +' DD scaling +' drift +' turnover cap
    L4  Execution        +' No-trade region +' kill switches +' TWAP slicing

    RL is ONE alpha among TEN (Tier 1: Amihud + Hurst, Tier 3: STReversal). It NEVER outputs a position size.
    Only L3 (PortfolioConstructor) determines target_exposure.
================================================================================
"""

import sys
import os
import argparse
import time
import json
import math
from copy import copy
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# -- Ensure sibling imports work --
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# -- Import base trading system --
try:
    from alphago_trading_system import (
        Config, AlphaTradeSystem, SyntheticMarketGenerator, TradingEnv,
        prepare_datasets, download_data, load_from_dir, load_from_norgate,
        NORGATE_DIR, NORGATE_DATABASES, build_network,
        unwrap_net, DEFAULT_SYMBOLS, SECTOR_MAP, HAS_YF, NUM_FEATURES,
        print_box, print_divider, print_gpu_info, tprint, C, progress_bar,
        build_feature_matrix, compute_indicators,  # For RL observation construction
        hbar_chart, line_chart, dual_line_chart,  # Terminal charts
    )
    HAS_BASE = True
except ImportError as e:
    print(f"ERROR: Cannot import alphago_trading_system.py")
    print(f"  Detail: {e}")
    print(f"  Ensure it's in: {script_dir}")
    sys.exit(1)

# -- Import institutional architecture --
try:
    from alphago_architecture import (
        ArchitectureConfig, InstitutionalPipeline,
        AlphaSignal, PortfolioOrder, PipelineResult,
        RLAlphaAdapter, TrendAlpha, MeanReversionAlpha,
        ValueAlpha, CarryAlpha, SeasonalityAlpha, VolatilityPremiumAlpha,
        TrendVolRegimeDetector, FutureLeakGuard, build_default_pipeline,
    )
    HAS_ARCH = True
except ImportError as e:
    print(f"ERROR: Cannot import alphago_architecture.py")
    print(f"  Detail: {e}")
    print(f"  Ensure it's in: {script_dir}")
    sys.exit(1)

import numpy as np

try:
    from data_quality import DataLoadMeta, analyze_ohlcv, print_quality_table, hash_df
    HAS_DATA_QUALITY = True
except ImportError:
    HAS_DATA_QUALITY = False

# -- v7.0 WS2: Data Infrastructure modules --
try:
    from data_quality import TradingCalendar, DataIngestionGate, MissingDataPolicy
    HAS_WS2 = True
except ImportError:
    HAS_WS2 = False

# -- v7.0 WS1/WS6: Validation & Reproducibility --
try:
    from validation_engine import (
        ReproducibilityManager, VersionedConfig,
        PurgedWalkForwardCV, SignificanceGate, MultipleTestingTracker,
        HoldoutGuard,
    )
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False

# -- v7.0 WS4: Evaluation & Reporting --
try:
    from backtest_report import (
        BacktestReport, compute_backtest_report,
        StressTester, CapacityEstimator,
    )
    HAS_BACKTEST_REPORT = True
except ImportError:
    HAS_BACKTEST_REPORT = False

try:
    from run_artifacts import RunWriter
    HAS_RUN_ARTIFACTS = True
except ImportError:
    HAS_RUN_ARTIFACTS = False

try:
    from table_formatter import TableFormatter, format_alpha_results
    HAS_TABLE_FORMATTER = True
except ImportError:
    HAS_TABLE_FORMATTER = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"


# ============================================================================
# ALPHA REGISTRY
# ============================================================================

ALPHA_REGISTRY = {
    'rl':          ('RL Agent (PPO+MCTS)',       RLAlphaAdapter),
    'trend':       ('Trend Following',           TrendAlpha),
    'mr':          ('Mean Reversion',            MeanReversionAlpha),
    'value':       ('Value',                     ValueAlpha),
    'carry':       ('Carry',                     CarryAlpha),
    'seasonality': ('Seasonality',               SeasonalityAlpha),
    'vol_premium': ('Volatility Premium',        VolatilityPremiumAlpha),
}


# ============================================================================
# OUTPUT FORMATTING HELPERS
# ============================================================================

_section_counter = 0

def print_section(title, reset=False):
    """Numbered section divider for major pipeline stages."""
    global _section_counter
    if reset:
        _section_counter = 0
        return
    _section_counter += 1
    print_divider(f"{_section_counter}. {title}")

def print_subsection(title):
    """Lightweight sub-section divider within a numbered section."""
    print(f"\n  {C.CYAN}{C.BOLD}--- {title} ---{C.RESET}")


# ============================================================================
# PIPELINE-BASED EVALUATION
# ============================================================================

def evaluate_with_pipeline(net, datasets, pipeline, cfg, acfg, label="eval",
                           verbose=0, spy_returns_lookup=None):
    """
    Evaluate a trained network through the full institutional pipeline.

    Instead of: network +' action +' env.step(action)
    We now do:  network +' L1(alpha) +' L2(ensemble) +' L3(portfolio) +' L4(exec) +' env.step

    This is the KEY difference from v5.0 -- the RL agent's output is treated
    as one alpha signal among many, and L3 makes all sizing decisions.
    """
    if HAS_TORCH:
        net.eval()

    # -- Run logging (run_artifacts) --
    writer = None
    if HAS_RUN_ARTIFACTS:
        out_dir = f"run_output/{label}_{int(time.time())}"
        writer = RunWriter(out_dir=out_dir)
        writer.write_manifest({
            "label": label,
            "architecture_config": {k: str(v) for k, v in vars(acfg).items()} if hasattr(acfg, '__dict__') else str(acfg),
            "starting_capital": cfg.starting_capital,
            "n_symbols": len(datasets),
        })

    all_pnls = []
    all_trades = []
    all_dds = []
    all_sharpes = []
    all_audit = []
    per_sym = {}

    # -- FIX: Collect ACTUAL per-bar returns for institutional backtest report --
    # These replace the np.random.randn proxy returns that previously made
    # every BacktestReport metric fabricated.
    all_bar_returns_gross = []   # Per-bar portfolio return BEFORE costs
    all_bar_returns_net = []     # Per-bar portfolio return AFTER costs (realized)
    all_spy_returns = []         # SPY return for same dates (benchmark)
    # FIX Ãƒâ€šÃ‚Â§4.7: Track per-symbol return series separately to avoid
    # inflated observation counts from cross-symbol concatenation.
    per_symbol_returns_net = {}   # symbol -> list of per-bar net returns
    per_symbol_returns_gross = {} # symbol -> list of per-bar gross returns
    per_symbol_date_returns = {}  # symbol -> [(date_str, net_return), ...] for date-aligned equity

    for d in datasets:
        # -- Create environment as PURE SIMULATOR --
        # When running through the pipeline, the env must not independently
        # manage risk. Kill switches belong to L4. Drawdown-based position
        # restrictions belong to L3. Enabling both would cause double-gating
        # and "who killed it?" audit ambiguity.
        #
        # Ownership model (pipeline mode):
        #   Kill switches:     L4 ExecutionEngine  (not env)
        #   Drawdown scaling:  L3 PortfolioConstructor  (not env action masking)
        #   Position sizing:   L3 only  (env just executes discrete_action)
        #
        # In v3.0 standalone training mode, the env keeps its own risk logic
        # because no pipeline exists.
        sim_cfg = copy(cfg)
        sim_cfg.use_kill_switches = False
        sim_cfg.use_action_masking = False
        sim_cfg.use_trailing_stops = False       # L4 owns risk — env stops interfere
        sim_cfg.use_asymmetric_stops = False     # L4 owns risk — env stops interfere
        # Match env action space to Pipeline's 7-level discretization
        sim_cfg.action_targets = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
        sim_cfg.n_actions = 7                    # Must match action_targets length
        env = TradingEnv(d.features_test, d.prices_test, sim_cfg, d.symbol, ev=True)
        # FIX Ãƒâ€šÃ‚Â§4.4: Flag env as pipeline eval mode ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â prevents get_risk_target() future access
        env._pipeline_eval_mode = True
        obs, _ = env.reset()
        done = False

        # -- Reset pipeline for this symbol --
        pipeline.reset(cfg.starting_capital)

        # -- Inject trained network into RL alpha --
        rl_alpha = None
        for name in pipeline.alpha_factory.alpha_names:
            alpha = pipeline.alpha_factory._alphas.get(name)
            if isinstance(alpha, RLAlphaAdapter):
                alpha.set_network(unwrap_net(net) if HAS_TORCH else None)
                rl_alpha = alpha
                break

        step_count = 0
        prev_close = None
        episode_audit = []
        sym_bar_returns_gross = []   # Per-bar returns for this symbol
        sym_bar_returns_net = []
        sym_spy_returns = []         # SPY return for same dates
        sym_date_returns = []        # [(date_str, net_return), ...] for equity curve
        bars_in_cash = 0             # bars where not in a position (for cash yield)
        peak_notional = 0.0          # max abs(shares * price) for $ Used

        while not done:
            bar_idx = env.cs
            # Phase 1: Extract OHLC data for Tier 1 alphas
            opens = env.prices[:bar_idx + 1, 0] if env.prices.shape[1] > 0 else None
            highs = env.prices[:bar_idx + 1, 1] if env.prices.shape[1] > 1 else None
            lows = env.prices[:bar_idx + 1, 2] if env.prices.shape[1] > 2 else None
            closes = env.prices[:bar_idx + 1, 3]
            volumes = env.prices[:bar_idx + 1, 4] if env.prices.shape[1] > 4 else None

            # Bar return
            bar_ret = 0.0
            if prev_close is not None and closes[-1] > 0:
                bar_ret = math.log(closes[-1] / (prev_close + 1e-12))
            prev_close = float(closes[-1])

            # Capture portfolio value BEFORE this step for actual return computation
            pv_before = env._portfolio_value()

            # Realized vol (for L4 no-trade region)
            if len(closes) > 20:
                log_rets = np.diff(np.log(closes[-20:] + 1e-12))
                realized_vol = float(np.std(log_rets)) * np.sqrt(252)
            else:
                realized_vol = 0.15

            # -- SET L3 COST MODEL CONTEXT (canonical cost model) --
            mid_price = float(closes[-1])
            pv = env._portfolio_value()
            pc = pipeline.portfolio
            pc._last_mid = mid_price
            pc._last_portfolio_value = pv
            if volumes is not None and len(volumes) > 20:
                adv_shares = float(np.mean(volumes[-20:]))
                pc._adv_dollars = adv_shares * mid_price
            # else: falls back to acfg.cost_default_adv_dollars

            # -- RUN FULL L1 -> L2 -> L3 -> L4 PIPELINE --
            # FIX Ãƒâ€šÃ‚Â§5.2: Compute per-bar L0 data quality score.
            # Prefer data_quality.analyze_ohlcv() when available; fallback uses
            # configurable penalty weights instead of hardcoded magic numbers.
            _dq_score = None
            if closes is not None and len(closes) > 1:
                if HAS_DATA_QUALITY and volumes is not None:
                    try:
                        import pandas as _pd
                        _ohlcv_df = _pd.DataFrame({
                            'Open': closes, 'High': closes, 'Low': closes,
                            'Close': closes, 'Volume': volumes if volumes is not None else 1e6,
                        })
                        _dq_meta = analyze_ohlcv(_ohlcv_df, label="live_bar")
                        _dq_score = _dq_meta.quality_score
                    except Exception:
                        _dq_score = None  # Fall through to fallback

                if _dq_score is None:
                    # Fallback: configurable penalty weights
                    _n = len(closes)
                    _nan_frac = float(np.sum(np.isnan(closes))) / _n if _n > 0 else 0.0
                    _zero_frac = float(np.sum(closes <= 0)) / _n if _n > 0 else 0.0
                    _stale = 0
                    if _n > 5:
                        _recent = closes[-5:]
                        _stale = int(np.all(_recent == _recent[0]))
                    # Configurable penalties (from acfg if available, else defaults)
                    _nan_penalty = getattr(acfg, 'dq_nan_penalty', 500.0)
                    _zero_penalty = getattr(acfg, 'dq_zero_penalty', 500.0)
                    _stale_penalty = getattr(acfg, 'dq_stale_penalty', 40.0)
                    _dq_score = max(0.0, 100.0 - _nan_frac * _nan_penalty - _zero_frac * _zero_penalty - _stale * _stale_penalty)

            result = pipeline.step(
                observation=obs,
                closes=closes,
                volumes=volumes,
                bar_idx=bar_idx,
                bar_return=bar_ret,
                current_exposure=env.exposure,
                portfolio_value=pv,
                regime_probs=None,
                realized_vol=realized_vol,
                data_quality_score=_dq_score,
                opens=opens,   # Phase 1: Pass OHLC to pipeline
                highs=highs,   # Phase 1: Pass OHLC to pipeline
                lows=lows,     # Phase 1: Pass OHLC to pipeline
            )

            # -- Use L4's discretized action directly --
            action = result.discrete_action

            # -- Step environment --
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step_count += 1

            # -- Feed REAL execution costs to L4 from env info --
            traded_notional = info.get("step_traded_notional", 0.0)
            step_commission = info.get("step_commission", 0.0)
            step_slippage_bps = info.get("step_slippage_bps", 0.0)
            step_mid = info.get("step_mid_price", 0.0)

            pipeline.execution_engine.ingest_execution(
                traded_notional=traded_notional,
                commission=step_commission,
                slippage_bps=step_slippage_bps,
                mid_price=step_mid,
            )

            # -- FIX: Compute ACTUAL per-bar returns (net and gross) --
            pv_after = env._portfolio_value()
            if pv_before > 0 and step_count > 0:
                # Net return: what the portfolio actually earned (after all costs)
                bar_return_net = (pv_after - pv_before) / pv_before
                # Gross return: add back costs to get pre-cost return
                cost_this_bar = step_commission + (
                    step_slippage_bps * 1e-4 * traded_notional if traded_notional > 0 else 0.0
                )
                bar_return_gross = bar_return_net + cost_this_bar / pv_before
                sym_bar_returns_net.append(float(bar_return_net))
                sym_bar_returns_gross.append(float(bar_return_gross))
                # Date-aligned return for equity curve
                if d.timestamps_test is not None:
                    sym_date_returns.append((str(d.timestamps_test[bar_idx])[:10],
                                             float(bar_return_net)))

            # -- Track position state for cash yield + capital used --
            _notional = abs(env.shares * float(closes[-1]))
            if _notional > peak_notional:
                peak_notional = _notional
            if abs(env.exposure) < 0.01:
                bars_in_cash += 1

            # -- SPY benchmark return for this bar's date --
            spy_ret_this_bar = 0.0
            if spy_returns_lookup and d.timestamps_test is not None:
                date_str = str(d.timestamps_test[bar_idx])[:10]
                spy_ret_this_bar = spy_returns_lookup.get(date_str, 0.0)
            sym_spy_returns.append(spy_ret_this_bar)

            # -- FIX Ãƒâ€šÃ‚Â§5.4: Position reconciliation --
            # In backtesting, "external" = env ground truth. Verifies L4 tracking.
            if hasattr(pipeline, '_reconciler') and pipeline._reconciler is not None:
                pipeline._reconciler.reconcile(
                    bar_idx=bar_idx,
                    internal_position=result.target_exposure,
                    external_position=env.exposure,
                    internal_nav=pv_after if 'pv_after' in dir() else pv,
                    external_nav=env._portfolio_value(),
                )

            # -- Log trade to run_artifacts --
            if writer and traded_notional > 0:
                writer.log_trade({
                    "t": bar_idx,
                    "side": "buy" if result.target_exposure > env.exposure else "sell",
                    "trade_notional": traded_notional,
                    "fill": step_mid * (1 + step_slippage_bps * 1e-4) if step_mid > 0 else 0.0,
                    "mid": step_mid,
                    "slippage_bps": step_slippage_bps,
                    "fees": step_commission,
                    "impact": 0.0,
                    "half_spread": 0.0,
                })


            # -- Record audit --
            if verbose >= 2 and step_count % 200 == 0:
                pipeline.print_status()

            episode_audit.append(result.audit)

        # -- Episode-scoped L4 stats (correctly reset per symbol) --
        ep_stats = pipeline.execution_engine.stats

        # -- Collect episode results --
        all_pnls.append(info["net_pnl"])
        all_trades.append(info["total_trades"])
        all_dds.append(info["max_drawdown"])
        all_sharpes.append(info.get("sharpe_ratio", 0))
        all_audit.extend(episode_audit)

        # -- FIX Ãƒâ€šÃ‚Â§4.7: Store per-symbol returns separately --
        # Concatenation is kept for backward compatibility but per-symbol data
        # enables correct per-symbol metrics (vol, Sharpe, drawdown).
        per_symbol_returns_net[d.symbol] = sym_bar_returns_net
        per_symbol_returns_gross[d.symbol] = sym_bar_returns_gross
        per_symbol_date_returns[d.symbol] = sym_date_returns
        all_bar_returns_gross.extend(sym_bar_returns_gross)
        all_bar_returns_net.extend(sym_bar_returns_net)
        all_spy_returns.extend(sym_spy_returns)

        per_sym[d.symbol] = {
            "pnl": info["net_pnl"],
            "trades": info["total_trades"],
            "win_rate": info.get("win_rate", 0),
            "profit_factor": info.get("profit_factor", 0),
            "sharpe": info.get("sharpe_ratio", 0),
            "max_dd": info["max_drawdown"],
            "turnover": info.get("turnover", 0.0),
            # FIX: Cash yield only for bars sitting in cash (not in position)
            "step_count": step_count,
            "bars_in_cash": bars_in_cash,
            "peak_notional": peak_notional,
            "cash_yield_pnl": (
                (cfg.cash_yield_bps_annual * 1e-4 / max(cfg.bars_per_year, 1))
                * cfg.starting_capital * bars_in_cash
                if getattr(cfg, 'cash_yield_bps_annual', 0) > 0 else 0.0
            ),
            # Episode-scoped L4 stats
            "pipeline_trades": ep_stats["n_trades"],
            "pipeline_suppressed": ep_stats["n_suppressed"],
            "pipeline_suppression_rate": ep_stats["suppression_rate"],
            "pipeline_fills": ep_stats["n_fills"],
            "pipeline_avg_slippage_bps": ep_stats["avg_slippage_bps"],
            "pipeline_worst_slippage_bps": ep_stats["worst_slippage_bps"],
            "pipeline_total_commission": ep_stats["total_commission"],
            "pipeline_kill": ep_stats["kill_triggered"],
            "pipeline_kill_context": ep_stats["kill_context"],
        }

        # -- Episode boundary log (good-to-have 3) --
        if verbose >= 1:
            lt_so_far = pipeline.execution_engine.lifetime_stats
            sym_info = per_sym[d.symbol]
            sup_rate = sym_info["pipeline_suppressed"] / max(
                sym_info["pipeline_trades"] + sym_info["pipeline_suppressed"], 1
            )
            # FIX §SA-5: Decompose PnL into trading alpha vs cash yield
            cash_yield = sym_info.get("cash_yield_pnl", 0.0)
            trading_pnl = sym_info["pnl"] - cash_yield
            pnl_color = C.GREEN if sym_info["pnl"] > 0 else C.RED
            trade_color = C.GREEN if trading_pnl > 0 else (C.RED if trading_pnl < -0.01 else C.YELLOW)
            tprint(
                f"  {d.symbol:>12s}  PnL:{pnl_color}${sym_info['pnl']:>+10,.2f}{C.RESET}  "
                f"(trade:{trade_color}${trading_pnl:>+8,.2f}{C.RESET} "
                f"cash:{C.CYAN}${cash_yield:>+8,.2f}{C.RESET})  "
                f"Sh:{sym_info['sharpe']:>+.2f}  DD:{sym_info['max_dd']:.1f}%  "
                f"Trades:{sym_info['trades']}  Suppressed:{sup_rate:.0%}",
                "info"
            )

    # -- Aggregate results --
    start_cap = float(cfg.starting_capital)
    rets = np.array(all_pnls) / max(start_cap, 1e-9)
    sp = sum(1 for p in all_pnls if p > 0)
    br = sp / max(len(all_pnls), 1)
    avg_sh = float(np.mean(all_sharpes)) if all_sharpes else 0.0
    dd_arr = np.array(all_dds) if all_dds else np.array([100.0])
    dd_q75 = float(np.quantile(dd_arr, 0.75))
    dd_max = float(np.max(dd_arr))

    mean_ret = float(np.mean(rets)) if len(rets) else 0.0
    med_ret = float(np.median(rets)) if len(rets) else 0.0

    # Win rate and profit factor (from active symbols only, matching base)
    active_syms = [s for s in per_sym.values() if s["trades"] > 0]
    avg_wr = float(np.mean([s["win_rate"] for s in active_syms])) if active_syms else 0.0
    avg_pf = float(np.mean([s["profit_factor"] for s in active_syms])) if active_syms else 0.0

    # Turnover penalty (matching base formula for apples-to-apples scoring)
    total_turnover = sum(s.get("turnover", 0.0) for s in per_sym.values())
    if total_turnover > 0:
        turnover_pen = (total_turnover / max(start_cap * len(datasets), 1e-9)) * 0.05
    else:
        avg_trades = sum(all_trades) / max(len(all_trades), 1)
        turnover_pen = avg_trades * 0.01

    score = (
        100.0 * (0.7 * mean_ret + 0.3 * med_ret) * br
        + 0.20 * avg_sh
        - 0.35 * (dd_q75 / 100.0)
        - 0.25 * (dd_max / 100.0)
        - 0.10 * turnover_pen
    )

    # Penalize no-trading behavior (matching base)
    if sum(all_trades) == 0:
        score -= 1.0

    # -- Close run_artifacts writer --
    run_output_dir = None
    if writer:
        writer.log_step({
            "_type": "eval_summary",
            "total_pnl": sum(all_pnls),
            "total_trades": sum(all_trades),
            "avg_sharpe": avg_sh,
            "dd_max": dd_max,
            "score": score,
        })
        writer.close()
        run_output_dir = writer.out_dir
        tprint(f"Run artifacts saved to: {run_output_dir}", "info")

    return {
        "label": label,
        "total_pnl": sum(all_pnls),
        "avg_pnl": float(np.mean(all_pnls)) if all_pnls else 0,
        "total_trades": sum(all_trades),
        # Schema-compatible with base evaluate (same keys)
        "avg_sh": avg_sh,
        "avg_sharpe": avg_sh,             # Alias for pipeline-specific code
        "avg_wr": avg_wr,
        "avg_pf": avg_pf,
        "avg_dd": float(np.mean(dd_arr)),
        "dd_q75": dd_q75,
        "dd_max": dd_max,
        "sp": sp,
        "st": len(datasets),
        "breadth": br * 100,
        "profitable_symbols": sp,          # Alias for pipeline-specific code
        "total_symbols": len(datasets),    # Alias for pipeline-specific code
        "turnover_pen": turnover_pen,
        "score": score,
        "per_sym": per_sym,
        "audit_log_size": len(all_audit),
        "run_output_dir": run_output_dir,
        # FIX Ãƒâ€šÃ‚Â§4.7: Per-bar returns ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â per-symbol series for correct metrics,
        # plus concatenated series for backward compatibility (with warning).
        "daily_returns_gross": np.array(all_bar_returns_gross, dtype=np.float64),
        "daily_returns_net": np.array(all_bar_returns_net, dtype=np.float64),
        "daily_returns_concatenated_warning": (
            "These are concatenated across symbols ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â observation count is inflated. "
            "Use per_symbol_returns_* for correct per-symbol Sharpe/vol/drawdown."
        ),
        "per_symbol_returns_net": {
            sym: np.array(rets, dtype=np.float64) for sym, rets in per_symbol_returns_net.items()
        },
        "per_symbol_returns_gross": {
            sym: np.array(rets, dtype=np.float64) for sym, rets in per_symbol_returns_gross.items()
        },
        "per_symbol_date_returns": per_symbol_date_returns,
        "spy_returns": np.array(all_spy_returns, dtype=np.float64),
    }


def print_results(r):
    """Pretty-print evaluation results."""
    pnl_color = C.GREEN if r["total_pnl"] > 0 else C.RED
    print(f"\n  {C.BOLD}{C.CYAN}=== {r['label']} ==={C.RESET}")
    print(f"  P&L:     {pnl_color}${r['total_pnl']:>+12,.2f}{C.RESET}  "
          f"(${r['avg_pnl']:>+,.2f}/sym)")
    print(f"  Sharpe:  {r['avg_sh']:>+.3f}   "
          f"WR: {r.get('avg_wr', 0):.1f}%   PF: {r.get('avg_pf', 0):.2f}")
    print(f"  DD: avg={r['avg_dd']:.1f}%  q75={r['dd_q75']:.1f}%  max={r['dd_max']:.1f}%")
    print(f"  Breadth: {r.get('sp', r.get('profitable_symbols', 0))}/{r.get('st', r.get('total_symbols', 0))}  "
          f"({r['breadth']:.0f}%)   Trades: {r['total_trades']}")
    print(f"  Score:   {C.BOLD}{r['score']:>+.3f}{C.RESET}")


# ============================================================================
# DATA SPLIT SUMMARY -- Shows train/val/test date ranges & bar counts
# ============================================================================

def _ts_range(timestamps):
    """Extract first and last date string from a timestamp array, or 'N/A' if empty/None."""
    if timestamps is None or len(timestamps) == 0:
        return "N/A", "N/A"
    first = str(timestamps[0])[:10]   # Trim to YYYY-MM-DD
    last = str(timestamps[-1])[:10]
    return first, last


def print_data_split_summary(datasets, cfg):
    """
    Print a clear table of train / validation / test (holdout) date ranges
    and data-point counts for every symbol, plus aggregate totals.

    Also shows the embargo gap size and split ratios used.

    Called immediately after prepare_datasets() so the user can verify
    that splits are correct before any training or evaluation begins.
    """
    if not datasets:
        return

    # ---- Header ----
    print(f"\n  {C.BOLD}{C.CYAN}=== DATA SPLIT SUMMARY ==={C.RESET}")
    train_frac = 1.0 - cfg.val_ratio - cfg.holdout_ratio
    n_total = sum(d.n_train + d.n_val + d.n_test for d in datasets)
    print(f"  Split ratios: Train={train_frac:.0%} | Val={cfg.val_ratio:.0%} | "
          f"Test/OOS={cfg.holdout_ratio:.0%}   "
          f"Embargo={cfg.embargo_bars} bars (configured, scaled per series)")
    print(f"  Total data points: {n_total:,} across {len(datasets)} symbol(s)")
    print()

    # ---- Per-symbol table ----
    # Header row
    hdr = (f"  {'Symbol':<20s} {'TF':>4s}  {'Pool':>7s}  "
           f"{'Train Start':>11s} {'Train End':>11s} {'#Bars':>6s}  "
           f"{'Val Start':>11s} {'Val End':>11s} {'#Bars':>6s}  "
           f"{'Test Start':>11s} {'Test End':>11s} {'#Bars':>6s}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(f"  {C.DIM}{'(Ticker)':<20s} {'':>4s}  {'(group)':>7s}  "
          f"{'':>11s} {'(learn from)':>11s} {'':>6s}  "
          f"{'':>11s} {'(tune with)':>11s} {'':>6s}  "
          f"{'':>11s} {'(final exam)':>11s} {'':>6s}{C.RESET}")
    print(sep)

    # Aggregates
    tot_train = tot_val = tot_test = 0
    agg_train_start = None
    agg_train_end = None
    agg_test_start = None
    agg_test_end = None

    for d in datasets:
        tr_s, tr_e = _ts_range(d.timestamps_train)
        va_s, va_e = _ts_range(d.timestamps_val)
        te_s, te_e = _ts_range(d.timestamps_test)
        pool = getattr(d, 'split_group', '?')

        sym_display = d.symbol[:20]
        tf_display = getattr(d, 'timeframe', '?')

        print(f"  {sym_display:<20s} {tf_display:>4s}  {pool:>7s}  "
              f"{tr_s:>11s} {tr_e:>11s} {d.n_train:>6,d}  "
              f"{va_s:>11s} {va_e:>11s} {d.n_val:>6,d}  "
              f"{te_s:>11s} {te_e:>11s} {d.n_test:>6,d}")

        tot_train += d.n_train
        tot_val += d.n_val
        tot_test += d.n_test

        # Track aggregate date range
        if d.timestamps_train is not None and len(d.timestamps_train) > 0:
            s = str(d.timestamps_train[0])[:10]
            e = str(d.timestamps_train[-1])[:10]
            if agg_train_start is None or s < agg_train_start:
                agg_train_start = s
            if agg_train_end is None or e > agg_train_end:
                agg_train_end = e
        if d.timestamps_test is not None and len(d.timestamps_test) > 0:
            s = str(d.timestamps_test[0])[:10]
            e = str(d.timestamps_test[-1])[:10]
            if agg_test_start is None or s < agg_test_start:
                agg_test_start = s
            if agg_test_end is None or e > agg_test_end:
                agg_test_end = e

    # ---- Totals row ----
    print(sep)
    print(f"  {'TOTAL':<20s} {'':>4s}  {'':>7s}  "
          f"{agg_train_start or 'N/A':>11s} {agg_train_end or 'N/A':>11s} {tot_train:>6,d}  "
          f"{'':>11s} {'':>11s} {tot_val:>6,d}  "
          f"{agg_test_start or 'N/A':>11s} {agg_test_end or 'N/A':>11s} {tot_test:>6,d}")
    print()

    # ---- Embargo visualization (for first dataset as representative) ----
    d0 = datasets[0]
    _, tr_e = _ts_range(d0.timestamps_train)
    va_s, va_e = _ts_range(d0.timestamps_val)
    te_s, _ = _ts_range(d0.timestamps_test)

    # Compute actual embargo used (same formula as _split_indices)
    n0 = d0.n_train + d0.n_val + d0.n_test
    emb_actual = min(int(cfg.embargo_bars), max(5, int((n0 + 200) * 0.03)))  # Approx
    # Better: infer from gaps between last train date and first val date
    print(f"  {C.DIM}Timeline ({d0.symbol}):{C.RESET}")
    print(f"  {C.DIM}  [TRAIN {tr_e}] --embargo~{emb_actual}bars--> "
          f"[VAL {va_s}..{va_e}] --embargo~{emb_actual}bars--> "
          f"[TEST/OOS {te_s}]{C.RESET}")
    print()


def print_wf_fold_summary(folds, timestamps, symbol, purge_gap, embargo):
    """
    Print walk-forward CV fold boundaries with date ranges.

    Args:
        folds:      List of CVFold objects from PurgedWalkForwardCV
        timestamps: numpy array of timestamp strings (from prices_train)
        symbol:     Symbol name for display
        purge_gap:  Configured purge gap in bars
        embargo:    Configured embargo in bars
    """
    if not folds or timestamps is None or len(timestamps) == 0:
        return

    n_obs = len(timestamps)

    def ts_at(idx):
        """Safely get timestamp at index, clamped to valid range."""
        idx = max(0, min(idx, n_obs - 1))
        return str(timestamps[idx])[:10]

    print(f"\n  {C.DIM}Walk-Forward Folds ({symbol}, purge={purge_gap}, embargo={embargo}):{C.RESET}")
    print(f"  {'Fold':>4s}  {'Train':>11s} -> {'End':>11s} {'#Bars':>6s}  "
          f"{'Purge':>6s}  "
          f"{'Test':>11s} -> {'End':>11s} {'#Bars':>6s}")
    print(f"  {'----':>4s}  {'-'*11} -- {'-'*11} {'-'*6}  "
          f"{'-'*6}  "
          f"{'-'*11} -- {'-'*11} {'-'*6}")

    for f in folds:
        tr_bars = f.train_end - f.train_start
        te_bars = f.test_end - f.test_start
        purge_bars = f.purge_end - f.train_end + f.embargo_end - f.purge_end

        print(f"  {f.fold_id:>4d}  "
              f"{ts_at(f.train_start):>11s} -> {ts_at(f.train_end - 1):>11s} {tr_bars:>6,d}  "
              f"{purge_bars:>6d}  "
              f"{ts_at(f.test_start):>11s} -> {ts_at(f.test_end - 1):>11s} {te_bars:>6,d}")

    # Holdout info
    if folds:
        holdout_start = folds[-1].test_end
        if holdout_start < n_obs:
            ho_bars = n_obs - holdout_start
            print(f"  {'HO':>4s}  "
                  f"{'':>11s}    {'':>11s} {'':>6s}  "
                  f"{'':>6s}  "
                  f"{ts_at(holdout_start):>11s} -> {ts_at(n_obs - 1):>11s} {ho_bars:>6,d}  (holdout 20%)")
    print()


# ============================================================================
# WS1A INTEGRATION: WALK-FORWARD ALPHA VALIDATION
# ============================================================================

def _process_single_dataset_validation(args):
    """
    Worker function to process walk-forward validation for a single dataset.
    Returns per-alpha results for this dataset to be aggregated by parent.

    This function is extracted to enable parallel processing of datasets.

    Args:
        args: Tuple of (d, acfg, alpha_names, verbose)
              pipeline and net are NOT passed (to avoid pickle issues)
    """
    d, acfg, alpha_names, verbose = args

    # Suppress import-time print noise from worker processes (verbose < 2)
    import io as _io, warnings as _warnings
    _suppress = verbose < 2
    if _suppress:
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = _io.StringIO()
        sys.stderr = _io.StringIO()
        _warnings.filterwarnings("ignore", category=UserWarning)

    try:
        from alphago_architecture import build_default_pipeline
        pipeline = build_default_pipeline(acfg, net=None)  # No RL net in validation
    finally:
        if _suppress:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
    # Use training prices for walk-forward validation.
    # prices_train columns: [open, high, low, close, volume, ...]
    prices = d.prices_train
    if prices is None or len(prices) < 200:
        return None  # Skip datasets with insufficient data

    n_obs = len(prices)
    close_col = 3  # Standard OHLCV column index for close

    # Purge gap = max alpha horizon (21 bars for monthly alphas).
    # Embargo = 5 bars for autocorrelation bleed.
    purge_gap = max(21, acfg.meta_learner_retrain_freq // 10)
    embargo = 5
    n_splits = min(5, max(2, n_obs // 200))  # Scale splits to data size

    # Reserve 20% as holdout (touched only once in final eval)
    cv = PurgedWalkForwardCV(
        n_obs=n_obs,
        purge_gap=purge_gap,
        embargo=embargo,
        n_splits=n_splits,
        holdout_pct=0.20,
    )
    folds = cv.generate_folds()
    if not folds:
        return None

    # Show fold boundaries with dates (verbose >= 2 only to avoid N*8 lines)
    if verbose >= 2 and d.timestamps_train is not None:
        print_wf_fold_summary(
            folds=folds,
            timestamps=d.timestamps_train,
            symbol=d.symbol,
            purge_gap=purge_gap,
            embargo=embargo,
        )

    # Precompute full close array and log returns (shifted by +1 for
    # next-bar prediction targets)
    closes_full = prices[:, close_col].copy()
    log_rets = np.zeros(n_obs)
    for i in range(1, n_obs):
        if closes_full[i] > 0 and closes_full[i - 1] > 0:
            log_rets[i] = math.log(closes_full[i] / closes_full[i - 1])

    # Generate alpha signals for every bar ONCE (signals are deterministic
    # given the price history up to that point -- no future leakage).
    # This is much faster than re-running the pipeline per fold.
    all_signals = {}  # bar_idx -> {alpha_name -> mu}
    volumes_full = prices[:, 4] if prices.shape[1] > 4 else None
    timestamps_full = d.timestamps_train  # For calendar features

    # ---- FIX: Build feature matrix for RL observation construction ----
    # Per RL_zero_forensic.md Section 2.1 - P0 Fix:
    # Construct observations to enable RL alpha signal generation.
    # Without this, RLAlphaAdapter.generate() always returns mu=0.0.
    features_full = None
    window_size = 60  # TradingEnv default window size
    try:
        import pandas as pd
        # Convert prices array to DataFrame
        df_prices = pd.DataFrame(
            prices,
            columns=['Open', 'High', 'Low', 'Close', 'Volume'][:prices.shape[1]]
        )
        if 'Volume' not in df_prices.columns:
            df_prices['Volume'] = 1_000_000  # Dummy volume
        # Compute technical indicators
        df_with_indicators = compute_indicators(df_prices)
        # Build feature matrix (n_obs × NUM_FEATURES)
        features_full = build_feature_matrix(df_with_indicators)
    except Exception as e:
        if verbose >= 1:
            tprint(f"Warning: Could not build features for {d.symbol}: {e}", "warn")

    # Note: RL alpha is not used in walk-forward validation
    # (we're validating traditional alphas only, net=None in pipeline creation above)

    # ---- OPTIMIZED: Subsample signal generation (5x speedup) ----
    # Alpha signals use 21-63 bar lookbacks and change slowly.
    # Computing every 5th bar loses <1% IC accuracy but cuts cost 5x.
    SIGNAL_STRIDE = 5
    signal_bars = list(range(50, n_obs, SIGNAL_STRIDE))
    if n_obs - 1 > 50 and (n_obs - 1) not in signal_bars:
        signal_bars.append(n_obs - 1)

    for t in signal_bars:
        closes_to_t = closes_full[:t + 1]
        vols_to_t = volumes_full[:t + 1] if volumes_full is not None else None
        ts_at_t = timestamps_full[t] if timestamps_full is not None and t < len(timestamps_full) else None

        observation = None
        if features_full is not None and t >= window_size:
            try:
                obs_window = features_full[t - window_size:t].copy()
                obs_window[:, -4:] = 0.0
                observation = obs_window.flatten().astype(np.float32)
                if not np.isfinite(observation).all():
                    np.nan_to_num(observation, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
            except Exception:
                observation = None

        try:
            sigs = pipeline.alpha_factory.generate_all(
                closes=closes_to_t, volumes=vols_to_t,
                bar_idx=t, timestamp=ts_at_t, observation=observation,
            )
            all_signals[t] = {
                name: sig.mu for name, sig in sigs.items() if sig.is_active
            }
        except Exception:
            all_signals[t] = {}

    # Forward-fill: bars between samples inherit the last computed signal
    last_computed = {}
    for t in range(50, n_obs):
        if t in all_signals:
            last_computed = all_signals[t]
        else:
            all_signals[t] = last_computed

    # ---- Build signals matrix for vectorized fold evaluation ----
    n_alphas = len(alpha_names)
    alpha_idx = {name: i for i, name in enumerate(alpha_names)}
    signals_matrix = np.zeros((n_alphas, n_obs), dtype=np.float32)
    for t, sigs_dict in all_signals.items():
        for name, mu in sigs_dict.items():
            if name in alpha_idx:
                signals_matrix[alpha_idx[name], t] = mu

    # ---- Multi-Horizon Profiling (Task: WS1B) ----
    alpha_horizon_data = {name: {1: {'mus': [], 'rets': []},
                                  5: {'mus': [], 'rets': []},
                                  15: {'mus': [], 'rets': []}}
                           for name in alpha_names}

    # Vectorized cumulative returns at 5-bar and 15-bar horizons
    cum_rets_5bar = np.zeros(n_obs)
    cum_rets_15bar = np.zeros(n_obs)
    for t in range(50, n_obs - 5):
        cum_rets_5bar[t] = log_rets[t+1:t+6].sum()
    for t in range(50, n_obs - 15):
        cum_rets_15bar[t] = log_rets[t+1:t+16].sum()

    # ---- VECTORIZED fold evaluation (replaces nested Python loops) ----
    dataset_alpha_returns = {name: {'is': [], 'oos': []} for name in alpha_names}
    n_folds = len(folds)

    for fold in folds:
        train_end = fold.train_end
        test_start = fold.test_start
        test_end = min(fold.test_end, n_obs - 1)

        for i, alpha_name in enumerate(alpha_names):
            # IS returns: vectorized
            is_end = min(train_end, n_obs - 1)
            mus_is = signals_matrix[i, 50:is_end]
            rets_is = log_rets[51:is_end + 1]
            n_common = min(len(mus_is), len(rets_is))
            if n_common > 0:
                mus_is = mus_is[:n_common]
                rets_is = rets_is[:n_common]
                mask = np.abs(mus_is) > 1e-10
                dataset_alpha_returns[alpha_name]['is'].extend(
                    (mus_is[mask] * rets_is[mask]).tolist())

            # OOS returns: vectorized
            mus_oos = signals_matrix[i, test_start:test_end]
            rets_oos = log_rets[test_start + 1:test_end + 1]
            n_common = min(len(mus_oos), len(rets_oos))
            if n_common > 0:
                mus_oos = mus_oos[:n_common]
                rets_oos = rets_oos[:n_common]
                mask = np.abs(mus_oos) > 1e-10
                dataset_alpha_returns[alpha_name]['oos'].extend(
                    (mus_oos[mask] * rets_oos[mask]).tolist())

            # Multi-horizon profiling on OOS data: vectorized
            # 1-bar
            h1_mus = signals_matrix[i, test_start:test_end]
            h1_rets = log_rets[test_start + 1:test_end + 1]
            n_common = min(len(h1_mus), len(h1_rets))
            if n_common > 0:
                alpha_horizon_data[alpha_name][1]['mus'].extend(h1_mus[:n_common].tolist())
                alpha_horizon_data[alpha_name][1]['rets'].extend(h1_rets[:n_common].tolist())

            # 5-bar: non-overlapping (stride=5)
            h5_idx = np.arange(test_start, min(test_end, n_obs - 5), 5)
            if len(h5_idx) > 0:
                alpha_horizon_data[alpha_name][5]['mus'].extend(signals_matrix[i, h5_idx].tolist())
                alpha_horizon_data[alpha_name][5]['rets'].extend(cum_rets_5bar[h5_idx].tolist())

            # 15-bar: non-overlapping (stride=15)
            h15_idx = np.arange(test_start, min(test_end, n_obs - 15), 15)
            if len(h15_idx) > 0:
                alpha_horizon_data[alpha_name][15]['mus'].extend(signals_matrix[i, h15_idx].tolist())
                alpha_horizon_data[alpha_name][15]['rets'].extend(cum_rets_15bar[h15_idx].tolist())

    return {
        'alpha_returns': dataset_alpha_returns,
        'alpha_horizon_data': alpha_horizon_data,
        'n_folds': n_folds,
        'symbol': d.symbol,
    }


def validate_alphas_walkforward(datasets, pipeline, net, cfg, acfg, verbose=1):
    """
    Run purged walk-forward cross-validation on training data to validate
    that each alpha has statistically significant predictive power BEFORE
    the final out-of-sample evaluation.

    For each dataset:
      1. Use the TRAINING price series as the validation universe.
      2. Generate purged walk-forward folds (PurgedWalkForwardCV).
      3. On each fold's test window, run L1 alphas bar-by-bar and collect
         per-alpha mu predictions vs next-bar realized returns.
      4. Compute per-alpha "alpha return" = mu_predicted[t] * realized_return[t+1].
      5. Aggregate IS / OOS alpha-return series across folds.
      6. Evaluate each alpha through SignificanceGate (t-stat, deflated Sharpe,
         OOS/IS decay, PBO).
      7. Report results and optionally demote failing alphas.

    This bridges the gap between the validation_engine infrastructure
    (which was implemented but unwired) and the live pipeline.

    Args:
        datasets:   List of dataset objects from prepare_datasets()
        pipeline:   InstitutionalPipeline instance
        net:        Trained network (for RL alpha)
        cfg:        Base system Config
        acfg:       ArchitectureConfig
        verbose:    Print detail level

    Returns:
        Dict with per-alpha significance reports and aggregate summary.
    """
    if not HAS_VALIDATION:
        tprint("validation_engine not available -- skipping walk-forward CV", "warn")
        return {}

    sig_gate = SignificanceGate()
    mt_tracker = MultipleTestingTracker(method="holm_bonferroni")

    alpha_names = pipeline.alpha_factory.alpha_names
    n_alphas = len(alpha_names)

    # Accumulate per-alpha returns across all datasets and folds
    # Structure: {alpha_name: {'is': [...], 'oos': [...]}}
    alpha_returns_agg = {name: {'is': [], 'oos': []} for name in alpha_names}

    # Accumulate multi-horizon IC data across all datasets
    # Structure: {alpha_name: {horizon: {'mus': [...], 'rets': [...]}}}
    alpha_horizon_data = {name: {1: {'mus': [], 'rets': []},
                                  5: {'mus': [], 'rets': []},
                                  15: {'mus': [], 'rets': []}}
                           for name in alpha_names}

    # Purge gap and embargo (same as used in workers)
    purge_gap = max(21, acfg.meta_learner_retrain_freq // 10)
    embargo = 5

    n_folds_total = 0
    n_datasets_used = 0

    # ============================================================================
    # PARALLELIZED WALK-FORWARD VALIDATION
    # ============================================================================
    # Process datasets in parallel using ThreadPoolExecutor.
    # Each dataset is independent, so we can parallelize across CPU cores.
    # On a 20-core CPU, this gives ~10-15x speedup (250 datasets / 20 workers).
    # Sequential: 250 × 20s ≈ 83 minutes
    # Parallel:   (250/20) × 20s ≈ 4-5 minutes

    # Use ProcessPoolExecutor to bypass Python's GIL completely
    # Each worker process gets its own Python interpreter for true parallelism
    n_workers = min(max(1, multiprocessing.cpu_count() - 2), len(datasets))  # Leave 2 cores for system
    if verbose >= 1:
        tprint(f"Parallelizing walk-forward validation across {n_workers} workers ({len(datasets)} datasets)", "info")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all dataset processing tasks
        # Pack args as tuple to avoid pickle issues with pipeline/net objects
        task_args = [(d, acfg, alpha_names, verbose) for d in datasets]
        futures = {
            executor.submit(_process_single_dataset_validation, args): args[0]
            for args in task_args
        }

        # Collect results as they complete
        cv_start_time = time.time()
        n_completed = 0
        n_total = len(futures)
        for future in as_completed(futures):
            n_completed += 1
            if verbose >= 1 and (n_completed % 10 == 0 or n_completed == n_total):
                progress_bar(n_completed, n_total, "WF-CV", start_time=cv_start_time)
            try:
                result = future.result(timeout=300)
            except Exception as e:
                dataset = futures[future]
                tprint(f"Worker failed on {getattr(dataset, 'symbol', '?')}: {e}", "warn")
                continue
            if result is None:
                continue  # Dataset skipped (insufficient data)

            n_datasets_used += 1
            n_folds_total += result['n_folds']

            # Aggregate alpha returns from this dataset
            for alpha_name in alpha_names:
                alpha_returns_agg[alpha_name]['is'].extend(result['alpha_returns'][alpha_name]['is'])
                alpha_returns_agg[alpha_name]['oos'].extend(result['alpha_returns'][alpha_name]['oos'])

            # Aggregate multi-horizon IC data from this dataset
            if 'alpha_horizon_data' in result:
                for alpha_name in alpha_names:
                    for horizon in [1, 5, 15]:
                        alpha_horizon_data[alpha_name][horizon]['mus'].extend(
                            result['alpha_horizon_data'][alpha_name][horizon]['mus']
                        )
                        alpha_horizon_data[alpha_name][horizon]['rets'].extend(
                            result['alpha_horizon_data'][alpha_name][horizon]['rets']
                        )

    if verbose >= 1 and n_datasets_used > 0:
        folds_per = n_folds_total // max(n_datasets_used, 1)
        tprint(f"CV complete: {n_datasets_used} datasets, {n_folds_total} folds "
               f"({folds_per} folds/dataset avg)", "ok")

    # ---- Significance evaluation per alpha ----
    reports = {}
    n_configs_tested = max(n_alphas, 1)  # Each alpha is one hypothesis

    for alpha_name in alpha_names:
        is_rets = np.array(alpha_returns_agg[alpha_name]['is'], dtype=np.float64)
        oos_rets = np.array(alpha_returns_agg[alpha_name]['oos'], dtype=np.float64)

        if len(is_rets) < 30 or len(oos_rets) < 30:
            reports[alpha_name] = {
                'verdict': 'INSUFFICIENT_DATA',
                'n_is': len(is_rets),
                'n_oos': len(oos_rets),
            }
            continue

        try:
            report = sig_gate.evaluate(
                alpha_name=alpha_name,
                is_returns=is_rets,
                oos_returns=oos_rets,
                n_configs_tested=n_configs_tested,
            )
            reports[alpha_name] = {
                'verdict': report.verdict,
                't_stat': report.t_stat,
                'sharpe_is': report.sharpe_is,
                'sharpe_oos': report.sharpe_oos,
                'deflated_sharpe': report.deflated_sharpe,
                'oos_is_decay': report.oos_is_decay_ratio,
                'pbo': report.pbo,
                'n_is': report.n_obs_is,
                'n_oos': report.n_obs_oos,
                'adjusted_p': report.adjusted_p_value,
            }

            # Register with multiple testing tracker
            mt_tracker.register_test(
                name=alpha_name,
                p_value=report.adjusted_p_value,
                context={'t_stat': report.t_stat, 'verdict': report.verdict},
            )
        except Exception as e:
            reports[alpha_name] = {
                'verdict': 'ERROR',
                'error': str(e),
                'n_is': len(is_rets),
                'n_oos': len(oos_rets),
            }

    # ---- Multiple testing correction ----
    adjusted_pvals = mt_tracker.adjusted_p_values()
    for alpha_name in alpha_names:
        if alpha_name in adjusted_pvals and alpha_name in reports:
            reports[alpha_name]['holm_bonf_p'] = adjusted_pvals[alpha_name]

    # ---- Multi-Horizon IC Profiling (Task: WS1B) ----
    # Compute Information Coefficient (IC = correlation between mu and realized return)
    # at 1-bar, 5-bar, and 15-bar horizons for each alpha.
    horizon_ics = {}  # {alpha_name: {horizon: {'ic': float, 'n': int, 't_stat': float, 'p_val': float}}}
    horizon_mt_tracker = MultipleTestingTracker(method="holm_bonferroni")

    for alpha_name in alpha_names:
        horizon_ics[alpha_name] = {}

        for horizon in [1, 5, 15]:
            mus = np.array(alpha_horizon_data[alpha_name][horizon]['mus'])
            rets = np.array(alpha_horizon_data[alpha_name][horizon]['rets'])

            if len(mus) < 10:
                horizon_ics[alpha_name][horizon] = {
                    'ic': 0.0, 'n': len(mus), 't_stat': 0.0, 'p_val': 1.0
                }
                continue

            # Information Coefficient = Pearson correlation(mu, realized_return)
            if np.std(mus) > 1e-10 and np.std(rets) > 1e-10:
                ic = np.corrcoef(mus, rets)[0, 1]
                n = len(mus)

                # t-statistic for correlation: t = IC * sqrt((n-2) / (1 - IC^2))
                if abs(ic) < 0.9999:
                    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                    # Two-tailed p-value
                    try:
                        from scipy import stats
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
                    except ImportError:
                        # Fallback: use normal approximation for large n
                        p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / np.sqrt(2))))
                else:
                    t_stat = 100.0 if ic > 0 else -100.0
                    p_val = 0.0

                horizon_ics[alpha_name][horizon] = {
                    'ic': ic, 'n': n, 't_stat': t_stat, 'p_val': p_val
                }

                # Register with multiple testing tracker
                test_name = f"{alpha_name}_{horizon}bar"
                horizon_mt_tracker.register_test(
                    name=test_name,
                    p_value=p_val,
                    context={'ic': ic, 't_stat': t_stat, 'horizon': horizon}
                )
            else:
                horizon_ics[alpha_name][horizon] = {
                    'ic': 0.0, 'n': len(mus), 't_stat': 0.0, 'p_val': 1.0
                }

    # Apply Holm-Bonferroni correction across all 21 tests (7 alphas × 3 horizons)
    horizon_adjusted_pvals = horizon_mt_tracker.adjusted_p_values()

    # ---- Horizon-Aware Demotion Logic (P1: Priority Implementation) ----
    # Override verdicts: evaluate each alpha at its native horizon, not just 1-bar.
    # This prevents unfairly rejecting long-horizon alphas based on noisy 1-bar IC.

    def get_native_horizon(alpha_name: str, acfg) -> int:
        """Map alpha name to its native horizon from config."""
        horizon_map = {
            'rl_ppo_mcts': acfg.rl_alpha_horizon,           # 5
            'trend_follow': acfg.trend_horizon,             # 15
            'mean_reversion': acfg.mr_horizon,              # 5 (after P0 fix)
            'value': acfg.value_horizon,                    # 15
            'carry': acfg.carry_horizon,                    # 15
            'seasonality': acfg.seasonality_horizon,        # 15
            'calendar': acfg.calendar_horizon,              # 5 (after P0 fix)
            'vol_premium': acfg.vol_premium_horizon,        # 15
            'amihud_liquidity': acfg.amihud_horizon,        # 15
            'hurst_regime': acfg.hurst_horizon,             # 15
            'short_term_reversal': acfg.reversal_horizon,   # 5
            'vol_term_structure': 15,                       # NEW: Vol term structure
            'volume_price_divergence': 10,                  # NEW: Volume-price divergence
        }
        return horizon_map.get(alpha_name, 1)

    def evaluate_at_native_horizon(alpha_name: str, horizon_ics: dict,
                                     acfg, available_horizons: list) -> dict:
        """
        Evaluate alpha at its native horizon ± adjacent horizons.
        Returns best IC stats across the 3 nearest horizons to native.
        """
        native_h = get_native_horizon(alpha_name, acfg)

        # Find 3 nearest profiling horizons to native horizon
        sorted_horizons = sorted(available_horizons, key=lambda h: abs(h - native_h))
        test_horizons = sorted_horizons[:min(3, len(sorted_horizons))]

        best_ic = 0.0
        best_t_stat = 0.0
        best_horizon = native_h
        best_n = 0
        best_abs_ic = 0.0

        alpha_ics = horizon_ics.get(alpha_name, {})
        for h in test_horizons:
            ic_data = alpha_ics.get(h, {})
            ic = ic_data.get('ic', 0.0)
            t_stat = ic_data.get('t_stat', 0.0)
            n = ic_data.get('n', 0)

            # Select based on ABSOLUTE IC (negative IC is as valuable as positive)
            if abs(ic) > best_abs_ic:
                best_ic = ic
                best_abs_ic = abs(ic)
                best_t_stat = t_stat
                best_horizon = h
                best_n = n

        return {
            'best_ic': best_ic,
            'best_t_stat': best_t_stat,
            'best_horizon': best_horizon,
            'best_n': best_n,
            'native_horizon': native_h,
            'tested_horizons': test_horizons,
        }

    # Override verdicts based on horizon-aware evaluation
    horizon_overrides = {}
    for alpha_name in alpha_names:
        horizon_eval = evaluate_at_native_horizon(
            alpha_name, horizon_ics, acfg, [1, 5, 15]
        )

        best_ic = horizon_eval['best_ic']
        best_t = horizon_eval['best_t_stat']
        best_h = horizon_eval['best_horizon']

        # FIX #5: Negative IC is just as valuable as positive IC
        # A signal that reliably predicts the opposite direction is informative
        # What's useless is IC ≈ 0.0 (no predictive relationship)
        #
        # New verdict logic based on |IC| (absolute value):
        # - |IC| > threshold AND t-stat significant → PASS (flip sign if IC < 0)
        # - |IC| ≈ 0 → REJECT (no information)
        #
        # Thresholds: |IC| > 0.03 AND |t| > 2.5 for PASS
        #            |IC| > 0.015 AND |t| > 1.5 for MARGINAL

        flip_suggestion = False
        if abs(best_ic) > 0.03 and abs(best_t) > 2.5:
            new_verdict = 'PASS'
            if best_ic < 0:
                flip_suggestion = True  # Strong negative IC → suggest signal flip
        elif abs(best_ic) > 0.015 and abs(best_t) > 1.5:
            new_verdict = 'MARGINAL'
            if best_ic < 0:
                flip_suggestion = True  # Moderate negative IC → suggest signal flip
        else:
            new_verdict = 'REJECT'  # |IC| ≈ 0 or not statistically significant

        old_verdict = reports.get(alpha_name, {}).get('verdict', '?')

        # Store horizon evaluation metadata
        horizon_overrides[alpha_name] = {
            'old_verdict': old_verdict,
            'new_verdict': new_verdict,
            'best_ic': best_ic,
            'best_t_stat': best_t,
            'best_horizon': best_h,
            'native_horizon': horizon_eval['native_horizon'],
            'changed': (old_verdict != new_verdict),
            'flip_suggestion': flip_suggestion,  # Flag for negative IC
        }

        # Override the verdict in reports
        if alpha_name in reports:
            reports[alpha_name]['verdict'] = new_verdict
            reports[alpha_name]['horizon_aware'] = True
            reports[alpha_name]['best_ic_horizon'] = best_h
            reports[alpha_name]['best_ic'] = best_ic

    # ---- Print results ----
    if verbose >= 1:
        print_subsection("Alpha Significance")
        print(f"  {C.DIM}Testing each strategy on data it wasn't trained on (like a blind taste test).{C.RESET}")
        print(f"  {C.DIM}If a strategy only works on data it's seen before, it's memorizing, not learning.{C.RESET}")
        print(f"  Datasets: {n_datasets_used}  |  "
              f"CV folds: {n_folds_total}  |  "
              f"Purge gap: {purge_gap} bars  |  "
              f"Embargo: {embargo} bars")
        print()

        # Build table data
        if HAS_TABLE_FORMATTER:
            table = TableFormatter(title="ALPHA VALIDATION RESULTS")
            table.add_column('Alpha', align='left')
            table.add_column('Verdict', align='center')
            table.add_column('t-stat', align='right', format_spec='+.2f')
            table.add_column('Sh(IS)', align='right', format_spec='+.2f')
            table.add_column('Sh(OOS)', align='right', format_spec='+.3f')
            table.add_column('Decay', align='right', format_spec='.2f')
            table.add_column('DSR', align='right', format_spec='.2f')
            table.add_column('PBO', align='right', format_spec='.2f')
            table.add_column('n_IS', align='right')
            table.add_column('n_OOS', align='right')

            n_pass = n_marginal = n_reject = n_other = 0
            for alpha_name in alpha_names:
                r = reports.get(alpha_name, {})
                verdict = r.get('verdict', '?')
                if verdict == 'PASS':
                    n_pass += 1
                elif verdict == 'MARGINAL':
                    n_marginal += 1
                elif verdict == 'REJECT':
                    n_reject += 1
                else:
                    n_other += 1

                table.add_row([
                    alpha_name,
                    verdict,
                    r.get('t_stat', 0.0),
                    r.get('sharpe_is', 0.0),
                    r.get('sharpe_oos', 0.0),
                    r.get('oos_is_decay', 0.0),
                    r.get('deflated_sharpe', 0.0),
                    r.get('pbo', 0.0),
                    r.get('n_is', 0),
                    r.get('n_oos', 0),
                ])

            rendered = "  " + table.render().replace("\n", "\n  ")
            try:
                print(rendered)
            except UnicodeEncodeError:
                print(rendered.encode('ascii', errors='replace').decode('ascii'))
            print(f"  {C.DIM}Column guide: Alpha=strategy name | Verdict=pass/fail grade | "
                  f"t-stat=statistical confidence (higher=more reliable){C.RESET}")
            print(f"  {C.DIM}Sh(IS)=performance on training data | Sh(OOS)=performance on unseen data "
                  f"(this is the real test){C.RESET}")
            print(f"  {C.DIM}Decay=how much performance drops on new data (lower=better) | "
                  f"DSR=deflated Sharpe (adjusted for luck){C.RESET}")
            print(f"  {C.DIM}PBO=probability it's overfitted (lower=better, like odds of a fluke) | "
                  f"n=sample size{C.RESET}")
        else:
            # Fallback to old format if table_formatter not available
            print(f"  {'Alpha':<26s}  {'Verdict':>10s} {'t-stat':>7s}  "
                  f"{'Sh(IS)':>7s} {'Sh(OOS)':>8s} {'Decay':>6s}  "
                  f"{'DSR':>5s} {'PBO':>5s}  {'n_IS':>6s} {'n_OOS':>6s}")
            print(f"  {'-'*26}  {'-'*10} {'-'*7}  {'-'*7} {'-'*8} {'-'*6}  "
                  f"{'-'*5} {'-'*5}  {'-'*6} {'-'*6}")

            n_pass = n_marginal = n_reject = n_other = 0
            for alpha_name in alpha_names:
                r = reports.get(alpha_name, {})
                verdict = r.get('verdict', '?')
                if verdict == 'PASS':
                    badge = f"{C.GREEN}{verdict:>10s}{C.RESET}"
                    n_pass += 1
                elif verdict == 'MARGINAL':
                    badge = f"{C.YELLOW}{verdict:>10s}{C.RESET}"
                    n_marginal += 1
                elif verdict == 'REJECT':
                    badge = f"{C.RED}{verdict:>10s}{C.RESET}"
                    n_reject += 1
                else:
                    badge = f"{verdict:>10s}"
                    n_other += 1

                t_stat = r.get('t_stat', 0.0)
                sh_is = r.get('sharpe_is', 0.0)
                sh_oos = r.get('sharpe_oos', 0.0)
                decay = r.get('oos_is_decay', 0.0)
                dsr = r.get('deflated_sharpe', 0.0)
                pbo = r.get('pbo', 0.0)
                n_is = r.get('n_is', 0)
                n_oos = r.get('n_oos', 0)

                print(f"  {alpha_name:<26s}  {badge} {t_stat:>+6.2f}  "
                      f"{sh_is:>+6.2f} {sh_oos:>+7.3f} {decay:>5.2f}  "
                      f"{dsr:>5.2f} {pbo:>5.2f}  {n_is:>6d} {n_oos:>6d}")

        # Summary
        mt_summary = mt_tracker.summary()
        print(f"\n  {C.BOLD}Summary:{C.RESET}  "
              f"PASS={n_pass}  MARGINAL={n_marginal}  "
              f"REJECT={n_reject}  OTHER={n_other}")
        print(f"  Multiple testing: {mt_summary['total_tests']} hypotheses tested "
              f"({mt_summary['method']})")
        print(f"  Significant at 0.05 (raw): {mt_summary['significant_at_005_raw']}  "
              f"(adjusted): {mt_summary['significant_at_005_adjusted']}")

        # Horizon-Aware Verdict Changes Summary
        changed = [name for name in alpha_names if horizon_overrides.get(name, {}).get('changed', False)]
        if changed:
            print_subsection("Horizon-Aware Adjustments")
            for name in changed:
                hdata = horizon_overrides[name]
                old_v = hdata['old_verdict']
                new_v = hdata['new_verdict']
                best_h = hdata['best_horizon']
                native_h = hdata['native_horizon']
                best_ic = hdata['best_ic']
                best_t = hdata['best_t_stat']

                # Color code the change
                if new_v == 'PASS':
                    verdict_color = C.GREEN
                elif new_v == 'MARGINAL':
                    verdict_color = C.YELLOW
                else:
                    verdict_color = C.RED

                change_symbol = "UP" if (new_v == 'PASS' or (new_v == 'MARGINAL' and old_v == 'REJECT')) else "DN"

                # FIX #5: Flag negative IC alphas for signal flip
                flip_flag = " [FLIP SIGNAL]" if hdata.get('flip_suggestion', False) else ""

                print(f"    {name:<26s}: {old_v:>10s} -> {verdict_color}{new_v:>10s}{C.RESET} {change_symbol}  "
                      f"(IC={best_ic:+.3f} t={best_t:+.2f} @ H={best_h}, native H={native_h}){flip_flag}")

            # Summary stats
            rescued_count = sum(1 for h in horizon_overrides.values() if h['old_verdict']=='REJECT' and h['new_verdict']!='REJECT')
            flip_count = sum(1 for h in horizon_overrides.values() if h.get('flip_suggestion', False))

            print(f"  Rescued alphas: {rescued_count}")
            if flip_count > 0:
                flip_names = [name for name, h in horizon_overrides.items() if h.get('flip_suggestion', False)]
                print(f"  {C.YELLOW}Negative IC detected ({flip_count} alphas): {', '.join(flip_names)}{C.RESET}")
                print(f"  {C.YELLOW}-> Consider inverting these signals (negative IC = predictive, just inverted){C.RESET}")
            print(f"  Evaluated at native horizon ± adjacent horizons")

        # Warn about rejected alphas but do NOT auto-disable
        # (spec: "no automatic re-entry without human review" applies to
        #  alpha retirement too -- flag for review, don't silently remove)
        if n_reject > 0:
            print_subsection("Demotion Actions")
            rejected = [n for n in alpha_names
                        if reports.get(n, {}).get('verdict') == 'REJECT']
            tprint(f"  Flagged for review: {', '.join(rejected)}", "warn")
            tprint(f"  These alphas failed significance (t<3.0 or OOS decay>50%). "
                   f"Consider disabling via --disable-alphas", "warn")

            # FIX §SA-2: Auto-demote (not fully disable) rejected alphas.
            # Rejected alphas with negative/noisy mu pull ensemble mu_hat
            # toward zero, contributing to the signal attenuation problem.
            # Demotion to 5% confidence preserves the "flag for review"
            # principle while preventing ensemble pollution.
            if hasattr(pipeline, '_ensemble') and pipeline._ensemble is not None:
                ml = pipeline._ensemble.meta_learner
                for name in rejected:
                    if name in ml._alpha_demotion:
                        ml._alpha_demotion[name] = 0.05  # 95% confidence reduction
                tprint(f"  Auto-demoted rejected alphas to 5% confidence: "
                       f"{', '.join(rejected)}", "warn")

    # ---- Compute ICIR, Hit Rate, and Signal Persistence (A3/A4/A5) ----
    # These metrics complement the point-estimate IC with stability and directionality.
    icir_window = 63  # ~3 months rolling window for ICIR
    for alpha_name in alpha_names:
        mus_1bar = np.array(alpha_horizon_data[alpha_name][1]['mus'])
        rets_1bar = np.array(alpha_horizon_data[alpha_name][1]['rets'])

        # A3: ICIR = mean(rolling IC) / std(rolling IC) at H=1 only
        # ICIR measures IC consistency -- an alpha with IC=+0.03 every quarter
        # is far better than one with IC=+0.12 then -0.06 alternating.
        if len(mus_1bar) >= icir_window * 2:
            n_windows = len(mus_1bar) // icir_window
            rolling_ics = []
            for w in range(n_windows):
                start = w * icir_window
                end = start + icir_window
                w_mus = mus_1bar[start:end]
                w_rets = rets_1bar[start:end]
                if np.std(w_mus) > 1e-10 and np.std(w_rets) > 1e-10:
                    wic = float(np.corrcoef(w_mus, w_rets)[0, 1])
                    if np.isfinite(wic):
                        rolling_ics.append(wic)
            if len(rolling_ics) >= 2:
                ic_mean = float(np.mean(rolling_ics))
                ic_std = float(np.std(rolling_ics))
                icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
                reports[alpha_name]['icir'] = float(icir)
                reports[alpha_name]['icir_mean'] = ic_mean
                reports[alpha_name]['icir_std'] = ic_std
                reports[alpha_name]['icir_n_windows'] = len(rolling_ics)
            else:
                reports[alpha_name]['icir'] = None
        else:
            reports[alpha_name]['icir'] = None

        # A4: Hit Rate = fraction of active bars where sign(mu) == sign(return)
        # More intuitive than IC for practitioners: "how often is this alpha right?"
        active_mask = np.abs(mus_1bar) > 1e-10
        n_active = int(active_mask.sum())
        if n_active >= 30:
            active_mus = mus_1bar[active_mask]
            active_rets = rets_1bar[active_mask]
            hits = int(np.sum(np.sign(active_mus) == np.sign(active_rets)))
            reports[alpha_name]['hit_rate'] = float(hits / n_active)
            reports[alpha_name]['hit_rate_n'] = n_active
        else:
            reports[alpha_name]['hit_rate'] = None

        # A5: Signal Persistence = avg consecutive bars with same signal direction
        # Low persistence + high no_trade_threshold = alpha changes direction but
        # can't clear the threshold to trade, resulting in trade suppression.
        if n_active >= 30:
            signs = np.sign(mus_1bar[active_mask])
            run_lengths = []
            current_run = 1
            for i in range(1, len(signs)):
                if signs[i] == signs[i - 1]:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1
            run_lengths.append(current_run)
            reports[alpha_name]['persistence'] = float(np.mean(run_lengths))
            reports[alpha_name]['persistence_median'] = float(np.median(run_lengths))
            reports[alpha_name]['persistence_n_runs'] = len(run_lengths)
        else:
            reports[alpha_name]['persistence'] = None

    # ---- Print Multi-Horizon IC Profile ----
    if verbose >= 1:
        print_subsection("Multi-Horizon IC Profiling")
        print(f"  Information Coefficient (IC) at 1-bar, 5-bar, and 15-bar horizons.")
        print(f"  IC = correlation(mu, realized_return). Higher |IC| = stronger predictive skill.")
        print(f"  Non-overlapping windows for 5-bar and 15-bar. Holm-Bonferroni correction across 21 tests.")
        print()

        if HAS_TABLE_FORMATTER:
            # Create table with horizon groups
            table = TableFormatter(title="MULTI-HORIZON IC PROFILING")
            table.add_column('Alpha', align='left')
            table.add_column('1-bar', align='center')
            table.add_column('5-bar', align='center')
            table.add_column('15-bar', align='center')

            # Add sub-header row (will be plain text spanning the column)
            table.add_row(['', 'IC     t     n', 'IC     t     n', 'IC     t     n'])
        else:
            print(f"  {'Alpha':<26s}  {'1-bar':>18s}  {'5-bar':>18s}  {'15-bar':>18s}")
            print(f"  {'':<22s}  {'IC':>6s} {'t':>5s} {'n':>5s}  {'IC':>6s} {'t':>5s} {'n':>5s}  {'IC':>6s} {'t':>5s} {'n':>5s}")
            print(f"  {'-'*26}  {'-'*18}  {'-'*18}  {'-'*18}")

        for alpha_name in alpha_names:
            h1 = horizon_ics[alpha_name][1]
            h5 = horizon_ics[alpha_name][5]
            h15 = horizon_ics[alpha_name][15]

            # Color-code IC by magnitude (|IC| > 0.05 is noteworthy for daily data)
            def format_ic(ic, t_stat, p_val, horizon):
                test_name = f"{alpha_name}_{horizon}bar"
                adj_p = horizon_adjusted_pvals.get(test_name, 1.0)

                # Format IC value first (fixed width: 6 chars for "+0.123")
                ic_plain = f"{ic:+6.3f}"

                # Significant after correction (p < 0.05)?
                if adj_p < 0.05 and abs(ic) > 0.02:
                    color = C.GREEN if ic > 0 else C.RED
                    return f"{color}{ic_plain}{C.RESET}"
                elif abs(ic) > 0.05:
                    return f"{C.YELLOW}{ic_plain}{C.RESET}"
                else:
                    return ic_plain

            ic1_str = format_ic(h1['ic'], h1['t_stat'], h1['p_val'], 1)
            ic5_str = format_ic(h5['ic'], h5['t_stat'], h5['p_val'], 5)
            ic15_str = format_ic(h15['ic'], h15['t_stat'], h15['p_val'], 15)

            if HAS_TABLE_FORMATTER:
                # Format as single strings for each horizon group
                h1_str = f"{ic1_str} {h1['t_stat']:>+5.1f} {h1['n']:>5d}"
                h5_str = f"{ic5_str} {h5['t_stat']:>+5.1f} {h5['n']:>5d}"
                h15_str = f"{ic15_str} {h15['t_stat']:>+5.1f} {h15['n']:>5d}"
                table.add_row([alpha_name, h1_str, h5_str, h15_str])
            else:
                # Print with fixed column widths and visual separators (IC strings already contain color codes)
                print(f"  {alpha_name:<26s}  "
                      f"{ic1_str} {h1['t_stat']:>+5.1f} {h1['n']:>5d}  "
                      f"{ic5_str} {h5['t_stat']:>+5.1f} {h5['n']:>5d}  "
                      f"{ic15_str} {h15['t_stat']:>+5.1f} {h15['n']:>5d}")

        if HAS_TABLE_FORMATTER:
            rendered = "  " + table.render().replace("\n", "\n  ")
            try:
                print(rendered)
            except UnicodeEncodeError:
                print(rendered.encode('ascii', errors='replace').decode('ascii'))

        horizon_summary = horizon_mt_tracker.summary()
        print(f"\n  {C.BOLD}Multi-Horizon Summary:{C.RESET}")
        print(f"  Total hypotheses tested: {horizon_summary['total_tests']} (7 alphas x 3 horizons)")
        print(f"  Significant at alpha=0.05 (raw): {horizon_summary['significant_at_005_raw']}  "
              f"(Holm-Bonferroni adjusted): {horizon_summary['significant_at_005_adjusted']}")
        print(f"  Method: {horizon_summary['method']}")
        print()
        print(f"  {C.BOLD}Interpretation:{C.RESET}")
        print(f"  • Green IC: Statistically significant after correction (p < 0.05)")
        print(f"  • Yellow IC: |IC| > 0.05 but not significant after correction")
        print(f"  • Plain IC: Weak or no signal")
        print(f"  • Positive IC: Alpha correctly predicts direction (up = up, down = down)")
        print(f"  • Negative IC: Alpha anti-predicts (consider inverting or disabling)")
        print()

        # ---- Print Alpha Quality Metrics (ICIR, Hit Rate, Persistence) ----
        print_subsection("Alpha Quality Metrics")
        print(f"  ICIR = IC consistency (mean/std of rolling 63-bar IC). |ICIR| > 0.5 = good, > 1.0 = excellent.")
        print(f"  Hit Rate = directional accuracy. 50% = coin flip. >53% with n>500 is meaningful.")
        print(f"  Persistence = avg bars holding same direction. Low + high no_trade_threshold = suppression.")
        print()

        if HAS_TABLE_FORMATTER:
            table = TableFormatter(title="ALPHA QUALITY METRICS")
            table.add_column('Alpha', align='left')
            table.add_column('ICIR', align='right')
            table.add_column('IC_m', align='right')
            table.add_column('IC_s', align='right')
            table.add_column('Win', align='right')
            table.add_column('HitRt', align='right')
            table.add_column('n', align='right')
            table.add_column('Persist', align='right')
            table.add_column('MedP', align='right')
        else:
            print(f"  {'Alpha':<26s}  {'ICIR':>7s} {'IC_m':>7s} {'IC_s':>7s}  {'Win':>6s} {'HitRt':>6s} {'n':>5s}  {'Persist':>7s} {'MedP':>5s}")
            print(f"  {'-'*26}  {'-'*7} {'-'*7} {'-'*7}  {'-'*6} {'-'*6} {'-'*5}  {'-'*7} {'-'*5}")

        for alpha_name in alpha_names:
            r = reports.get(alpha_name, {})
            icir = r.get('icir')
            icir_mean = r.get('icir_mean')
            icir_std = r.get('icir_std')
            hit_rate = r.get('hit_rate')
            hit_n = r.get('hit_rate_n')
            persist = r.get('persistence')
            persist_med = r.get('persistence_median')
            n_windows = r.get('icir_n_windows', 0)

            # Format ICIR with color
            if icir is not None:
                if abs(icir) >= 1.0:
                    icir_str = f"{C.GREEN}{icir:+6.2f}{C.RESET}"
                elif abs(icir) >= 0.5:
                    icir_str = f"{C.YELLOW}{icir:+6.2f}{C.RESET}"
                else:
                    icir_str = f"{icir:+6.2f}"
                icir_m_str = f"{icir_mean:+6.3f}"
                icir_s_str = f"{icir_std:6.3f}"
            else:
                icir_str = "   N/A"
                icir_m_str = "    N/A"
                icir_s_str = "    N/A"

            # Format Hit Rate with color
            if hit_rate is not None:
                n_wins = int(hit_rate * hit_n)
                if hit_rate >= 0.53:
                    hr_str = f"{C.GREEN}{hit_rate:5.1%}{C.RESET}"
                elif hit_rate <= 0.47:
                    hr_str = f"{C.RED}{hit_rate:5.1%}{C.RESET}"
                else:
                    hr_str = f"{hit_rate:5.1%}"
                win_str = f"{n_wins:>5d}"
                n_str = f"{hit_n:>5d}"
            else:
                hr_str = "   N/A"
                win_str = "   N/A"
                n_str = "   N/A"

            # Format Persistence
            if persist is not None:
                if persist >= 10:
                    p_str = f"{C.GREEN}{persist:6.1f}{C.RESET}"
                elif persist <= 3:
                    p_str = f"{C.RED}{persist:6.1f}{C.RESET}"
                else:
                    p_str = f"{persist:6.1f}"
                pm_str = f"{persist_med:4.0f}"
            else:
                p_str = "   N/A"
                pm_str = "  N/A"

            if HAS_TABLE_FORMATTER:
                table.add_row([alpha_name, icir_str, icir_m_str, icir_s_str, win_str, hr_str, n_str, p_str, pm_str])
            else:
                print(f"  {alpha_name:<26s}  {icir_str} {icir_m_str} {icir_s_str}  {win_str} {hr_str} {n_str}  {p_str} {pm_str}")

        if HAS_TABLE_FORMATTER:
            rendered = "  " + table.render().replace("\n", "\n  ")
            try:
                print(rendered)
            except UnicodeEncodeError:
                print(rendered.encode('ascii', errors='replace').decode('ascii'))
            print(f"  {C.DIM}Column guide:{C.RESET}")
            print(f"  {C.DIM}  - ICIR   : How reliably the model makes good predictions, day after day.{C.RESET}")
            print(f"  {C.DIM}             Think of it like a weather forecaster's track record — not just{C.RESET}")
            print(f"  {C.DIM}             \"are they right sometimes?\" but \"can you count on them daily?\"{C.RESET}")
            print(f"  {C.DIM}             > 0.5 = dependable forecaster  |  < 0.2 = unreliable, skip it{C.RESET}")
            print(f"  {C.DIM}  - IC_m   : The average quality of predictions (higher = better).{C.RESET}")
            print(f"  {C.DIM}             Like a student's average test score across the whole semester.{C.RESET}")
            print(f"  {C.DIM}             e.g. IC_m=0.05 means the model has a small but real skill at{C.RESET}")
            print(f"  {C.DIM}             picking winners over losers — even 0.03 can be profitable{C.RESET}")
            print(f"  {C.DIM}  - IC_s   : How wildly the prediction quality swings from day to day.{C.RESET}")
            print(f"  {C.DIM}             Like comparing two archers — both average 8/10, but one scores{C.RESET}")
            print(f"  {C.DIM}             7-9 every time (low IC_s, reliable) while the other scores{C.RESET}")
            print(f"  {C.DIM}             3 one day and 10 the next (high IC_s, erratic). Lower = better{C.RESET}")
            print(f"  {C.DIM}  - Win    : Count of days where the model correctly guessed the direction.{C.RESET}")
            print(f"  {C.DIM}             e.g. Win=180 means out of ~252 trading days, it got the{C.RESET}")
            print(f"  {C.DIM}             \"will it go up or down?\" question right on 180 of them{C.RESET}")
            print(f"  {C.DIM}  - HitRt  : Win expressed as a percentage — the real scorecard.{C.RESET}")
            print(f"  {C.DIM}             Like free-throw shooting: >53%% = you have an edge over the{C.RESET}")
            print(f"  {C.DIM}             market, >58%% = strong skill, <50%% = you'd do better flipping a coin{C.RESET}")
            print(f"  {C.DIM}  - Persist : How many bars into the future a signal stays useful.{C.RESET}")
            print(f"  {C.DIM}             Like milk's expiry date — Persist=1 means the signal is stale{C.RESET}")
            print(f"  {C.DIM}             by tomorrow; Persist=5 means it's still fresh 5 bars later{C.RESET}")
        else:
            pass  # Already printed in loop above

        print()
        print(f"  {C.BOLD}Quality Metric Notes:{C.RESET}")
        print(f"  • ICIR computed at H=1 only (H>1 has overlapping-return bias)")
        print(f"  • Hit Rate excludes inactive bars (mu=0). n = active bar count")
        print(f"  • Persistence < 5 bars with 2% no-trade threshold causes excessive suppression")
        print()

    return {
        'per_alpha': reports,
        'multiple_testing': mt_tracker.summary(),
        'horizon_ics': horizon_ics,
        'horizon_adjusted_pvals': horizon_adjusted_pvals,
        'n_folds': n_folds_total,
        'n_datasets': n_datasets_used,
    }

def parse_args():
    p = argparse.ArgumentParser(
        description="Alpha-Trade v6.0 -- Institutional Architecture Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Daily timeframe (default, recommended)
  python alphago_layering.py --synthetic
  python alphago_layering.py --symbols AAPL,MSFT,NVDA --iterations 3

  # Multiple timeframes (slower, for multi-TF analysis)
  python alphago_layering.py --symbols AAPL,MSFT --timeframes 1h,1d
  python alphago_layering.py --synthetic --timeframes 5m,15m,30m,1h,1d

  # Other options
  python alphago_layering.py --synthetic --eval-only
  python alphago_layering.py --data-dir ./my_csv_data
  python alphago_layering.py --synthetic --disable-alphas carry,seasonality
        """)

    # Data source
    g = p.add_argument_group("Data Source")
    g.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic market data (fastest)")
    g.add_argument("--n-synthetic", type=int, default=20,
                   help="Number of synthetic symbols (default: 20)")
    g.add_argument("--symbols", type=str, default="",
                   help="Comma-separated symbols (e.g. AAPL,MSFT,NVDA)")
    g.add_argument("--sector", type=str, default="",
                   help="Comma-separated sectors (e.g. technology,financials). "
                        "Use --sector list to show available sectors.")
    g.add_argument("--n-symbols", type=int, default=50)
    g.add_argument("--data-dir", type=str, default="",
                   help="Load from a directory of CSV/Parquet files")
    g.add_argument("--norgate-dir", type=str, default="",
                   help="Override Norgate data path (default: D:\\Experiments\\norgate_data)")
    g.add_argument("--norgate-db", type=str, default="",
                   help="Comma-separated Norgate databases (e.g. US_Equities,US_Equities_Delisted)")
    g.add_argument("--yahoo", action="store_true",
                   help="Use Yahoo Finance instead of Norgate (survivorship-biased)")

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--iterations", type=int, default=3)        # v7.0 optimal: 3 iterations
    g.add_argument("--steps-per-iter", type=int, default=50_000)  # v7.0 optimal: 50k/iter (150k total)
    g.add_argument("--capital", type=float, default=10_000)
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--batch-size", type=int, default=2048)  # GPU-optimized (Config default=4096)
    g.add_argument("--n-envs", type=int, default=64)
    g.add_argument("--mcts-rollouts", type=int, default=32)
    g.add_argument("--n-epochs", type=int, default=4,
                   help="PPO epochs per update (default: 4, target_kl guard prevents over-optimization)")
    g.add_argument("--timeframes", type=str, default="1d",
                   help="Comma-separated timeframes (default: 1d). Options: 1m,5m,15m,30m,1h,1d,1wk")
    g.add_argument("--backbone", type=str, default="conv_attn",
                   choices=["conv_attn", "transformer"])
    g.add_argument("--eval-only", action="store_true",
                   help="Skip training, evaluate untrained network through pipeline")

    # Performance
    g = p.add_argument_group("Performance")
    g.add_argument("--no-amp", action="store_true")
    g.add_argument("--no-compile", action="store_true")
    g.add_argument("--no-dsr", action="store_true")
    g.add_argument("--no-subproc", action="store_true",
                   help="Disable shared-memory subprocess env stepping (fall back to sequential)")

    # Architecture
    g = p.add_argument_group("Institutional Architecture")
    g.add_argument("--disable-alphas", type=str, default="",
                   help='Disable specific alphas (e.g. "carry,seasonality")')
    g.add_argument("--invert-seasonality", action="store_true",
                   help="INVESTIGATION: Invert seasonality signal (test if IC becomes positive)")
    g.add_argument("--invert-vol-premium", action="store_true",
                   help="INVESTIGATION: Invert vol premium signal (buy vol instead of sell)")
    g.add_argument("--invert-trend", action="store_true",
                   help="OPTIMIZED: Invert trend signal (IC: -0.059 -> +0.036, 175%% improvement)")
    g.add_argument("--target-vol", type=float, default=0.15,
                   help="L3 target annual volatility (default: 0.15)")
    g.add_argument("--max-leverage", type=float, default=2.0)

    # Data Caching (Yahoo Finance only)
    g = p.add_argument_group("Data Caching")
    g.add_argument("--cache", action="store_true",
                   help="Enable Yahoo Finance data caching (disabled by default)")
    g.add_argument("--no-cache", action="store_true",
                   help="(Deprecated, caching is already off by default)")
    g.add_argument("--kelly-fraction", type=float, default=0.50)
    g.add_argument("--no-trade-threshold", type=float, default=0.005)  # Lowered from 0.02 (2%) to 0.005 (0.5%) based on persistence analysis

    # Output
    g = p.add_argument_group("Output")
    g.add_argument("--export-pine", action="store_true")
    g.add_argument("--verbose", type=int, default=1)
    g.add_argument("--alert-level", default="CRITICAL",
                   choices=["INFO", "WARNING", "CRITICAL", "FATAL"],
                   help="Min alert level to print on terminal (default: CRITICAL). "
                        "All alerts are always saved to alerts.jsonl in output dir.")
    g.add_argument("--self-test", action="store_true",
                   help="Run integrity tests (cost model + data quality) before launch")

    return p.parse_args()


# ============================================================================
# MAIN
# ============================================================================


# ============================================================================
# SELF-TEST: Cost model + data quality integrity checks
# ============================================================================

def _run_self_tests():
    """Run integrity tests for cost model and data quality modules.
    Called via --self-test flag. Halts on failure."""
    import traceback

    tests_passed = 0
    tests_failed = 0

    def _check(name, condition, msg=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
            tprint(f"  PASS: {name}", "ok")
        else:
            tests_failed += 1
            tprint(f"  FAIL: {name} -- {msg}", "err")

    print_divider("SELF-TEST: Integrity Checks")

    # -- Cost model tests --
    try:
        from alphago_cost_model import estimate_cost_dollars, fill_price_from_cost

        cb1 = estimate_cost_dollars(mid=100, trade_notional=10_000, adv_dollars=10_000_000,
                                    spread_bps=1, impact_coeff=0.1, fees_pct=0)
        cb2 = estimate_cost_dollars(mid=100, trade_notional=100_000, adv_dollars=10_000_000,
                                    spread_bps=1, impact_coeff=0.1, fees_pct=0)
        _check("Cost monotonicity (larger trade = higher cost)",
               cb2.total_cost > cb1.total_cost,
               f"{cb2.total_cost} <= {cb1.total_cost}")

        cb_lo = estimate_cost_dollars(mid=100, trade_notional=100_000, adv_dollars=1_000_000,
                                      spread_bps=1, impact_coeff=0.1, fees_pct=0)
        cb_hi = estimate_cost_dollars(mid=100, trade_notional=100_000, adv_dollars=100_000_000,
                                      spread_bps=1, impact_coeff=0.1, fees_pct=0)
        _check("Higher ADV reduces cost",
               cb_hi.total_cost < cb_lo.total_cost,
               f"{cb_hi.total_cost} >= {cb_lo.total_cost}")

        cb0 = estimate_cost_dollars(mid=100, trade_notional=0, adv_dollars=10_000_000,
                                    spread_bps=1, impact_coeff=0.1, fees_pct=0.001)
        _check("Zero trade = zero cost", cb0.total_cost == 0.0)

        fb, _ = fill_price_from_cost(mid=100, side="buy", trade_notional=50_000,
                                     adv_dollars=10_000_000, spread_bps=1, impact_coeff=0.1)
        fs, _ = fill_price_from_cost(mid=100, side="sell", trade_notional=50_000,
                                     adv_dollars=10_000_000, spread_bps=1, impact_coeff=0.1)
        _check("Buy fill above mid", fb > 100.0, f"fill={fb}")
        _check("Sell fill below mid", fs < 100.0, f"fill={fs}")

        cb = estimate_cost_dollars(mid=100, trade_notional=50_000, adv_dollars=10_000_000,
                                   spread_bps=1.5, impact_coeff=0.12, fees_pct=0.0005)
        _check("Components sum to total",
               abs(cb.half_spread_cost + cb.impact_cost + cb.fees_cost - cb.total_cost) < 1e-10)

    except Exception as e:
        tprint(f"  FAIL: Cost model import/test error: {e}", "err")
        tests_failed += 1

    # -- Data quality tests --
    try:
        from data_quality import analyze_ohlcv
        import pandas as pd

        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        prices = 100 + np.cumsum(np.random.randn(252) * 0.5)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices + np.random.randn(252) * 0.1,
            "close": prices,
            "volume": np.abs(np.random.randn(252) * 1e6 + 5e6),
        })
        df["high"] = df[["open", "close"]].max(axis=1) + 0.5
        df["low"] = df[["open", "close"]].min(axis=1) - 0.5
        q = analyze_ohlcv(df, ts_col="timestamp")
        _check("Clean data => PASS", q.verdict == "PASS",
               f"got {q.verdict} score={q.score_0_100}")

        df_bad = pd.DataFrame({
            "timestamp": pd.date_range("2020-01-01", periods=10, freq="B"),
            "open": [100]*10, "high": [101]*10, "low": [99]*10,
            "close": [100, -5, 100, 100, 100, 100, 100, 100, 100, 100],
            "volume": [1_000_000]*10,
        })
        q2 = analyze_ohlcv(df_bad, ts_col="timestamp")
        _check("Negative prices => FAIL", q2.verdict == "FAIL",
               f"got {q2.verdict}")

        q3 = analyze_ohlcv(pd.DataFrame())
        _check("Empty DataFrame => FAIL", q3.verdict == "FAIL")

    except Exception as e:
        tprint(f"  FAIL: Data quality import/test error: {e}", "err")
        tests_failed += 1

    # -- Summary --
    total = tests_passed + tests_failed
    if tests_failed == 0:
        tprint(f"All {total} integrity tests PASSED", "ok")
    else:
        tprint(f"{tests_failed}/{total} tests FAILED -- fix before trading", "err")
        sys.exit(1)


def resolve_symbols(args):
    """Resolve symbol list from --sector, --symbols, or DEFAULT_SYMBOLS."""
    # --sector list: print available sectors and exit
    if args.sector.strip().lower() == "list":
        print("Available sectors:")
        for name, syms in SECTOR_MAP.items():
            print(f"  {name:25s} ({len(syms):2d} symbols)  {', '.join(syms[:6])}"
                  + (f" ..." if len(syms) > 6 else ""))
        print(f"\nTotal: {sum(len(s) for s in SECTOR_MAP.values())} symbols across "
              f"{len(SECTOR_MAP)} sectors")
        sys.exit(0)
    # --sector X,Y: collect symbols from named sectors
    if args.sector:
        sector_syms = []
        for s in args.sector.split(","):
            key = s.strip().lower().replace(" ", "_")
            if key not in SECTOR_MAP:
                print(f"ERROR: Unknown sector '{s.strip()}'. Use --sector list to see options.")
                sys.exit(1)
            sector_syms.extend(SECTOR_MAP[key])
        # If --symbols also given, intersect
        if args.symbols:
            explicit = {x.strip().upper() for x in args.symbols.split(",") if x.strip()}
            sector_syms = [s for s in sector_syms if s in explicit]
        return sector_syms
    # --symbols only
    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    # Default: first n_symbols from DEFAULT_SYMBOLS
    return DEFAULT_SYMBOLS[:args.n_symbols]


def main():
    args = parse_args()

    # -- Handle --sector list early (before heavy init) --
    if args.sector.strip().lower() == "list":
        resolve_symbols(args)  # prints and exits

    # -- Fixed random seed for reproducible results --
    import random, shutil, glob as _glob
    _SEED = 42
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -- Clear stale caches at start of each run --
    _cache_dir = os.path.join(os.path.dirname(__file__), "__pycache__")
    if os.path.isdir(_cache_dir):
        shutil.rmtree(_cache_dir, ignore_errors=True)
    _data_cache = os.path.join(os.path.dirname(__file__), "data_cache")
    for _f in _glob.glob(os.path.join(_data_cache, "*.pkl")) + _glob.glob(os.path.join(_data_cache, "*.txt")):
        try:
            os.remove(_f)
        except OSError:
            pass

    # -- Self-test: run integrity checks before launch --
    if args.self_test:
        _run_self_tests()


    # -- Validate timeframes --
    tfs = tuple(t.strip() for t in args.timeframes.split(",") if t.strip())
    valid_tfs = {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m",
                 "1d", "5d", "1wk", "1mo"}
    for t in tfs:
        if t not in valid_tfs:
            print(f"ERROR: Invalid timeframe '{t}'")
            return

    # **********************************************************************
    # STEP 1: Configure base system
    # **********************************************************************
    cfg = Config(
        starting_capital=args.capital,
        n_iterations=args.iterations,
        total_timesteps_per_iter=args.steps_per_iter,
        mcts_rollouts=args.mcts_rollouts,
        mcts_schedule_base=min(args.mcts_rollouts, 64),  # schedule base respects user's limit
        mcts_schedule_max=args.mcts_rollouts,              # schedule max respects user's limit
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_envs=args.n_envs,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        use_dsr=not args.no_dsr,
        no_subproc=args.no_subproc,
        backbone_type=args.backbone,
        timeframes=tfs,
        download_period="max",
        verbose=args.verbose,
    )

    # **********************************************************************
    # STEP 2: Configure institutional architecture
    # **********************************************************************
    acfg = ArchitectureConfig(
        target_annual_vol=args.target_vol,
        max_leverage=args.max_leverage,
        kelly_fraction=args.kelly_fraction,
        no_trade_threshold_pct=args.no_trade_threshold,
        backtest_mode=True,  # Enable backtest mode: allow cross-regime testing, disable drift kill
    )

    # -- v7.0 WS6B: Deterministic reproducibility --
    if HAS_VALIDATION:
        ReproducibilityManager.set_global_seed(42)
        vcfg = VersionedConfig(acfg, version="v7.0")
        config_hash = vcfg.config_hash()
    else:
        config_hash = "N/A"

    # FIX Ãƒâ€šÃ‚Â§6.1/Ãƒâ€šÃ‚Â§6.2: Log library versions for reproducibility (Absolute Rule #6).
    # In production, pin via requirements.txt; here we log versions per run.
    _lib_versions = {
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    if HAS_TORCH:
        _lib_versions["torch"] = torch.__version__
        _lib_versions["cuda"] = torch.version.cuda if torch.version.cuda else "N/A"
    try:
        import pandas as _pd_ver
        _lib_versions["pandas"] = _pd_ver.__version__
    except Exception:
        pass

    # **********************************************************************
    # BANNER
    # **********************************************************************
    print_section("", reset=True)  # Reset section counter for this run
    print_box(
        "ALPHA-TRADE v7.0 -- Institutional Architecture",
        f"L1->L2->L3->L4 | RL is one alpha of many | cfg:{config_hash} | {datetime.now():%Y-%m-%d %H:%M}"
    )
    print_gpu_info()
    tprint("This system combines multiple trading strategies (alphas) like a team of analysts,", "info")
    tprint("then filters their ideas through risk controls before placing trades.", "info")
    if HAS_TORCH:
        tprint(f"PyTorch {torch.__version__} | Device: {DEVICE} | "
               f"AMP:{'ON' if cfg.use_amp else 'OFF'}", "info")

    # **********************************************************************
    # STEP 3: Build institutional pipeline
    # **********************************************************************
    print_section("SETUP")
    tprint("Setting up the assembly line: L1=signal generation, L2=combining signals,", "info")
    tprint("L3=position sizing (how much to bet), L4=execution (when to actually trade).", "info")

    # Wire alert logging to file Ã¢â‚¬â€ keeps terminal clean, captures everything for analysis
    alert_log_path = os.path.join(cfg.output_dir, "alerts.jsonl")
    acfg.alert_log_file = alert_log_path
    acfg.alert_terminal_level = getattr(args, 'alert_level', 'CRITICAL')

    pipeline = build_default_pipeline(
        acfg,
        bars_per_year=252,
        net=None,
        invert_seasonality=getattr(args, 'invert_seasonality', False),
        invert_vol_premium=getattr(args, 'invert_vol_premium', False),
        invert_trend=getattr(args, 'invert_trend', False)
    )

    # Wire verbose flag to execution engine for debug output control
    if hasattr(pipeline, 'execution_engine'):
        pipeline.execution_engine.verbose = args.verbose

    # Handle --disable-alphas
    if args.disable_alphas:
        for name in args.disable_alphas.split(","):
            name = name.strip().lower()
            # Map user-friendly names to internal alpha names
            name_map = {
                'rl': 'rl_ppo_mcts', 'trend': 'trend_follow',
                'mr': 'mean_reversion', 'value': 'value',
                'carry': 'carry', 'seasonality': 'seasonality',
                'vol_premium': 'vol_premium', 'vol': 'vol_premium',
            }
            internal = name_map.get(name, name)
            pipeline.alpha_factory.unregister(internal)
            pipeline.rebuild_ensemble()
            tprint(f"  Disabled alpha: {internal}", "warn")

    tprint(f"Active alphas ({pipeline.alpha_factory.n_alphas}):", "ok")
    for name in pipeline.alpha_factory.alpha_names:
        tprint(f"    [OK] {name}", "info")

    tprint(f"L3 config: vol_target={acfg.target_annual_vol:.0%}  "
           f"kelly={acfg.kelly_fraction}  max_leverage={acfg.max_leverage}x  "
           f"dd_scale=[{acfg.drawdown_scale_threshold:.0%},{acfg.drawdown_max_threshold:.0%}]",
           "info")
    tprint(f"L4 config: no_trade={acfg.no_trade_threshold_pct:.0%}  "
           f"kill_loss={acfg.kill_max_loss_pct:.0%}  "
           f"slicing={'TWAP' if acfg.order_slicing else 'OFF'}--{acfg.n_slices}",
           "info")

    # **********************************************************************
    # STEP 4: Load data
    # **********************************************************************
    print_section("DATA LOADING")
    tprint("Loading historical price data for training.", "info")
    data = None
    data_source = "unknown"

    if args.yahoo:
        data_source = "Yahoo Finance (survivorship-biased)"
        if not HAS_YF:
            tprint("--yahoo requires yfinance: pip install yfinance", "err"); return
        syms = resolve_symbols(args)
        tprint(f"Data source: {data_source}", "warn")
        tprint("  Backtest results will be upward-biased. Use Norgate PIT data for valid results.", "warn")
        data = download_data(syms, cfg, use_cache=args.cache)
    elif args.data_dir:
        data_source = f"Local directory: {args.data_dir}"
        tprint(f"Data source: {data_source}", "info")
        data = load_from_dir(args.data_dir, cfg.min_bars)
    elif args.synthetic:
        data_source = f"Synthetic ({args.n_synthetic} instruments)"
        tprint(f"Data source: {data_source}", "info")
        data = SyntheticMarketGenerator(cfg).generate_multiple(args.n_synthetic, 2000)
    else:
        norgate_dir = args.norgate_dir or NORGATE_DIR
        norgate_dbs = ([db.strip() for db in args.norgate_db.split(",") if db.strip()]
                       if args.norgate_db else None)
        norgate_syms = resolve_symbols(args)

        if not Path(norgate_dir).exists():
            tprint(f"Norgate data not found at: {norgate_dir}", "err")
            tprint("Run norgate_download_all.py first, or use --yahoo / --synthetic", "err")
            return

        db_list = norgate_dbs or [db for db, info in NORGATE_DATABASES.items() if info["has_volume"]]
        data_source = f"Norgate PIT data: {norgate_dir}"
        tprint(f"Data source: {data_source}", "ok")
        tprint(f"  Databases: {', '.join(db_list)}", "info")
        if norgate_syms:
            tprint(f"  Symbols filter: {', '.join(norgate_syms[:10])}"
                   f"{'...' if len(norgate_syms) > 10 else ''}", "info")
        data = load_from_norgate(norgate_dir, databases=norgate_dbs,
                                 symbols=norgate_syms, mb=cfg.min_bars)

    if not data:
        tprint(f"No data loaded from: {data_source}", "err")
        tprint("Check your data source or use --synthetic", "err"); return

    tprint(f"Loaded {len(data)} datasets", "ok")

    # -- Load SPY benchmark returns --
    spy_returns_lookup = {}  # date_str -> daily return
    try:
        _spy_base = norgate_dir if 'norgate_dir' in dir() else NORGATE_DIR
        spy_path = Path(_spy_base) / "US_Equities" / "SPY.parquet"
        if spy_path.exists():
            import pandas as _spy_pd
            spy_df = _spy_pd.read_parquet(spy_path)
            col_map = {}
            for c in spy_df.columns:
                cl = c.lower().strip()
                if cl == "close" and "unadj" not in cl.replace(" ", ""):
                    col_map[c] = "Close"
            spy_df = spy_df.rename(columns=col_map)
            if "Close" in spy_df.columns:
                spy_closes = spy_df["Close"].values.astype(np.float64)
                spy_dates = np.array(spy_df.index.astype(str))
                spy_rets = np.diff(spy_closes) / (spy_closes[:-1] + 1e-12)
                for i, r in enumerate(spy_rets):
                    spy_returns_lookup[spy_dates[i + 1][:10]] = float(r)
                tprint(f"SPY benchmark loaded: {len(spy_returns_lookup)} daily returns", "ok")
            else:
                tprint("SPY parquet missing Close column, using cash benchmark", "warn")
        else:
            tprint(f"SPY data not found at {spy_path}, using cash benchmark", "warn")
    except Exception as e:
        tprint(f"Failed to load SPY benchmark: {e}, using cash benchmark", "warn")

    # -- L0 DATA QUALITY CHECKS --
    quality_results = []
    trading_calendar = None
    ingestion_gate = None

    # v7.0 WS2C: Exchange calendar
    if HAS_WS2:
        trading_calendar = TradingCalendar("NYSE")
        # v7.1: Policy driven from ArchitectureConfig.missing_data_policy
        _mdp = MissingDataPolicy(acfg.missing_data_policy)
        ingestion_gate = DataIngestionGate(policy=_mdp)
        tprint(f"  L0: Trading calendar loaded (NYSE), ingestion gate active "
               f"(missing_data_policy={_mdp.value})", "ok")

    if HAS_DATA_QUALITY:
        import pandas as pd
        for key, df in data.items():
            sym = key.rsplit("_", 1)[0] if "_" in key else key
            tf = key.rsplit("_", 1)[-1] if "_" in key else "1D"

            # v7.0 WS2D: Data ingestion gate -- reject/quarantine before analysis
            if ingestion_gate is not None:
                df_gated, gate_report = ingestion_gate.ingest(df, calendar=trading_calendar)
                if gate_report.get('rejected', 0) > 0:
                    tprint(f"  {sym}: {gate_report['rejected']} bars rejected, "
                           f"{gate_report.get('quarantined', 0)} quarantined", "warn")
                # Strip internal metadata column before downstream use
                if '_quarantined' in df_gated.columns:
                    df_gated = df_gated.drop(columns=['_quarantined'])
                data[key] = df_gated  # Preserves original column names (Close, Open, etc.)

            # Normalise column names to lowercase FOR QUALITY CHECKS ONLY
            # Use the latest version of data (post-gating if gated)
            df_q = data[key].copy()
            df_q.columns = [c.lower() for c in df_q.columns]
            if df_q.index.name or not any(c == "timestamp" for c in df_q.columns):
                df_q = df_q.reset_index()
                df_q.rename(columns={df_q.columns[0]: "timestamp"}, inplace=True)

            fields = [c for c in ("open", "high", "low", "close", "volume") if c in df_q.columns]
            meta = DataLoadMeta(
                source=data_source,
                symbols=[sym], timeframe=tf,
                requested_start=str(args.start) if hasattr(args, "start") else "N/A",
                requested_end=str(args.end) if hasattr(args, "end") else "N/A",
                loaded_start=str(df_q["timestamp"].min()) if "timestamp" in df_q.columns else "N/A",
                loaded_end=str(df_q["timestamp"].max()) if "timestamp" in df_q.columns else "N/A",
                timezone="UTC", calendar="NYSE",
                adjusted=False, fields=fields,
                rows_total=len(df_q),
            )
            meta.dataset_hash = hash_df(df_q, cols=fields)
            q = analyze_ohlcv(df_q, ts_col="timestamp")
            quality_results.append((meta, q))

        # Print one consolidated table for all symbols
        print_quality_table(quality_results)
    else:
        tprint("  data_quality module not found -- skipping quality checks", "warn")

    # **********************************************************************
    # STEP 5: Feature engineering
    # **********************************************************************
    print_section("FEATURE ENGINEERING")
    tprint("Turning raw prices into useful signals (like converting temperature to 'hot/cold').", "info")
    tprint("Splitting data into: Train (textbook), Validation (practice test), Test (final exam).", "info")
    datasets = prepare_datasets(data, cfg)
    if not datasets:
        tprint("No valid datasets!", "err")
        return

    # Show detailed train/val/test date ranges and bar counts
    print_data_split_summary(datasets, cfg)

    # **********************************************************************
    # STEP 6: Train (or skip if --eval-only)
    # **********************************************************************
    base_results = None

    training_hist = []  # populated by training loop (empty in eval-only mode)
    if args.eval_only:
        print_section("EVAL-ONLY MODE")
        tprint("Building untrained network for pipeline demonstration...", "info")
        net = build_network(cfg)
    else:
        print_section("TRAINING (PPO + MCTS)")
        system = AlphaTradeSystem(datasets, cfg)
        tprint("Starting self-play training loop...", "info")
        champion = system.train()
        training_hist = getattr(system, 'hist', [])
        net = system.champ

        # -- Show base system's own evaluation (for comparison) --
        print_section("BASE EVALUATION (v3.0)")
        tprint("Testing the AI's raw trading ability (no risk filters, just its own decisions).", "info")
        base_results = system.final_eval()

    # **********************************************************************
    # STEP 6.5: Walk-forward alpha validation (WS1A -- fix 5.2)
    # **********************************************************************
    # Run purged walk-forward CV on TRAINING data to validate each alpha's
    # statistical significance BEFORE the final OOS evaluation.
    # This wires the PurgedWalkForwardCV + SignificanceGate infrastructure
    # from validation_engine.py into the actual pipeline.
    wf_validation = None
    if HAS_VALIDATION and not args.eval_only:
        print_section("WALK-FORWARD ALPHA VALIDATION")
        tprint("Running purged walk-forward CV on training data...", "info")
        tprint("  Each alpha evaluated for significance via t-stat, deflated Sharpe,", "info")
        tprint("  OOS/IS decay ratio, and probability of backtest overfitting (PBO)", "info")
        print()
        wf_validation = validate_alphas_walkforward(
            datasets=datasets,
            pipeline=pipeline,
            net=unwrap_net(net) if HAS_TORCH else net,
            cfg=cfg,
            acfg=acfg,
            verbose=args.verbose,
        )
    elif not HAS_VALIDATION:
        tprint("validation_engine not available -- skipping walk-forward CV", "warn")

    # FIX §SA-3: Seed meta-learner from validation results BEFORE holdout.
    # Without this, the ensemble falls back to equal-weight combination
    # during holdout evaluation because no training data has been recorded.
    # This gives the Ridge meta-learner an informed starting point based
    # on each alpha's OOS Sharpe from walk-forward CV.
    if wf_validation and hasattr(pipeline, '_ensemble') and pipeline._ensemble is not None:
        try:
            ml = pipeline._ensemble.meta_learner
            ml.fit_from_validation_priors(wf_validation)
            n_priors = sum(1 for r in wf_validation.get('per_alpha', {}).values()
                          if r.get('sharpe_oos', 0) > 0)
            tprint(f"Meta-learner seeded from validation priors "
                   f"({n_priors} alphas with positive OOS Sharpe)", "ok")
        except Exception as e:
            tprint(f"Meta-learner seeding warning: {e}", "warn")

    # B2: Auto-flip negative IC signals based on WF validation results.
    # If an alpha reliably predicts the opposite direction (IC < 0 with
    # |IC| > threshold), flipping its mu sign converts anti-predictive
    # signal into predictive signal.
    if wf_validation:
        per_alpha = wf_validation.get('per_alpha', {})
        flipped_alphas = []
        for aname, rpt in per_alpha.items():
            best_ic = rpt.get('best_ic', 0)
            verdict = rpt.get('verdict', 'REJECT')
            # Only flip alphas with significant negative IC (PASS or MARGINAL with flip)
            if best_ic < -0.015 and verdict in ('PASS', 'MARGINAL'):
                pipeline.alpha_factory.set_signal_flip(aname, flip=True)
                flipped_alphas.append(f"{aname} (IC={best_ic:+.3f})")
        if flipped_alphas:
            tprint(f"Auto-flipped {len(flipped_alphas)} negative-IC alphas: "
                   f"{', '.join(flipped_alphas)}", "ok")

    # **********************************************************************
    # STEP 7: Pipeline evaluation
    # **********************************************************************
    # Suppress noisy feature discipline warnings at default verbosity
    if args.verbose < 2:
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Feature discipline")

    print_section("PIPELINE EVALUATION (v7.0)")
    tprint("Now testing with ALL safety layers active (risk limits, position sizing, trade filters).", "info")
    tprint("This is the full production system, not just the raw AI.", "info")

    # FIX Ãƒâ€šÃ‚Â§5.5: Enforce HoldoutGuard ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â final evaluation is single-use.
    # The holdout test set (most recent 20% of each dataset) must be touched
    # EXACTLY ONCE. This guard prevents repeated tuning on the holdout.
    holdout_guard = None
    if HAS_VALIDATION and datasets:
        try:
            # Compute holdout boundaries from first dataset's test split
            sample_n = len(datasets[0].prices_test) if hasattr(datasets[0], 'prices_test') else 1000
            holdout_start = int(sample_n * 0.80)
            holdout_end = sample_n
            holdout_guard = HoldoutGuard(holdout_start=holdout_start, holdout_end=holdout_end)
            holdout_range = holdout_guard.consume_holdout()
            tprint(f"HoldoutGuard: consuming holdout range [{holdout_range[0]}:{holdout_range[1]}] (single-use)", "info")
        except Exception as e:
            tprint(f"HoldoutGuard setup warning: {e}", "warn")

    tprint("Evaluating through institutional pipeline...", "info")
    tprint("  RL agent outputs alpha signals (mu, sigma, confidence)", "info")
    tprint("  L2 combines with 6 other alphas via meta-learner", "info")
    tprint("  L3 (PortfolioConstructor) makes ALL sizing decisions", "info")
    tprint("  L4 filters noise trades and enforces kill switches", "info")
    print()

    pipeline_results = evaluate_with_pipeline(
        net=unwrap_net(net) if HAS_TORCH else net,
        datasets=datasets,
        pipeline=pipeline,
        cfg=cfg,
        acfg=acfg,
        label="Pipeline v7.0",
        verbose=args.verbose,
        spy_returns_lookup=spy_returns_lookup,
    )
    # Save SMA pipeline lifetime stats before no-SMA run
    pip_sma_lt = dict(pipeline.execution_engine.lifetime_stats)

    print_results(pipeline_results)

    # -- Pipeline v7.0 WITHOUT SMA overlay (A/B comparison) --
    tprint("Re-evaluating without SMA overlay (for comparison)...", "info")
    pipeline.use_sma = False
    pipeline.execution_engine.reset_lifetime_stats()
    nosma_results = evaluate_with_pipeline(
        net=unwrap_net(net) if HAS_TORCH else net,
        datasets=datasets,
        pipeline=pipeline,
        cfg=cfg,
        acfg=acfg,
        label="Pipeline v7.0 (no SMA)",
        verbose=0,
        spy_returns_lookup=spy_returns_lookup,
    )
    nosma_lt = dict(pipeline.execution_engine.lifetime_stats)
    pipeline.use_sma = True  # Restore for diagnostics
    # Restore SMA lifetime stats for diagnostics section
    pipeline.execution_engine.reset_lifetime_stats()
    pipeline.execution_engine._lifetime_trades = pip_sma_lt['n_trades']
    pipeline.execution_engine._lifetime_suppressed = pip_sma_lt['n_suppressed']

    # FIX Ãƒâ€šÃ‚Â§5.5: Record holdout result in guard (audit trail)
    if holdout_guard is not None:
        try:
            holdout_guard.record_holdout_result(
                sharpe=pipeline_results.get('avg_sh', 0),
                total_pnl=pipeline_results.get('total_pnl', 0),
                dd_max=pipeline_results.get('dd_max', 0),
                score=pipeline_results.get('score', 0),
                n_symbols=pipeline_results.get('total_symbols', 0),
            )
            tprint(f"HoldoutGuard: result recorded (consumed={holdout_guard.is_consumed})", "ok")
        except Exception as e:
            tprint(f"HoldoutGuard result recording warning: {e}", "warn")

    # -- v7.0 WS4: Institutional Backtest Report + Stress Testing --
    if HAS_BACKTEST_REPORT:
        print_section("INSTITUTIONAL REPORT")
        tprint("Generating a professional-grade report (the kind hedge funds show to investors).", "info")

        # FIX: Use ACTUAL per-bar returns collected during evaluation
        # (previously used np.random.randn proxy -- every metric was fabricated)
        actual_gross = pipeline_results.get("daily_returns_gross", np.array([]))
        actual_net = pipeline_results.get("daily_returns_net", np.array([]))

        if len(actual_gross) < 5:
            tprint("Insufficient per-bar returns for institutional report "
                   f"({len(actual_gross)} bars). Need >= 5.", "warn")
        else:
            tprint(f"Computing report from {len(actual_gross)} actual per-bar returns", "ok")

            # Extract period_start and period_end from datasets
            period_start_str = ""
            period_end_str = ""
            if datasets and len(datasets) > 0:
                # Collect all timestamps from all datasets
                all_timestamps = []
                for d in datasets:
                    if hasattr(d, 'timestamps_train') and d.timestamps_train is not None:
                        all_timestamps.extend(d.timestamps_train)
                    if hasattr(d, 'timestamps_val') and d.timestamps_val is not None:
                        all_timestamps.extend(d.timestamps_val)
                    if hasattr(d, 'timestamps_test') and d.timestamps_test is not None:
                        all_timestamps.extend(d.timestamps_test)

                if all_timestamps:
                    # Sort to get first and last
                    all_timestamps = sorted(all_timestamps)
                    period_start_str = str(all_timestamps[0])[:10]   # YYYY-MM-DD
                    period_end_str = str(all_timestamps[-1])[:10]

            spy_rets = pipeline_results.get('spy_returns', np.array([]))
            has_spy = len(spy_rets) > 0 and np.any(spy_rets != 0.0)

            report = compute_backtest_report(
                daily_returns_gross=actual_gross,
                daily_returns_net=actual_net,
                benchmark_returns=spy_rets if has_spy else None,
                strategy_name="Alpha-Trade v7.0",
                benchmark="SPY" if has_spy else "cash",
                period_start=period_start_str,
                period_end=period_end_str,
                n_configs=1,
            )
            report.print_summary()

            # Stress test on ACTUAL returns
            st = StressTester()
            stress_report = st.full_stress_report(actual_net)
            st.print_stress_report(stress_report)

            # Capacity estimate
            avg_sharpe = pipeline_results.get('avg_sh', 0)
            ce = CapacityEstimator()
            cap = ce.estimate(
                gross_alpha_bps=max(avg_sharpe * 100, 10),
                annual_turnover=12.0,
                avg_adv_dollars=50_000_000,
            )
            tprint(f"Estimated capacity (Sharpe>0.5): "
                   f"${cap['capacity_at_sharpe_05']/1e6:.0f}M", "info")

    # -- Compute SPY buy-and-hold P&L for comparison table --
    # NOTE: spy_rets_arr is concatenated across symbols (duplicated dates) — correct
    # for benchmark-relative metrics (aligned 1:1 with strategy returns), but NOT
    # for computing actual SPY buy-and-hold P&L. Use unique test-period dates instead.
    spy_rets_arr = pipeline_results.get('spy_returns', np.array([]))
    has_spy_bench = len(spy_rets_arr) > 0 and np.any(spy_rets_arr != 0.0)
    spy_total_pnl = 0.0
    spy_cagr_str = "N/A"
    if has_spy_bench and spy_returns_lookup:
        # Collect unique TRADING dates (after lookback window) from per_symbol_date_returns.
        # This matches the equity curve chart which uses the same date source.
        # Using all timestamps_test would include the lookback window (~w bars)
        # where no trading occurs, inflating SPY compounding vs the chart.
        _date_rets_for_spy = pipeline_results.get("per_symbol_date_returns", {})
        _test_dates = set()
        for _pairs in _date_rets_for_spy.values():
            for _dt, _ret in _pairs:
                _test_dates.add(_dt)
        _test_dates_sorted = sorted(_test_dates)
        # Get SPY returns for those unique dates only
        _spy_unique = [spy_returns_lookup.get(dt, 0.0) for dt in _test_dates_sorted]
        if _spy_unique:
            _spy_unique_arr = np.array(_spy_unique, dtype=np.float64)
            spy_cum = float(np.prod(1 + _spy_unique_arr) - 1)
            # SPY P&L on $10k (same allocation as one symbol)
            spy_total_pnl = spy_cum * cfg.starting_capital
            n_spy = len(_spy_unique_arr)
            spy_cagr = float(np.prod(1 + _spy_unique_arr) ** (252 / max(n_spy, 1)) - 1)
            spy_cagr_str = f"{spy_cagr:+.2%}"

    # -- Build per-symbol test period date ranges --
    _sym_periods = {}  # symbol -> (start_date, end_date)
    for d in datasets:
        if hasattr(d, 'timestamps_test') and d.timestamps_test is not None and len(d.timestamps_test) > 0:
            _sym_periods[d.symbol] = (str(d.timestamps_test[0])[:10], str(d.timestamps_test[-1])[:10])

    # SPY test period (union of all symbol test dates)
    _spy_period_str = ""
    if has_spy_bench and '_test_dates_sorted' in dir() and _test_dates_sorted:
        _spy_period_str = f"{_test_dates_sorted[0]} to {_test_dates_sorted[-1]}"

    # -- Show comparison if training was done --
    if base_results is not None:
        # Compute cash yield for both systems
        _pip_per_sym = pipeline_results.get('per_sym', {})
        pip_cash_yield = sum(s.get('cash_yield_pnl', 0) for s in _pip_per_sym.values())
        pip_trade_pnl = pipeline_results['total_pnl'] - pip_cash_yield

        # Estimate base cash yield (same formula: cash_bps * capital * steps)
        # Base v3.0 doesn't track cash_yield_pnl per symbol, so we estimate it
        # from step counts (same test data = same step count per symbol).
        _cash_bps = getattr(cfg, 'cash_yield_bps_annual', 0) * 1e-4
        _cash_per_bar = (_cash_bps / max(cfg.bars_per_year, 1)) * cfg.starting_capital if _cash_bps > 0 else 0.0
        _base_total_steps = sum(
            s.get('step_count', 0) for s in _pip_per_sym.values()
        )  # Use pipeline step count (same test data length)
        base_cash_yield = _cash_per_bar * _base_total_steps
        # Per-symbol cash yield estimate for base (used in per-symbol rows)
        _base_cash_per_sym = {
            sym: _cash_per_bar * _pip_per_sym.get(sym, {}).get('step_count', 0)
            for sym in _pip_per_sym
        }
        base_trade_pnl = base_results.get('total_pnl', 0) - base_cash_yield

        # Pipeline trades from L4 stats
        pip_lt = pipeline.execution_engine.lifetime_stats
        pip_n_trades = pip_lt['n_trades']
        pip_n_suppressed = pip_lt['n_suppressed']
        pip_suppression_pct = pip_lt['suppression_rate'] * 100

        # Compute Win Rate and Profit Factor from per_sym data (aggregated across all symbols)
        # NOTE: 'trades' here means CLOSED round-trip trades (n_wins + n_losses), not position changes!
        # If system has open position, these will be 0 even if Trade P&L is positive (unrealized gains).
        total_wins_pip = sum(s.get('trades', 0) * s.get('win_rate', 0) / 100.0 for s in _pip_per_sym.values())
        total_trades_pip = sum(s.get('trades', 0) for s in _pip_per_sym.values())
        pip_win_rate = (total_wins_pip / max(total_trades_pip, 1)) * 100 if total_trades_pip > 0 else float('nan')

        # Profit factor: need individual trade PnLs (not available in per_sym)
        # Use per-symbol profit factors as proxy (weighted avg by number of trades)
        weighted_pf_pip = sum(
            s.get('profit_factor', 0) * s.get('trades', 0)
            for s in _pip_per_sym.values()
        )
        pip_profit_factor = weighted_pf_pip / max(total_trades_pip, 1) if total_trades_pip > 0 else float('nan')

        # Track number of closed vs executed trades for clarity
        pip_n_closed_trades = total_trades_pip

        # Base system trades
        base_n_trades = base_results.get('total_trades', 0)
        base_win_rate = base_results.get('avg_wr', 0.0)  # This is already computed correctly in base system
        base_profit_factor = base_results.get('avg_pf', 0.0)

        # No-SMA pipeline stats
        _nosma_per_sym = nosma_results.get('per_sym', {})
        nosma_cash_yield = sum(s.get('cash_yield_pnl', 0) for s in _nosma_per_sym.values())
        nosma_trade_pnl = nosma_results['total_pnl'] - nosma_cash_yield
        nosma_n_trades = nosma_lt.get('n_trades', 0)
        nosma_n_suppressed = nosma_lt.get('n_suppressed', 0)
        nosma_suppression_pct = nosma_lt.get('suppression_rate', 0) * 100

        print_section("COMPARISON: v3.0 vs v7.0")
        print(f"  {C.DIM}Side-by-side: raw AI decisions vs. the full system with risk controls.{C.RESET}")
        print(f"  {C.DIM}Like comparing a driver without seatbelts (v3.0) vs. with full safety gear (v7.0).{C.RESET}")

        if HAS_TABLE_FORMATTER:
            _E = ""  # shorthand for empty trailing columns
            table = TableFormatter(title="BASE V3.0 VS PIPELINE V7.0")
            table.add_column('Metric', align='left')
            table.add_column('Base v3.0', align='right')
            table.add_column('v7.0 (no SMA)', align='right')
            table.add_column('Pipeline v7.0', align='right')
            table.add_column('# Trades', align='right')
            table.add_column('$ Used', align='right')
            table.add_column('CAGR', align='right')

            # Prepare colored strings
            base_pnl_c = C.GREEN if base_results.get('total_pnl', 0) > 0 else C.RED
            nosma_pnl_c = C.GREEN if nosma_results['total_pnl'] > 0 else C.RED
            pip_pnl_c = C.GREEN if pipeline_results['total_pnl'] > 0 else C.RED
            base_trade_c = C.GREEN if base_trade_pnl > 0 else C.RED
            nosma_trade_c = C.GREEN if nosma_trade_pnl > 0 else C.RED
            pip_trade_c = C.GREEN if pip_trade_pnl > 0 else C.RED

            # Starting capital — show per-symbol and total
            _n_syms_table = len(_pip_per_sym) if _pip_per_sym else len(datasets)
            _total_cap = cfg.starting_capital * _n_syms_table
            table.add_row(['Capital / Symbol',
                          f"${cfg.starting_capital:>13,.0f}",
                          f"${cfg.starting_capital:>13,.0f}",
                          f"${cfg.starting_capital:>13,.0f}", _E, _E, _E])
            table.add_row([f'Total Capital ({_n_syms_table} syms)',
                          f"${_total_cap:>13,.0f}",
                          f"${_total_cap:>13,.0f}",
                          f"${_total_cap:>13,.0f}", _E, _E, _E])

            # Add rows with section headers
            table.add_row([f"{C.BOLD}--- P&L ---{C.RESET}", _E, _E, _E, _E, _E, _E])
            table.add_row(['Total P&L',
                          f"{base_pnl_c}${base_results.get('total_pnl', 0):>+13,.2f}{C.RESET}",
                          f"{nosma_pnl_c}${nosma_results['total_pnl']:>+13,.2f}{C.RESET}",
                          f"{pip_pnl_c}${pipeline_results['total_pnl']:>+13,.2f}{C.RESET}", _E, _E, _E])
            table.add_row(['  Trade P&L',
                          f"{base_trade_c}${base_trade_pnl:>+13,.2f}{C.RESET}",
                          f"{nosma_trade_c}${nosma_trade_pnl:>+13,.2f}{C.RESET}",
                          f"{pip_trade_c}${pip_trade_pnl:>+13,.2f}{C.RESET}", _E, _E, _E])
            table.add_row(['  Cash Yield',
                          f"{C.CYAN}${base_cash_yield:>+13,.2f}{C.RESET}",
                          f"{C.CYAN}${nosma_cash_yield:>+13,.2f}{C.RESET}",
                          f"{C.CYAN}${pip_cash_yield:>+13,.2f}{C.RESET}", _E, _E, _E])

            # Per-symbol Trade P&L (cash yield stripped out) + $ Used + CAGR
            _base_per_sym = base_results.get('per_sym', {})
            all_syms = sorted(set(list(_base_per_sym.keys()) + list(_pip_per_sym.keys())))
            if all_syms:
                table.add_row([f"{C.BOLD}--- Trade P&L by Symbol ---{C.RESET}", _E, _E, _E, _E, _E, _E])
                for sym in all_syms:
                    b_total = _base_per_sym.get(sym, {}).get('pnl', 0)
                    b_cash = _base_per_sym.get(sym, {}).get('cash_yield_pnl', 0) or _base_cash_per_sym.get(sym, 0)
                    b_trade = b_total - b_cash
                    n_total = _nosma_per_sym.get(sym, {}).get('pnl', 0)
                    n_cash = _nosma_per_sym.get(sym, {}).get('cash_yield_pnl', 0)
                    n_trade = n_total - n_cash
                    p_total = _pip_per_sym.get(sym, {}).get('pnl', 0)
                    p_cash = _pip_per_sym.get(sym, {}).get('cash_yield_pnl', 0)
                    p_trade = p_total - p_cash
                    b_c = C.GREEN if b_trade > 0 else C.RED
                    n_c = C.GREEN if n_trade > 0 else C.RED
                    p_c = C.GREEN if p_trade > 0 else C.RED
                    sym_trades = _pip_per_sym.get(sym, {}).get('pipeline_trades', 0)
                    period = _sym_periods.get(sym, None)
                    period_str = f" ({period[0]} to {period[1]})" if period else ""
                    # $ Used (peak notional) and CAGR
                    _peak = _pip_per_sym.get(sym, {}).get('peak_notional', 0)
                    _steps = _pip_per_sym.get(sym, {}).get('step_count', 0)
                    if _peak > 0 and _steps > 0:
                        _years = _steps / 252
                        _return_ratio = 1 + p_trade / _peak
                        if _return_ratio > 0:
                            _sym_cagr = _return_ratio ** (1 / max(_years, 0.01)) - 1
                            _cagr_c = C.GREEN if _sym_cagr > 0 else C.RED
                            cagr_str = f"{_cagr_c}{_sym_cagr:>+8.1%}{C.RESET}"
                        else:
                            cagr_str = f"{C.RED}  -100%+{C.RESET}"
                        used_str = f"${_peak:>10,.0f}"
                    else:
                        cagr_str = "N/A"
                        used_str = "N/A"
                    table.add_row([f"  {sym}{period_str}",
                                  f"{b_c}${b_trade:>+13,.2f}{C.RESET}",
                                  f"{n_c}${n_trade:>+13,.2f}{C.RESET}",
                                  f"{p_c}${p_trade:>+13,.2f}{C.RESET}",
                                  f"{sym_trades:>8d}", used_str, cagr_str])

            if has_spy_bench:
                spy_pnl_c = C.GREEN if spy_total_pnl > 0 else C.RED
                _spy_hdr = f"--- SPY Benchmark ({_spy_period_str}) ---" if _spy_period_str else "--- SPY Benchmark ---"
                table.add_row([f"{C.BOLD}{_spy_hdr}{C.RESET}", _E, _E, _E, _E, _E, _E])
                table.add_row(['  SPY Buy & Hold P&L',
                              f"{spy_pnl_c}${spy_total_pnl:>+13,.2f}{C.RESET}",
                              f"{spy_pnl_c}${spy_total_pnl:>+13,.2f}{C.RESET}",
                              f"{spy_pnl_c}${spy_total_pnl:>+13,.2f}{C.RESET}", _E, _E, _E])
                table.add_row(['  SPY CAGR', spy_cagr_str, spy_cagr_str, spy_cagr_str, _E, _E, _E])

            table.add_row([f"{C.BOLD}--- Trading ---{C.RESET}", _E, _E, _E, _E, _E, _E])
            table.add_row(['Trades Executed',
                          f"{base_n_trades:>14d}",
                          f"{nosma_n_trades:>14d}",
                          f"{pip_n_trades:>14d}",
                          f"{pip_n_trades:>8d}", _E, _E])

            base_n_suppressed = 0
            table.add_row(['Trades Suppressed',
                          f"{base_n_suppressed:>14d}",
                          f"{nosma_n_suppressed:>14d}",
                          f"{pip_n_suppressed:>14d}", _E, _E, _E])
            table.add_row(['Suppression Rate',
                          f"{0.0:>13.1f}%",
                          f"{nosma_suppression_pct:>13.1f}%",
                          f"{pip_suppression_pct:>13.1f}%", _E, _E, _E])

            base_wr_str = f"{base_win_rate:>13.1f}%" if not np.isnan(base_win_rate) else "N/A (0 closed)"
            pip_wr_str = f"{pip_win_rate:>13.1f}%" if not np.isnan(pip_win_rate) else f"N/A ({pip_n_closed_trades} closed)"
            table.add_row(['Win Rate', base_wr_str, _E, pip_wr_str, _E, _E, _E])

            base_pf_str = f"{base_profit_factor:>14.2f}" if not np.isnan(base_profit_factor) else "N/A"
            pip_pf_str = f"{pip_profit_factor:>14.2f}" if not np.isnan(pip_profit_factor) else "N/A"
            table.add_row(['Profit Factor', base_pf_str, _E, pip_pf_str, _E, _E, _E])

            # --- Risk ---
            table.add_row([f"{C.BOLD}--- Risk ---{C.RESET}", _E, _E, _E, _E, _E, _E])
            table.add_row(['Sharpe',
                          f"{base_results.get('avg_sh', 0):>+14.3f}",
                          f"{nosma_results['avg_sh']:>+14.3f}",
                          f"{pipeline_results['avg_sh']:>+14.3f}", _E, _E, _E])
            table.add_row(['Max Drawdown',
                          f"{base_results.get('dd_max', 0):>13.1f}%",
                          f"{nosma_results['dd_max']:>13.1f}%",
                          f"{pipeline_results['dd_max']:>13.1f}%", _E, _E, _E])
            table.add_row(['Breadth',
                          f"{base_results.get('breadth', 0):>13.0f}%",
                          f"{nosma_results['breadth']:>13.0f}%",
                          f"{pipeline_results['breadth']:>13.0f}%", _E, _E, _E])

            # --- Score ---
            table.add_row([f"{C.BOLD}--- Score ---{C.RESET}", _E, _E, _E, _E, _E, _E])
            table.add_row([f"{C.BOLD}Score{C.RESET}",
                          f"{C.BOLD}{base_results.get('score', 0):>+14.3f}{C.RESET}",
                          f"{C.BOLD}{nosma_results['score']:>+14.3f}{C.RESET}",
                          f"{C.BOLD}{pipeline_results['score']:>+14.3f}{C.RESET}", _E, _E, _E])

            rendered = "  " + table.render().replace("\n", "\n  ")
            try:
                print(rendered)
            except UnicodeEncodeError:
                print(rendered.encode('ascii', errors='replace').decode('ascii'))
        else:
            # Fallback to old format (5 columns)
            W = 16  # column width
            TW = 10  # trades column width
            print(f"  {'Metric':<22s}  {'Base v3.0':>{W}s}  {'v7.0 (no SMA)':>{W}s}  {'Pipeline v7.0':>{W}s}  {'# Trades':>{TW}s}")
            print(f"  {'='*22}  {'='*W}  {'='*W}  {'='*W}  {'='*TW}")

            # Starting Capital — show per-symbol and total
            _n_syms_fb_tbl = len(_pip_per_sym) if _pip_per_sym else len(datasets)
            _total_cap_fb = cfg.starting_capital * _n_syms_fb_tbl
            print(f"  {'Capital / Symbol':<22s}  "
                  f"${cfg.starting_capital:>{W-1},.0f}  "
                  f"${cfg.starting_capital:>{W-1},.0f}  "
                  f"${cfg.starting_capital:>{W-1},.0f}")
            print(f"  {f'Total Capital ({_n_syms_fb_tbl} syms)':<22s}  "
                  f"${_total_cap_fb:>{W-1},.0f}  "
                  f"${_total_cap_fb:>{W-1},.0f}  "
                  f"${_total_cap_fb:>{W-1},.0f}")

            # --- P&L Breakdown ---
            print(f"  {C.BOLD}{'--- P&L ---':<22s}{C.RESET}")
            base_pnl_c = C.GREEN if base_results.get('total_pnl', 0) > 0 else C.RED
            nosma_pnl_c = C.GREEN if nosma_results['total_pnl'] > 0 else C.RED
            pip_pnl_c = C.GREEN if pipeline_results['total_pnl'] > 0 else C.RED
            print(f"  {'Total P&L':<22s}  "
                  f"{base_pnl_c}${base_results.get('total_pnl', 0):>+13,.2f}{C.RESET}  "
                  f"{nosma_pnl_c}${nosma_results['total_pnl']:>+13,.2f}{C.RESET}  "
                  f"{pip_pnl_c}${pipeline_results['total_pnl']:>+13,.2f}{C.RESET}")
            base_trade_c = C.GREEN if base_trade_pnl > 0 else C.RED
            nosma_trade_c = C.GREEN if nosma_trade_pnl > 0 else C.RED
            pip_trade_c = C.GREEN if pip_trade_pnl > 0 else C.RED
            print(f"  {'  Trade P&L':<22s}  "
                  f"{base_trade_c}${base_trade_pnl:>+13,.2f}{C.RESET}  "
                  f"{nosma_trade_c}${nosma_trade_pnl:>+13,.2f}{C.RESET}  "
                  f"{pip_trade_c}${pip_trade_pnl:>+13,.2f}{C.RESET}")
            print(f"  {'  Cash Yield':<22s}  "
                  f"{C.CYAN}${base_cash_yield:>+13,.2f}{C.RESET}  "
                  f"{C.CYAN}${nosma_cash_yield:>+13,.2f}{C.RESET}  "
                  f"{C.CYAN}${pip_cash_yield:>+13,.2f}{C.RESET}")

            # Per-symbol Trade P&L (cash yield stripped out) + $ Used + CAGR
            _base_per_sym_fb = base_results.get('per_sym', {})
            _all_syms_fb = sorted(set(list(_base_per_sym_fb.keys()) + list(_pip_per_sym.keys())))
            if _all_syms_fb:
                print(f"  {C.BOLD}{'--- Trade P&L by Symbol ---':<22s}{C.RESET}")
                for sym in _all_syms_fb:
                    b_total = _base_per_sym_fb.get(sym, {}).get('pnl', 0)
                    b_cash = _base_per_sym_fb.get(sym, {}).get('cash_yield_pnl', 0) or _base_cash_per_sym.get(sym, 0)
                    b_trade = b_total - b_cash
                    n_total = _nosma_per_sym.get(sym, {}).get('pnl', 0)
                    n_cash = _nosma_per_sym.get(sym, {}).get('cash_yield_pnl', 0)
                    n_trade = n_total - n_cash
                    p_total = _pip_per_sym.get(sym, {}).get('pnl', 0)
                    p_cash = _pip_per_sym.get(sym, {}).get('cash_yield_pnl', 0)
                    p_trade = p_total - p_cash
                    b_c = C.GREEN if b_trade > 0 else C.RED
                    n_c = C.GREEN if n_trade > 0 else C.RED
                    p_c = C.GREEN if p_trade > 0 else C.RED
                    sym_trades = _pip_per_sym.get(sym, {}).get('pipeline_trades', 0)
                    period = _sym_periods.get(sym, None)
                    period_str = f" ({period[0]} to {period[1]})" if period else ""
                    _peak_fb = _pip_per_sym.get(sym, {}).get('peak_notional', 0)
                    _steps_fb = _pip_per_sym.get(sym, {}).get('step_count', 0)
                    if _peak_fb > 0 and _steps_fb > 0:
                        _yrs_fb = _steps_fb / 252
                        _rr_fb = 1 + p_trade / _peak_fb
                        if _rr_fb > 0:
                            _cagr_fb = _rr_fb ** (1 / max(_yrs_fb, 0.01)) - 1
                            cagr_fb_str = f"{_cagr_fb:>+8.1%}"
                        else:
                            cagr_fb_str = "  -100%+"
                        used_fb_str = f"${_peak_fb:>10,.0f}"
                    else:
                        cagr_fb_str = "     N/A"
                        used_fb_str = "       N/A"
                    print(f"  {'  ' + sym + period_str:<22s}  "
                          f"{b_c}${b_trade:>+13,.2f}{C.RESET}  "
                          f"{n_c}${n_trade:>+13,.2f}{C.RESET}  "
                          f"{p_c}${p_trade:>+13,.2f}{C.RESET}  "
                          f"{sym_trades:>10d}  {used_fb_str}  {cagr_fb_str}")

            if has_spy_bench:
                spy_pnl_c = C.GREEN if spy_total_pnl > 0 else C.RED
                _spy_hdr_fb = f"--- SPY Benchmark ({_spy_period_str}) ---" if _spy_period_str else "--- SPY Benchmark ---"
                print(f"  {C.BOLD}{_spy_hdr_fb}{C.RESET}")
                print(f"  {'  SPY Buy & Hold P&L':<22s}  "
                      f"{spy_pnl_c}${spy_total_pnl:>+13,.2f}{C.RESET}  "
                      f"{spy_pnl_c}${spy_total_pnl:>+13,.2f}{C.RESET}  "
                      f"{spy_pnl_c}${spy_total_pnl:>+13,.2f}{C.RESET}")
                print(f"  {'  SPY CAGR':<22s}  "
                      f"{spy_cagr_str:>14s}  {spy_cagr_str:>14s}  {spy_cagr_str:>14s}")

            # --- Trading Activity ---
            print(f"  {C.BOLD}{'--- Trading ---':<22s}{C.RESET}")
            print(f"  {'Trades Executed':<22s}  "
                  f"{base_n_trades:>14d}  "
                  f"{nosma_n_trades:>14d}  "
                  f"{pip_n_trades:>14d}")

            base_n_suppressed = 0
            print(f"  {'Trades Suppressed':<22s}  "
                  f"{base_n_suppressed:>14d}  "
                  f"{nosma_n_suppressed:>14d}  "
                  f"{pip_n_suppressed:>14d}")
            print(f"  {'Suppression Rate':<22s}  "
                  f"{0.0:>13.1f}%  "
                  f"{nosma_suppression_pct:>13.1f}%  "
                  f"{pip_suppression_pct:>13.1f}%")

            base_wr_str = f"{base_win_rate:>13.1f}%" if not np.isnan(base_win_rate) else "      N/A (no closed trades)"
            pip_wr_str = f"{pip_win_rate:>13.1f}%" if not np.isnan(pip_win_rate) else f"      N/A ({pip_n_closed_trades} closed)"

            print(f"  {'Win Rate':<22s}  {base_wr_str}  {'':>14s}  {pip_wr_str}")

            base_pf_str = f"{base_profit_factor:>14.2f}" if not np.isnan(base_profit_factor) else "           N/A"
            pip_pf_str = f"{pip_profit_factor:>14.2f}" if not np.isnan(pip_profit_factor) else "           N/A"

            print(f"  {'Profit Factor':<22s}  {base_pf_str}  {'':>14s}  {pip_pf_str}")

            # --- Risk ---
            print(f"  {C.BOLD}{'--- Risk ---':<22s}{C.RESET}")
            print(f"  {'Sharpe':<22s}  "
                  f"{base_results.get('avg_sh', 0):>+14.3f}  "
                  f"{nosma_results['avg_sh']:>+14.3f}  "
                  f"{pipeline_results['avg_sh']:>+14.3f}")
            print(f"  {'Max Drawdown':<22s}  "
                  f"{base_results.get('dd_max', 0):>13.1f}%  "
                  f"{nosma_results['dd_max']:>13.1f}%  "
                  f"{pipeline_results['dd_max']:>13.1f}%")
            print(f"  {'Breadth':<22s}  "
                  f"{base_results.get('breadth', 0):>13.0f}%  "
                  f"{nosma_results['breadth']:>13.0f}%  "
                  f"{pipeline_results['breadth']:>13.0f}%")

            # --- Score ---
            print(f"  {'='*22}  {'='*14}  {'='*14}  {'='*14}")
            print(f"  {C.BOLD}{'Score':<22s}  "
                  f"{base_results.get('score', 0):>+14.3f}  "
                  f"{nosma_results['score']:>+14.3f}  "
                  f"{pipeline_results['score']:>+14.3f}{C.RESET}")

        # Add explanatory note if pipeline has 0 closed trades but positive Trade P&L
        # Also show detailed trade entry breakdown
        if pip_n_closed_trades == 0 and abs(pip_trade_pnl) > 1.0:
            print(f"\n  {C.YELLOW}Note: Win Rate/PF = N/A because pipeline has open position with unrealized P&L.{C.RESET}")
            print(f"        Executed trades: {pip_n_trades} | Closed round-trips: {pip_n_closed_trades}")
            print(f"        Trade P&L (${pip_trade_pnl:,.2f}) includes unrealized gains/losses.")

            # Show detailed trade entries (BUY/SELL actions)
            trade_entries = []
            for sym_data in _pip_per_sym.values():
                entries = sym_data.get('trade_entries', [])
                trade_entries.extend(entries)

            if trade_entries:
                buys = sum(1 for e in trade_entries if e['action'] in ['BUY', 'BUY_COVER'])
                sells = sum(1 for e in trade_entries if e['action'] in ['SELL', 'SELL_SHORT'])
                total_realized = sum(e.get('realized_pnl', 0) for e in trade_entries)
                total_cost = sum(e.get('cost', 0) for e in trade_entries)

                print(f"\n  {C.BOLD}Trade Entry Breakdown:{C.RESET}")
                print(f"        Total entries: {len(trade_entries)} ({buys} BUY, {sells} SELL)")
                print(f"        Realized P&L: ${total_realized:,.2f}")
                print(f"        Total costs: ${total_cost:,.2f}")

                # Show last 5 trades
                if len(trade_entries) > 0:
                    print(f"\n        Last {min(5, len(trade_entries))} trade entries:")
                    for entry in trade_entries[-5:]:
                        action_color = C.GREEN if entry['action'] in ['BUY', 'BUY_COVER'] else C.RED
                        print(f"          Bar {entry['bar']:>4}: {action_color}{entry['action']:<10}{C.RESET} "
                              f"{entry['shares']:>6.0f} @ ${entry['price']:>7.2f}  "
                              f"Pos: {entry['position_before']:>+6.0f} -> {entry['position_after']:>+6.0f}  "
                              f"P&L: ${entry.get('realized_pnl', 0):>+8.2f}")

    # **********************************************************************
    # STEP 7.5: Terminal charts
    # **********************************************************************
    print_section("CHARTS")
    tprint("Visual summaries — quick-glance performance charts.", "info")

    # (a) Per-symbol Trade P&L horizontal bars (green/red)
    _chart_per_sym = pipeline_results.get('per_sym', {})
    _chart_syms = sorted(_chart_per_sym.keys())
    if _chart_syms:
        _sym_pnls = []
        for sym in _chart_syms:
            s = _chart_per_sym[sym]
            trade_pnl = s.get('pnl', 0) - s.get('cash_yield_pnl', 0)
            _sym_pnls.append((sym, trade_pnl))
        hbar_chart(_sym_pnls, title="Per-Symbol Trade P&L (Pipeline v7.0)")

    # (b) Sector performance horizontal bars (green/red)
    if _chart_per_sym:
        _sector_pnls = []
        for sector, tickers in SECTOR_MAP.items():
            total = sum(
                _chart_per_sym.get(t, {}).get('pnl', 0) - _chart_per_sym.get(t, {}).get('cash_yield_pnl', 0)
                for t in tickers if t in _chart_per_sym
            )
            if total != 0:
                _sector_pnls.append((sector.replace("_", " ").title(), total))
        if _sector_pnls:
            _sector_pnls.sort(key=lambda x: x[1], reverse=True)
            hbar_chart(_sector_pnls, title="Sector Performance (Pipeline v7.0)")

    # (c) Cumulative % return chart — equal-weight portfolio vs SPY
    # Each symbol gets $10k. We average per-symbol daily returns by date
    # (equal-weight portfolio), then cumprod to get cumulative % return.
    # SPY: cumprod of its daily returns over the same dates.
    # Both start at 0% — directly comparable regardless of dollar amounts.
    _date_rets = pipeline_results.get("per_symbol_date_returns", {})
    if _date_rets:
        from collections import defaultdict
        _by_date = defaultdict(list)
        for _pairs in _date_rets.values():
            for _dt, _ret in _pairs:
                _by_date[_dt].append(_ret)
        _dates_sorted = sorted(_by_date.keys())

        if len(_dates_sorted) > 2:
            # Equal-weight portfolio return per date (average across active symbols)
            _port_rets = np.array([np.mean(_by_date[_dt]) for _dt in _dates_sorted])
            _port_cum_pct = (np.cumprod(1 + _port_rets) - 1) * 100  # cumulative %

            # SPY cumulative % over the same dates
            _spy_cum_pct = None
            if has_spy_bench and spy_returns_lookup:
                _spy_rets = np.array([spy_returns_lookup.get(_dt, 0.0) for _dt in _dates_sorted])
                _spy_cum_pct = (np.cumprod(1 + _spy_rets) - 1) * 100

            # Compute summary values for annotation
            _n_syms_chart = len(_date_rets)
            _invested = cfg.starting_capital * _n_syms_chart
            _port_final_pct = float(_port_cum_pct[-1])
            _port_return_dollar = _invested * _port_final_pct / 100

            if _spy_cum_pct is not None:
                dual_line_chart(_port_cum_pct, _spy_cum_pct, width=70, height=14,
                               title="Cumulative Return: Portfolio vs SPY (equal $10k base)",
                               label1="Pipeline v7.0", label2="SPY", fmt="%")
                _spy_final_pct = float(_spy_cum_pct[-1])
                _spy_return_dollar = cfg.starting_capital * _spy_final_pct / 100
                # Summary: % return (Return $ / Invested $)
                _p_c = C.GREEN if _port_final_pct > 0 else C.RED
                _s_c = C.GREEN if _spy_final_pct > 0 else C.RED
                print(f"    Pipeline v7.0: {_p_c}{_port_final_pct:+.1f}%{C.RESET}"
                      f" (${_port_return_dollar:+,.0f} / ${_invested:,.0f})"
                      f"    SPY: {_s_c}{_spy_final_pct:+.1f}%{C.RESET}"
                      f" (${_spy_return_dollar:+,.0f} / ${cfg.starting_capital:,.0f})")
            else:
                line_chart(_port_cum_pct, width=70, height=14,
                          title="Cumulative Return: Portfolio ($10k/symbol)",
                          fmt="%")
                _p_c = C.GREEN if _port_final_pct > 0 else C.RED
                print(f"    Portfolio: {_p_c}{_port_final_pct:+.1f}%{C.RESET}"
                      f" (${_port_return_dollar:+,.0f} / ${_invested:,.0f})")

    # (d) Training loss chart (only when training was run)
    if training_hist:
        _vl_values = []
        for h in training_hist:
            if isinstance(h, dict):
                vl = h.get("vl", 0)
                if vl == 0:
                    um = h.get("update_metrics", {})
                    vl = um.get("vl", 0) if isinstance(um, dict) else 0
                _vl_values.append(float(vl))
        if any(v != 0 for v in _vl_values):
            line_chart(_vl_values, width=70, height=10,
                      title="Value Loss vs Iteration", fmt="f")

    # **********************************************************************
    # STEP 8: Pipeline diagnostics
    # **********************************************************************
    print_section("DIAGNOSTICS")
    tprint("Under the hood: how many trades were attempted, blocked, or modified by safety filters.", "info")
    # Use lifetime stats for cross-symbol aggregates
    lt = pipeline.execution_engine.lifetime_stats
    total_decisions = lt['n_trades'] + lt['n_suppressed']

    tprint(f"Risk ownership:       L4 ExecutionEngine (env kill switches OFF)", "info")
    tprint(f"Total bar decisions:  {total_decisions}", "info")
    tprint(f"Trades executed:      {lt['n_trades']}", "ok")
    tprint(f"Trades suppressed:    {lt['n_suppressed']} "
           f"({lt['suppression_rate']:.0%} -- no-trade region)", "info")

    # Kill status from the last episode (most recent symbol)
    ep = pipeline.execution_engine.stats
    tprint(f"Kill switch fired:    "
           f"{'YES -- ' + ep['kill_reason'] if ep['kill_triggered'] else 'No (last symbol)'}",
           "info")

    # -- Execution quality monitoring --
    # Aggregate fill quality from per-symbol data
    _per_sym = pipeline_results["per_sym"]
    total_commission = sum(
        s.get("pipeline_total_commission", 0) for s in _per_sym.values()
    )
    all_avg_slips = [
        s["pipeline_avg_slippage_bps"] for s in _per_sym.values()
        if s.get("pipeline_fills", 0) > 0
    ]
    all_worst_slips = [
        s["pipeline_worst_slippage_bps"] for s in _per_sym.values()
        if s.get("pipeline_fills", 0) > 0
    ]
    if all_avg_slips:
        tprint(f"Avg slippage:         {np.mean(all_avg_slips):.2f} bps "
               f"(worst fill: {max(all_worst_slips):.2f} bps)", "info")
    tprint(f"Total commission:     ${total_commission:,.2f}", "info")

    # -- Future leak guard audit --
    leak_log = FutureLeakGuard.get_audit_log()
    tprint(f"Future leak checks:   {len(leak_log)} (all passed)", "ok")

    # -- Alpha lifecycle report (Good-to-have A) --
    lifecycle = pipeline.get_lifecycle_report()
    if lifecycle:
        print(f"\n  {C.BOLD}Alpha Lifecycle Health:{C.RESET}")
        print(f"  {C.DIM}How healthy is each strategy? Like a doctor's checkup for your trading signals.{C.RESET}")
        if HAS_TABLE_FORMATTER:
            table = TableFormatter(title="ALPHA LIFECYCLE HEALTH")
            table.add_column('Alpha', align='left')
            table.add_column('HitRate', align='right')
            table.add_column('AvgWt', align='right', format_spec='.1%')
            table.add_column('WtVol', align='right', format_spec='.3f')
            table.add_column('Turnover', align='right', format_spec='.1%')
            table.add_column('SinceGood', align='right')

            for name, health in lifecycle.items():
                # Color hit rate: Green if >= 50%, Red if < 48%
                hr = health['hit_rate']
                if hr >= 0.50:
                    hr_str = f"{C.GREEN}{hr:6.1%}{C.RESET}"
                elif hr < 0.48:
                    hr_str = f"{C.RED}{hr:6.1%}{C.RESET}"
                else:
                    hr_str = f"{hr:6.1%}"

                table.add_row([
                    name,
                    hr_str,  # Formatted string with color
                    health['avg_weight'],
                    health['weight_vol'],
                    health['turnover_share'],
                    health['bars_since_good']
                ])

            rendered = "  " + table.render().replace("\n", "\n  ")
            try:
                print(rendered)
            except UnicodeEncodeError:
                print(rendered.encode('ascii', errors='replace').decode('ascii'))
            print(f"  {C.DIM}HitRate=% correct direction | AvgWt=how much influence it has | "
                  f"WtVol=weight stability{C.RESET}")
            print(f"  {C.DIM}Turnover=how often it changes its mind | "
                  f"SinceGood=bars since last good prediction{C.RESET}")
        else:
            # Fallback to old format
            print(f"  {'Alpha':<26s}  {'HitRate':>7s}  {'AvgWt':>6s} {'WtVol':>6s}  "
                  f"{'Turnover':>8s}  {'SinceGood':>9s}")
            print(f"  {'-'*26}  {'-'*7}  {'-'*6} {'-'*6}  {'-'*8}  {'-'*9}")
            for name, health in lifecycle.items():
                hr_color = C.GREEN if health['hit_rate'] >= 0.50 else (
                    C.RED if health['hit_rate'] < 0.48 else C.RESET
                )
                decay_warn = " " if health['bars_since_good'] > 50 else ""
                print(f"  {name:<26s}  {hr_color}{health['hit_rate']:>6.1%}{C.RESET}  "
                      f"{health['avg_weight']:>6.1%} {health['weight_vol']:>6.3f}  "
                      f"{health['turnover_share']:>7.1%}  "
                      f"{health['bars_since_good']:>8d}{decay_warn}")

    # -- Attribution report (Good-to-have C) --
    attr = pipeline.get_attribution_report()
    if attr.get('n_bars', 0) > 0:
        print(f"\n  {C.BOLD}PnL Attribution (paper vs realized):{C.RESET}")
        print(f"  {C.DIM}Where did the money go? Breaking down profit into: raw signals, costs, and filters.{C.RESET}")
        print(f"  Gross alpha PnL:      {attr['gross_alpha_pnl']:>+.6f}")
        print(f"  Realized PnL:         {attr['realized_pnl']:>+.6f}")
        print(f"  Cost drag:            {attr['cost_drag']:>+.6f}")
        print(f"  Suppression cost:     {attr['suppression_cost']:>+.6f}")
        print(f"  Kill cost:            {attr['kill_cost']:>+.6f}")
        print(f"  Discretization gap:   {attr['discretization_gap']:>+.6f}")
        eff = attr['implementation_efficiency']
        eff_color = C.GREEN if eff > 0.5 else C.RED
        print(f"  Implementation eff:   {eff_color}{eff:>+.1%}{C.RESET}")

    # -- v7.0 WS5A: Alpha Correlation Report --
    try:
        corr_report = pipeline.get_correlation_report()
        if corr_report:
            n_eff = corr_report.get('n_eff_independent', 0)
            n_eff_color = C.GREEN if n_eff > 2.0 else C.RED
            tprint(f"Alpha diversity:      N_eff={n_eff_color}{n_eff:.1f}{C.RESET} "
                   f"independent signals", "info")
    except AttributeError:
        pass

    # -- v7.0 WS1A: Walk-Forward Validation Summary --
    if wf_validation and wf_validation.get('per_alpha'):
        per_a = wf_validation['per_alpha']
        n_pass = sum(1 for r in per_a.values() if r.get('verdict') == 'PASS')
        n_total = len(per_a)
        mt = wf_validation.get('multiple_testing', {})
        badge_color = C.GREEN if n_pass == n_total else (
            C.YELLOW if n_pass > 0 else C.RED
        )
        tprint(f"WF validation:        {badge_color}{n_pass}/{n_total} alphas passed{C.RESET} "
               f"({mt.get('method', 'N/A')}, "
               f"{wf_validation.get('n_folds', 0)} folds)", "info")

    # -- v7.0 WS5C: Alert Summary --
    try:
        alert_counts = pipeline.get_alert_counts()
        if alert_counts:
            total_alerts = sum(alert_counts.values())
            tprint(f"Alerts:               "
                   f"INFO={alert_counts.get('INFO', 0)} "
                   f"WARN={alert_counts.get('WARNING', 0)} "
                   f"CRIT={alert_counts.get('CRITICAL', 0)} "
                   f"FATAL={alert_counts.get('FATAL', 0)}", "info")
            if acfg.alert_log_file and total_alerts > 0:
                tprint(f"Alert log:            {acfg.alert_log_file} "
                       f"({total_alerts} entries)", "info")
    except AttributeError:
        pass

    # **********************************************************************
    # STEP 9: Save results
    # **********************************************************************
    output_dir = cfg.output_dir
    results_path = os.path.join(output_dir, "results_v7.json")

    save_data = {k: v for k, v in pipeline_results.items()
                 if k not in ("per_sym", "audit_log_size")}
    save_data["version"] = "v7.0"
    save_data["architecture"] = "L1->L2->L3->L4"
    save_data["active_alphas"] = pipeline.alpha_factory.alpha_names
    save_data["timeframes"] = list(tfs)
    save_data["execution_stats"] = pipeline.execution_engine.lifetime_stats
    save_data["per_symbol"] = pipeline_results["per_sym"]
    save_data["alpha_lifecycle"] = pipeline.get_lifecycle_report()
    save_data["attribution"] = pipeline.get_attribution_report()

    # Persist L0 quality results so they survive terminal scrolling
    if quality_results:
        from dataclasses import asdict
        save_data["l0_data_quality"] = [
            {
                "symbol": meta.symbols[0] if meta.symbols else "?",
                "timeframe": meta.timeframe,
                "start": str(meta.loaded_start)[:10],
                "end": str(meta.loaded_end)[:10],
                "bars": q.received_bars,
                "score": round(q.score_0_100, 1),
                "verdict": q.verdict,
                "ohlc_violations": q.ohlc_violations,
                "stale_max_run": q.stale_close_runs_max,
                "extreme_returns": q.extreme_return_count,
            }
            for meta, q in quality_results
        ]

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    tprint(f"Results saved: {results_path}", "ok")

    if HAS_TORCH and torch.cuda.is_available():
        tprint(f"Peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB", "gpu")

    print_box(
        "COMPLETE -- ALPHA-TRADE v7.0",
        f"Score:{pipeline_results['score']:+.3f} | "
        f"Alphas:{pipeline.alpha_factory.n_alphas} | "
        f"Suppression:{lt['suppression_rate']:.0%} | "
        f"TFs:{','.join(tfs)}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Terminated by user.{C.RESET}")
    except SystemExit:
        pass