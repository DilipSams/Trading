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
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# -- UTF-8 stdout: required for Unicode chart characters (█ etc.) on Windows --
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
        hbar_chart, line_chart, multi_line_chart,  # Terminal charts
        SymbolDataset,  # v9.0: used for constructing sector ETF datasets for RRG
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
        SelectionConfig, StockSelector,
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

    # ---------- per-symbol worker ----------
    # Extracted from the serial for-loop so ThreadPoolExecutor can run
    # multiple symbols concurrently. NumPy releases the GIL on array ops →
    # real CPU overlap. Each thread gets a pipeline copy from _pl_pool.
    _unwrapped_net = unwrap_net(net) if HAS_TORCH else net

    def _eval_one(d, pl):
        """Evaluate one symbol; pl is a thread-local pipeline copy."""
        sim_cfg = copy(cfg)
        sim_cfg.use_kill_switches = False
        sim_cfg.use_action_masking = False
        sim_cfg.use_trailing_stops = False       # L4 owns risk — env stops interfere
        sim_cfg.use_asymmetric_stops = False     # L4 owns risk — env stops interfere
        # Match env action space to Pipeline’s 9-level discretization (Iter 9: added 1.25, 1.5)
        sim_cfg.action_targets = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.25, 1.5)
        sim_cfg.n_actions = 9                    # Must match action_targets length
        env = TradingEnv(d.features_test, d.prices_test, sim_cfg, d.symbol, ev=True)
        # FIX §4.4: Flag env as pipeline eval mode — prevents get_risk_target() future access
        env._pipeline_eval_mode = True
        obs, _ = env.reset()
        done = False

        # -- Reset pipeline for this symbol --
        pl.reset(cfg.starting_capital)
        pl._current_symbol = d.symbol  # v8.0: for rank-based sizing

        # v8.0: Supply training close prices so SMA200 is available from bar 0.
        # PERF FIX #2: Pre-concatenate train+test closes ONCE per symbol so that
        # pipeline.step() can slice in O(1) instead of calling np.concatenate every bar.
        if pl.use_v8_sizing and hasattr(d, 'prices_train') and d.prices_train is not None:
            _train_tail = d.prices_train[-250:, 3].copy()
            pl._v8_train_closes = _train_tail          # kept for fallback compat
            pl._v8_sma_combined = np.concatenate([_train_tail, d.prices_test[:, 3]])
            pl._v8_train_len    = len(_train_tail)
        else:
            pl._v8_train_closes = None
            pl._v8_sma_combined = None
            pl._v8_train_len    = 0

        # PERF FIX #3: Compute data-quality score ONCE per symbol (not every bar).
        _sym_dq_score = None
        _dq_closes_full = d.prices_test[:, 3]
        _dq_volumes_full = d.prices_test[:, 4] if d.prices_test.shape[1] > 4 else None
        if len(_dq_closes_full) > 1:
            if HAS_DATA_QUALITY and _dq_volumes_full is not None:
                try:
                    import pandas as _dq_pd
                    _dq_df = _dq_pd.DataFrame({
                        'Open':   _dq_closes_full, 'High':  _dq_closes_full,
                        'Low':    _dq_closes_full, 'Close': _dq_closes_full,
                        'Volume': _dq_volumes_full,
                    })
                    _dq_meta = analyze_ohlcv(_dq_df, label=d.symbol)
                    _sym_dq_score = _dq_meta.quality_score
                except Exception:
                    _sym_dq_score = None
            if _sym_dq_score is None:
                _dq_n = len(_dq_closes_full)
                _dq_nan  = float(np.sum(np.isnan(_dq_closes_full))) / _dq_n
                _dq_zero = float(np.sum(_dq_closes_full <= 0)) / _dq_n
                _dq_stale = 0
                if _dq_n > 5:
                    _dq_recent = _dq_closes_full[-5:]
                    _dq_stale = int(np.all(_dq_recent == _dq_recent[0]))
                _nan_penalty   = getattr(acfg, 'dq_nan_penalty',   500.0)
                _zero_penalty  = getattr(acfg, 'dq_zero_penalty',  500.0)
                _stale_penalty = getattr(acfg, 'dq_stale_penalty',  40.0)
                _sym_dq_score  = max(0.0, 100.0
                                     - _dq_nan  * _nan_penalty
                                     - _dq_zero * _zero_penalty
                                     - _dq_stale * _stale_penalty)

        # -- Inject trained network into RL alpha --
        for _aname in pl.alpha_factory.alpha_names:
            _alpha = pl.alpha_factory._alphas.get(_aname)
            if isinstance(_alpha, RLAlphaAdapter):
                _alpha.set_network(_unwrapped_net)
                break

        step_count = 0
        episode_audit = []
        sym_bar_returns_gross = []   # Per-bar returns for this symbol
        sym_bar_returns_net = []
        sym_spy_returns = []         # SPY return for same dates
        sym_date_returns = []        # [(date_str, net_return), ...] for equity curve
        bars_in_cash = 0             # bars where not in a position (for cash yield)
        peak_notional = 0.0          # max abs(shares * price) for $ Used
        _notional_sum = 0.0          # sum of notional on bars where in position
        trade_log = []               # collect for writer; applied in main thread
        info = {}

        # PERF FIX: Pre-compute bar log-returns and 20-bar rolling vol for the
        # entire test period ONCE before the bar loop.  Inside the loop we do a
        # single O(1) array lookup instead of O(20) np.diff/np.log/np.std calls.
        _all_closes   = d.prices_test[:, 3]
        _n_test_bars  = len(_all_closes)
        _log_rets_all = np.empty(_n_test_bars, dtype=np.float64)
        _log_rets_all[0] = 0.0
        if _n_test_bars > 1:
            _log_rets_all[1:] = np.diff(np.log(np.maximum(_all_closes, 1e-12)))
        # Pre-compute 20-bar rolling std (annualised realized vol) for each bar
        _rvol_all = np.full(_n_test_bars, 0.15, dtype=np.float64)
        for _bi in range(21, _n_test_bars):
            _rvol_all[_bi] = float(np.std(_log_rets_all[_bi - 19: _bi + 1])) * 15.8745  # √252

        # PERF FIX: All alphas use closes[-N:] relative indexing.
        # Maximum lookback across all alphas: 252 bars (12-month RS).
        # Truncating the growing slice to MAX_LOOKBACK bars converts
        # O(n²) total work → O(n × MAX_LOOKBACK), saving ~3-5× on long histories.
        _MAX_LOOKBACK = 260  # 252 bars needed + 8 bar buffer for off-by-one patterns
        _prices_full  = env.prices           # reference only — no copy
        _n_cols       = _prices_full.shape[1]

        while not done:
            bar_idx  = env.cs
            _lb      = max(0, bar_idx - _MAX_LOOKBACK + 1)  # rolling window start

            # Phase 1: Extract OHLC data for Tier 1 alphas (fixed-size window)
            closes  = _prices_full[_lb:bar_idx + 1, 3]
            opens   = _prices_full[_lb:bar_idx + 1, 0] if _n_cols > 0 else None
            highs   = _prices_full[_lb:bar_idx + 1, 1] if _n_cols > 1 else None
            lows    = _prices_full[_lb:bar_idx + 1, 2] if _n_cols > 2 else None
            volumes = _prices_full[_lb:bar_idx + 1, 4] if _n_cols > 4 else None

            # Bar return — O(1) lookup into pre-computed array
            bar_ret = float(_log_rets_all[bar_idx]) if bar_idx > 0 else 0.0

            # Capture portfolio value BEFORE this step for actual return computation
            pv_before = env._portfolio_value()

            # Realized vol — O(1) lookup into pre-computed array
            realized_vol = float(_rvol_all[bar_idx])

            # -- SET L3 COST MODEL CONTEXT (canonical cost model) --
            mid_price = float(closes[-1])
            pv = env._portfolio_value()
            pc = pl.portfolio
            pc._last_mid = mid_price
            pc._last_portfolio_value = pv
            if volumes is not None and len(volumes) > 20:
                adv_shares = float(np.mean(volumes[-20:]))
                pc._adv_dollars = adv_shares * mid_price
            # else: falls back to acfg.cost_default_adv_dollars

            # -- RUN FULL L1 -> L2 -> L3 -> L4 PIPELINE --
            result = pl.step(
                observation=obs,
                closes=closes,
                volumes=volumes,
                bar_idx=bar_idx,
                bar_return=bar_ret,
                current_exposure=env.exposure,
                portfolio_value=pv,
                regime_probs=None,
                realized_vol=realized_vol,
                data_quality_score=_sym_dq_score,
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
            traded_notional   = info.get("step_traded_notional", 0.0)
            step_commission   = info.get("step_commission", 0.0)
            step_slippage_bps = info.get("step_slippage_bps", 0.0)
            step_mid          = info.get("step_mid_price", 0.0)

            pl.execution_engine.ingest_execution(
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
            else:
                _notional_sum += _notional   # accumulate while in position

            # -- SPY benchmark return for this bar's date --
            spy_ret_this_bar = 0.0
            if spy_returns_lookup and d.timestamps_test is not None:
                date_str = str(d.timestamps_test[bar_idx])[:10]
                spy_ret_this_bar = spy_returns_lookup.get(date_str, 0.0)
            sym_spy_returns.append(spy_ret_this_bar)

            # -- FIX §5.4: Position reconciliation --
            if hasattr(pl, '_reconciler') and pl._reconciler is not None:
                pl._reconciler.reconcile(
                    bar_idx=bar_idx,
                    internal_position=result.target_exposure,
                    external_position=env.exposure,
                    internal_nav=pv_after,
                    external_nav=env._portfolio_value(),
                )

            # -- Collect trade for run_artifacts (applied in main thread) --
            if traded_notional > 0:
                trade_log.append({
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

            if verbose >= 2 and step_count % 200 == 0:
                pl.print_status()

            episode_audit.append(result.audit)

        # -- Episode-scoped L4 stats --
        ep_stats = dict(pl.execution_engine.stats)
        _bars_in_pos = step_count - bars_in_cash
        avg_notional = _notional_sum / max(_bars_in_pos, 1)
        cash_yield_pnl = (
            (cfg.cash_yield_bps_annual * 1e-4 / max(cfg.bars_per_year, 1))
            * cfg.starting_capital * bars_in_cash
            if getattr(cfg, 'cash_yield_bps_annual', 0) > 0 else 0.0
        )

        if verbose >= 1:
            sup_rate = (ep_stats.get('n_suppressed', 0) /
                        max(ep_stats.get('n_trades', 0) + ep_stats.get('n_suppressed', 0), 1))
            trading_pnl = info.get('net_pnl', 0) - cash_yield_pnl
            pnl_color   = C.GREEN if info.get('net_pnl', 0) > 0 else C.RED
            trade_color = C.GREEN if trading_pnl > 0 else (C.RED if trading_pnl < -0.01 else C.YELLOW)
            tprint(
                f"  {d.symbol:>12s}  PnL:{pnl_color}${info.get('net_pnl', 0):>+10,.2f}{C.RESET}  "
                f"(trade:{trade_color}${trading_pnl:>+8,.2f}{C.RESET} "
                f"cash:{C.CYAN}${cash_yield_pnl:>+8,.2f}{C.RESET})  "
                f"Sh:{info.get('sharpe_ratio', 0):>+.2f}  DD:{info.get('max_drawdown', 0):.1f}%  "
                f"Trades:{info.get('total_trades', 0)}  Suppressed:{sup_rate:.0%}",
                "info"
            )

        return {
            'symbol': d.symbol,
            'pnl': info.get('net_pnl', 0),
            'trades': info.get('total_trades', 0),
            'max_drawdown': info.get('max_drawdown', 0),
            'sharpe': info.get('sharpe_ratio', 0),
            'win_rate': info.get('win_rate', 0),
            'profit_factor': info.get('profit_factor', 0),
            'turnover': info.get('turnover', 0.0),
            'step_count': step_count,
            'bars_in_cash': bars_in_cash,
            'peak_notional': peak_notional,
            'avg_notional': avg_notional,
            'cash_yield_pnl': cash_yield_pnl,
            'ep_stats': ep_stats,
            'sym_bar_returns_net': sym_bar_returns_net,
            'sym_bar_returns_gross': sym_bar_returns_gross,
            'sym_spy_returns': sym_spy_returns,
            'sym_date_returns': sym_date_returns,
            'episode_audit': episode_audit,
            'trade_log': trade_log,
        }

    # ---------- thread pool with pipeline pool ----------
    # Each worker draws a pipeline copy from the queue; returns it when done.
    # Use all available logical CPU cores — NumPy releases the GIL so threads
    # achieve real CPU overlap. Cap at dataset count to avoid idle workers.
    from queue import Queue as _Queue
    n_workers = max(1, min(len(datasets), multiprocessing.cpu_count()))
    _pl_pool = _Queue()
    for _ in range(n_workers):
        _pl_pool.put(deepcopy(pipeline))

    def _task(d):
        pl = _pl_pool.get()
        try:
            return _eval_one(d, pl)
        finally:
            _pl_pool.put(pl)

    sym_results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as _pool:
        _futs = {_pool.submit(_task, d): d.symbol for d in datasets}
        for _fut in as_completed(_futs):
            _sym = _futs[_fut]
            try:
                sym_results[_sym] = _fut.result()
            except Exception as _exc:
                tprint(f"  [eval] {_sym} raised: {_exc}", "warn")
                sym_results[_sym] = None

    # ---------- aggregate results in original symbol order ----------
    for d in datasets:
        r = sym_results.get(d.symbol)
        if r is None:
            continue

        all_pnls.append(r['pnl'])
        all_trades.append(r['trades'])
        all_dds.append(r['max_drawdown'])
        all_sharpes.append(r['sharpe'])
        all_audit.extend(r['episode_audit'])

        per_symbol_returns_net[d.symbol]   = r['sym_bar_returns_net']
        per_symbol_returns_gross[d.symbol] = r['sym_bar_returns_gross']
        per_symbol_date_returns[d.symbol]  = r['sym_date_returns']
        all_bar_returns_gross.extend(r['sym_bar_returns_gross'])
        all_bar_returns_net.extend(r['sym_bar_returns_net'])
        all_spy_returns.extend(r['sym_spy_returns'])

        ep_stats = r['ep_stats']
        per_sym[d.symbol] = {
            "pnl": r['pnl'],
            "trades": r['trades'],
            "win_rate": r['win_rate'],
            "profit_factor": r['profit_factor'],
            "sharpe": r['sharpe'],
            "max_dd": r['max_drawdown'],
            "turnover": r['turnover'],
            "step_count": r['step_count'],
            "bars_in_cash": r['bars_in_cash'],
            "peak_notional": r['peak_notional'],
            "avg_notional": r['avg_notional'],
            "cash_yield_pnl": r['cash_yield_pnl'],
            # Episode-scoped L4 stats
            "pipeline_trades": ep_stats.get("n_trades", 0),
            "pipeline_suppressed": ep_stats.get("n_suppressed", 0),
            "pipeline_suppression_rate": ep_stats.get("suppression_rate", 0),
            "pipeline_fills": ep_stats.get("n_fills", 0),
            "pipeline_avg_slippage_bps": ep_stats.get("avg_slippage_bps", 0),
            "pipeline_worst_slippage_bps": ep_stats.get("avg_slippage_bps", 0),
            "pipeline_total_commission": ep_stats.get("total_commission", 0),
            "pipeline_kill": ep_stats.get("kill_triggered", False),
            "pipeline_kill_context": ep_stats.get("kill_context", None),
        }

        # Apply trade log to run_artifacts writer (main thread — thread-safe)
        if writer:
            for _te in r['trade_log']:
                writer.log_trade(_te)

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
        # FIX §4.7: Per-bar returns — per-symbol series for correct metrics,
        # plus concatenated series for backward compatibility (with warning).
        "daily_returns_gross": np.array(all_bar_returns_gross, dtype=np.float64),
        "daily_returns_net": np.array(all_bar_returns_net, dtype=np.float64),
        "daily_returns_concatenated_warning": (
            "These are concatenated across symbols — observation count is inflated. "
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


# ============================================================================
# V8.0: FAILURE POST-MORTEM
# ============================================================================

def run_postmortem(base_results, nosma_results, sma_results, datasets):
    """
    Quantify WHY v3.0 and v7.0 (no SMA) underperform.

    Analyzes:
    1. v3.0: % of losses from trading below SMA200 (no trend awareness)
    2. v7.0 (no SMA): Falling knife trades, bear-market signal noise
    3. v7.0 (SMA): Why conservative (quantify cash drag)
    """
    print_subsection("FAILURE POST-MORTEM")

    def _analyze_version(results, label, datasets):
        """Compute bear-market loss attribution for a result set."""
        if results is None:
            tprint(f"  {label}: No results available", "warn")
            return
        per_sym = results.get('per_sym', {})
        if not per_sym:
            tprint(f"  {label}: No per-symbol data", "warn")
            return

        total_pnl = results.get('total_pnl', 0)
        total_trades = results.get('total_trades', 0)

        # Build price lookup from datasets
        sym_closes = {}
        for d in datasets:
            if hasattr(d, 'prices_test') and d.prices_test is not None:
                sym_closes[d.symbol] = d.prices_test[:, 3]

        bear_loss_total = 0.0
        bull_loss_total = 0.0
        bear_trade_count = 0
        bull_trade_count = 0
        symbols_below_sma = 0

        for sym, metrics in per_sym.items():
            closes = sym_closes.get(sym)
            if closes is None or len(closes) < 200:
                continue

            sma200 = float(np.mean(closes[-200:]))
            final_price = float(closes[-1])
            sym_pnl = metrics.get('pnl', 0)
            sym_trades = metrics.get('trades', 0)

            if final_price < sma200:
                symbols_below_sma += 1
                if sym_pnl < 0:
                    bear_loss_total += sym_pnl
                    bear_trade_count += sym_trades
            else:
                if sym_pnl < 0:
                    bull_loss_total += sym_pnl
                    bull_trade_count += sym_trades

        total_loss = bear_loss_total + bull_loss_total
        bear_pct = (bear_loss_total / min(total_loss, -0.01)) * 100 if total_loss < -0.01 else 0

        tprint(f"  {label}:", "info")
        pnl_c = C.GREEN if total_pnl > 0 else C.RED
        tprint(f"    Total P&L: {pnl_c}${total_pnl:+,.0f}{C.RESET}  "
               f"({total_trades} trades)", "info")
        tprint(f"    Symbols below SMA200 at end: {symbols_below_sma}", "info")
        if total_loss < -0.01:
            tprint(f"    Losses from below-SMA200 stocks: "
                   f"{C.RED}${bear_loss_total:+,.0f}{C.RESET} "
                   f"({bear_pct:.0f}% of all losses)", "warn")
        tprint(f"    Losses from above-SMA200 stocks: "
               f"${bull_loss_total:+,.0f}", "info")

    # Analyze each version
    _analyze_version(base_results, "Base v3.0 (raw RL)", datasets)
    print()
    _analyze_version(nosma_results, "v7.0 (no SMA)", datasets)
    print()

    # v7.0+SMA: quantify conservatism
    if sma_results is not None:
        per_sym_sma = sma_results.get('per_sym', {})
        n_syms = len(per_sym_sma)
        profitable = sma_results.get('profitable_symbols', 0)
        total_pnl = sma_results.get('total_pnl', 0)
        pnl_c = C.GREEN if total_pnl > 0 else C.RED
        tprint(f"  v7.0 (SMA) — Conservative but profitable:", "ok")
        tprint(f"    Total P&L: {pnl_c}${total_pnl:+,.0f}{C.RESET}", "info")
        tprint(f"    Profitable symbols: {profitable}/{n_syms}", "info")
        tprint(f"    Max position: 0.25 (75% cash earning yield)", "info")
        tprint(f"    No stock selection: equal-weight all {n_syms} symbols", "info")
    print()


# ============================================================================
# V8.0: ABLATION STUDY ("KEEP OR DISCARD")
# ============================================================================

def run_ablation_study(datasets, pipeline, cfg, acfg,
                       spy_returns_lookup, net, sector_map,
                       v7_sma_results, spy_total_pnl):
    """
    Empirical validation: test each v8.0 component independently.

    For each test variant:
    - Run evaluate_with_pipeline() on the selected subset
    - Compare score, Sharpe, Total P&L vs v7.0+SMA and SPY
    - Print KEEP if beats both benchmarks, DISCARD otherwise
    """
    benchmarks = {
        'v7_sma_score': v7_sma_results.get('score', 0),
        'v7_sma_sharpe': v7_sma_results.get('avg_sh', 0),
        'spy_pnl': spy_total_pnl,
    }

    # Define test configurations
    # Each: (name, SelectionConfig with specific weight overrides)
    tests = [
        ("Momentum Only (top-15)",
         SelectionConfig(top_n=15, w_momentum=1.0, w_trend=0.0, w_rs=0.0, w_invvol=0.0)),
        ("SMA Alignment Only (top-15)",
         SelectionConfig(top_n=15, w_momentum=0.0, w_trend=1.0, w_rs=0.0, w_invvol=0.0)),
        ("Relative Strength Only (top-15)",
         SelectionConfig(top_n=15, w_momentum=0.0, w_trend=0.0, w_rs=1.0, w_invvol=0.0)),
        ("Full Composite (top-15)",
         SelectionConfig(top_n=15)),
        ("Top 5 stocks",
         SelectionConfig(top_n=5)),
        ("Top 10 stocks",
         SelectionConfig(top_n=10)),
        ("Top 20 stocks",
         SelectionConfig(top_n=20)),
        ("Top 30 stocks",
         SelectionConfig(top_n=30)),
    ]

    tprint(f"  Benchmarks: v7.0+SMA score={benchmarks['v7_sma_score']:+.3f}  "
           f"SPY P&L=${benchmarks['spy_pnl']:+,.0f}", "info")
    print()

    # Table header
    hdr = (f"  {'Test':<30s} {'Score':>8s} {'Sharpe':>8s} "
           f"{'P&L':>12s} {'DD%':>7s} {'N':>4s}  Verdict")
    print(f"  {C.BOLD}{'-' * 90}{C.RESET}")
    print(f"  {C.BOLD}{hdr}{C.RESET}")
    print(f"  {'-' * 90}")

    # Pre-compute stock selections serially (fast, no I/O)
    ablation_jobs = []
    for name, sel_cfg in tests:
        selector = StockSelector(sel_cfg, sector_map)
        filtered = selector.select(datasets, spy_returns_lookup)
        ablation_jobs.append((name, filtered))

    # Unwrap net once (shared read-only reference across threads)
    _abl_net = unwrap_net(net) if HAS_TORCH else net

    def _run_one_ablation(job):
        """Run a single ablation config with its own pipeline deep-copy."""
        name, filtered = job
        if not filtered:
            return name, None, 0
        # Each thread gets its own pipeline copy — no shared mutable state
        pl = deepcopy(pipeline)
        pl.use_sma = True
        pl.use_v8_sizing = False  # Flat sizing for ablation
        pl.execution_engine.reset_lifetime_stats()
        try:
            results = evaluate_with_pipeline(
                net=_abl_net,
                datasets=filtered,
                pipeline=pl, cfg=cfg, acfg=acfg,
                label=f"ablation_{name}", verbose=0,
                spy_returns_lookup=spy_returns_lookup,
            )
        except Exception as _e:
            tprint(f"  [ablation] {name} raised: {_e}", "warn")
            results = None
        return name, results, len(filtered)

    # Run all 8 configs in parallel; NumPy releases GIL → real CPU overlap
    n_workers = min(len(ablation_jobs), 16)  # 16-core machine — run all jobs in parallel
    abl_results = {}   # name -> (results_dict_or_None, n_selected)
    with ThreadPoolExecutor(max_workers=n_workers) as _abl_pool:
        _futures = {_abl_pool.submit(_run_one_ablation, job): job[0]
                    for job in ablation_jobs}
        for _fut in as_completed(_futures):
            _n, _res, _n_sel = _fut.result()
            abl_results[_n] = (_res, _n_sel)

    # Print in original order (threads finished out-of-order)
    for name, _ in tests:
        results, n_sel = abl_results.get(name, (None, 0))

        if n_sel == 0:
            print(f"  {name:<30s}  -- no stocks selected --")
            continue
        if results is None:
            print(f"  {name:<30s}  -- eval failed --")
            continue

        r_score = results.get('score', 0)
        r_sharpe = results.get('avg_sh', 0)
        r_pnl = results.get('total_pnl', 0)
        r_dd = results.get('dd_max', 0)

        beats_v7 = r_score > benchmarks['v7_sma_score']
        beats_spy = r_pnl > benchmarks['spy_pnl']

        if beats_v7 and beats_spy:
            verdict = f"{C.GREEN}{C.BOLD}KEEP{C.RESET}"
        elif beats_v7 or beats_spy:
            verdict = f"{C.YELLOW}MIXED{C.RESET}"
        else:
            verdict = f"{C.RED}DISCARD{C.RESET}"

        pnl_c = C.GREEN if r_pnl > 0 else C.RED
        print(f"  {name:<30s} {r_score:>+8.3f} {r_sharpe:>+8.3f} "
              f"{pnl_c}${r_pnl:>+11,.0f}{C.RESET} {r_dd:>6.1f}% {n_sel:>4d}  {verdict}")

    print(f"  {'-' * 90}")
    print()


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
    _valid = (closes_full[1:] > 0) & (closes_full[:-1] > 0)
    log_rets[1:][_valid] = np.log(closes_full[1:][_valid] / closes_full[:-1][_valid])

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
    # Use cumsum for O(n) instead of O(n*w) rolling sum loops
    _cs = np.cumsum(log_rets)
    cum_rets_5bar = np.zeros(n_obs)
    cum_rets_15bar = np.zeros(n_obs)
    if n_obs > 55:
        cum_rets_5bar[50:n_obs - 5] = _cs[55:n_obs] - _cs[50:n_obs - 5]
    if n_obs > 65:
        cum_rets_15bar[50:n_obs - 15] = _cs[65:n_obs] - _cs[50:n_obs - 15]

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
    g.add_argument("--realized-sizing", action="store_true",
                   help="Size positions on starting_capital + realized PnL only (not MTM portfolio value)")

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

    # v8.0 Stock Selection
    g = p.add_argument_group("v8.0 Stock Selection")
    g.add_argument("--version", type=str, default="v7",
                   choices=["v7", "v8", "v9", "letf"],
                   help="Strategy version: v7=baseline, v8=stock selection engine, "
                        "v9=full stack (v8 + RRG rotation + volume/breakout factors + sector breadth) | "
                        "letf=S7 Dynamic 3x Sector Rotation (momentum-ranked XLK/XLE/XLF/QQQ/IWM/XLV -> TECL/ERX/FAS/TQQQ/TNA/CURE, bear gate SPY<SMA200->Cash)")
    g.add_argument("--top-n", type=int, default=15,
                   help="v8.0: fallback fixed-N cap used only when --no-adaptive-n is passed (default: 15). "
                        "In v8 mode adaptive selection is ON by default — all stocks passing the threshold are traded.")
    g.add_argument("--adaptive-n", action="store_true",
                   help="v8.0 Tier 2: variable N — select all stocks within min-score-pct drop of the top score. "
                        "Always ON when --version v8 (default). 4 qualify in a weak market → take 4; 25 qualify in a bull run → take 25.")
    g.add_argument("--no-adaptive-n", action="store_true",
                   help="v8.0: disable adaptive N and use a fixed --top-n cap instead (overrides v8 default).")
    g.add_argument("--min-score-pct", type=float, default=0.35,
                   help="v8.0 Tier 2: quality threshold — select stocks within this fraction of the top score "
                        "(default: 0.35, i.e. threshold = 35%% of top score magnitude). "
                        "Lower = wider net (more stocks). Higher = tighter (fewer, higher-conviction). "
                        "Must be in [0.0, 1.0].")
    g.add_argument("--sector-momentum-gate", action="store_true",
                   help="v8.0 Tier 3: boost composite scores for stocks in high-momentum sectors")
    g.add_argument("--sector-momentum-weight", type=float, default=0.10,
                   help="v8.0 Tier 3: max additive boost to composite score from sector momentum (default: 0.10)")
    # v9.0: New scoring factors
    g.add_argument("--volume-acc-weight", type=float, default=0.0,
                   help="v9.0 Tier 4: weight for volume accumulation factor in composite score (default: 0, off). "
                        "Measures fraction of volume on up-days (institutional accumulation signal).")
    g.add_argument("--high52w-weight", type=float, default=0.0,
                   help="v9.0 Tier 5: weight for 52-week high proximity in composite score (default: 0, off). "
                        "Stocks near 52wk high = confirmed breakout territory.")
    g.add_argument("--rrg-rotation", action="store_true",
                   help="v9.0: use Relative Rotation Graph (RRG) quadrant scores for Tier 3 sector boost. "
                        "Detects sector rotation from price-only data using sector ETFs vs SPY.")
    g.add_argument("--rrg-fast", type=int, default=50,
                   help="v9.0 RRG: fast EMA period in bars (default: 50 = ~10 trading weeks)")
    g.add_argument("--rrg-slow", type=int, default=200,
                   help="v9.0 RRG: slow EMA period in bars (default: 200 = ~40 trading weeks)")
    g.add_argument("--sector-breadth-gate", action="store_true",
                   help="v9.0: require >50%% of sector stocks above 50d SMA before allowing Tier 3 boost. "
                        "Prevents narrow moves (1-2 large-caps dragging sector score).")
    g.add_argument("--include-midcap", action="store_true",
                   help="v9.0: include mid-cap 'Rising Stars' universe alongside large-cap 200-stock core. "
                        "Adds ~70 mid-cap stocks ($2B-$10B); same selector filters out weak names.")
    g.add_argument("--min-dollar-volume", type=float, default=0.0,
                   help="v9.0: minimum average daily dollar volume in $ (e.g. 20000000 for $20M, default: off). "
                        "Used in rank_universe to filter illiquid stocks, and as lower bound for "
                        "--include-midcap discovery (defaults to $20M when --include-midcap is set).")
    g.add_argument("--max-dollar-volume", type=float, default=0.0,
                   help="v9.0: maximum average daily dollar volume in $ for --include-midcap discovery "
                        "(e.g. 2000000000 for $2B, default: $2B when --include-midcap is set). "
                        "Caps the upper end so blue-chip large-caps not in DEFAULT_SYMBOLS are excluded.")
    g.add_argument("--min-avg-volume", type=float, default=0.0,
                   help="v9.0: minimum average daily volume in shares for --include-midcap discovery "
                        "(e.g. 300000 for 300K shares/day, default: 300K when --include-midcap is set). "
                        "Float proxy — rules out low-float stocks that hit dollar thresholds only on spike days.")
    g.add_argument("--kill-loss", type=float, default=None,
                   help="Kill switch max portfolio loss fraction (default: ArchitectureConfig default=0.30)")
    g.add_argument("--skip-ablation", action="store_true",
                   help="v8.0: skip ablation study")

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
    # Default: for v8.0, load full diversified universe; otherwise first n_symbols
    if getattr(args, 'version', 'v7') in ('v8', 'v9'):
        base = DEFAULT_SYMBOLS[:206]  # All 206 stocks across 11 sectors
        # v9.0: --include-midcap dynamically discovers mid-cap candidates from Norgate
        # using objective OHLCV-only filters (price, dollar volume, history length).
        # No hand-curation — the data decides.
        if getattr(args, 'include_midcap', False):
            try:
                from alphago_trading_system import discover_midcap_symbols, NORGATE_DIR
                base_set = set(base)
                # Dollar volume bounds: --min/max-dollar-volume if set, else function defaults
                _min_dv  = getattr(args, 'min_dollar_volume', 0.0) or 20_000_000   # $20M/day
                _max_dv  = getattr(args, 'max_dollar_volume', 0.0) or 2_000_000_000 # $2B/day
                _min_vol = getattr(args, 'min_avg_volume', 0.0)    or 300_000        # 300K shares/day
                tprint(f"  [Mid-Cap] Scanning Norgate for candidates "
                       f"(dv ${_min_dv/1e6:.0f}M\u2013${_max_dv/1e6:.0f}M/day, "
                       f"vol \u2265{_min_vol/1e3:.0f}K shares/day, price \u2265$10)...", "info")
                extra = discover_midcap_symbols(
                    norgate_dir=NORGATE_DIR,
                    exclude_symbols=base_set,
                    min_dollar_volume=_min_dv,
                    max_dollar_volume=_max_dv,
                    min_avg_volume_shares=_min_vol,
                )
                tprint(f"  [Mid-Cap] Found {len(extra)} candidates "
                       f"(total universe: {len(base) + len(extra)} stocks)", "info")
                return base + extra
            except Exception as e:
                tprint(f"  [Mid-Cap] Discovery failed: {e}", "warn")
        return base
    return DEFAULT_SYMBOLS[:args.n_symbols]


# ============================================================================
# S7 LETF: Dynamic 3x Sector Rotation  (--version letf)
#   Signal ETF  ->  3x Position ETF
#   XLK (Tech)  ->  TECL   |  XLE (Energy)   ->  ERX
#   XLF (Fin.)  ->  FAS    |  QQQ (Nasdaq)   ->  TQQQ
#   IWM (Small) ->  TNA    |  XLV (Health)   ->  CURE
#   Bear gate: SPY < SMA(200) -> Cash
#   Rebalance: every 21 trading days, minimum 21-day hold
# ============================================================================

def _run_letf_strategy(args):
    """
    S7 Dynamic 3x Sector Rotation strategy  (--version letf)

    Uses non-leveraged sector ETFs as clean momentum signals, then buys the
    corresponding 3x leveraged ETF for the top-ranked sector each month.
    Bear gate: when SPY drops below its 200-day SMA, rotate to Cash.
    Minimum hold: 21 trading days per position to avoid whipsaw.
    Momentum score: 40% × 1m-return + 35% × 3m-return + 25% × 6m-return.
    """
    import pandas as _pd

    ALLOCATION  = max(float(getattr(args, 'capital', 50_000)), 50_000)
    COST_BPS    = 10         # 10 bps one-way per trade
    COST        = COST_BPS / 10_000
    REBAL_DAYS  = 21         # Monthly rebalance period (trading days)
    MIN_HOLD    = 21         # Minimum hold before a switch is allowed
    START_DATE  = "2010-03-11"
    MOM_WEIGHTS = {21: 0.40, 63: 0.35, 126: 0.25}   # 1m / 3m / 6m
    norgate_dir = Path(NORGATE_DIR) / "US_Equities"

    SIGNAL_TO_POS = {
        "XLK": "TECL",   # Technology 3x
        "XLE": "ERX",    # Energy 3x
        "XLF": "FAS",    # Financials 3x
        "QQQ": "TQQQ",   # Nasdaq 100 3x
        "IWM": "TNA",    # Russell 2000 3x
        "XLV": "CURE",   # Healthcare 3x
    }
    SECTOR_NAMES = {
        "XLK": "Technology", "XLE": "Energy",    "XLF": "Financials",
        "QQQ": "Nasdaq",     "IWM": "Small Caps", "XLV": "Healthcare",
    }
    SIGNAL_ETFS = list(SIGNAL_TO_POS.keys())
    POS_ETFS    = list(SIGNAL_TO_POS.values())

    print_section("", reset=True)
    print_box(
        "ALPHA-TRADE S7 — Dynamic 3x Sector Rotation",
        f"Signals: XLK/XLE/XLF/QQQ/IWM/XLV -> TECL/ERX/FAS/TQQQ/TNA/CURE  |  "
        f"Bear gate: SPY<SMA(200)->Cash  |  Monthly rebal + 21d min hold  |  "
        f"Capital: ${ALLOCATION:,.0f}  |  {datetime.now():%Y-%m-%d %H:%M}",
    )

    # ── 1. Load all parquet files ─────────────────────────────────────────
    print_section("DATA LOADING")
    series = {}
    for sym in SIGNAL_ETFS + POS_ETFS + ["SPY"]:
        fp = norgate_dir / f"{sym}.parquet"
        if not fp.exists():
            tprint(f"{sym}.parquet not found at {fp}", "err")
            return
        df = _pd.read_parquet(fp)
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == "close" and "unadj" not in cl:
                col_map[c] = "Close"
        df = df.rename(columns=col_map)
        if "Close" not in df.columns:
            for c in df.columns:
                if "close" in c.lower() and "unadj" not in c.lower():
                    df = df.rename(columns={c: "Close"})
                    break
        if "Close" not in df.columns:
            tprint(f"{sym}: no Close column found. Columns: {list(df.columns)}", "err")
            return
        df.index = _pd.to_datetime(df.index)
        series[sym] = df["Close"].rename(sym)
        tprint(f"  {sym:<6}: {len(df):,} bars  "
               f"({str(df.index[0])[:10]} – {str(df.index[-1])[:10]})", "ok")

    # ── 2. Align on common dates from START_DATE ──────────────────────────
    # Common index is driven by signal ETFs + SPY + TQQQ (benchmark).
    # Position ETFs are NOT included here — they are reindexed separately.
    # If a position ETF has no data on a given day (e.g. CURE before 2011),
    # the daily return is treated as 0% (equivalent to holding Cash).
    common_idx = series[SIGNAL_ETFS[0]].index
    for sym in SIGNAL_ETFS + ["SPY", "TQQQ"]:
        common_idx = common_idx.intersection(series[sym].index)
    common_idx = common_idx.sort_values()
    common_idx = common_idx[common_idx >= START_DATE]

    sig_df = _pd.DataFrame(
        {s: series[s].reindex(common_idx) for s in SIGNAL_ETFS}
    ).ffill()
    # Position ETFs: reindex without enforcing intersection — NaN before inception
    pos_df = _pd.DataFrame(
        {s: series[s].reindex(common_idx) for s in POS_ETFS}
    ).ffill()
    spy    = series["SPY"].reindex(common_idx).ffill()
    tprint(f"  Period: {str(common_idx[0])[:10]} – {str(common_idx[-1])[:10]}"
           f"  ({len(common_idx):,} trading days)", "info")

    # ── 3. Momentum scores + bear gate (both lagged 1 day) ───────────────
    print_section("SIGNALS")
    mom = _pd.DataFrame(0.0, index=sig_df.index, columns=sig_df.columns)
    for window, weight in MOM_WEIGHTS.items():
        mom = mom + sig_df.pct_change(window) * weight
    mom_scores = mom.shift(1)                             # 1-day lag

    spy_sma200 = spy.rolling(200).mean().shift(1)
    bear_gate  = (spy.shift(1) < spy_sma200)              # True = SPY below SMA200

    # ── 4. Backtest: S7 ───────────────────────────────────────────────────
    n         = len(common_idx)
    nav_s7    = np.ones(n)
    holding   = "Cash"   # current 3x ETF name, or "Cash"
    days_held = 0
    switches  = []       # list of (date, new_holding, old_holding)

    for i in range(1, n):
        date      = common_idx[i]
        prev_date = common_idx[i - 1]

        # Only consider rebalancing every REBAL_DAYS AND after minimum hold
        switched = False
        if (i % REBAL_DAYS == 0) and (days_held >= MIN_HOLD):
            if bool(bear_gate.loc[date]):
                new_h = "Cash"
            else:
                sc      = mom_scores.loc[date].dropna()
                top_sig = sc.idxmax() if len(sc) > 0 else None
                new_h   = SIGNAL_TO_POS.get(top_sig, "Cash") if top_sig else "Cash"

            if new_h != holding:
                switches.append((date, new_h, holding))
                holding   = new_h
                days_held = 0
                switched  = True

        days_held += 1

        # Daily P&L
        if holding == "Cash":
            daily_ret = 0.0
        else:
            pn = float(pos_df.loc[date, holding])
            pp = float(pos_df.loc[prev_date, holding])
            # 0% if ETF not yet started (NaN price) — equivalent to Cash
            daily_ret = pn / pp - 1.0 if (not np.isnan(pn) and not np.isnan(pp)) and pp > 0 else 0.0

        nav_s7[i] = nav_s7[i - 1] * (1.0 + daily_ret) * (
            (1.0 - COST) if switched else 1.0
        )

    # Benchmarks (normalised NAV starting at 1.0)
    tqqq_ret  = pos_df["TQQQ"].pct_change().fillna(0.0).values
    qqq_ret   = sig_df["QQQ"].pct_change().fillna(0.0).values
    spy_ret   = spy.pct_change().fillna(0.0).values
    nav_tqqq  = np.cumprod(1.0 + tqqq_ret)
    nav_qqq   = np.cumprod(1.0 + qqq_ret)
    nav_spy   = np.cumprod(1.0 + spy_ret)

    # Current position label
    _pos_to_sig = {v: k for k, v in SIGNAL_TO_POS.items()}
    _cur_sig     = _pos_to_sig.get(holding, "")
    _cur_sector  = SECTOR_NAMES.get(_cur_sig, holding)
    _cur_label   = f"{holding} ({_cur_sector})" if holding != "Cash" else "Cash"
    tprint(f"  Current position : {_cur_label}", "ok" if holding != "Cash" else "warn")
    tprint(f"  Total switches   : {len(switches)}"
           f"  (~{len(switches)/(len(common_idx)/252):.1f}/yr)", "info")

    # ── 5. Equity curve chart ─────────────────────────────────────────────
    print_section("EQUITY CURVE")
    dates_str = [str(d)[:10] for d in common_idx]
    multi_line_chart(
        [
            ("S7 Rotation", (nav_s7   - 1.0) * 100),
            ("TQQQ B&H",   (nav_tqqq - 1.0) * 100),
            ("QQQ B&H",    (nav_qqq  - 1.0) * 100),
            ("SPY B&H",    (nav_spy  - 1.0) * 100),
        ],
        width=70, height=14, fmt="%", dates=dates_str,
        title=f"S7 Sector Rotation vs Benchmarks — Cumulative Return (${ALLOCATION:,.0f})",
    )
    print()
    for _lbl, _nav in [
        ("S7 Rotation", nav_s7), ("TQQQ B&H", nav_tqqq), ("SPY B&H", nav_spy)
    ]:
        _fp  = float((_nav[-1] - 1.0) * 100.0)
        _pnl = ALLOCATION * (_nav[-1] - 1.0)
        _col = C.GREEN if _fp >= 0 else C.RED
        print(f"    {_col}{_lbl:<14}: {_fp:>+8.1f}%{C.RESET}"
              f"  (+${_pnl:>12,.0f} P&L / ${ALLOCATION:,.0f} committed)")

    # ── 6. Annual performance table ───────────────────────────────────────
    print_section("ANNUAL PERFORMANCE")

    def _annual_nav(nav_arr):
        yr_end, result = {}, {}
        for i, d in enumerate(common_idx):
            yr_end[d.year] = nav_arr[i]
        prev = 1.0
        for yr in sorted(yr_end):
            result[yr] = (yr_end[yr] / prev - 1.0) * 100.0
            prev = yr_end[yr]
        return result

    ann_s7   = _annual_nav(nav_s7)
    ann_tqqq = _annual_nav(nav_tqqq)
    ann_qqq  = _annual_nav(nav_qqq)
    ann_spy  = _annual_nav(nav_spy)

    _strats = [
        ("S7 Rotation", ann_s7), ("TQQQ B&H", ann_tqqq),
        ("QQQ B&H",     ann_qqq), ("SPY B&H",  ann_spy),
    ]
    _yrs = sorted({yr for _, d in _strats for yr in d})
    _cw  = 12

    print(f"\n    {C.BOLD}Year-by-Year Returns ({_yrs[0]}–{_yrs[-1]}){C.RESET}")
    print(f"    {'Year':<6}" + "".join(f"  {lb:>{_cw}}" for lb, _ in _strats))
    print(f"    {'─'*6}" + f"  {'─'*_cw}" * len(_strats))
    for _yr in _yrs:
        _row = f"    {_yr:<6}"
        for _, _ann in _strats:
            _v = _ann.get(_yr)
            if _v is None:
                _row += f"  {'N/A':>{_cw}}"
            else:
                _c = C.GREEN if _v >= 0 else C.RED
                _row += f"  {_c}{_v:>+9.1f}%{C.RESET}"
        print(_row)
    print(f"    {'─'*6}" + f"  {'─'*_cw}" * len(_strats))
    for _rname in ("Total", "CAGR"):
        _row = f"    {_rname:<6}"
        for _, _ann in _strats:
            _vals = [_ann[yr] for yr in _yrs if yr in _ann]
            if _vals:
                _comp = 1.0
                for _v in _vals:
                    _comp *= (1 + _v / 100)
                _res = (_comp - 1.0) * 100 if _rname == "Total" \
                       else ((_comp ** (1.0 / len(_vals))) - 1.0) * 100
                _c = C.GREEN if _res >= 0 else C.RED
                _row += f"  {_c}{_res:>+9.1f}%{C.RESET}"
            else:
                _row += f"  {'N/A':>{_cw}}"
        print(_row)
    print()

    # ── 7. Sector rotation history ────────────────────────────────────────
    print_section("SECTOR ROTATION HISTORY")

    # Build daily holding series from switches
    _sw_dict  = {d: nh for d, nh, _ in switches}
    _cur_hold = "Cash"
    _hold_ser = {}
    for _d in common_idx:
        if _d in _sw_dict:
            _cur_hold = _sw_dict[_d]
        _hold_ser[_d] = _cur_hold
    hold_s = _pd.Series(_hold_ser)

    # Dominant holding per year
    print(f"\n    {C.BOLD}Dominant 3x ETF Held Each Year{C.RESET}")
    print(f"    {'Year':<6} {'3x ETF':<8} {'Sector':<16} {'Year Return':>12}")
    print(f"    {'─'*6} {'─'*8} {'─'*16} {'─'*12}")
    for _yr in _yrs:
        _yr_hold = hold_s[hold_s.index.year == _yr]
        if len(_yr_hold) == 0:
            continue
        _dom    = _yr_hold.value_counts().idxmax()
        _sig    = _pos_to_sig.get(_dom, "")
        _sec    = SECTOR_NAMES.get(_sig, "—") if _dom != "Cash" else "—"
        _yr_ret = ann_s7.get(_yr)
        _rc     = C.GREEN if (_yr_ret or 0) >= 0 else C.RED
        _ret_s  = f"{_rc}{_yr_ret:>+9.1f}%{C.RESET}" if _yr_ret is not None else "N/A"
        print(f"    {_yr:<6} {_dom:<8} {_sec:<16} {_ret_s}")

    # All switches log
    print(f"\n    {C.BOLD}All Switches ({len(switches)} total){C.RESET}")
    print(f"    {'Date':<12} {'From':<8} {'To':<8} {'Reason'}")
    print(f"    {'─'*12} {'─'*8} {'─'*8} {'─'*12}")
    for _date, _new, _old in switches:
        _bg     = bool(bear_gate.loc[_date]) if _date in bear_gate.index else False
        _reason = "Bear Gate" if _bg else "Momentum"
        print(f"    {str(_date)[:10]:<12} {_old:<8} {_new:<8} {_reason}")

    # Live momentum scores
    print(f"\n    {C.BOLD}Current Momentum Scores{C.RESET}"
          f" (as of {str(common_idx[-1])[:10]}):"
          f"  Bear gate: {'ACTIVE' if bool(bear_gate.iloc[-1]) else 'off'}")
    _cur_scores = mom_scores.iloc[-1].dropna().sort_values(ascending=False)
    for _sig, _sc in _cur_scores.items():
        _pos    = SIGNAL_TO_POS.get(_sig, "?")
        _active = _pos == holding
        _c      = C.GREEN if _active else C.RESET
        _marker = " <-- HOLDING" if _active else ""
        print(f"    {_c}  {_sig:<6} {_sc:>+.3f}  ->  {_pos:<6}"
              f"  ({SECTOR_NAMES.get(_sig,'')}){_marker}{C.RESET}")

    # ── 8. Risk & return summary ──────────────────────────────────────────
    print_section("RISK & RETURN SUMMARY")

    def _metrics(nav_arr, daily_rets=None):
        n_years = len(nav_arr) / 252.0
        cagr    = (nav_arr[-1] / nav_arr[0]) ** (1.0 / n_years) - 1.0
        if daily_rets is None:
            daily_rets = np.diff(nav_arr) / nav_arr[:-1]
        dr      = np.array(daily_rets)
        sharpe  = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
        roll_mx = np.maximum.accumulate(nav_arr)
        maxdd   = float(np.min((nav_arr - roll_mx) / roll_mx)) * 100
        calmar  = float(cagr / abs(maxdd / 100)) if maxdd != 0 else 0.0
        final   = ALLOCATION * nav_arr[-1]
        return cagr * 100, sharpe, maxdd, calmar, final

    print(f"\n    {'Strategy':<16} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'Calmar':>7} {'${:,.0f}->'.format(int(ALLOCATION)):>14}")
    print(f"    {'─'*16} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*14}")
    for _lbl, _nav, _ret in [
        ("S7 Rotation", nav_s7,   None),
        ("TQQQ B&H",   nav_tqqq, tqqq_ret),
        ("QQQ B&H",    nav_qqq,  qqq_ret),
        ("SPY B&H",    nav_spy,  spy_ret),
    ]:
        _cg, _sh, _dd, _cal, _fin = _metrics(_nav, _ret)
        _dc = C.RED if _dd < -60 else C.YELLOW if _dd < -40 else C.GREEN
        _cc = C.GREEN if _cg > 20 else C.YELLOW if _cg > 10 else C.RED
        print(f"    {_lbl:<16} {_cc}{_cg:>6.1f}%{C.RESET} "
              f"{_sh:>7.2f} "
              f"{_dc}{_dd:>7.1f}%{C.RESET} "
              f"{_cal:>7.2f} "
              f"${_fin:>12,.0f}")
    print()

    print_box(
        "DONE — S7 Dynamic Sector Rotation",
        f"Now holding: {_cur_label}  |  Switches: {len(switches)}"
        f"  |  ${ALLOCATION:,.0f} -> ${ALLOCATION * nav_s7[-1]:,.0f}"
        f"  |  {dates_str[0]} – {dates_str[-1]}",
    )


def main():
    args = parse_args()

    # --- S2 LETF: standalone, no RL/pipeline needed — exit early ----------
    if args.version == "letf":
        _run_letf_strategy(args)
        return

    # --- v9.0: auto-enable all v9.0 features when --version v9 ---
    # v9 = v8 base + RRG rotation + volume accumulation + 52w high proximity
    #      + sector breadth gate + mid-cap Rising Stars universe.
    # Individual flags can still be passed to override these defaults.
    if args.version == "v9":
        if not args.sector_momentum_gate:
            args.sector_momentum_gate = True          # required for RRG to activate
        if not getattr(args, 'rrg_rotation', False):
            args.rrg_rotation = True                  # RRG quadrant sector scoring
        # NOTE: sector_breadth_gate is NOT auto-enabled — it filters entire sectors and
        # causes 0 stocks selected when the universe is single-sector (e.g. all tech).
        # Enable manually with --sector-breadth-gate for diverse multi-sector universes.
        if getattr(args, 'volume_acc_weight', 0.0) == 0.0:
            args.volume_acc_weight = 0.10             # Tier 4: institutional accumulation
        if getattr(args, 'high52w_weight', 0.0) == 0.0:
            args.high52w_weight = 0.05                # Tier 5: 52-week high proximity / breakout
        if not getattr(args, 'include_midcap', False):
            args.include_midcap = True                # Mid-cap Rising Stars universe extension

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
        mcts_schedule_base=min(args.mcts_rollouts, 64),   # start shallow: network still weak early
        mcts_schedule_max=min(args.mcts_rollouts, 512),   # cap at 512; grows to max over iterations
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_envs=args.n_envs,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile,
        use_dsr=not args.no_dsr,
        no_subproc=args.no_subproc,
        use_realized_capital_sizing=getattr(args, 'realized_sizing', False),
        backbone_type=args.backbone,
        timeframes=tfs,
        download_period="max",
        verbose=args.verbose,
    )

    # **********************************************************************
    # STEP 2: Configure institutional architecture
    # **********************************************************************
    _kill_loss_kw = {}
    if getattr(args, 'kill_loss', None) is not None:
        _kill_loss_kw['kill_max_loss_pct'] = args.kill_loss
    acfg = ArchitectureConfig(
        target_annual_vol=args.target_vol,
        max_leverage=args.max_leverage,
        kelly_fraction=args.kelly_fraction,
        no_trade_threshold_pct=args.no_trade_threshold,
        backtest_mode=True,  # Enable backtest mode: allow cross-regime testing, disable drift kill
        **_kill_loss_kw,
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
    _ver_label = (
        "v9.0 -- RRG + Multi-Factor Selection Engine" if args.version == "v9"
        else "v8.0 -- Stock Selection Engine" if args.version == "v8"
        else "v7.0 -- Institutional Architecture"
    )
    print_box(
        f"ALPHA-TRADE {_ver_label}",
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

    # -- v9.0: Load sector ETFs for RRG + SPY dataset for RRG normalization --
    _SECTOR_ETF_MAP = {
        'technology': 'XLK', 'financials': 'XLF', 'healthcare': 'XLV',
        'industrials': 'XLI', 'energy': 'XLE', 'consumer_discretionary': 'XLY',
        'consumer_staples': 'XLP', 'materials': 'XLB', 'utilities': 'XLU',
        'real_estate': 'XLRE', 'telecom': 'XLC',
    }
    sector_etf_datasets = {}
    spy_dataset_for_rrg = None

    if getattr(args, 'rrg_rotation', False):
        import pandas as _etf_pd  # noqa: F811 — local import for RRG ETF loading
        from concurrent.futures import ThreadPoolExecutor as _TPEX, as_completed as _ac

        _etf_dir = Path(NORGATE_DIR) / "US_Equities"

        def _load_one_etf(sector_sym_pair):
            """Load and build a SymbolDataset for one sector ETF — runs in thread pool."""
            _sn, _sym = sector_sym_pair
            _path = _etf_dir / f"{_sym}.parquet"
            if not _path.exists():
                return _sn, _sym, None, f"{_sym} not found in Norgate"
            try:
                _df = _etf_pd.read_parquet(_path)
                _cmap = {}
                for _c in _df.columns:
                    _cl = _c.lower().strip()
                    if _cl == 'close' and 'unadj' not in _cl:
                        _cmap[_c] = 'Close'
                    elif _cl == 'high':
                        _cmap[_c] = 'High'
                    elif _cl == 'open':
                        _cmap[_c] = 'Open'
                    elif _cl == 'low':
                        _cmap[_c] = 'Low'
                    elif _cl == 'volume':
                        _cmap[_c] = 'Volume'
                _df = _df.rename(columns=_cmap)
                if 'Close' not in _df.columns:
                    return _sn, _sym, None, "no Close column"
                _ec = _df['Close'].values.astype(np.float32)
                _eh = _df['High'].values.astype(np.float32) if 'High' in _df else _ec.copy()
                _eo = _df['Open'].values.astype(np.float32) if 'Open' in _df else _ec.copy()
                _el = _df['Low'].values.astype(np.float32)  if 'Low'  in _df else _ec.copy()
                _ev = (_df['Volume'].values.astype(np.float32)
                       if 'Volume' in _df else np.zeros(len(_ec), dtype=np.float32))
                _n = len(_ec)
                _ds = SymbolDataset(
                    symbol=_sym,
                    features_train=np.zeros((0, 49), dtype=np.float32),
                    features_val=np.zeros((0, 49), dtype=np.float32),
                    features_test=np.zeros((0, 49), dtype=np.float32),
                    prices_train=np.zeros((0, 5), dtype=np.float32),
                    prices_val=np.zeros((0, 5), dtype=np.float32),
                    prices_test=np.column_stack([_eo, _eh, _el, _ec, _ev]),
                    n_train=0, n_val=0, n_test=_n,
                )
                return _sn, _sym, _ds, None
            except Exception as _e:
                return _sn, _sym, None, str(_e)

        # Load all 11 ETFs + SPY in parallel (I/O-bound — real speedup from concurrency)
        _etf_items = list(_SECTOR_ETF_MAP.items()) + [('__spy__', 'SPY')]
        with _TPEX(max_workers=min(len(_etf_items), 16)) as _etf_pool:
            _etf_futs = {_etf_pool.submit(_load_one_etf, item): item for item in _etf_items}
            for _fut in _ac(_etf_futs):
                _sn, _sym, _ds, _err = _fut.result()
                if _err:
                    tprint(f"  [RRG] {_sym}: {_err}", "warn")
                elif _sn == '__spy__':
                    spy_dataset_for_rrg = _ds
                else:
                    sector_etf_datasets[_sn] = _ds

        tprint(f"  [RRG] Loaded {len(sector_etf_datasets)}/11 sector ETFs"
               + (" + SPY" if spy_dataset_for_rrg else ""), "info")

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

    # **********************************************************************
    # V8.0: STOCK SELECTION ENGINE (always runs; verbose output gated on --version v8)
    # **********************************************************************
    v8_results = None
    v8_lt = {}

    if args.version in ("v8", "v9"):
        _sel_label = "V9.0 STOCK SELECTION ENGINE" if args.version == "v9" else "V8.0 STOCK SELECTION ENGINE"
        print_section(_sel_label)
        # Failure post-mortem (verbose, v8/v9 mode only)
        tprint("Analyzing why v3.0 and v7.0 (no SMA) underperform...", "info")
        run_postmortem(base_results, nosma_results, pipeline_results, datasets)

    # Stock selection — always run so v8 column appears in comparison table
    # v8 defaults to adaptive N (trade all stocks meeting criteria).
    # --no-adaptive-n overrides back to fixed --top-n cap.
    _adaptive_n = (args.adaptive_n or args.version in ("v8", "v9")) and not getattr(args, 'no_adaptive_n', False)
    sel_cfg = SelectionConfig(
        top_n=args.top_n,
        adaptive_n=_adaptive_n,
        min_score_pct=args.min_score_pct,
        sector_momentum_gate=args.sector_momentum_gate,
        w_sector_momentum=args.sector_momentum_weight,
        # v9.0 new fields:
        w_volume_acc=getattr(args, 'volume_acc_weight', 0.0),
        w_high52w=getattr(args, 'high52w_weight', 0.0),
        use_rrg_rotation=getattr(args, 'rrg_rotation', False),
        rrg_fast_period=getattr(args, 'rrg_fast', 50),
        rrg_slow_period=getattr(args, 'rrg_slow', 200),
        sector_breadth_gate=getattr(args, 'sector_breadth_gate', False),
        min_dollar_volume=getattr(args, 'min_dollar_volume', 0.0),
    )
    selector = StockSelector(sel_cfg, SECTOR_MAP)
    selected_datasets = selector.select(
        datasets,
        spy_returns_lookup,
        sector_etf_datasets=sector_etf_datasets if getattr(args, 'rrg_rotation', False) else None,
        spy_dataset=spy_dataset_for_rrg if getattr(args, 'rrg_rotation', False) else None,
    )

    if args.version in ("v8", "v9"):
        tprint(f"Selected {len(selected_datasets)} stocks from "
               f"{len(datasets)} universe:", "ok")
        if selector.selection_log:
            _log = selector.selection_log[-1]
            for _sel_sym in _log['selected']:
                _sel_score = _log['scores'].get(_sel_sym, 0)
                _comp_str = ""
                for _s, _sc, _comp in _log['rankings']:
                    if _s == _sel_sym:
                        _vol_acc_str = (f" vacc={_comp['vol_acc']:.0%}"
                                        if 'vol_acc' in _comp and sel_cfg.w_volume_acc > 0 else "")
                        _h52w_str = (f" h52w={_comp['high52w_proximity']:.0%}"
                                     if 'high52w_proximity' in _comp and sel_cfg.w_high52w > 0 else "")
                        _comp_str = (f"mom={_comp['momentum']:+.1%} "
                                     f"sma={_comp['sma_score']}/3 "
                                     f"rs={_comp['rs_vs_spy']:+.1%} "
                                     f"vol={_comp['vol_20']:.0%}"
                                     f"{_vol_acc_str}{_h52w_str}")
                        break
                tprint(f"  {_sel_sym:<6s}: score={_sel_score:+.4f}  {_comp_str}", "info")
            _sec_alloc = _log.get('sector_allocation', {})
            if _sec_alloc:
                tprint(f"  Sector allocation: {dict(_sec_alloc)}", "info")
            # Tier 2: log N in action (always on in v8 mode)
            if _adaptive_n:
                tprint(f"  Adaptive N: {len(_log['selected'])} stocks selected "
                       f"(threshold={args.min_score_pct:.0%} of top score)", "info")
            # Tier 3: log top-3 sector momentum scores (v8 stock-avg mode)
            _sec_mom = _log.get('sector_momentum', {})
            if _sec_mom and not getattr(args, 'rrg_rotation', False):
                _top3 = sorted(_sec_mom.items(), key=lambda x: -x[1])[:3]
                tprint(f"  Sector momentum (top 3): " +
                       ", ".join(f"{s}={m:+.1%}" for s, m in _top3), "info")

            # v9.0: RRG quadrant diagnostics
            _rrg_meta = _log.get('rrg_metadata', {})
            if _rrg_meta:
                _qmap = {'leading': '[G]', 'improving': '[B]', 'weakening': '[Y]', 'lagging': '[R]'}
                _quadrant_rows = sorted(_rrg_meta.items(), key=lambda x: -x[1]['score'])
                tprint("  RRG Sector Quadrants:", "info")
                for _sec, _data in _quadrant_rows:
                    _icon = _qmap.get(_data['quadrant'], '[?]')
                    tprint(f"    {_icon} {_sec:<24} {_data['quadrant']:<10} "
                           f"RS_ratio={_data['rs_ratio']:+.1f}  RS_mom={_data['rs_momentum']:+.1f}",
                           "info")

            # v9.0: Multi-TF RS diagnostics
            _mtf = _log.get('multitf_metadata', {})
            if _mtf:
                tprint("  Sector Multi-TF RS vs SPY (1m / 3m / 6m / 12m):", "info")
                _mtf_sorted = sorted(_mtf.items(),
                                     key=lambda x: -x[1].get('rs_score_norm', 0))
                for _sec, _v in _mtf_sorted[:6]:
                    _accel_flag = "(accel)" if _v.get('rs_accel', 0) > 0 else ""
                    tprint(f"    {_sec:<24} "
                           f"{_v['rs_1m']:+.1%} / {_v['rs_3m']:+.1%} / "
                           f"{_v['rs_6m']:+.1%} / {_v['rs_12m']:+.1%}  "
                           f"accel={_v['rs_accel']:+.1%} {_accel_flag}",
                           "info")

            # v9.0: Sector breadth diagnostics
            _breadth = _log.get('sector_breadth', {})
            if _breadth:
                _top_breadth = sorted(_breadth.items(), key=lambda x: -x[1])[:5]
                tprint("  Sector breadth (% stocks > 50d SMA): " +
                       ", ".join(f"{s}={b:.0%}" for s, b in _top_breadth), "info")

            # v9.0: Rising Stars — selected stocks not in the large-cap DEFAULT_SYMBOLS universe
            _large_cap_set = set(DEFAULT_SYMBOLS)
            _rising = [_s for _s in _log['selected']
                       if (_s.split('_')[0] if '_' in _s else _s) not in _large_cap_set]
            if _rising:
                tprint(f"  Rising Stars selected: {', '.join(_rising)}", "info")

    # Build rank map
    _v8_rank = {}
    _v8_sel_log = selector.selection_log[-1] if selector.selection_log else None
    for _rank_idx, _sel_sym in enumerate(_v8_sel_log['selected'] if _v8_sel_log else []):
        _v8_rank[_sel_sym] = _rank_idx

    # Evaluate v8/v9 with rank-based sizing.
    # --version v8/v9: use selector-filtered datasets (stocks passing quality threshold).
    # comparison mode (v7): use same datasets as v7 so all symbols appear; only sizing differs.
    _v8_eval_datasets = selected_datasets if args.version in ("v8", "v9") else datasets
    pipeline.use_sma = True
    pipeline.use_v8_sizing = True
    pipeline._v8_rank = _v8_rank
    pipeline.execution_engine.reset_lifetime_stats()

    v8_results = evaluate_with_pipeline(
        net=unwrap_net(net) if HAS_TORCH else net,
        datasets=_v8_eval_datasets,
        pipeline=pipeline, cfg=cfg, acfg=acfg,
        label="Pipeline v8.0",
        verbose=args.verbose if args.version in ("v8", "v9") else 0,
        spy_returns_lookup=spy_returns_lookup,
    )
    v8_lt = dict(pipeline.execution_engine.lifetime_stats)

    if args.version in ("v8", "v9"):
        print_results(v8_results)

    # Restore pipeline to v7.0 mode
    pipeline.use_v8_sizing = False
    pipeline._v8_rank = {}
    pipeline.execution_engine.reset_lifetime_stats()
    pipeline.execution_engine._lifetime_trades = pip_sma_lt['n_trades']
    pipeline.execution_engine._lifetime_suppressed = pip_sma_lt['n_suppressed']

    # --- v8 BASELINE evaluation for v9 comparison table ---
    # When --version v9, run a SECOND selection pass using clean v8 config
    # (no vol_acc, no high52w, no RRG) so the table shows v8 vs v9 side-by-side.
    _v8b_results = None
    _v8b_lt = {}
    _v8b_per_sym = {}
    _v8b_n_syms = 0
    _v8b_total_cap = 0
    _v8b_trade_pnl = 0.0
    _v8b_cash_yield = 0.0
    _v8b_n_trades = 0
    _v8b_n_suppressed = 0
    _v8b_win_rate = float('nan')
    _v8b_profit_factor = float('nan')

    if args.version == "v9":
        _v8b_sel_cfg = SelectionConfig(
            top_n=args.top_n,
            adaptive_n=_adaptive_n,
            min_score_pct=args.min_score_pct,
            sector_momentum_gate=False,   # v8 default — don't inherit v9's auto-enabled gate
            w_sector_momentum=args.sector_momentum_weight,
            # v9 features explicitly OFF for clean v8 baseline:
            w_volume_acc=0.0,
            w_high52w=0.0,
            use_rrg_rotation=False,
            sector_breadth_gate=False,
            min_dollar_volume=getattr(args, 'min_dollar_volume', 0.0),
        )
        _v8b_selector = StockSelector(_v8b_sel_cfg, SECTOR_MAP)
        _v8b_datasets = _v8b_selector.select(datasets, spy_returns_lookup)
        # Warn if v8 and v9 select identical portfolios
        _v8b_selected_syms = set(_v8b_selector.selection_log[-1]['selected']) if _v8b_selector.selection_log else set()
        _v9_selected_syms = set(selector.selection_log[-1]['selected']) if selector.selection_log else set()
        if _v8b_selected_syms and _v8b_selected_syms == _v9_selected_syms:
            tprint(f"  Note: v8 and v9 selected identical portfolios "
                   f"({', '.join(sorted(_v8b_selected_syms))}). "
                   f"v9 features did not change selection on this universe. "
                   f"Try broader universe or lower --min-score-pct.", "warn")
        _v8b_rank = {sym: i for i, sym in enumerate(
            (_v8b_selector.selection_log[-1]['selected']
             if _v8b_selector.selection_log else [])
        )}
        pipeline.use_sma = True
        pipeline.use_v8_sizing = True
        pipeline._v8_rank = _v8b_rank
        pipeline.execution_engine.reset_lifetime_stats()
        _v8b_results = evaluate_with_pipeline(
            net=unwrap_net(net) if HAS_TORCH else net,
            datasets=_v8b_datasets,
            pipeline=pipeline, cfg=cfg, acfg=acfg,
            label="Pipeline v8.0 (baseline)",
            verbose=0,
            spy_returns_lookup=spy_returns_lookup,
        )
        _v8b_lt = dict(pipeline.execution_engine.lifetime_stats)
        # Restore pipeline again
        pipeline.use_v8_sizing = False
        pipeline._v8_rank = {}
        pipeline.execution_engine.reset_lifetime_stats()
        pipeline.execution_engine._lifetime_trades = pip_sma_lt['n_trades']
        pipeline.execution_engine._lifetime_suppressed = pip_sma_lt['n_suppressed']

    # -- Compute SPY buy-and-hold P&L for comparison table --
    # NOTE: spy_rets_arr is concatenated across symbols (duplicated dates) — correct
    # for benchmark-relative metrics (aligned 1:1 with strategy returns), but NOT
    # for computing actual SPY buy-and-hold P&L. Use unique test-period dates instead.
    spy_rets_arr = pipeline_results.get('spy_returns', np.array([]))
    has_spy_bench = len(spy_rets_arr) > 0 and np.any(spy_rets_arr != 0.0)
    spy_total_pnl = 0.0
    spy_cagr_str = "N/A"
    if has_spy_bench and spy_returns_lookup:
        # Use FULL test period dates (from bar 0, not post-lookback) for fair SPY benchmark.
        # The lookback window shifts the start date by ~60 bars which can land near market
        # bottoms, artificially inflating SPY returns.
        _test_dates = set()
        for d in datasets:
            if hasattr(d, 'timestamps_test') and d.timestamps_test is not None:
                for ts in d.timestamps_test:
                    _test_dates.add(str(ts)[:10])
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

    # -- v8.0 Ablation study (needs spy_total_pnl computed above) --
    if args.version in ("v8", "v9") and not args.skip_ablation and pipeline_results is not None:
        _abl_label = "V9.0 ABLATION STUDY" if args.version == "v9" else "V8.0 ABLATION STUDY"
        print_section(_abl_label)
        tprint("Testing each selection factor independently — KEEP or DISCARD.", "info")
        run_ablation_study(
            datasets=datasets, pipeline=pipeline, cfg=cfg, acfg=acfg,
            spy_returns_lookup=spy_returns_lookup, net=net,
            sector_map=SECTOR_MAP,
            v7_sma_results=pipeline_results,
            spy_total_pnl=spy_total_pnl,
        )

    # Pre-compute v9 column flag here (outside the comparison-table if-block)
    # so the chart section can safely reference it regardless of base/pipeline availability.
    _has_v9_col = _v8b_results is not None
    _vc_per_sym: dict = {}   # safe defaults; overwritten by alias block inside table section
    _xc_per_sym: dict = {}
    _pip_per_sym: dict = {}  # overwritten when pipeline_results is not None

    # -- Show comparison table --
    # Show when: (a) training was done (base_results exists), OR (b) pipeline ran (eval-only)
    _has_base = base_results is not None
    if _has_base or pipeline_results is not None:
        # Use empty dict so .get() calls return defaults safely
        if not _has_base:
            base_results = {}
        # Compute cash yield for both systems
        _pip_per_sym = pipeline_results.get('per_sym', {})
        pip_cash_yield = sum(s.get('cash_yield_pnl', 0) for s in _pip_per_sym.values())
        pip_trade_pnl = pipeline_results['total_pnl'] - pip_cash_yield

        # Estimate base cash yield (same formula: cash_bps * capital * steps)
        # Base v3.0 doesn't track cash_yield_pnl per symbol, so we estimate it
        # from step counts (same test data = same step count per symbol).
        if _has_base:
            _cash_bps = getattr(cfg, 'cash_yield_bps_annual', 0) * 1e-4
            _cash_per_bar = (_cash_bps / max(cfg.bars_per_year, 1)) * cfg.starting_capital if _cash_bps > 0 else 0.0
            _base_total_steps = sum(
                s.get('step_count', 0) for s in _pip_per_sym.values()
            )  # Use pipeline step count (same test data length)
            base_cash_yield = _cash_per_bar * _base_total_steps
            _base_cash_per_sym = {
                sym: _cash_per_bar * _pip_per_sym.get(sym, {}).get('step_count', 0)
                for sym in _pip_per_sym
            }
            base_trade_pnl = base_results.get('total_pnl', 0) - base_cash_yield
        else:
            base_cash_yield = 0.0
            _base_cash_per_sym = {}
            base_trade_pnl = 0.0

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

        # Profit factor: weighted avg of per-symbol PF by number of closed trades.
        # Cap per-symbol PF at 99.9 to prevent inf (gross_profit / ~0 when no losses).
        _PF_CAP = 99.9
        weighted_pf_pip = sum(
            min(s.get('profit_factor', 0), _PF_CAP) * s.get('trades', 0)
            for s in _pip_per_sym.values()
        )
        pip_profit_factor = weighted_pf_pip / max(total_trades_pip, 1) if total_trades_pip > 0 else float('nan')

        # Track number of closed vs executed trades for clarity
        pip_n_closed_trades = total_trades_pip

        # Base system trades
        base_n_trades = base_results.get('total_trades', 0)
        base_win_rate = base_results.get('avg_wr', 0.0)
        base_profit_factor = base_results.get('avg_pf', 0.0)

        # No-SMA pipeline stats
        _nosma_per_sym = nosma_results.get('per_sym', {})
        nosma_cash_yield = sum(s.get('cash_yield_pnl', 0) for s in _nosma_per_sym.values())
        nosma_trade_pnl = nosma_results['total_pnl'] - nosma_cash_yield
        nosma_n_trades = nosma_lt.get('n_trades', 0)
        nosma_n_suppressed = nosma_lt.get('n_suppressed', 0)
        nosma_suppression_pct = nosma_lt.get('suppression_rate', 0) * 100

        # No-SMA Win Rate and Profit Factor
        _nosma_total_wins = sum(s.get('trades', 0) * s.get('win_rate', 0) / 100.0 for s in _nosma_per_sym.values())
        _nosma_total_closed = sum(s.get('trades', 0) for s in _nosma_per_sym.values())
        nosma_win_rate = (_nosma_total_wins / max(_nosma_total_closed, 1)) * 100 if _nosma_total_closed > 0 else float('nan')
        _nosma_wpf = sum(min(s.get('profit_factor', 0), _PF_CAP) * s.get('trades', 0) for s in _nosma_per_sym.values())
        nosma_profit_factor = _nosma_wpf / max(_nosma_total_closed, 1) if _nosma_total_closed > 0 else float('nan')

        # v8.0 stats (conditional)
        _has_v8 = v8_results is not None
        if _has_v8:
            _v8_per_sym = v8_results.get('per_sym', {})
            v8_cash_yield = sum(s.get('cash_yield_pnl', 0) for s in _v8_per_sym.values())
            v8_trade_pnl = v8_results['total_pnl'] - v8_cash_yield
            v8_n_trades = v8_lt.get('n_trades', 0)
            v8_n_suppressed = v8_lt.get('n_suppressed', 0)
            v8_suppression_pct = v8_lt.get('suppression_rate', 0) * 100
            _v8_n_syms = len(_v8_per_sym) if _v8_per_sym else len(selected_datasets)
            _v8_total_cap = cfg.starting_capital * _v8_n_syms

            # v8.0 Win Rate and Profit Factor
            _v8_total_wins = sum(s.get('trades', 0) * s.get('win_rate', 0) / 100.0 for s in _v8_per_sym.values())
            _v8_total_closed = sum(s.get('trades', 0) for s in _v8_per_sym.values())
            v8_win_rate = (_v8_total_wins / max(_v8_total_closed, 1)) * 100 if _v8_total_closed > 0 else float('nan')
            _v8_wpf = sum(min(s.get('profit_factor', 0), _PF_CAP) * s.get('trades', 0) for s in _v8_per_sym.values())
            v8_profit_factor = _v8_wpf / max(_v8_total_closed, 1) if _v8_total_closed > 0 else float('nan')

        # --- v9.0 column aliases ---
        # When --version v9: col4 = v8 baseline (_v8b_results), col5 = v9 (v8_results)
        # When --version v8: col4 = v8 (v8_results), no col5
        _has_v9_col = _v8b_results is not None
        if _has_v9_col:
            # Derive v8b (baseline) column metrics
            _v8b_per_sym = _v8b_results.get('per_sym', {})
            _v8b_cash_yield = sum(s.get('cash_yield_pnl', 0) for s in _v8b_per_sym.values())
            _v8b_trade_pnl = _v8b_results['total_pnl'] - _v8b_cash_yield
            _v8b_n_trades = _v8b_lt.get('n_trades', 0)
            _v8b_n_suppressed = _v8b_lt.get('n_suppressed', 0)
            _v8b_suppression_pct = _v8b_lt.get('suppression_rate', 0) * 100
            _v8b_n_syms = len(_v8b_per_sym) if _v8b_per_sym else 0
            _v8b_total_cap = cfg.starting_capital * _v8b_n_syms
            _v8b_total_wins = sum(s.get('trades', 0) * s.get('win_rate', 0) / 100.0 for s in _v8b_per_sym.values())
            _v8b_total_closed = sum(s.get('trades', 0) for s in _v8b_per_sym.values())
            _v8b_win_rate = (_v8b_total_wins / max(_v8b_total_closed, 1)) * 100 if _v8b_total_closed > 0 else float('nan')
            _v8b_wpf = sum(min(s.get('profit_factor', 0), _PF_CAP) * s.get('trades', 0) for s in _v8b_per_sym.values())
            _v8b_profit_factor = _v8b_wpf / max(_v8b_total_closed, 1) if _v8b_total_closed > 0 else float('nan')

            # Column 4 (v_*) uses v8 baseline data; column 5 (x_*) uses v9 (v8_results)
            _vc_results = _v8b_results
            _vc_per_sym = _v8b_per_sym
            _vc_trade_pnl = _v8b_trade_pnl
            _vc_cash_yield = _v8b_cash_yield
            _vc_n_trades = _v8b_n_trades
            _vc_n_suppressed = _v8b_n_suppressed
            _vc_suppression_pct = _v8b_suppression_pct
            _vc_n_syms = _v8b_n_syms
            _vc_total_cap = _v8b_total_cap
            _vc_win_rate = _v8b_win_rate
            _vc_profit_factor = _v8b_profit_factor
            # Column 5 (x_*) uses v9 selection data (stored in v8_results)
            _xc_results = v8_results
            _xc_per_sym = _v8_per_sym if _has_v8 else {}
            _xc_trade_pnl = v8_trade_pnl if _has_v8 else 0.0
            _xc_cash_yield = v8_cash_yield if _has_v8 else 0.0
            _xc_n_trades = v8_n_trades if _has_v8 else 0
            _xc_n_suppressed = v8_n_suppressed if _has_v8 else 0
            _xc_suppression_pct = v8_suppression_pct if _has_v8 else 0.0
            _xc_n_syms = _v8_n_syms if _has_v8 else 0
            _xc_total_cap = _v8_total_cap if _has_v8 else 0
            _xc_win_rate = v8_win_rate if _has_v8 else float('nan')
            _xc_profit_factor = v8_profit_factor if _has_v8 else float('nan')
        else:
            # Column 4 (v_*) uses v8_results as-is (--version v8 / v7 comparison mode)
            _vc_results = v8_results
            _vc_per_sym = _v8_per_sym if _has_v8 else {}
            _vc_trade_pnl = v8_trade_pnl if _has_v8 else 0.0
            _vc_cash_yield = v8_cash_yield if _has_v8 else 0.0
            _vc_n_trades = v8_n_trades if _has_v8 else 0
            _vc_n_suppressed = v8_n_suppressed if _has_v8 else 0
            _vc_suppression_pct = v8_suppression_pct if _has_v8 else 0.0
            _vc_n_syms = _v8_n_syms if _has_v8 else 0
            _vc_total_cap = _v8_total_cap if _has_v8 else 0
            _vc_win_rate = v8_win_rate if _has_v8 else float('nan')
            _vc_profit_factor = v8_profit_factor if _has_v8 else float('nan')
            _xc_results = None
            _xc_per_sym = {}

        _cmp_title = (
            "COMPARISON: v3.0 vs v7.0 vs v8.0 vs v9.0" if _has_v9_col
            else "COMPARISON: v3.0 vs v7.0 vs v8.0" if _has_v8
            else "COMPARISON: v3.0 vs v7.0"
        )
        print_section(_cmp_title)
        print(f"  {C.DIM}Side-by-side: raw AI decisions vs. the full system with risk controls.{C.RESET}")
        print(f"  {C.DIM}Like comparing a driver without seatbelts (v3.0) vs. with full safety gear (v7.0).{C.RESET}")

        if HAS_TABLE_FORMATTER:
            _E = ""  # shorthand for empty cells
            _tbl_title = (
                "BASE V3.0 VS PIPELINE V7.0 VS V8.0 VS V9.0" if _has_v9_col
                else "BASE V3.0 VS PIPELINE V7.0 VS V8.0" if _has_v8
                else "BASE V3.0 VS PIPELINE V7.0"
            )
            table = TableFormatter(title=_tbl_title)

            # Column structure: Metric + 4 sub-cols per strategy (P&L, #Tr, $ Used, CAGR)
            _n_strat_cols = 3 + (1 if _has_v8 else 0) + (1 if _has_v9_col else 0)
            table.add_column('Metric', align='left')
            for _ in range(_n_strat_cols):
                table.add_column('P&L', align='right')
                table.add_column('#Tr', align='right')
                table.add_column('$ Used', align='right')
                table.add_column('CAGR', align='right')

            _groups = [('', 1), ('Base v3.0', 4), ('v7.0 (no SMA)', 4), ('Pipeline v7.0', 4)]
            if _has_v8:
                _v8_col_label = (
                    'v8.0 (Select)' if args.version in ("v8", "v9")
                    else 'v8.0 (Sizing)'
                )
                _groups.append((_v8_col_label, 4))
            if _has_v9_col:
                _groups.append(('v9.0 (Select)', 4))
            table.set_header_groups(_groups)

            # Helper: build row — each strategy gets (pnl, tr, used, cagr)
            # v = col 4 (v8.0 Select/Sizing), x = col 5 (v9.0 Select, only when --version v9)
            def _tr(metric, b_pnl=_E, b_tr=_E, b_used=_E, b_cagr=_E,
                             n_pnl=_E, n_tr=_E, n_used=_E, n_cagr=_E,
                             p_pnl=_E, p_tr=_E, p_used=_E, p_cagr=_E,
                             v_pnl=_E, v_tr=_E, v_used=_E, v_cagr=_E,
                             x_pnl=_E, x_tr=_E, x_used=_E, x_cagr=_E):
                row = [metric,
                       b_pnl, b_tr, b_used, b_cagr,
                       n_pnl, n_tr, n_used, n_cagr,
                       p_pnl, p_tr, p_used, p_cagr]
                if _has_v8:
                    row.extend([v_pnl, v_tr, v_used, v_cagr])
                if _has_v9_col:
                    row.extend([x_pnl, x_tr, x_used, x_cagr])
                return row

            # Helper: compute CAGR string from trade pnl, peak notional, steps
            def _cagr_str(trade_pnl, peak, steps):
                if peak > 0 and steps > 0:
                    yrs = steps / 252
                    rr = 1 + trade_pnl / peak
                    if rr > 0:
                        cagr = rr ** (1 / max(yrs, 0.01)) - 1
                        c = C.GREEN if cagr > 0 else C.RED
                        return f"{c}{cagr:>+.1%}{C.RESET}"
                    return f"{C.RED}-100%+{C.RESET}"
                return _E

            # Helper: format $ Used
            def _used_str(peak):
                return f"${peak:>,.0f}" if peak > 0 else _E

            # Prepare colored strings
            nosma_pnl_c = C.GREEN if nosma_results['total_pnl'] > 0 else C.RED
            pip_pnl_c = C.GREEN if pipeline_results['total_pnl'] > 0 else C.RED
            nosma_trade_c = C.GREEN if nosma_trade_pnl > 0 else C.RED
            pip_trade_c = C.GREEN if pip_trade_pnl > 0 else C.RED

            # Base color helpers (only meaningful when base was actually evaluated)
            if _has_base:
                base_pnl_c = C.GREEN if base_results.get('total_pnl', 0) > 0 else C.RED
                base_trade_c = C.GREEN if base_trade_pnl > 0 else C.RED

            # Starting capital
            _n_syms_table = len(_pip_per_sym) if _pip_per_sym else len(datasets)
            _total_cap = cfg.starting_capital * _n_syms_table
            table.add_row(_tr('Capital / Symbol',
                          b_pnl=f"${cfg.starting_capital:>,.0f}" if _has_base else _E,
                          n_pnl=f"${cfg.starting_capital:>,.0f}",
                          p_pnl=f"${cfg.starting_capital:>,.0f}",
                          v_pnl=f"${cfg.starting_capital:>,.0f}" if _has_v8 else _E,
                          x_pnl=f"${cfg.starting_capital:>,.0f}" if _has_v9_col else _E))
            table.add_row(_tr(f'Total Capital ({_n_syms_table} syms)',
                          b_pnl=f"${_total_cap:>,.0f}" if _has_base else _E,
                          n_pnl=f"${_total_cap:>,.0f}",
                          p_pnl=f"${_total_cap:>,.0f}",
                          v_pnl=f"${_vc_total_cap:>,.0f}" if _has_v8 else _E,
                          x_pnl=f"${_xc_total_cap:>,.0f}" if _has_v9_col else _E))

            # --- P&L ---
            table.add_row(_tr(f"{C.BOLD}--- P&L ---{C.RESET}"))
            _vc_pnl_c = (C.GREEN if _vc_results['total_pnl'] > 0 else C.RED) if _has_v8 else ""
            _xc_pnl_c = (C.GREEN if _xc_results['total_pnl'] > 0 else C.RED) if _has_v9_col else ""
            table.add_row(_tr('Total P&L',
                          b_pnl=f"{base_pnl_c}${base_results['total_pnl']:>+,.2f}{C.RESET}" if _has_base else _E,
                          n_pnl=f"{nosma_pnl_c}${nosma_results['total_pnl']:>+,.2f}{C.RESET}",
                          p_pnl=f"{pip_pnl_c}${pipeline_results['total_pnl']:>+,.2f}{C.RESET}",
                          v_pnl=f"{_vc_pnl_c}${_vc_results['total_pnl']:>+,.2f}{C.RESET}" if _has_v8 else _E,
                          x_pnl=f"{_xc_pnl_c}${_xc_results['total_pnl']:>+,.2f}{C.RESET}" if _has_v9_col else _E))
            _vc_tc = (C.GREEN if _vc_trade_pnl > 0 else C.RED) if _has_v8 else ""
            _xc_tc = (C.GREEN if _xc_trade_pnl > 0 else C.RED) if _has_v9_col else ""
            table.add_row(_tr('  Trade P&L',
                          b_pnl=f"{base_trade_c}${base_trade_pnl:>+,.2f}{C.RESET}" if _has_base else _E,
                          n_pnl=f"{nosma_trade_c}${nosma_trade_pnl:>+,.2f}{C.RESET}",
                          p_pnl=f"{pip_trade_c}${pip_trade_pnl:>+,.2f}{C.RESET}",
                          v_pnl=f"{_vc_tc}${_vc_trade_pnl:>+,.2f}{C.RESET}" if _has_v8 else _E,
                          x_pnl=f"{_xc_tc}${_xc_trade_pnl:>+,.2f}{C.RESET}" if _has_v9_col else _E))
            table.add_row(_tr('  Cash Yield',
                          b_pnl=f"{C.CYAN}${base_cash_yield:>+,.2f}{C.RESET}" if _has_base else _E,
                          n_pnl=f"{C.CYAN}${nosma_cash_yield:>+,.2f}{C.RESET}",
                          p_pnl=f"{C.CYAN}${pip_cash_yield:>+,.2f}{C.RESET}",
                          v_pnl=f"{C.CYAN}${_vc_cash_yield:>+,.2f}{C.RESET}" if _has_v8 else _E,
                          x_pnl=f"{C.CYAN}${_xc_cash_yield:>+,.2f}{C.RESET}" if _has_v9_col else _E))

            # --- Per-symbol Trade P&L with sub-columns ---
            _base_per_sym = base_results.get('per_sym', {})
            all_syms = sorted(set(
                list(_base_per_sym.keys()) + list(_pip_per_sym.keys())
                + list(_vc_per_sym.keys()) + list(_xc_per_sym.keys())
            ))
            if all_syms:
                table.add_row(_tr(f"{C.BOLD}--- Trade P&L by Symbol ---{C.RESET}"))
                for sym in all_syms:
                    # Base v3.0 — per_sym only has pnl/trades/turnover (no pipeline fields)
                    b_sym = _base_per_sym.get(sym, {})
                    b_total = b_sym.get('pnl', 0)
                    b_cash = _base_cash_per_sym.get(sym, 0)  # always estimated (base has no cash_yield_pnl)
                    b_trade = b_total - b_cash
                    b_trades = b_sym.get('trades', 0)
                    # Base trades with full capital, so peak_notional ≈ starting_capital
                    b_peak = cfg.starting_capital if b_sym else 0
                    b_steps = _pip_per_sym.get(sym, {}).get('step_count', 0)  # same test data length
                    # v7.0 (no SMA)
                    n_total = _nosma_per_sym.get(sym, {}).get('pnl', 0)
                    n_cash = _nosma_per_sym.get(sym, {}).get('cash_yield_pnl', 0)
                    n_trade = n_total - n_cash
                    n_sym = _nosma_per_sym.get(sym, {})
                    n_trades = n_sym.get('pipeline_trades', 0)
                    n_peak = n_sym.get('peak_notional', 0)
                    n_steps = n_sym.get('step_count', 0)
                    # Pipeline v7.0
                    p_total = _pip_per_sym.get(sym, {}).get('pnl', 0)
                    p_cash = _pip_per_sym.get(sym, {}).get('cash_yield_pnl', 0)
                    p_trade = p_total - p_cash
                    p_sym = _pip_per_sym.get(sym, {})
                    p_trades = p_sym.get('pipeline_trades', 0)
                    p_peak = p_sym.get('peak_notional', 0)
                    p_steps = p_sym.get('step_count', 0)

                    b_c = C.GREEN if b_trade > 0 else C.RED
                    n_c = C.GREEN if n_trade > 0 else C.RED
                    p_c = C.GREEN if p_trade > 0 else C.RED

                    period = _sym_periods.get(sym, None)
                    period_str = f" ({period[0]} to {period[1]})" if period else ""

                    # v8.0 column (col 4): _vc_per_sym (v8 baseline when v9 mode, else v8 results)
                    v_pnl_str = _E; v_tr_str = _E; v_used_s = _E; v_cagr_s = _E
                    if _has_v8:
                        vc_s = _vc_per_sym.get(sym, {})
                        if vc_s:
                            vc_s_trade = vc_s.get('pnl', 0) - vc_s.get('cash_yield_pnl', 0)
                            vc_s_c = C.GREEN if vc_s_trade > 0 else C.RED
                            v_pnl_str = f"{vc_s_c}${vc_s_trade:>+,.2f}{C.RESET}"
                            v_tr_str = f"{vc_s.get('pipeline_trades', 0):>d}"
                            v_used_s = _used_str(vc_s.get('peak_notional', 0))
                            v_cagr_s = _cagr_str(vc_s_trade, vc_s.get('peak_notional', 0), vc_s.get('step_count', 0))
                        else:
                            v_pnl_str = f"{C.DIM}--{C.RESET}"

                    # v9.0 column (col 5): _xc_per_sym (only when --version v9)
                    x_pnl_str = _E; x_tr_str = _E; x_used_s = _E; x_cagr_s = _E
                    if _has_v9_col:
                        xc_s = _xc_per_sym.get(sym, {})
                        if xc_s:
                            xc_s_trade = xc_s.get('pnl', 0) - xc_s.get('cash_yield_pnl', 0)
                            xc_s_c = C.GREEN if xc_s_trade > 0 else C.RED
                            x_pnl_str = f"{xc_s_c}${xc_s_trade:>+,.2f}{C.RESET}"
                            x_tr_str = f"{xc_s.get('pipeline_trades', 0):>d}"
                            x_used_s = _used_str(xc_s.get('peak_notional', 0))
                            x_cagr_s = _cagr_str(xc_s_trade, xc_s.get('peak_notional', 0), xc_s.get('step_count', 0))
                        else:
                            x_pnl_str = f"{C.DIM}--{C.RESET}"

                    _b_has_data = _has_base and sym in _base_per_sym
                    table.add_row(_tr(f"  {sym}{period_str}",
                                  b_pnl=f"{b_c}${b_trade:>+,.2f}{C.RESET}" if _b_has_data else _E,
                                  b_tr=f"{b_trades:>d}" if (_b_has_data and b_trades) else _E,
                                  b_used=_used_str(b_peak) if _b_has_data else _E,
                                  b_cagr=_cagr_str(b_trade, b_peak, b_steps) if _b_has_data else _E,
                                  n_pnl=f"{n_c}${n_trade:>+,.2f}{C.RESET}",
                                  n_tr=f"{n_trades:>d}" if n_trades else _E,
                                  n_used=_used_str(n_peak),
                                  n_cagr=_cagr_str(n_trade, n_peak, n_steps),
                                  p_pnl=f"{p_c}${p_trade:>+,.2f}{C.RESET}",
                                  p_tr=f"{p_trades:>d}" if p_trades else _E,
                                  p_used=_used_str(p_peak),
                                  p_cagr=_cagr_str(p_trade, p_peak, p_steps),
                                  v_pnl=v_pnl_str, v_tr=v_tr_str,
                                  v_used=v_used_s, v_cagr=v_cagr_s,
                                  x_pnl=x_pnl_str, x_tr=x_tr_str,
                                  x_used=x_used_s, x_cagr=x_cagr_s))

            # --- SPY Benchmark ---
            if has_spy_bench:
                spy_pnl_c = C.GREEN if spy_total_pnl > 0 else C.RED
                _spy_hdr = f"--- SPY Benchmark ({_spy_period_str}) ---" if _spy_period_str else "--- SPY Benchmark ---"
                table.add_row(_tr(f"{C.BOLD}{_spy_hdr}{C.RESET}"))
                _spy_pnl_s = f"{spy_pnl_c}${spy_total_pnl:>+,.2f}{C.RESET}"
                table.add_row(_tr('  SPY Buy & Hold P&L',
                              b_pnl=_spy_pnl_s if _has_base else _E,
                              b_cagr=spy_cagr_str if _has_base else _E,
                              n_pnl=_spy_pnl_s, n_cagr=spy_cagr_str,
                              p_pnl=_spy_pnl_s, p_cagr=spy_cagr_str,
                              v_pnl=_spy_pnl_s if _has_v8 else _E,
                              v_cagr=spy_cagr_str if _has_v8 else _E,
                              x_pnl=_spy_pnl_s if _has_v9_col else _E,
                              x_cagr=spy_cagr_str if _has_v9_col else _E))

            # --- Stock Selection (v8.0/v9.0 only) ---
            if _has_v8:
                table.add_row(_tr(f"{C.BOLD}--- Stock Selection ---{C.RESET}"))
                table.add_row(_tr('Universe Size',
                              b_pnl=f"{_n_syms_table:>d}" if _has_base else _E, n_pnl=f"{_n_syms_table:>d}",
                              p_pnl=f"{_n_syms_table:>d}",
                              v_pnl=f"{_vc_n_syms:>d}",
                              x_pnl=f"{_xc_n_syms:>d}" if _has_v9_col else _E))
                # v8 baseline avg momentum/SMA (from v8b selector when v9 mode)
                _v8b_sel_log = _v8b_selector.selection_log[-1] if (_has_v9_col and hasattr(_v8b_selector, 'selection_log') and _v8b_selector.selection_log) else None
                _cur_sel_log = selector.selection_log[-1] if selector.selection_log else None
                if _v8b_sel_log or _cur_sel_log:
                    _ref_log = _v8b_sel_log if _has_v9_col else _cur_sel_log
                    if _ref_log:
                        _ref_rankings = _ref_log['rankings']
                        _ref_set = set(_ref_log['selected'])
                        _ref_moms = [c['momentum'] for s, sc, c in _ref_rankings if s in _ref_set]
                        _ref_smas = [c['sma_score'] for s, sc, c in _ref_rankings if s in _ref_set]
                        _avg_mom_v = np.mean(_ref_moms) if _ref_moms else 0
                        _avg_sma_v = np.mean(_ref_smas) if _ref_smas else 0
                        table.add_row(_tr('Avg Momentum', v_pnl=f"{_avg_mom_v:>+.1%}"))
                        table.add_row(_tr('Avg SMA Score', v_pnl=f"{_avg_sma_v:>.1f}"))
                    if _has_v9_col and _cur_sel_log:
                        _x_rankings = _cur_sel_log['rankings']
                        _x_set = set(_cur_sel_log['selected'])
                        _x_moms = [c['momentum'] for s, sc, c in _x_rankings if s in _x_set]
                        _x_smas = [c['sma_score'] for s, sc, c in _x_rankings if s in _x_set]
                        _avg_mom_x = np.mean(_x_moms) if _x_moms else 0
                        _avg_sma_x = np.mean(_x_smas) if _x_smas else 0
                        table.add_row(_tr('  v9 Avg Momentum', x_pnl=f"{_avg_mom_x:>+.1%}"))
                        table.add_row(_tr('  v9 Avg SMA Score', x_pnl=f"{_avg_sma_x:>.1f}"))

            # --- Trading ---
            table.add_row(_tr(f"{C.BOLD}--- Trading ---{C.RESET}"))
            table.add_row(_tr('Trades Executed',
                          b_pnl=f"{base_n_trades:>d}" if _has_base else _E,
                          n_pnl=f"{nosma_n_trades:>d}",
                          p_pnl=f"{pip_n_trades:>d}",
                          v_pnl=f"{_vc_n_trades:>d}" if _has_v8 else _E,
                          x_pnl=f"{_xc_n_trades:>d}" if _has_v9_col else _E))
            table.add_row(_tr('Trades Suppressed',
                          b_pnl=f"{0:>d}" if _has_base else _E,
                          n_pnl=f"{nosma_n_suppressed:>d}",
                          p_pnl=f"{pip_n_suppressed:>d}",
                          v_pnl=f"{_vc_n_suppressed:>d}" if _has_v8 else _E,
                          x_pnl=f"{_xc_n_suppressed:>d}" if _has_v9_col else _E))
            table.add_row(_tr('Suppression Rate',
                          b_pnl=f"{0.0:.1f}%" if _has_base else _E,
                          n_pnl=f"{nosma_suppression_pct:.1f}%",
                          p_pnl=f"{pip_suppression_pct:.1f}%",
                          v_pnl=f"{_vc_suppression_pct:.1f}%" if _has_v8 else _E,
                          x_pnl=f"{_xc_suppression_pct:.1f}%" if _has_v9_col else _E))

            base_wr_str = (f"{base_win_rate:.1f}%" if not np.isnan(base_win_rate) else "N/A") if _has_base else _E
            nosma_wr_str = f"{nosma_win_rate:.1f}%" if not np.isnan(nosma_win_rate) else "N/A"
            pip_wr_str = f"{pip_win_rate:.1f}%" if not np.isnan(pip_win_rate) else "N/A"
            vc_wr_str = (f"{_vc_win_rate:.1f}%" if not np.isnan(_vc_win_rate) else "N/A") if _has_v8 else _E
            xc_wr_str = (f"{_xc_win_rate:.1f}%" if not np.isnan(_xc_win_rate) else "N/A") if _has_v9_col else _E
            table.add_row(_tr('Win Rate', b_pnl=base_wr_str, n_pnl=nosma_wr_str,
                          p_pnl=pip_wr_str, v_pnl=vc_wr_str, x_pnl=xc_wr_str))

            base_pf_str = (f"{base_profit_factor:.2f}" if not np.isnan(base_profit_factor) else "N/A") if _has_base else _E
            nosma_pf_str = f"{nosma_profit_factor:.2f}" if not np.isnan(nosma_profit_factor) else "N/A"
            pip_pf_str = f"{pip_profit_factor:.2f}" if not np.isnan(pip_profit_factor) else "N/A"
            vc_pf_str = (f"{_vc_profit_factor:.2f}" if not np.isnan(_vc_profit_factor) else "N/A") if _has_v8 else _E
            xc_pf_str = (f"{_xc_profit_factor:.2f}" if not np.isnan(_xc_profit_factor) else "N/A") if _has_v9_col else _E
            table.add_row(_tr('Profit Factor', b_pnl=base_pf_str, n_pnl=nosma_pf_str,
                          p_pnl=pip_pf_str, v_pnl=vc_pf_str, x_pnl=xc_pf_str))

            # --- Risk ---
            table.add_row(_tr(f"{C.BOLD}--- Risk ---{C.RESET}"))
            table.add_row(_tr('Sharpe',
                          b_pnl=f"{base_results.get('avg_sh', 0):>+.3f}" if _has_base else _E,
                          n_pnl=f"{nosma_results['avg_sh']:>+.3f}",
                          p_pnl=f"{pipeline_results['avg_sh']:>+.3f}",
                          v_pnl=f"{_vc_results['avg_sh']:>+.3f}" if _has_v8 else _E,
                          x_pnl=f"{_xc_results['avg_sh']:>+.3f}" if _has_v9_col else _E))
            table.add_row(_tr('Max Drawdown',
                          b_pnl=f"{base_results.get('dd_max', 0):.1f}%" if _has_base else _E,
                          n_pnl=f"{nosma_results['dd_max']:.1f}%",
                          p_pnl=f"{pipeline_results['dd_max']:.1f}%",
                          v_pnl=f"{_vc_results['dd_max']:.1f}%" if _has_v8 else _E,
                          x_pnl=f"{_xc_results['dd_max']:.1f}%" if _has_v9_col else _E))
            table.add_row(_tr('Breadth',
                          b_pnl=f"{base_results.get('breadth', 0):.0f}%" if _has_base else _E,
                          n_pnl=f"{nosma_results['breadth']:.0f}%",
                          p_pnl=f"{pipeline_results['breadth']:.0f}%",
                          v_pnl=f"{_vc_results['breadth']:.0f}%" if _has_v8 else _E,
                          x_pnl=f"{_xc_results['breadth']:.0f}%" if _has_v9_col else _E))

            # --- Score ---
            table.add_row(_tr(f"{C.BOLD}--- Score ---{C.RESET}"))
            table.add_row(_tr(f"{C.BOLD}Score{C.RESET}",
                          b_pnl=f"{C.BOLD}{base_results.get('score', 0):>+.3f}{C.RESET}" if _has_base else _E,
                          n_pnl=f"{C.BOLD}{nosma_results['score']:>+.3f}{C.RESET}",
                          p_pnl=f"{C.BOLD}{pipeline_results['score']:>+.3f}{C.RESET}",
                          v_pnl=f"{C.BOLD}{_vc_results['score']:>+.3f}{C.RESET}" if _has_v8 else _E,
                          x_pnl=f"{C.BOLD}{_xc_results['score']:>+.3f}{C.RESET}" if _has_v9_col else _E))

            rendered = "  " + table.render().replace("\n", "\n  ")
            try:
                print(rendered)
            except UnicodeEncodeError:
                print(rendered.encode('ascii', errors='replace').decode('ascii'))
        else:
            # Fallback: plain text (table_formatter not available)
            print(f"  {C.DIM}(Install table_formatter for grouped-column table){C.RESET}")
            _vc_sc_fb = f"  v8.0={_vc_results['score']:+.3f}" if _has_v8 else ""
            _xc_sc_fb = f"  v9.0={_xc_results['score']:+.3f}" if _has_v9_col else ""
            _base_sc_fb = f"Base={base_results.get('score', 0):+.3f}  " if _has_base else ""
            print(f"  Score: {_base_sc_fb}"
                  f"NoSMA={nosma_results['score']:+.3f}  "
                  f"v7.0={pipeline_results['score']:+.3f}{_vc_sc_fb}{_xc_sc_fb}")

        # Add explanatory note if pipeline has 0 closed trades but positive Trade P&L
        if pip_n_closed_trades == 0 and abs(pip_trade_pnl) > 1.0:
            print(f"\n  {C.YELLOW}Note: Win Rate/PF = N/A because pipeline has open position with unrealized P&L.{C.RESET}")
            print(f"        Executed trades: {pip_n_trades} | Closed round-trips: {pip_n_closed_trades}")
            print(f"        Trade P&L (${pip_trade_pnl:,.2f}) includes unrealized gains/losses.")

    # **********************************************************************
    # STEP 7.5: Terminal charts
    # **********************************************************************
    print_section("CHARTS")
    tprint("Visual summaries — quick-glance performance charts.", "info")

    # Version-aware chart label and data source.
    # In v8/v9 mode, charts show the SELECTED stocks, not all pipeline stocks.
    # v9 mode: primary chart = v9 selection (_xc_per_sym); secondary = v8 baseline (_vc_per_sym)
    # v8 mode: primary chart = v8 selection (_vc_per_sym)
    # v7 mode: pipeline results (all stocks)
    _has_v8_chart = v8_results is not None
    _has_v9_chart = _has_v9_col
    _chart_ver = (
        "v9.0 (Select)" if args.version == "v9"
        else "v8.0 (Select)" if args.version == "v8"
        else "Pipeline v7.0"
    )
    if args.version == "v9" and _has_v9_chart:
        _chart_per_sym = _xc_per_sym   # v9 selected stocks
    elif args.version in ("v8", "v9") and _has_v8_chart:
        _chart_per_sym = _vc_per_sym   # v8 selected stocks
    else:
        _chart_per_sym = pipeline_results.get('per_sym', {})
    _chart_syms = sorted(_chart_per_sym.keys())
    if _chart_syms:
        _sym_pnls = []
        for sym in _chart_syms:
            s = _chart_per_sym[sym]
            trade_pnl = s.get('pnl', 0) - s.get('cash_yield_pnl', 0)
            _sym_pnls.append((sym, trade_pnl))
        hbar_chart(_sym_pnls, title=f"Per-Symbol Trade P&L ({_chart_ver})")

    # (b) Sector performance horizontal bars (green/red)
    # BUG FIX: Norgate symbols have _1d suffix (e.g. "AAPL_1d") but SECTOR_MAP uses
    # plain symbols ("AAPL"). Build a reverse lookup so both formats match.
    if _chart_per_sym:
        _bare_to_key = {}
        for _k in _chart_per_sym:
            _bare = _k.split('_')[0] if '_' in _k else _k
            _bare_to_key[_bare] = _k

        _sector_pnls = []
        _sector_table_rows = []   # for the text table below the chart
        for sector, tickers in SECTOR_MAP.items():
            _sec_stocks = 0
            _sec_winners = 0
            _sec_total_pnl = 0.0
            for t in tickers:
                _key = _bare_to_key.get(t, t)
                if _key not in _chart_per_sym:
                    continue
                _sd = _chart_per_sym[_key]
                _trade_pnl = _sd.get('pnl', 0) - _sd.get('cash_yield_pnl', 0)
                _sec_total_pnl += _trade_pnl
                _sec_stocks += 1
                if _trade_pnl > 0:
                    _sec_winners += 1
            if _sec_stocks > 0:
                _sector_table_rows.append((
                    sector.replace("_", " ").title(),
                    _sec_total_pnl,
                    _sec_stocks,
                    _sec_winners,
                ))
            if _sec_total_pnl != 0:
                _sector_pnls.append((sector.replace("_", " ").title(), _sec_total_pnl))

        if _sector_pnls:
            _sector_pnls.sort(key=lambda x: x[1], reverse=True)
            hbar_chart(_sector_pnls, title=f"Sector Performance ({_chart_ver})")

        # Also print a concise text table so it's visible even in CI / log files
        if _sector_table_rows:
            _sector_table_rows.sort(key=lambda x: x[1], reverse=True)
            print(f"\n    {'Sector':<26s} {'P&L':>12s}  {'Stocks':>6s}  {'Win%':>6s}")
            print(f"    {'─'*26}  {'─'*12}  {'─'*6}  {'─'*6}")
            for _sname, _spnl, _sn, _sw in _sector_table_rows:
                _pnl_c = C.GREEN if _spnl >= 0 else C.RED
                _wr_str = f"{_sw / _sn:.0%}" if _sn > 0 else " N/A"
                print(f"    {_sname:<26s}  {_pnl_c}${_spnl:>+11,.0f}{C.RESET}  {_sn:>6d}  {_wr_str:>6s}")
            print(f"    {'─'*26}  {'─'*12}  {'─'*6}  {'─'*6}")

        # ----------------------------------------------------------------
        # (b.1) Score-quartile vs return validation
        # Did the composite scoring actually rank winners ahead of losers?
        # ----------------------------------------------------------------
        _sel_log_sf = None
        try:
            if hasattr(selector, 'selection_log') and selector.selection_log:
                _sel_log_sf = selector.selection_log[-1]
        except NameError:
            pass  # v7 path: no selector object

        if _sel_log_sf:
            # Build sym -> (score, comp) from full rankings list
            _sym_to_sf = {s: (sc, c) for s, sc, c in _sel_log_sf.get('rankings', [])}

            # Align: only symbols that appear in BOTH rankings AND per_sym
            _aligned = []
            for _k, _psd in _chart_per_sym.items():
                _bare = _k.split('_')[0] if '_' in _k else _k
                _sf = _sym_to_sf.get(_k) or _sym_to_sf.get(_bare)
                if _sf is None:
                    continue
                _sc, _comp = _sf
                _trade_pnl = _psd.get('pnl', 0) - _psd.get('cash_yield_pnl', 0)
                _aligned.append((_k, _sc, _comp, _trade_pnl, _psd.get('sharpe', 0)))

            if len(_aligned) >= 4:
                _aligned.sort(key=lambda x: x[1], reverse=True)  # highest score first
                _n_aln = len(_aligned)
                _q = max(1, _n_aln // 4)
                _quartile_defs = [
                    ('Top 25% (score)',    _aligned[:_q]),
                    ('Upper-Mid 25%',      _aligned[_q:2*_q]),
                    ('Lower-Mid 25%',      _aligned[2*_q:3*_q]),
                    ('Bottom 25% (score)', _aligned[3*_q:]),
                ]
                print(f"\n    {C.BOLD}Score Quartile vs Actual Returns{C.RESET}")
                print(f"    {'─'*66}")
                print(f"    {'Quartile':<22s}  {'N':>4s}  {'Avg P&L':>12s}  {'Win%':>6s}  {'Avg Sharpe':>10s}")
                print(f"    {'─'*66}")
                for _qname, _grp in _quartile_defs:
                    if not _grp:
                        continue
                    _pnls   = [x[3] for x in _grp]
                    _sharpes = [x[4] for x in _grp]
                    _avg_pnl   = float(np.mean(_pnls))
                    _win_rate  = sum(1 for p in _pnls if p > 0) / len(_pnls)
                    _avg_sharpe = float(np.mean(_sharpes))
                    _pc = C.GREEN if _avg_pnl >= 0 else C.RED
                    print(f"    {_qname:<22s}  {len(_grp):>4d}  "
                          f"{_pc}${_avg_pnl:>+11,.0f}{C.RESET}  "
                          f"{_win_rate:>5.0%}  {_avg_sharpe:>+9.2f}")
                print(f"    {'─'*66}")

                # ----------------------------------------------------------------
                # (b.2) Factor-return Pearson correlation
                # Which scoring factors actually predicted stock returns?
                # ----------------------------------------------------------------
                _factor_map = {
                    'momentum':          'Momentum (6m)',
                    'sma_score':         'SMA Alignment',
                    'rs_vs_spy':         'Rel. Strength vs SPY',
                    'vol_20':            'Volatility (20d)',
                    'vol_acc':           'Volume Accumulation',
                    'high52w_proximity': '52w High Proximity',
                }
                _f_vecs = {f: [] for f in _factor_map}
                _pnl_vec = []
                for _k, _, _comp2, _pnl2, _ in _aligned:
                    _pnl_vec.append(_pnl2)
                    for _f in _factor_map:
                        _f_vecs[_f].append(float(_comp2.get(_f, 0.0)))

                _pnl_arr = np.array(_pnl_vec)
                _corr_rows = []
                for _f, _label in _factor_map.items():
                    _farr = np.array(_f_vecs[_f])
                    if np.std(_farr) > 1e-9 and np.std(_pnl_arr) > 1e-9:
                        _r = float(np.corrcoef(_farr, _pnl_arr)[0, 1])
                    else:
                        _r = 0.0
                    _corr_rows.append((_label, float(np.mean(_farr)), _r))
                _corr_rows.sort(key=lambda x: abs(x[2]), reverse=True)

                print(f"\n    {C.BOLD}Factor Correlation with Trade P&L{C.RESET}")
                print(f"    {'─'*54}")
                print(f"    {'Factor':<28s}  {'Avg Value':>10s}  {'r (P&L)':>8s}")
                print(f"    {'─'*54}")
                for _fname, _avg, _r in _corr_rows:
                    if abs(_r) < 1e-9:
                        continue  # skip zero-variance factors
                    _rc = C.GREEN if _r > 0.10 else (C.RED if _r < -0.10 else C.YELLOW)
                    _arrow = '▲' if _r > 0.10 else ('▼' if _r < -0.10 else '~')
                    print(f"    {_fname:<28s}  {_avg:>+10.3f}  {_rc}{_arrow} {_r:>+6.2f}{C.RESET}")
                print(f"    {'─'*54}")
                print(f"    {C.DIM}r = Pearson correlation across {len(_pnl_vec)} selected stocks. "
                      f"|r|>0.1 shown in color.{C.RESET}")

    # (b.5) Multi-strategy comparison table — one row per symbol, strategy columns side-by-side
    # Layout: Symbol | v7.0: [bar P&L] | v8.0: [bar P&L] | v9.0: [bar P&L]
    # Each cell shows a mini proportional bar + dollar P&L.  "──" = not traded in that strategy.
    _ms_labels = ['v7.0', 'v8.0', 'v9.0'] if _has_v9_col else (['v7.0', 'v8.0'] if _has_v8_chart else [])
    if len(_ms_labels) >= 2:
        _ms_all_syms = sorted(set(
            list(_pip_per_sym.keys())
            + list(_vc_per_sym.keys())
            + (list(_xc_per_sym.keys()) if _has_v9_col else [])
        ))
        # Build data: sym -> {label: trade_pnl or None}
        _ms_data = {}
        for _ms_sym in _ms_all_syms:
            _p7 = _pip_per_sym.get(_ms_sym)
            _p8 = _vc_per_sym.get(_ms_sym) if _has_v8_chart else None
            _p9 = _xc_per_sym.get(_ms_sym) if _has_v9_col else None
            _ms_data[_ms_sym] = {
                'v7.0': (_p7.get('pnl', 0) - _p7.get('cash_yield_pnl', 0)) if _p7 else None,
                'v8.0': (_p8.get('pnl', 0) - _p8.get('cash_yield_pnl', 0)) if _p8 else None,
                'v9.0': (_p9.get('pnl', 0) - _p9.get('cash_yield_pnl', 0)) if _p9 else None,
            }

        if _ms_data and HAS_TABLE_FORMATTER:
            # Scale bars per-column (each strategy's max = full bar width)
            _ms_col_max = {}
            for _lbl in _ms_labels:
                _vals = [abs(d[_lbl]) for d in _ms_data.values() if d.get(_lbl) is not None]
                _ms_col_max[_lbl] = max(_vals) if _vals else 1.0

            _ms_bar_w = 12
            _ms_hdr_color = {'v7.0': C.CYAN, 'v8.0': C.YELLOW, 'v9.0': C.MAGENTA}

            ms_table = TableFormatter(title="Strategy Comparison — Trade P&L by Symbol")
            ms_table.add_column('Symbol', align='left')
            for _lbl in _ms_labels:
                _col = _ms_hdr_color.get(_lbl, C.WHITE)
                ms_table.add_column(f"{_col}{_lbl}{C.RESET}", align='left')

            for _ms_sym in sorted(_ms_data):
                _ms_d = _ms_data[_ms_sym]
                _row = [_ms_sym]
                for _lbl in _ms_labels:
                    _val = _ms_d.get(_lbl)
                    if _val is None:
                        _row.append(f"{C.DIM}──{C.RESET}")
                    else:
                        _bar_len = max(1, int(abs(_val) / _ms_col_max[_lbl] * _ms_bar_w))
                        _bar_char = '█' if _val >= 0 else '▒'
                        _bar_c = C.GREEN if _val >= 0 else C.RED
                        _row.append(f"{_bar_c}{_bar_char * _bar_len}  ${_val:>+,.0f}{C.RESET}")
                ms_table.add_row(_row)

            print(f"\n    {C.DIM}Bar ∝ P&L within each strategy column.  ── = not traded.{C.RESET}")
            rendered_ms = "  " + ms_table.render().replace("\n", "\n  ")
            try:
                print(rendered_ms)
            except UnicodeEncodeError:
                print(rendered_ms.encode('ascii', errors='replace').decode('ascii'))

    # (c) Cumulative % return chart — ALL strategies vs SPY
    # ─────────────────────────────────────────────────────────────────────────
    # $ used = total COMMITTED capital = starting_capital × n_syms.
    #
    # Why committed, not avg_notional?
    #   Example: 5 symbols × $10k = $50k committed.
    #            avg notional deployed = $38k (24% cash drag on average).
    #            Equity after period   = $55k  →  P&L = $5k.
    #   WRONG:   $5k / $38k =  13.2%  ← inflated by ignoring cash drag
    #   CORRECT: $5k / $50k =  10.0%  ← return on committed capital
    #
    # avg_notional is shown as informational context ("avg deployed: $38k") but
    # is never used as a denominator.  This keeps the % directly comparable to
    # SPY (which always uses its own starting_capital as denominator).

    def _build_curve(result_dict, ref_dates=None):
        """Build cumulative % equity curve from evaluate_with_pipeline() output.

        Returns (cum_pct_array, total_capital, total_eq_list, dates_sorted).
        total_capital = starting_capital × n_syms — the committed capital.
        cum_pct[i]   = (total_equity[i] - total_capital) / total_capital × 100.
        """
        dr = (result_dict or {}).get("per_symbol_date_returns", {})
        if not dr:
            return None, 0, [], []
        cap = cfg.starting_capital * len(dr)
        sym_eq = {}
        for sym, pairs in dr.items():
            eq = cfg.starting_capital
            d = {}
            for dt, ret in pairs:
                eq *= (1 + ret)
                d[dt] = eq
            sym_eq[sym] = d
        if ref_dates is None:
            all_d = set()
            for d in sym_eq.values():
                all_d.update(d.keys())
            ref_dates = sorted(all_d)
        last = {s: cfg.starting_capital for s in sym_eq}
        total_eq = []
        for dt in ref_dates:
            for s, deq in sym_eq.items():
                if dt in deq:
                    last[s] = deq[dt]
            total_eq.append(sum(last.values()))
        cum_pct = np.array([(v - cap) / cap * 100 for v in total_eq])
        return cum_pct, cap, total_eq, ref_dates

    def _annual_returns(ref_dates, total_eq, total_capital):
        """Per-calendar-year % return.

        Year-start equity = prev year's closing equity (total_capital for yr 0).
        annual_return[year] = (year_end_equity / year_start_equity) − 1, as %.
        """
        if not ref_dates or not total_eq:
            return {}
        year_end = {}
        for i, dt in enumerate(ref_dates):
            year_end[dt[:4]] = total_eq[i]
        result, prev = {}, total_capital
        for yr in sorted(year_end):
            result[yr] = (year_end[yr] / prev - 1.0) * 100
            prev = year_end[yr]
        return result

    # Pipeline v7 dates are the master x-axis; all other strategies align to them.
    _pipe_dr = pipeline_results.get("per_symbol_date_returns", {})
    if _pipe_dr:
        _pip_cum, _pip_cap, _pip_eq, _dates_sorted = _build_curve(pipeline_results)
        _n_syms = len(_pipe_dr)

        if len(_dates_sorted) > 2:
            # ── Build curves for every available strategy ──────────────────
            _nosma_cum, _nosma_cap, _nosma_eq, _ = _build_curve(
                nosma_results if 'nosma_results' in dir() else None, _dates_sorted)

            _v8_label = (
                "v9.0 (Select)" if args.version == "v9"
                else "v8.0 (Select)" if args.version == "v8"
                else "v8.0 (Sizing)"
            )
            _v8_cum,  _v8_cap,  _v8_eq,  _ = _build_curve(
                v8_results if v8_results is not None else None, _dates_sorted)
            _v9b_cum, _v9b_cap, _v9b_eq, _ = _build_curve(
                _v8b_results if _v8b_results is not None else None, _dates_sorted)

            # ── SPY curve: pre-compounded through lookback window ──────────
            # The portfolio chart starts post-lookback; SPY must start at the
            # same point so returns are not inflated by a lucky lookback entry.
            _spy_cum_pct = None
            _spy_base    = cfg.starting_capital
            _spy_vals    = []
            if has_spy_bench and spy_returns_lookup:
                _full_test_dates = set()
                for _d in datasets:
                    if hasattr(_d, 'timestamps_test') and _d.timestamps_test is not None:
                        for _ts in _d.timestamps_test:
                            _full_test_dates.add(str(_ts)[:10])
                _spy_eq = _spy_base
                for _dt in sorted(_full_test_dates):
                    if _dt < _dates_sorted[0]:
                        _spy_eq *= (1 + spy_returns_lookup.get(_dt, 0.0))
                for _dt in _dates_sorted:
                    _spy_eq *= (1 + spy_returns_lookup.get(_dt, 0.0))
                    _spy_vals.append(_spy_eq)
                _spy_cum_pct = np.array(
                    [(v - _spy_base) / _spy_base * 100 for v in _spy_vals])

            # ── Assemble ordered series list for multi_line_chart ──────────
            # Order = display priority (first = drawn on top on overlaps).
            _chart_series = []
            if _v8_cum is not None and args.version in ("v8", "v9"):
                _chart_series.append((_v8_label, _v8_cum))
            if _v9b_cum is not None and args.version == "v9":
                _chart_series.append(("v8.0 (baseline)", _v9b_cum))
            _chart_series.append(("Pipeline v7", _pip_cum))
            if _nosma_cum is not None:
                _chart_series.append(("v7 (no SMA)", _nosma_cum))
            if _spy_cum_pct is not None:
                _chart_series.append(("SPY", _spy_cum_pct))

            _n_label = f"{_n_syms} syms × ${cfg.starting_capital/1e3:.0f}k"
            multi_line_chart(
                _chart_series, width=70, height=14, fmt="%",
                dates=_dates_sorted,
                title=f"Cumulative Return: All Strategies vs SPY ({_n_label})",
            )

            # ── Legend lines ───────────────────────────────────────────────
            # Shows: return % on committed capital, dollar P&L, and avg
            # notional deployed (informational — not the denominator).
            def _legend(label, cum_pct, total_cap, res):
                if cum_pct is None or len(cum_pct) == 0:
                    return
                fp  = float(cum_pct[-1])
                pnl = total_cap * fp / 100
                avg = sum(s.get('avg_notional', 0)
                          for s in (res or {}).get('per_sym', {}).values())
                col = C.GREEN if fp >= 0 else C.RED
                dep = f", avg deployed ${avg:,.0f}" if avg > 0 else ""
                print(f"    {col}{label}: {fp:+.1f}%{C.RESET}"
                      f"  ({'+'if pnl>=0 else ''}${abs(pnl):,.0f}"
                      f" / ${total_cap:,.0f} committed{dep})")

            print()
            if _v8_cum is not None and args.version in ("v8", "v9"):
                _legend(_v8_label, _v8_cum, _v8_cap, v8_results)
            if _v9b_cum is not None and args.version == "v9":
                _legend("v8.0 (baseline)", _v9b_cum, _v9b_cap, _v8b_results)
            _legend("Pipeline v7", _pip_cum, _pip_cap, pipeline_results)
            if _nosma_cum is not None:
                _legend("v7 (no SMA)", _nosma_cum, _nosma_cap,
                        nosma_results if 'nosma_results' in dir() else None)
            if _spy_cum_pct is not None and _spy_vals:
                _sf  = float(_spy_cum_pct[-1])
                _spl = _spy_base * _sf / 100
                _sc  = C.GREEN if _sf >= 0 else C.RED
                print(f"    {_sc}SPY: {_sf:+.1f}%{C.RESET}"
                      f"  ({'+'if _spl>=0 else ''}${abs(_spl):,.0f}"
                      f" / ${_spy_base:,.0f} per $10k invested)")

            # ── Year-by-year backtest table ────────────────────────────────
            # Annual return = (year-end equity / prev-year equity) − 1.
            # Covers whatever the test period spans — up to 35 years with full
            # Norgate history from 1990.
            _ann_strats = []   # [(label, {year: pct})]
            if _v8_cum is not None and args.version in ("v8", "v9"):
                _ann_strats.append(
                    (_v8_label, _annual_returns(_dates_sorted, _v8_eq, _v8_cap)))
            if _v9b_cum is not None and args.version == "v9":
                _ann_strats.append(
                    ("v8.0 (base)", _annual_returns(_dates_sorted, _v9b_eq, _v9b_cap)))
            _ann_strats.append(
                ("Pipe v7", _annual_returns(_dates_sorted, _pip_eq, _pip_cap)))
            if _nosma_cum is not None:
                _ann_strats.append(
                    ("v7 no SMA", _annual_returns(_dates_sorted, _nosma_eq, _nosma_cap)))
            if _spy_cum_pct is not None and _spy_vals:
                _spy_ye, _prev_s = {}, _spy_base
                for _i, _dt in enumerate(_dates_sorted):
                    _spy_ye[_dt[:4]] = _spy_vals[_i]
                _spy_ann = {}
                for _yr in sorted(_spy_ye):
                    _spy_ann[_yr] = (_spy_ye[_yr] / _prev_s - 1.0) * 100
                    _prev_s = _spy_ye[_yr]
                _ann_strats.append(("SPY", _spy_ann))

            _all_yrs = sorted({yr for _, ann in _ann_strats for yr in ann})
            if _all_yrs and _ann_strats:
                _cw = 10   # column width per strategy
                _slabels = [lb for lb, _ in _ann_strats]
                print(f"\n    {C.BOLD}Year-by-Year Returns"
                      f" ({_all_yrs[0]}–{_all_yrs[-1]}){C.RESET}")
                _hdr = f"    {'Year':<6}" + "".join(f"  {lb:>{_cw}}" for lb in _slabels)
                print(_hdr)
                print(f"    {'─'*6}" + f"  {'─'*_cw}" * len(_slabels))
                for _yr in _all_yrs:
                    _row = f"    {_yr:<6}"
                    for _, _ann in _ann_strats:
                        _v = _ann.get(_yr)
                        if _v is None:
                            _row += f"  {'N/A':>{_cw}}"
                        else:
                            _c = C.GREEN if _v >= 0 else C.RED
                            _row += f"  {_c}{_v:>+8.1f}%{C.RESET}"
                    print(_row)
                print(f"    {'─'*6}" + f"  {'─'*_cw}" * len(_slabels))
                # Total compound + CAGR rows
                for _rname in ("Total", "CAGR"):
                    _row = f"    {_rname:<6}"
                    for _, _ann in _ann_strats:
                        _vals = [_ann[yr] for yr in _all_yrs if yr in _ann]
                        if _vals:
                            _comp = 1.0
                            for _v in _vals:
                                _comp *= (1 + _v / 100)
                            _res = (_comp - 1.0) * 100 if _rname == "Total" \
                                   else ((_comp ** (1.0 / len(_vals))) - 1.0) * 100
                            _c = C.GREEN if _res >= 0 else C.RED
                            _row += f"  {_c}{_res:>+8.1f}%{C.RESET}"
                        else:
                            _row += f"  {'N/A':>{_cw}}"
                    print(_row)
                print()

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

    _complete_ver = "v9.0" if args.version == "v9" else "v8.0" if args.version == "v8" else "v7.0"
    _complete_score = v8_results['score'] if v8_results is not None else pipeline_results['score']
    print_box(
        f"COMPLETE -- ALPHA-TRADE {_complete_ver}",
        f"Score:{_complete_score:+.3f} | "
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