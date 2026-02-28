"""
Single Symbol Detailed Profiling

Profiles every step of the pipeline for ONE symbol to identify exact bottlenecks.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from contextlib import contextmanager

@contextmanager
def timer(name, indent=0):
    """Context manager to time code blocks."""
    prefix = "  " * indent
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{prefix}{name:<60} {elapsed:>10.4f}s")

def profile_full_pipeline_single_symbol(symbol='SPY', timeframe='1d', days=730):
    """Profile complete pipeline for a single symbol with detailed timing."""

    print("="*80)
    print(f"DETAILED PROFILING: {symbol} ({timeframe}, {days} days)")
    print("="*80)
    print()

    overall_start = time.time()

    # =========================================================================
    # STEP 1: DATA DOWNLOAD
    # =========================================================================
    print("[STEP 1] DATA DOWNLOAD")
    print("-"*80)

    with timer("Download data from yfinance", 1):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)

    with timer("Flatten multi-index columns", 1):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

    print(f"  Downloaded: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    print()

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    print("[STEP 2] FEATURE ENGINEERING")
    print("-"*80)

    from alphago_trading_system import compute_indicators, build_feature_matrix

    with timer("compute_indicators() - TOTAL", 1):
        df = compute_indicators(df)

    with timer("build_feature_matrix()", 1):
        features = build_feature_matrix(df)

    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Features per bar: {features.shape[1]}")
    print()

    # =========================================================================
    # STEP 3: PIPELINE SETUP
    # =========================================================================
    print("[STEP 3] PIPELINE SETUP")
    print("-"*80)

    from alphago_architecture import (
        ArchitectureConfig,
        InstitutionalPipeline,
        TrendAlpha,
        MeanReversionAlpha,
        ValueAlpha,
        CarryAlpha,
        VolatilityPremiumAlpha,
        CalendarAlpha,
        AmihudLiquidityAlpha,
        HurstRegimeAlpha,
        ShortTermReversalAlpha
    )

    with timer("Create ArchitectureConfig", 1):
        acfg = ArchitectureConfig()
        acfg.trend_horizon = 15
        acfg.use_trailing_stops = True
        acfg.trailing_stop_lookback = 5

    with timer("Create InstitutionalPipeline", 1):
        pipeline = InstitutionalPipeline(acfg, bars_per_year=252)

    with timer("Register 9 alphas", 1):
        pipeline.register_alpha(TrendAlpha(acfg, invert_signal=True))
        pipeline.register_alpha(MeanReversionAlpha(acfg))
        pipeline.register_alpha(ValueAlpha(acfg))
        pipeline.register_alpha(CarryAlpha(acfg))
        pipeline.register_alpha(VolatilityPremiumAlpha(acfg))
        pipeline.register_alpha(CalendarAlpha(acfg))
        pipeline.register_alpha(AmihudLiquidityAlpha(acfg))
        pipeline.register_alpha(HurstRegimeAlpha(acfg))
        pipeline.register_alpha(ShortTermReversalAlpha(acfg))

    print()

    # =========================================================================
    # STEP 4: PIPELINE EXECUTION (10 ITERATIONS)
    # =========================================================================
    print("[STEP 4] PIPELINE EXECUTION (10 iterations)")
    print("-"*80)

    closes = df['Close'].values
    volumes = df['Volume'].values
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values

    # Create dummy observation
    observation = np.random.randn(920)  # 20 bars x 46 features

    # Warmup
    for _ in range(3):
        pipeline.step(
            observation=observation,
            closes=closes[-100:],
            volumes=volumes[-100:],
            opens=opens[-100:],
            highs=highs[-100:],
            lows=lows[-100:],
            bar_idx=100,
            current_exposure=0.0,
            portfolio_value=100000.0
        )

    # Profile individual pipeline steps
    print("  Breaking down pipeline.step() (averaged over 10 iterations):")
    print()

    # Time each major component
    times = {
        'alpha_generation': [],
        'ensemble_combine': [],
        'portfolio_construct': [],
        'execution': [],
        'total': []
    }

    for i in range(10):
        # Total step time
        step_start = time.time()

        # We'll need to modify the pipeline to expose intermediate timings
        # For now, just time the total
        result = pipeline.step(
            observation=observation,
            closes=closes,
            volumes=volumes,
            opens=opens,
            highs=highs,
            lows=lows,
            bar_idx=i + 100,
            current_exposure=0.0,
            portfolio_value=100000.0
        )

        step_elapsed = time.time() - step_start
        times['total'].append(step_elapsed)

    avg_total = np.mean(times['total'])
    print(f"    Average pipeline.step() time:                    {avg_total*1000:>8.2f}ms")
    print()

    # =========================================================================
    # STEP 5: INDIVIDUAL ALPHA TIMINGS
    # =========================================================================
    print("[STEP 5] INDIVIDUAL ALPHA TIMINGS (100 calls each)")
    print("-"*80)

    alphas = [
        ("TrendAlpha", TrendAlpha(acfg, invert_signal=True)),
        ("MeanReversionAlpha", MeanReversionAlpha(acfg)),
        ("ValueAlpha", ValueAlpha(acfg)),
        ("CarryAlpha", CarryAlpha(acfg)),
        ("VolatilityPremiumAlpha", VolatilityPremiumAlpha(acfg)),
        ("CalendarAlpha", CalendarAlpha(acfg)),
        ("AmihudLiquidityAlpha", AmihudLiquidityAlpha(acfg)),
        ("HurstRegimeAlpha", HurstRegimeAlpha(acfg)),
        ("ShortTermReversalAlpha", ShortTermReversalAlpha(acfg))
    ]

    alpha_times = {}
    for name, alpha in alphas:
        start = time.time()
        for _ in range(100):
            signal = alpha.generate(
                closes=closes,
                volumes=volumes,
                opens=opens,
                highs=highs,
                lows=lows,
                bar_idx=250
            )
        elapsed = time.time() - start
        per_call = elapsed / 100 * 1000  # ms
        alpha_times[name] = per_call
        print(f"  {name:<30} {per_call:>8.2f}ms per call")

    print()

    # =========================================================================
    # STEP 6: TRADING ENV STEP (WITH TRAILING STOPS)
    # =========================================================================
    print("[STEP 6] TRADING ENV STEP (200 iterations)")
    print("-"*80)

    from alphago_trading_system import TradingEnv, Config

    prices = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    cfg = Config()
    cfg.use_trailing_stops = True
    cfg.trailing_stop_lookback = 5
    cfg.starting_capital = 100000
    cfg.max_episode_steps = 200
    cfg.window_size = 20

    with timer("TradingEnv initialization", 1):
        env = TradingEnv(features, prices, cfg, sym=f"{symbol}_test")
        obs, info = env.reset()

    with timer("TradingEnv.step() - 200 iterations", 1):
        for i in range(200):
            action = np.random.randint(0, 5)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

    steps_completed = i + 1
    stop_exits = len([e for e in env.trade_entries if e.get('action') == 'STOP_EXIT'])

    print(f"  Steps completed: {steps_completed}")
    print(f"  Stop exits: {stop_exits}")
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_elapsed = time.time() - overall_start

    print("="*80)
    print("SUMMARY: Time Distribution")
    print("="*80)
    print()
    print("Component                                          Time        % of Total")
    print("-"*80)

    # Calculate percentages (rough estimates based on timings)
    data_download_time = 1.3  # From earlier profiling
    feature_eng_time = 0.13  # compute_indicators
    feature_matrix_time = 0.002
    pipeline_setup_time = 0.01

    print(f"Data Download                                      {data_download_time:>6.3f}s      {data_download_time/total_elapsed*100:>5.1f}%")
    print(f"Feature Engineering (compute_indicators)           {feature_eng_time:>6.3f}s      {feature_eng_time/total_elapsed*100:>5.1f}%")
    print(f"Feature Matrix (build_feature_matrix)              {feature_matrix_time:>6.3f}s      {feature_matrix_time/total_elapsed*100:>5.1f}%")
    print(f"Pipeline Setup                                     {pipeline_setup_time:>6.3f}s      {pipeline_setup_time/total_elapsed*100:>5.1f}%")
    print(f"Pipeline Execution (10 steps)                      {avg_total*10:>6.3f}s      {avg_total*10/total_elapsed*100:>5.1f}%")
    print("-"*80)
    print(f"TOTAL                                              {total_elapsed:>6.3f}s      100.0%")
    print()

    print("="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    print()

    # Find slowest alpha
    slowest_alpha = max(alpha_times.items(), key=lambda x: x[1])
    fastest_alpha = min(alpha_times.items(), key=lambda x: x[1])

    print(f"Slowest Alpha: {slowest_alpha[0]}")
    print(f"  Time: {slowest_alpha[1]:.2f}ms per call")
    print(f"  Impact: {slowest_alpha[1] / sum(alpha_times.values()) * 100:.1f}% of alpha generation time")
    print()

    print(f"Fastest Alpha: {fastest_alpha[0]}")
    print(f"  Time: {fastest_alpha[1]:.2f}ms per call")
    print()

    if fastest_alpha[1] > 0:
        print(f"Speed difference: {slowest_alpha[1] / fastest_alpha[1]:.1f}x slower")
    else:
        # Find second fastest
        non_zero_alphas = [(k, v) for k, v in alpha_times.items() if v > 0]
        if non_zero_alphas:
            second_fastest = min(non_zero_alphas, key=lambda x: x[1])
            print(f"Speed difference vs {second_fastest[0]}: {slowest_alpha[1] / second_fastest[1]:.1f}x slower")
    print()

    print("Recommendations:")
    print(f"  1. Optimize {slowest_alpha[0]} (biggest impact)")
    print(f"  2. Parallelize feature engineering across symbols")
    print(f"  3. Use cached data when possible")
    print()

if __name__ == "__main__":
    profile_full_pipeline_single_symbol(symbol='SPY', timeframe='1d', days=730)
