"""
Performance Profiling - Identify Bottlenecks

Profiles the main components of the trading system to identify where time is spent.
"""

import time
import cProfile
import pstats
import io
from contextlib import contextmanager
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

@contextmanager
def timer(name):
    """Context manager to time code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  {name:<50} {elapsed:>8.3f}s")

def profile_data_loading():
    """Profile data loading and caching."""
    print("\n" + "="*80)
    print("PROFILING: Data Loading")
    print("="*80)

    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)

    with timer("Download 5 symbols x 2 years"):
        data = {}
        for symbol in symbols:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if hasattr(df.columns, 'levels'):
                df.columns = df.columns.droplevel(1)
            data[symbol] = df

    print(f"\n  Total bars downloaded: {sum(len(df) for df in data.values())}")
    return data

def profile_feature_engineering(data):
    """Profile feature engineering (indicators + feature matrix)."""
    print("\n" + "="*80)
    print("PROFILING: Feature Engineering")
    print("="*80)

    from alphago_trading_system import compute_indicators, build_feature_matrix

    symbol = 'SPY'
    df = data[symbol].copy()

    with timer("compute_indicators (single symbol)"):
        df = compute_indicators(df)

    with timer("build_feature_matrix (single symbol)"):
        features = build_feature_matrix(df)

    print(f"\n  Feature matrix shape: {features.shape}")
    print(f"  Bars: {len(df)}, Features per bar: {features.shape[1]}")

    # Profile all symbols
    total_bars = 0
    with timer("compute_indicators (5 symbols)"):
        for symbol, df in data.items():
            df = compute_indicators(df)
            data[symbol] = df
            total_bars += len(df)

    with timer("build_feature_matrix (5 symbols)"):
        for symbol, df in data.items():
            features = build_feature_matrix(df)

    print(f"\n  Total bars processed: {total_bars}")

    return data

def profile_pipeline_components():
    """Profile individual pipeline components (L1-L4)."""
    print("\n" + "="*80)
    print("PROFILING: Pipeline Components (L1->L2->L3->L4)")
    print("="*80)

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

    # Setup
    acfg = ArchitectureConfig()
    acfg.trend_horizon = 15
    acfg.use_trailing_stops = True
    acfg.trailing_stop_lookback = 5

    pipeline = InstitutionalPipeline(acfg, bars_per_year=252)

    # Register alphas
    with timer("Register 9 alphas"):
        pipeline.register_alpha(TrendAlpha(acfg, invert_signal=True))
        pipeline.register_alpha(MeanReversionAlpha(acfg))
        pipeline.register_alpha(ValueAlpha(acfg))
        pipeline.register_alpha(CarryAlpha(acfg))
        pipeline.register_alpha(VolatilityPremiumAlpha(acfg))
        pipeline.register_alpha(CalendarAlpha(acfg))
        pipeline.register_alpha(AmihudLiquidityAlpha(acfg))
        pipeline.register_alpha(HurstRegimeAlpha(acfg))
        pipeline.register_alpha(ShortTermReversalAlpha(acfg))

    # Create synthetic data for testing
    n_bars = 500
    closes = np.random.randn(n_bars).cumsum() + 100
    closes = np.abs(closes)  # Keep positive
    volumes = np.random.uniform(1e6, 1e7, n_bars)
    highs = closes * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    lows = closes * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    opens = (highs + lows) / 2

    observation = np.random.randn(920)  # 20 bars x 46 features

    # Warmup (JIT compilation, etc.)
    for _ in range(5):
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

    # Profile single step
    with timer("Pipeline.step() - Single iteration"):
        result = pipeline.step(
            observation=observation,
            closes=closes,
            volumes=volumes,
            opens=opens,
            highs=highs,
            lows=lows,
            bar_idx=250,
            current_exposure=0.0,
            portfolio_value=100000.0
        )

    # Profile 100 steps
    start_time = time.time()
    for i in range(100):
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
    elapsed_100 = time.time() - start_time
    print(f"  Pipeline.step() - 100 iterations                  {elapsed_100:>8.3f}s")

    print(f"\n  Steps per second: {100 / elapsed_100:.1f}")

def profile_env_step():
    """Profile TradingEnv.step() with and without trailing stops."""
    print("\n" + "="*80)
    print("PROFILING: TradingEnv.step() - With/Without Trailing Stops")
    print("="*80)

    from alphago_trading_system import TradingEnv, Config, compute_indicators, build_feature_matrix
    import yfinance as yf

    # Download data
    print("  Downloading SPY data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = yf.download('SPY', start=start_date, end=end_date, progress=False)

    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.droplevel(1)

    df = compute_indicators(df)
    features = build_feature_matrix(df)
    prices = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # Test WITHOUT trailing stops
    cfg = Config()
    cfg.use_trailing_stops = False
    cfg.starting_capital = 100000
    cfg.max_episode_steps = 200
    cfg.window_size = 20

    env = TradingEnv(features, prices, cfg, sym="SPY_no_stops")
    obs, info = env.reset()

    with timer("TradingEnv.step() - 200 steps WITHOUT trailing stops"):
        for i in range(200):
            action = np.random.randint(0, 5)  # Random action
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break

    steps_no_stops = i + 1

    # Test WITH trailing stops
    cfg2 = Config()
    cfg2.use_trailing_stops = True
    cfg2.trailing_stop_lookback = 5
    cfg2.starting_capital = 100000
    cfg2.max_episode_steps = 200
    cfg2.window_size = 20

    env2 = TradingEnv(features, prices, cfg2, sym="SPY_with_stops")
    obs, info = env2.reset()

    with timer("TradingEnv.step() - 200 steps WITH trailing stops"):
        for i in range(200):
            action = np.random.randint(0, 5)  # Random action
            obs, reward, term, trunc, info = env2.step(action)
            if term or trunc:
                break

    steps_with_stops = i + 1

    print(f"\n  Steps completed (no stops): {steps_no_stops}")
    print(f"  Steps completed (with stops): {steps_with_stops}")

    # Check for stop exits
    stop_exits = [e for e in env2.trade_entries if e.get('action') == 'STOP_EXIT']
    print(f"  Stop exits triggered: {len(stop_exits)}")

def profile_alpha_generation():
    """Profile individual alpha generation times."""
    print("\n" + "="*80)
    print("PROFILING: Individual Alpha Generation Times")
    print("="*80)

    from alphago_architecture import (
        ArchitectureConfig,
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

    acfg = ArchitectureConfig()

    # Create test data
    n_bars = 500
    closes = np.random.randn(n_bars).cumsum() + 100
    closes = np.abs(closes)
    volumes = np.random.uniform(1e6, 1e7, n_bars)
    highs = closes * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    lows = closes * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    opens = (highs + lows) / 2

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

    print("\n  Timing 100 iterations per alpha:")
    print()

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
        print(f"    {name:<30} {elapsed:>8.3f}s total, {per_call:>6.2f}ms per call")

def main():
    print("="*80)
    print("PERFORMANCE PROFILING - Trading System Bottleneck Analysis")
    print("="*80)
    print()
    print("This will profile:")
    print("  1. Data loading (yfinance downloads)")
    print("  2. Feature engineering (indicators + feature matrix)")
    print("  3. Pipeline components (L1->L2->L3->L4)")
    print("  4. TradingEnv.step() with/without trailing stops")
    print("  5. Individual alpha generation times")
    print()

    # Run profiling
    data = profile_data_loading()
    data = profile_feature_engineering(data)
    profile_pipeline_components()
    profile_env_step()
    profile_alpha_generation()

    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print("  - Check which components take the most time")
    print("  - Compare env.step() with/without trailing stops for overhead")
    print("  - Identify slow alphas for potential optimization")
    print()

if __name__ == "__main__":
    main()
