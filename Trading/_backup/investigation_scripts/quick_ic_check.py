#!/usr/bin/env python
"""
Quick IC check for signal inversions.
Runs multi-horizon IC profiling without any RL training.
Expected runtime: ~2-5 minutes instead of 2-3 hours.
"""

import numpy as np
import sys
from collections import defaultdict
from alphago_architecture import build_default_pipeline, ArchitectureConfig
from alphago_trading_system import download_data, Config, DEFAULT_SYMBOLS


def quick_ic_profile(invert_seasonality=False, invert_vol_premium=False):
    """
    Calculate multi-horizon IC for alphas WITHOUT RL training.

    This extracts alpha signals (mu) from traditional alphas and correlates
    them with realized returns at 1-bar, 5-bar, and 15-bar horizons.
    """

    print(f"\n{'='*80}")
    print(f"  QUICK MULTI-HORIZON IC CHECK")
    print(f"  Seasonality inverted: {invert_seasonality}")
    print(f"  Vol premium inverted: {invert_vol_premium}")
    print(f"{'='*80}\n")

    # Download data for validation symbols (using same 5 val symbols as main test)
    print("Loading validation data (5 symbols)...")
    val_symbols = ["BLK", "PYPL", "META", "COST", "AAPL"]  # Same validation pool

    # Create minimal config for download
    cfg = Config(
        starting_capital=100000,
        n_iterations=1,
        total_timesteps_per_iter=1000,
        download_period="max",
    )

    datasets = download_data(val_symbols, cfg)

    if not datasets:
        print("ERROR: No data loaded")
        return

    print(f"Loaded {len(datasets)} datasets\n")

    # Build pipeline with inversions
    acfg = ArchitectureConfig()
    pipeline = build_default_pipeline(
        acfg,
        bars_per_year=252,
        net=None,
        invert_seasonality=invert_seasonality,
        invert_vol_premium=invert_vol_premium,
    )

    # Get alpha names (excluding RL agent since it needs training)
    alpha_names = [name for name in pipeline.alpha_factory.alpha_names if name != 'rl_ppo_mcts']

    print(f"Alphas to profile: {', '.join(alpha_names)}\n")

    # Data structures for multi-horizon profiling
    alpha_horizon_data = defaultdict(lambda: {
        1: {'mus': [], 'rets': []},
        5: {'mus': [], 'rets': []},
        15: {'mus': [], 'rets': []},
    })

    # Process each dataset (filter for 1d only)
    for ds_name, ds in datasets.items():
        if '1d' not in ds_name.lower():
            continue  # Skip non-daily data

        print(f"Processing {ds_name}...")

        # Get price column (could be 'Close', 'close', or 'Adj Close')
        if 'Adj Close' in ds.columns:
            prices = ds['Adj Close'].values
        elif 'Close' in ds.columns:
            prices = ds['Close'].values
        elif 'close' in ds.columns:
            prices = ds['close'].values
        else:
            print(f"  Skipping {ds_name}: no price column found")
            continue

        log_rets = np.diff(np.log(prices))
        n_obs = len(prices)

        # Precompute cumulative returns for 5-bar and 15-bar horizons
        cum_rets_5bar = np.zeros(n_obs)
        cum_rets_15bar = np.zeros(n_obs)

        for t in range(n_obs - 5):
            cum_rets_5bar[t] = np.sum(log_rets[t+1:t+6])

        for t in range(n_obs - 15):
            cum_rets_15bar[t] = np.sum(log_rets[t+1:t+16])

        # Use validation period (last 10% of data)
        val_start = int(0.8 * n_obs)
        val_end = n_obs - 1

        # Extract signals from each alpha
        all_signals = {}  # {t: {alpha_name: mu}}

        for t in range(max(0, val_start - 100), val_end):  # Start 100 bars early for warmup
            # Build observation at time t
            if t < 50:
                continue

            # Get OHLCV data with proper column names
            high_col = 'High' if 'High' in ds.columns else ('high' if 'high' in ds.columns else None)
            low_col = 'Low' if 'Low' in ds.columns else ('low' if 'low' in ds.columns else None)
            vol_col = 'Volume' if 'Volume' in ds.columns else ('volume' if 'volume' in ds.columns else None)

            obs = {
                'close': prices[:t+1],
                'high': ds[high_col].values[:t+1] if high_col else prices[:t+1],
                'low': ds[low_col].values[:t+1] if low_col else prices[:t+1],
                'volume': ds[vol_col].values[:t+1] if vol_col else np.ones(t+1),
                'timestamp': ds.index[t],
            }

            # Get alpha signals
            all_signals[t] = {}
            for alpha_name in alpha_names:
                alpha = pipeline.alpha_factory._alphas[alpha_name]

                try:
                    signal = alpha.generate_signal(obs, symbol=ds_name)
                    mu = signal.get('mu', 0.0) if signal else 0.0
                    all_signals[t][alpha_name] = mu
                except Exception as e:
                    all_signals[t][alpha_name] = 0.0

        # Collect multi-horizon data
        for alpha_name in alpha_names:
            # 1-bar: every bar in validation window
            for t in range(val_start, val_end):
                if t not in all_signals:
                    continue
                mu = all_signals[t].get(alpha_name, 0.0)
                if t + 1 < n_obs:
                    alpha_horizon_data[alpha_name][1]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][1]['rets'].append(log_rets[t])

            # 5-bar: non-overlapping windows (stride = 5)
            t = val_start
            while t + 5 < min(val_end, n_obs):
                if t in all_signals:
                    mu = all_signals[t].get(alpha_name, 0.0)
                    alpha_horizon_data[alpha_name][5]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][5]['rets'].append(cum_rets_5bar[t])
                t += 5

            # 15-bar: non-overlapping windows (stride = 15)
            t = val_start
            while t + 15 < min(val_end, n_obs):
                if t in all_signals:
                    mu = all_signals[t].get(alpha_name, 0.0)
                    alpha_horizon_data[alpha_name][15]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][15]['rets'].append(cum_rets_15bar[t])
                t += 15

    # Calculate ICs
    print(f"\n{'='*80}")
    print(f"  MULTI-HORIZON ALPHA PROFILING")
    print(f"  Information Coefficient (IC) at 1-bar, 5-bar, and 15-bar horizons")
    print(f"  IC = correlation(mu, realized_return). Higher |IC| = stronger predictive skill")
    print(f"  Non-overlapping windows for 5-bar and 15-bar")
    print(f"{'='*80}\n")

    print(f"  {'Alpha':<22s} {'1-bar':>18s} {'5-bar':>18s} {'15-bar':>18s}")
    print(f"  {'':<22s} {'IC':>7s} {'t':>5s} {'n':>4s} {'IC':>7s} {'t':>5s} {'n':>4s} {'IC':>7s} {'t':>5s} {'n':>4s}")
    print(f"  {'-'*22} {'-'*18} {'-'*18} {'-'*18}")

    results = {}

    for alpha_name in alpha_names:
        results[alpha_name] = {}

        row_data = []
        for horizon in [1, 5, 15]:
            mus = np.array(alpha_horizon_data[alpha_name][horizon]['mus'])
            rets = np.array(alpha_horizon_data[alpha_name][horizon]['rets'])

            if len(mus) < 10 or np.std(mus) < 1e-10 or np.std(rets) < 1e-10:
                ic, t_stat, n = 0.0, 0.0, len(mus)
            else:
                ic = np.corrcoef(mus, rets)[0, 1]
                n = len(mus)

                # t-statistic for correlation
                if abs(ic) < 0.9999:
                    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                else:
                    t_stat = 100.0 if ic > 0 else -100.0

            results[alpha_name][horizon] = {'ic': ic, 't_stat': t_stat, 'n': n}
            row_data.append((ic, t_stat, n))

        # Print row
        print(f"  {alpha_name:<22s} ", end='')
        for ic, t_stat, n in row_data:
            print(f"{ic:>+7.3f} {t_stat:>+5.1f} {n:>4d} ", end='')
        print()

    print(f"\n{'='*80}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick IC check for signal inversions")
    parser.add_argument("--invert-seasonality", action="store_true",
                        help="Invert seasonality signal")
    parser.add_argument("--invert-vol-premium", action="store_true",
                        help="Invert vol premium signal")
    args = parser.parse_args()

    results = quick_ic_profile(
        invert_seasonality=args.invert_seasonality,
        invert_vol_premium=args.invert_vol_premium,
    )

    # Print comparison if inversions are enabled
    if args.invert_seasonality or args.invert_vol_premium:
        print("\nExpected IC sign flips:")
        if args.invert_seasonality:
            print("  • Seasonality: Should flip from negative to positive")
        if args.invert_vol_premium:
            print("  • Vol premium: Should flip from negative to positive")
