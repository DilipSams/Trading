#!/usr/bin/env python
"""
Super quick IC test - just runs the alphas on 5 validation symbols.
Expected runtime: 1-2 minutes.
"""

import numpy as np
from alphago_architecture import build_default_pipeline, ArchitectureConfig
from alphago_trading_system import download_data, Config
from collections import defaultdict

def test_ic(invert_seasonality=False, invert_vol_premium=False, invert_trend=False):
    print(f"\n{'='*80}")
    print(f"  QUICK IC TEST")
    print(f"  Seasonality inverted: {invert_seasonality}")
    print(f"  Vol premium inverted: {invert_vol_premium}")
    print(f"  Trend inverted: {invert_trend}")
    print(f"{'='*80}\n")

    # Load 5 validation symbols
    cfg = Config(starting_capital=100000, n_iterations=1, total_timesteps_per_iter=1000)
    symbols = ["BLK", "PYPL", "META", "COST", "AAPL"]
    print(f"Loading {len(symbols)} symbols...")
    datasets = download_data(symbols, cfg)

    # Filter for 1d data only
    daily_data = {k: v for k, v in datasets.items() if '1d' in k.lower() or '_1D' in k}
    print(f"Loaded {len(daily_data)} daily datasets\n")

    # Build pipeline with inversions
    acfg = ArchitectureConfig()
    pipeline = build_default_pipeline(
        acfg, bars_per_year=252, net=None,
        invert_seasonality=invert_seasonality,
        invert_vol_premium=invert_vol_premium,
        invert_trend=invert_trend
    )

    # Alpha names (exclude RL)
    alpha_names = [n for n in pipeline.alpha_factory.alpha_names if n != 'rl_ppo_mcts']
    print(f"Testing alphas: {', '.join(alpha_names)}\n")

    # Collect IC data
    alpha_horizon_data = defaultdict(lambda: {1: {'mus': [], 'rets': []},
                                               5: {'mus': [], 'rets': []},
                                               15: {'mus': [], 'rets': []}})

    # Process each dataset
    for ds_name, df in daily_data.items():
        print(f"Processing {ds_name}...")

        # Get prices
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        vol_col = 'Volume' if 'Volume' in df.columns else None

        closes = df[price_col].values
        volumes = df[vol_col].values if vol_col else np.ones(len(closes))
        n_obs = len(closes)

        # Calculate returns
        log_rets = np.diff(np.log(closes))

        # Cumulative returns for multi-horizon
        cum_rets_5 = np.zeros(n_obs)
        cum_rets_15 = np.zeros(n_obs)
        for t in range(n_obs - 5):
            cum_rets_5[t] = np.sum(log_rets[t+1:t+6])
        for t in range(n_obs - 15):
            cum_rets_15[t] = np.sum(log_rets[t+1:t+16])

        # Validation period (last 10%)
        val_start = int(0.8 * n_obs)
        val_end = n_obs - 1

        # IMPORTANT: Warm up alphas by calling generate() on historical data first
        # This allows seasonality and other stateful alphas to accumulate history
        warmup_start = max(0, val_start - 300)  # 300 bars of warmup
        for t in range(warmup_start, val_start):
            if t < 50:
                continue
            # Call generate() to let alphas accumulate state, but don't save signals
            for alpha_name in alpha_names:
                alpha = pipeline.alpha_factory._alphas[alpha_name]
                try:
                    alpha.generate(closes[:t+1], volumes[:t+1], None, t)
                except:
                    pass

        # Generate signals for each bar in validation window
        for t in range(val_start, val_end):
            if t < 50:
                continue

            # Call each alpha
            for alpha_name in alpha_names:
                alpha = pipeline.alpha_factory._alphas[alpha_name]

                try:
                    # Call the generate method with price/volume arrays up to time t
                    signal = alpha.generate(
                        closes=closes[:t+1],
                        volumes=volumes[:t+1] if volumes is not None else None,
                        features=None,
                        bar_idx=t
                    )

                    mu = signal.mu if signal else 0.0
                except Exception as e:
                    mu = 0.0

                # Collect 1-bar horizon
                if t + 1 < n_obs:
                    alpha_horizon_data[alpha_name][1]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][1]['rets'].append(log_rets[t])

            # 5-bar horizon (non-overlapping)
            if t % 5 == 0 and t + 5 < val_end:
                for alpha_name in alpha_names:
                    alpha = pipeline.alpha_factory._alphas[alpha_name]
                    try:
                        signal = alpha.generate(closes[:t+1], volumes[:t+1], None, t)
                        mu = signal.mu if signal else 0.0
                    except:
                        mu = 0.0
                    alpha_horizon_data[alpha_name][5]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][5]['rets'].append(cum_rets_5[t])

            # 15-bar horizon (non-overlapping)
            if t % 15 == 0 and t + 15 < val_end:
                for alpha_name in alpha_names:
                    alpha = pipeline.alpha_factory._alphas[alpha_name]
                    try:
                        signal = alpha.generate(closes[:t+1], volumes[:t+1], None, t)
                        mu = signal.mu if signal else 0.0
                    except:
                        mu = 0.0
                    alpha_horizon_data[alpha_name][15]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][15]['rets'].append(cum_rets_15[t])

    # Calculate and print ICs
    print(f"\n{'='*80}")
    print(f"  MULTI-HORIZON IC RESULTS")
    print(f"{'='*80}\n")
    print(f"  {'Alpha':<22s} {'1-bar':>18s} {'5-bar':>18s} {'15-bar':>18s}")
    print(f"  {'':<22s} {'IC':>7s} {'t':>5s} {'n':>4s} {'IC':>7s} {'t':>5s} {'n':>4s} {'IC':>7s} {'t':>5s} {'n':>4s}")
    print(f"  {'-'*22} {'-'*18} {'-'*18} {'-'*18}")

    for alpha_name in alpha_names:
        row = []
        for horizon in [1, 5, 15]:
            mus = np.array(alpha_horizon_data[alpha_name][horizon]['mus'])
            rets = np.array(alpha_horizon_data[alpha_name][horizon]['rets'])

            if len(mus) < 10 or np.std(mus) < 1e-10 or np.std(rets) < 1e-10:
                ic, t_stat, n = 0.0, 0.0, len(mus)
            else:
                ic = np.corrcoef(mus, rets)[0, 1]
                n = len(mus)
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 0.9999 else (100.0 if ic > 0 else -100.0)

            row.append((ic, t_stat, n))

        print(f"  {alpha_name:<22s} ", end='')
        for ic, t_stat, n in row:
            print(f"{ic:>+7.3f} {t_stat:>+5.1f} {n:>4d} ", end='')
        print()

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--invert-seasonality", action="store_true")
    parser.add_argument("--invert-vol-premium", action="store_true")
    parser.add_argument("--invert-trend", action="store_true")
    args = parser.parse_args()

    test_ic(
        invert_seasonality=args.invert_seasonality,
        invert_vol_premium=args.invert_vol_premium,
        invert_trend=args.invert_trend
    )
