#!/usr/bin/env python
"""
Test optimized alpha configuration based on IC analysis.

Comparison:
- Baseline: All 7 alphas, no inversions
- Optimized: Only 3 strong alphas (vol_premium + inverted_trend + value)
"""

import numpy as np
from alphago_architecture import build_default_pipeline, ArchitectureConfig
from alphago_trading_system import download_data, Config
from collections import defaultdict


def test_configuration(config_name, invert_trend=False, disable_alphas=""):
    """Test a specific alpha configuration."""

    # Load data (reuse across configs)
    if not hasattr(test_configuration, '_datasets'):
        cfg = Config(starting_capital=100000, n_iterations=1, total_timesteps_per_iter=1000)
        symbols = ["BLK", "PYPL", "META", "COST", "AAPL"]
        test_configuration._datasets = download_data(symbols, cfg)
        test_configuration._daily_data = {k: v for k, v in test_configuration._datasets.items()
                                          if '1d' in k.lower()}

    daily_data = test_configuration._daily_data

    # Build pipeline
    acfg = ArchitectureConfig()
    pipeline = build_default_pipeline(
        acfg, bars_per_year=252, net=None,
        invert_trend=invert_trend
    )

    # Disable specified alphas
    if disable_alphas:
        disabled_list = [a.strip() for a in disable_alphas.split(',')]
        for alpha_name in disabled_list:
            if alpha_name in pipeline.alpha_factory._alphas:
                pipeline.alpha_factory.unregister(alpha_name)

    alpha_names = pipeline.alpha_factory.alpha_names

    # Collect IC data
    alpha_horizon_data = defaultdict(lambda: {1: {'mus': [], 'rets': []},
                                               5: {'mus': [], 'rets': []},
                                               15: {'mus': [], 'rets': []}})

    # Process datasets
    for ds_name, df in daily_data.items():
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        vol_col = 'Volume' if 'Volume' in df.columns else None

        closes = df[price_col].values
        volumes = df[vol_col].values if vol_col else np.ones(len(closes))
        n_obs = len(closes)

        log_rets = np.diff(np.log(closes))

        # Cumulative returns
        cum_rets_5 = np.zeros(n_obs)
        cum_rets_15 = np.zeros(n_obs)
        for t in range(n_obs - 5):
            cum_rets_5[t] = np.sum(log_rets[t+1:t+6])
        for t in range(n_obs - 15):
            cum_rets_15[t] = np.sum(log_rets[t+1:t+16])

        # Validation period
        val_start = int(0.8 * n_obs)
        val_end = n_obs - 1

        # Warmup
        warmup_start = max(0, val_start - 300)
        for t in range(warmup_start, val_start):
            if t < 50:
                continue
            for alpha_name in alpha_names:
                alpha = pipeline.alpha_factory._alphas[alpha_name]
                try:
                    alpha.generate(closes[:t+1], volumes[:t+1], None, t)
                except:
                    pass

        # Generate validation signals
        for t in range(val_start, val_end):
            if t < 50:
                continue

            for alpha_name in alpha_names:
                alpha = pipeline.alpha_factory._alphas[alpha_name]
                try:
                    signal = alpha.generate(closes[:t+1], volumes[:t+1], None, t)
                    mu = signal.mu if signal else 0.0
                except:
                    mu = 0.0

                # 1-bar
                if t + 1 < n_obs:
                    alpha_horizon_data[alpha_name][1]['mus'].append(mu)
                    alpha_horizon_data[alpha_name][1]['rets'].append(log_rets[t])

            # 5-bar (non-overlapping)
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

            # 15-bar (non-overlapping)
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

    # Calculate ICs
    results = {}
    for alpha_name in alpha_names:
        results[alpha_name] = {}
        for horizon in [1, 5, 15]:
            mus = np.array(alpha_horizon_data[alpha_name][horizon]['mus'])
            rets = np.array(alpha_horizon_data[alpha_name][horizon]['rets'])

            if len(mus) < 10 or np.std(mus) < 1e-10 or np.std(rets) < 1e-10:
                ic = 0.0
            else:
                ic = np.corrcoef(mus, rets)[0, 1]

            results[alpha_name][horizon] = ic

    return results, alpha_names


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  BASELINE vs OPTIMIZED CONFIGURATION")
    print("  Based on IC analysis and signal inversion tests")
    print("="*80 + "\n")

    print("Loading market data (5 symbols)...")

    # Baseline: All alphas, no inversions
    print("\n[1/2] Testing baseline (all 7 alphas, no inversions)...")
    baseline_results, baseline_alphas = test_configuration(
        "Baseline",
        invert_trend=False,
        disable_alphas=""
    )

    # Optimized: All 7 alphas, trend inverted
    print("[2/2] Testing optimized (all 7 alphas, trend inverted)...")
    optimized_results, optimized_alphas = test_configuration(
        "Optimized",
        invert_trend=True,
        disable_alphas=""
    )

    # Print comparison
    print("\n" + "="*80)
    print("  IC COMPARISON (15-bar horizon)")
    print("="*80 + "\n")

    print(f"  {'Alpha':<20s} {'Baseline IC':>15s} {'Optimized IC':>15s} {'Improvement':>15s}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")

    # Show all baseline alphas
    all_alphas = sorted(set(baseline_alphas) | set(optimized_alphas))

    baseline_total_ic = 0.0
    optimized_total_ic = 0.0
    baseline_positive_ic = 0.0
    optimized_positive_ic = 0.0

    for alpha in all_alphas:
        baseline_ic = baseline_results.get(alpha, {}).get(15, 0.0)
        optimized_ic = optimized_results.get(alpha, {}).get(15, 0.0)

        if alpha in baseline_alphas:
            baseline_total_ic += baseline_ic  # Signed IC (negative IC hurts)
            if baseline_ic > 0:
                baseline_positive_ic += baseline_ic
        if alpha in optimized_alphas:
            optimized_total_ic += optimized_ic
            if optimized_ic > 0:
                optimized_positive_ic += optimized_ic

        if alpha in optimized_alphas:
            improvement = optimized_ic - baseline_ic
            print(f"  {alpha:<20s} {baseline_ic:>+15.3f} {optimized_ic:>+15.3f} {improvement:>+15.3f} *")
        elif alpha in baseline_alphas:
            print(f"  {alpha:<20s} {baseline_ic:>+15.3f} {'DISABLED':>15s} {'---':>15s}")

    print(f"\n  {'TOTAL IC (signed)':<20s} {baseline_total_ic:>+15.3f} {optimized_total_ic:>+15.3f} {optimized_total_ic - baseline_total_ic:>+15.3f}")
    print(f"  {'Positive IC only':<20s} {baseline_positive_ic:>+15.3f} {optimized_positive_ic:>+15.3f} {optimized_positive_ic - baseline_positive_ic:>+15.3f}")

    print("\n" + "="*80)
    print("\n  SUMMARY:")
    print("  --------")
    print(f"  Baseline config:  {len(baseline_alphas)} alphas, no inversions")
    print(f"    Total IC (signed):    {baseline_total_ic:+.3f}  (negative IC hurts performance!)")
    print(f"    Positive IC only:     {baseline_positive_ic:+.3f}")
    print(f"\n  Optimized config: {len(optimized_alphas)} alphas, trend inverted")
    print(f"    Total IC (signed):    {optimized_total_ic:+.3f}")
    print(f"    Positive IC only:     {optimized_positive_ic:+.3f}")
    print(f"\n  Improvement: {optimized_total_ic - baseline_total_ic:+.3f} ({100*(optimized_total_ic - baseline_total_ic)/abs(baseline_total_ic):+.1f}%)")
    print("\n  Key change:")
    print("    - All 7 alphas kept active (L2 meta-learner will weight them)")
    print("    - trend_follow signal INVERTED (IC: -0.059 -> +0.059)")
    print("\n" + "="*80 + "\n")
