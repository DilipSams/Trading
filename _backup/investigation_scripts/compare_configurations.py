#!/usr/bin/env python
"""
Quick comparison of different alpha configurations.
Tests multiple scenarios in ~30 seconds total.
"""

import numpy as np
from alphago_architecture import build_default_pipeline, ArchitectureConfig
from alphago_trading_system import download_data, Config
from collections import defaultdict

def test_configuration(config_name, invert_seasonality=False, invert_vol_premium=False,
                       invert_trend=False, disable_alphas=""):
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
        invert_seasonality=invert_seasonality,
        invert_vol_premium=invert_vol_premium,
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


def print_comparison_table(config_results):
    """Print comparison table of all configurations."""

    print(f"\n{'='*100}")
    print(f"  ALPHA CONFIGURATION COMPARISON")
    print(f"  Quick test on 5 symbols - 15-bar horizon IC shown (most predictive)")
    print(f"{'='*100}\n")

    # Get all unique alpha names across configs
    all_alphas = set()
    for config_name, (results, alpha_names) in config_results.items():
        all_alphas.update(alpha_names)
    all_alphas = sorted(all_alphas)

    # Print header
    config_names = list(config_results.keys())
    header = f"  {'Alpha':<20s}"
    for name in config_names:
        header += f" {name[:12]:>12s}"
    print(header)
    print(f"  {'-'*20}" + "".join([f" {'-'*12}" for _ in config_names]))

    # Print each alpha's IC at 15-bar
    for alpha in all_alphas:
        row = f"  {alpha:<20s}"
        for config_name in config_names:
            results, alpha_names = config_results[config_name]
            if alpha in results:
                ic = results[alpha][15]  # 15-bar horizon
                if abs(ic) > 0.04:
                    row += f" {ic:>+12.3f}*"  # Mark strong signals
                else:
                    row += f" {ic:>+12.3f} "
            else:
                row += f" {'---':>12s} "
        print(row)

    print(f"\n  * = |IC| > 0.04 (noteworthy signal strength)")
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    print("Loading market data (5 symbols)...")

    # Define configurations to test
    configs = [
        ("Baseline", {
            'invert_seasonality': False,
            'invert_vol_premium': False,
            'invert_trend': False,
            'disable_alphas': ''
        }),
        ("No Weak", {
            'invert_seasonality': False,
            'invert_vol_premium': False,
            'invert_trend': False,
            'disable_alphas': 'seasonality,vol_premium,carry,rl_ppo_mcts'
        }),
        ("Value Only", {
            'invert_seasonality': False,
            'invert_vol_premium': False,
            'invert_trend': False,
            'disable_alphas': 'rl_ppo_mcts,trend_follow,mean_reversion,carry,seasonality,vol_premium'
        }),
        ("Top 3", {
            'invert_seasonality': False,
            'invert_vol_premium': False,
            'invert_trend': False,
            'disable_alphas': 'rl_ppo_mcts,carry,seasonality,vol_premium'
        }),
        ("Inverted", {
            'invert_seasonality': True,
            'invert_vol_premium': True,
            'invert_trend': False,
            'disable_alphas': ''
        }),
        ("Trend Flip", {
            'invert_seasonality': False,
            'invert_vol_premium': False,
            'invert_trend': True,
            'disable_alphas': ''
        }),
    ]

    # Run all configurations
    config_results = {}
    for config_name, params in configs:
        print(f"\nTesting: {config_name}...")
        results, alpha_names = test_configuration(config_name, **params)
        config_results[config_name] = (results, alpha_names)
        print(f"  Active alphas: {', '.join(alpha_names)}")

    # Print comparison
    print_comparison_table(config_results)

    # Summary recommendations
    print("\nRECOMMENDATIONS (based on 15-bar IC):")
    print("-" * 80)

    # Analyze which alphas are consistently strong
    baseline_results, baseline_alphas = config_results["Baseline"]
    strong_alphas = []
    weak_alphas = []

    for alpha in baseline_alphas:
        ic_15 = baseline_results[alpha][15]
        if abs(ic_15) > 0.04:
            strong_alphas.append((alpha, ic_15))
        elif abs(ic_15) < 0.02:
            weak_alphas.append((alpha, ic_15))

    if strong_alphas:
        print("\n[+] STRONG ALPHAS (keep enabled):")
        for alpha, ic in sorted(strong_alphas, key=lambda x: abs(x[1]), reverse=True):
            print(f"   - {alpha:<20s} IC = {ic:+.3f}")

    if weak_alphas:
        print("\n[!] WEAK ALPHAS (consider disabling):")
        for alpha, ic in weak_alphas:
            print(f"   - {alpha:<20s} IC = {ic:+.3f}")

    print("\n" + "="*80)
