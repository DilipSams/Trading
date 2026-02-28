#!/usr/bin/env python
"""
Comprehensive IC Test: 10 Symbols x 3 Horizons
Tests all 10 alphas with optimized configuration
"""

from alphago_architecture import build_default_pipeline, ArchitectureConfig
from alphago_trading_system import download_data, Config
from collections import defaultdict
import numpy as np

print('='*80)
print('  COMPREHENSIVE IC TEST: 10 SYMBOLS x 3 HORIZONS')
print('  Testing all 10 alphas with optimized configuration')
print('='*80)
print()

# Select 10 diverse symbols
symbols = ['AAPL', 'MSFT', 'NVDA', 'AMD',     # Tech
           'JPM', 'BAC', 'BLK',                # Finance
           'COST', 'META', 'TSLA']             # Consumer/Other

print(f'Loading {len(symbols)} symbols...')
cfg = Config(starting_capital=100000, n_iterations=1, total_timesteps_per_iter=1000)
datasets = download_data(symbols, cfg)
daily_data = {k: v for k, v in datasets.items() if '1d' in k.lower()}
print(f'Loaded {len(daily_data)} daily datasets')
print()

# Build pipeline with trend inversion
acfg = ArchitectureConfig()
pipeline = build_default_pipeline(acfg, bars_per_year=252, net=None, invert_trend=True)
alpha_names = [name for name in pipeline.alpha_factory.alpha_names if name != 'rl_ppo_mcts']

print(f'Testing alphas: {", ".join(alpha_names)}')
print()

# Collect IC data
alpha_horizon_data = defaultdict(lambda: {1: {'mus': [], 'rets': []},
                                           5: {'mus': [], 'rets': []},
                                           15: {'mus': [], 'rets': []}})

# Process datasets
for ds_name, df in daily_data.items():
    symbol = ds_name.split('_')[0]
    print(f'Processing {symbol}...', end=' ')

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    vol_col = 'Volume' if 'Volume' in df.columns else None

    closes = df[price_col].values
    volumes = df[vol_col].values if vol_col else np.ones(len(closes))

    # Extract OHLC for new features
    opens = df['Open'].values if 'Open' in df.columns else closes
    highs = df['High'].values if 'High' in df.columns else closes
    lows = df['Low'].values if 'Low' in df.columns else closes

    n_obs = len(closes)
    log_rets = np.diff(np.log(closes))

    # Cumulative returns
    cum_rets_5 = np.zeros(n_obs)
    cum_rets_15 = np.zeros(n_obs)
    for t in range(n_obs - 5):
        cum_rets_5[t] = np.sum(log_rets[t+1:t+6])
    for t in range(n_obs - 15):
        cum_rets_15[t] = np.sum(log_rets[t+1:t+16])

    # Validation period (last 20%)
    val_start = int(0.8 * n_obs)
    val_end = n_obs - 1

    # Warmup period
    warmup_start = max(0, val_start - 300)
    for t in range(warmup_start, val_start):
        if t < 50:
            continue
        for alpha_name in alpha_names:
            alpha = pipeline.alpha_factory._alphas[alpha_name]
            try:
                alpha.generate(closes[:t+1], volumes[:t+1], None, t,
                             opens=opens[:t+1], highs=highs[:t+1], lows=lows[:t+1])
            except:
                pass

    # Generate validation signals
    for t in range(val_start, val_end):
        if t < 50:
            continue

        for alpha_name in alpha_names:
            alpha = pipeline.alpha_factory._alphas[alpha_name]
            try:
                signal = alpha.generate(closes[:t+1], volumes[:t+1], None, t,
                                      opens=opens[:t+1], highs=highs[:t+1], lows=lows[:t+1])
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
                    signal = alpha.generate(closes[:t+1], volumes[:t+1], None, t,
                                          opens=opens[:t+1], highs=highs[:t+1], lows=lows[:t+1])
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
                    signal = alpha.generate(closes[:t+1], volumes[:t+1], None, t,
                                          opens=opens[:t+1], highs=highs[:t+1], lows=lows[:t+1])
                    mu = signal.mu if signal else 0.0
                except:
                    mu = 0.0
                alpha_horizon_data[alpha_name][15]['mus'].append(mu)
                alpha_horizon_data[alpha_name][15]['rets'].append(cum_rets_15[t])

    print('done')

print()
print('='*80)
print('  MULTI-HORIZON IC RESULTS (10 SYMBOLS)')
print('='*80)
print()

# Calculate ICs
results = {}
for alpha_name in alpha_names:
    results[alpha_name] = {}
    for horizon in [1, 5, 15]:
        mus = np.array(alpha_horizon_data[alpha_name][horizon]['mus'])
        rets = np.array(alpha_horizon_data[alpha_name][horizon]['rets'])

        if len(mus) < 10 or np.std(mus) < 1e-10 or np.std(rets) < 1e-10:
            ic = 0.0
            t_stat = 0.0
        else:
            ic = np.corrcoef(mus, rets)[0, 1]
            # T-statistic for IC
            n = len(mus)
            t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

        results[alpha_name][horizon] = (ic, t_stat, len(mus))

# Print table
print(f"  {'Alpha':<25s} {'1-bar':>18s} {'5-bar':>18s} {'15-bar':>18s}")
print(f"  {'':<25s} {'IC':>6s} {'t':>5s} {'n':>5s}  {'IC':>6s} {'t':>5s} {'n':>5s}  {'IC':>6s} {'t':>5s} {'n':>5s}")
print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*18}")

for alpha_name in alpha_names:
    ic1, t1, n1 = results[alpha_name][1]
    ic5, t5, n5 = results[alpha_name][5]
    ic15, t15, n15 = results[alpha_name][15]

    print(f"  {alpha_name:<25s} {ic1:+.3f} {t1:+4.1f} {n1:>5d}  {ic5:+.3f} {t5:+4.1f} {n5:>5d}  {ic15:+.3f} {t15:+4.1f} {n15:>5d} ")

print()
print('='*80)

# Summary statistics
print()
print('SUMMARY (15-bar horizon - IC optimized):')
print()
positive_ics = [(name, results[name][15][0]) for name in alpha_names if results[name][15][0] > 0]
negative_ics = [(name, results[name][15][0]) for name in alpha_names if results[name][15][0] < 0]

positive_ics.sort(key=lambda x: x[1], reverse=True)
negative_ics.sort(key=lambda x: x[1])

print(f'Positive IC alphas ({len(positive_ics)}/{len(alpha_names)}):')
for name, ic in positive_ics:
    t_stat = results[name][15][1]
    sig = '***' if abs(t_stat) > 2.5 else '**' if abs(t_stat) > 2.0 else '*' if abs(t_stat) > 1.5 else ''
    print(f'  {name:<25s} IC: {ic:+.3f}  t-stat: {t_stat:+.1f} {sig}')

if negative_ics:
    print()
    print(f'Negative IC alphas ({len(negative_ics)}/{len(alpha_names)}):')
    for name, ic in negative_ics:
        t_stat = results[name][15][1]
        print(f'  {name:<25s} IC: {ic:+.3f}  t-stat: {t_stat:+.1f}')

print()
total_ic = sum(ic for _, ic in positive_ics) + sum(ic for _, ic in negative_ics)
positive_sum = sum(ic for _, ic in positive_ics)
print(f'Total IC (signed): {total_ic:+.3f}')
print(f'Positive IC sum:   {positive_sum:+.3f}')
print()
print('='*80)
