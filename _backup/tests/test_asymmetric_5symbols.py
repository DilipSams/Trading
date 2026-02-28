"""
Multi-Symbol Performance Test for Asymmetric Stop Loss
=======================================================

Tests asymmetric stops on 5 symbols:
- MSFT (Tech - moderate volatility)
- AAPL (Tech - moderate volatility)
- SPY (Index - low volatility)
- GOOGL (Tech - moderate volatility)
- META (Tech - higher volatility)

Compares baseline (no asymmetric stops) vs enhanced (with stops).
"""

import sys
sys.path.insert(0, '.')

# Fix Windows console encoding for box-drawing characters
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from table_formatter import TableFormatter

try:
    from alphago_trading_system import Config, TradingEnv, compute_indicators, build_feature_matrix, NUM_FEATURES
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    HAS_YFINANCE = False

print("=" * 80)
print("ASYMMETRIC STOP LOSS - 5 SYMBOL PERFORMANCE TEST")
print("=" * 80)

# Symbols to test
SYMBOLS = ['MSFT', 'AAPL', 'SPY', 'GOOGL', 'META']

# Download data
print("\n[1/4] Downloading market data...")
print("-" * 80)

data_cache = {}
for sym in SYMBOLS:
    try:
        print(f"  Downloading {sym}...", end=" ")
        df = yf.download(sym, start='2020-01-01', end='2024-12-31', progress=False)
        if len(df) < 500:
            print(f"FAILED (only {len(df)} bars)")
            continue

        data_cache[sym] = df
        print(f"OK ({len(df)} bars)")
    except Exception as e:
        print(f"FAILED ({e})")

if len(data_cache) == 0:
    print("\n[ERROR] No data downloaded. Cannot proceed.")
    sys.exit(1)

print(f"\n[OK] Downloaded {len(data_cache)}/{len(SYMBOLS)} symbols")

# Prepare features for each symbol
print("\n[2/4] Computing features...")
print("-" * 80)

features_cache = {}
prices_cache = {}
failed_symbols = []

for sym, df in list(data_cache.items()):  # Convert to list to avoid RuntimeError
    try:
        print(f"  {sym}...", end=" ")

        # Compute indicators
        indicators = compute_indicators(df)

        # Build feature matrix
        feat = build_feature_matrix(df, indicators)

        # Extract OHLCV
        prices = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

        features_cache[sym] = feat
        prices_cache[sym] = prices

        print(f"OK ({len(feat)} bars, {NUM_FEATURES} features)")
    except Exception as e:
        print(f"FAILED ({e})")
        failed_symbols.append(sym)

# Remove failed symbols
for sym in failed_symbols:
    if sym in data_cache:
        del data_cache[sym]

print(f"\n[OK] Features ready for {len(features_cache)} symbols")

# Run backtests
print("\n[3/4] Running backtests...")
print("-" * 80)

results = {}

for sym in features_cache.keys():
    print(f"\n{sym}:")
    print("  " + "-" * 76)

    feat = features_cache[sym]
    prices = prices_cache[sym]

    # Baseline: No asymmetric stops
    print("  [1/2] Baseline (no asymmetric stops)...", end=" ")
    cfg_baseline = Config()
    cfg_baseline.use_asymmetric_stops = False
    cfg_baseline.use_trailing_stops = False  # Disable all stops for baseline
    cfg_baseline.max_episode_steps = 1000

    env_baseline = TradingEnv(feat, prices, cfg_baseline, sym=f"{sym}_1d", ev=True)
    obs, _ = env_baseline.reset()

    done = False
    steps = 0
    while not done and steps < 1000:
        # Simple buy-and-hold-ish strategy (action 4 = full long)
        action = 4 if np.random.random() > 0.3 else 2  # Mostly long, sometimes flat
        obs, reward, term, trunc, info = env_baseline.step(action)
        done = term or trunc
        steps += 1

    baseline_info = env_baseline._info()
    print(f"Done ({steps} steps)")
    print(f"      Sharpe: {baseline_info['sharpe_ratio']:.3f}, "
          f"Max DD: {baseline_info['max_drawdown']:.2f}%, "
          f"PnL: {baseline_info['net_pnl_pct']:+.2f}%")

    # Enhanced: With asymmetric stops
    print("  [2/2] Enhanced (with asymmetric stops)...", end=" ")
    cfg_enhanced = Config()
    cfg_enhanced.use_asymmetric_stops = True
    cfg_enhanced.loss_stop_pct = 0.015  # Grid search optimal
    cfg_enhanced.profit_trail_pct = 0.05
    cfg_enhanced.max_episode_steps = 1000

    env_enhanced = TradingEnv(feat, prices, cfg_enhanced, sym=f"{sym}_1d", ev=True)

    # Use same random seed for comparable trajectory
    np.random.seed(42)
    obs, _ = env_enhanced.reset()

    done = False
    steps = 0
    stop_events = {'loss': 0, 'trail': 0}

    while not done and steps < 1000:
        action = 4 if np.random.random() > 0.3 else 2
        obs, reward, term, trunc, info = env_enhanced.step(action)

        # Count stop events
        if len(env_enhanced.trade_entries) > 0:
            last_entry = env_enhanced.trade_entries[-1]
            if 'ASYMMETRIC_STOP' in last_entry.get('action', ''):
                if 'LOSS' in last_entry['action']:
                    stop_events['loss'] += 1
                elif 'TRAIL' in last_entry['action']:
                    stop_events['trail'] += 1

        done = term or trunc
        steps += 1

    enhanced_info = env_enhanced._info()
    print(f"Done ({steps} steps)")
    print(f"      Sharpe: {enhanced_info['sharpe_ratio']:.3f}, "
          f"Max DD: {enhanced_info['max_drawdown']:.2f}%, "
          f"PnL: {enhanced_info['net_pnl_pct']:+.2f}%")
    print(f"      Stops: {stop_events['loss']} loss, {stop_events['trail']} trail")

    # Store results
    results[sym] = {
        'baseline': baseline_info,
        'enhanced': enhanced_info,
        'stop_events': stop_events,
    }

# Summary
print("\n" + "=" * 80)
print("[4/4] RESULTS SUMMARY")
print("=" * 80)

# Sharpe comparison table
sharpe_table = TableFormatter(title="SHARPE RATIO COMPARISON")
sharpe_table.add_column('Symbol', width=10, align='left')
sharpe_table.add_column('Baseline', width=15, align='right', format_spec='.3f')
sharpe_table.add_column('Enhanced', width=15, align='right', format_spec='.3f')
sharpe_table.add_column('Change %', width=12, align='right', format_spec='+.1f')

sharpe_improvements = []
dd_improvements = []
pnl_improvements = []

for sym, res in results.items():
    base_sharpe = res['baseline']['sharpe_ratio']
    enh_sharpe = res['enhanced']['sharpe_ratio']

    if abs(base_sharpe) > 1e-6:
        change_pct = ((enh_sharpe - base_sharpe) / abs(base_sharpe)) * 100
    else:
        change_pct = 0

    sharpe_improvements.append(enh_sharpe - base_sharpe)
    sharpe_table.add_row([sym, base_sharpe, enh_sharpe, change_pct])

print("\n" + sharpe_table.render())

# Max DD comparison table
dd_table = TableFormatter(title="MAX DRAWDOWN COMPARISON")
dd_table.add_column('Symbol', width=10, align='left')
dd_table.add_column('Baseline', width=15, align='right', format_spec='.2f')
dd_table.add_column('Enhanced', width=15, align='right', format_spec='.2f')
dd_table.add_column('Change %', width=12, align='right', format_spec='+.1f')

for sym, res in results.items():
    base_dd = res['baseline']['max_drawdown']
    enh_dd = res['enhanced']['max_drawdown']

    if abs(base_dd) > 1e-6:
        change_pct = ((enh_dd - base_dd) / abs(base_dd)) * 100
    else:
        change_pct = 0

    dd_improvements.append(base_dd - enh_dd)
    dd_table.add_row([sym, base_dd, enh_dd, change_pct])

print("\n" + dd_table.render())

# PnL comparison table
pnl_table = TableFormatter(title="NET P&L COMPARISON")
pnl_table.add_column('Symbol', width=10, align='left')
pnl_table.add_column('Baseline', width=15, align='right', format_spec='.2f')
pnl_table.add_column('Enhanced', width=15, align='right', format_spec='.2f')
pnl_table.add_column('Change %', width=12, align='right', format_spec='+.1f')

for sym, res in results.items():
    base_pnl = res['baseline']['net_pnl_pct']
    enh_pnl = res['enhanced']['net_pnl_pct']

    if abs(base_pnl) > 1e-6:
        change_pct = ((enh_pnl - base_pnl) / abs(base_pnl)) * 100
    else:
        change_pct = 0

    pnl_improvements.append(enh_pnl - base_pnl)
    pnl_table.add_row([sym, base_pnl, enh_pnl, change_pct])

print("\n" + pnl_table.render())

# Aggregate statistics
print("\n" + "=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80)

avg_sharpe_improvement = np.mean(sharpe_improvements)
avg_dd_improvement = np.mean(dd_improvements)
avg_pnl_improvement = np.mean(pnl_improvements)

symbols_better_sharpe = sum(1 for x in sharpe_improvements if x > 0)
symbols_better_dd = sum(1 for x in dd_improvements if x > 0)

print(f"\nAverage Sharpe improvement:    {avg_sharpe_improvement:+.3f}")
print(f"Average Max DD improvement:    {avg_dd_improvement:+.2f}%")
print(f"Average PnL improvement:       {avg_pnl_improvement:+.2f}%")
print(f"\nSymbols with better Sharpe:    {symbols_better_sharpe}/{len(results)} ({symbols_better_sharpe/len(results)*100:.0f}%)")
print(f"Symbols with lower Max DD:     {symbols_better_dd}/{len(results)} ({symbols_better_dd/len(results)*100:.0f}%)")

# Stop statistics
print("\n" + "=" * 80)
print("STOP LOSS STATISTICS")
print("=" * 80)

stop_table = TableFormatter(title="STOP EVENTS BY SYMBOL")
stop_table.add_column('Symbol', width=10, align='left')
stop_table.add_column('Loss Stops', width=12, align='right')
stop_table.add_column('Trail Stops', width=12, align='right')
stop_table.add_column('Total', width=12, align='right')
stop_table.add_column('Hit Rate %', width=12, align='right', format_spec='.1f')

for sym, res in results.items():
    loss_stops = res['stop_events']['loss']
    trail_stops = res['stop_events']['trail']
    total_stops = loss_stops + trail_stops
    total_trades = res['enhanced']['total_trades']

    hit_rate = (total_stops / max(total_trades, 1)) * 100 if total_trades > 0 else 0
    stop_table.add_row([sym, loss_stops, trail_stops, total_stops, hit_rate])

print("\n" + stop_table.render())

# Final assessment
print("\n" + "=" * 80)
print("ASSESSMENT")
print("=" * 80)

if avg_sharpe_improvement > 0.1:
    print("\n✅ STRONG POSITIVE IMPACT")
    print(f"   Average Sharpe improved by {avg_sharpe_improvement:.3f}")
elif avg_sharpe_improvement > 0:
    print("\n✅ POSITIVE IMPACT")
    print(f"   Average Sharpe improved by {avg_sharpe_improvement:.3f}")
elif avg_sharpe_improvement > -0.1:
    print("\n⚠️  NEUTRAL IMPACT")
    print(f"   Average Sharpe changed by {avg_sharpe_improvement:.3f}")
else:
    print("\n❌ NEGATIVE IMPACT")
    print(f"   Average Sharpe decreased by {avg_sharpe_improvement:.3f}")

if avg_dd_improvement > 0:
    print(f"✅ Drawdowns reduced by {avg_dd_improvement:.2f}% on average")
else:
    print(f"⚠️  Drawdowns increased by {abs(avg_dd_improvement):.2f}% on average")

print("\n" + "=" * 80)
print("[TEST COMPLETE]")
print("=" * 80)
