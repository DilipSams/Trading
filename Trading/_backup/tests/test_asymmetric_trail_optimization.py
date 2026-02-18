"""
Trail Stop Optimization Test
=============================

Tests different trail stop widths to find optimal setting:
- 5% (current/grid search optimal)
- 7% (medium)
- 10% (wider)
- 12% (widest)

Loss stop held constant at 1.5% (working well).
"""

import sys
sys.path.insert(0, '.')

# Fix Windows console encoding for box-drawing characters
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from table_formatter import TableFormatter

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("[ERROR] yfinance not installed: pip install yfinance")
    sys.exit(1)

print("=" * 80)
print("TRAIL STOP OPTIMIZATION TEST")
print("=" * 80)

SYMBOLS = ['MSFT', 'AAPL', 'SPY', 'GOOGL', 'META']
TRAIL_STOPS = [0.05, 0.07, 0.10, 0.12]  # 5%, 7%, 10%, 12%
LOSS_STOP = 0.015  # Fixed at 1.5%

# Download data
print("\n[1/2] Downloading market data...")
print("-" * 80)

data_cache = {}
for sym in SYMBOLS:
    try:
        print(f"  {sym}...", end=" ")
        df = yf.download(sym, start='2020-01-01', end='2024-12-31', progress=False)
        if len(df) < 500:
            print(f"FAILED (only {len(df)} bars)")
            continue
        data_cache[sym] = df
        print(f"OK ({len(df)} bars)")
    except Exception as e:
        print(f"FAILED ({e})")

print(f"\n[OK] Downloaded {len(data_cache)} symbols")

def simulate_trading(prices, use_stops=False, loss_pct=0.015, trail_pct=0.05):
    """Simple trading simulation with asymmetric stops."""
    starting_equity = 100000.0
    equity = starting_equity
    position = 0
    entry_price = 0.0
    peak_pnl_pct = 0.0

    equity_curve = [float(equity)]
    stops_hit = {'loss': 0, 'trail': 0}

    for i in range(100, len(prices)):
        current_price = prices[i]

        if position != 0 and use_stops:
            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct > peak_pnl_pct:
                peak_pnl_pct = pnl_pct

            stop_triggered = False
            stop_type = None

            if pnl_pct < 0:
                if abs(pnl_pct) >= loss_pct:
                    stop_triggered = True
                    stop_type = 'loss'
            else:
                drawdown_from_peak = peak_pnl_pct - pnl_pct
                if drawdown_from_peak >= trail_pct:
                    stop_triggered = True
                    stop_type = 'trail'

            if stop_triggered:
                equity = position * current_price
                if stop_type:
                    stops_hit[stop_type] += 1
                position = 0
                entry_price = 0
                peak_pnl_pct = 0

        if position == 0 and i % 20 == 0:
            shares_to_buy = int(equity / current_price)
            if shares_to_buy > 0:
                position = shares_to_buy
                entry_price = current_price
                peak_pnl_pct = 0

        portfolio_value = position * current_price if position != 0 else equity
        equity_curve.append(float(portfolio_value))

    if position != 0:
        equity = position * prices[-1]

    # Metrics
    equity_arr = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)) if len(returns) > 0 else 0.0

    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_dd = float(abs(np.min(drawdowns)) * 100) if len(drawdowns) > 0 else 0.0

    final_return = float((equity - starting_equity) / starting_equity * 100)

    return sharpe, max_dd, final_return, stops_hit

# Run tests
print("\n[2/2] Testing trail stop widths...")
print("-" * 80)

results = {trail: {} for trail in TRAIL_STOPS}

for trail_pct in TRAIL_STOPS:
    print(f"\n>>> TRAIL STOP: {trail_pct*100:.0f}% (Loss stop: {LOSS_STOP*100:.1f}%)")
    print("    " + "-" * 72)

    for sym, df in data_cache.items():
        prices = df['Close'].values

        sharpe, dd, ret, stops = simulate_trading(
            prices,
            use_stops=True,
            loss_pct=LOSS_STOP,
            trail_pct=trail_pct
        )

        results[trail_pct][sym] = {
            'sharpe': sharpe,
            'dd': dd,
            'return': ret,
            'stops': stops
        }

        print(f"    {sym:<8} Sharpe: {sharpe:6.3f}  DD: {dd:5.2f}%  Return: {ret:+7.2f}%  "
              f"Stops: {stops['loss']:2d}L {stops['trail']:2d}T")

# Summary comparisons
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

# Aggregate by trail stop width
summary_table = TableFormatter(title="TRAIL STOP WIDTH COMPARISON")
summary_table.add_column('Trail %', width=8, align='center')
summary_table.add_column('Avg Sharpe', width=12, align='right', format_spec='.3f')
summary_table.add_column('Avg DD', width=10, align='right', format_spec='.2f')
summary_table.add_column('Avg Return', width=12, align='right', format_spec='.2f')
summary_table.add_column('Loss Stops', width=11, align='right')
summary_table.add_column('Trail Stops', width=12, align='right')

best_sharpe = {'trail': 0, 'value': -999}
best_dd = {'trail': 0, 'value': 999}
best_return = {'trail': 0, 'value': -999}

for trail_pct in TRAIL_STOPS:
    sharpes = [results[trail_pct][sym]['sharpe'] for sym in data_cache.keys()]
    dds = [results[trail_pct][sym]['dd'] for sym in data_cache.keys()]
    returns = [results[trail_pct][sym]['return'] for sym in data_cache.keys()]

    total_loss_stops = sum(results[trail_pct][sym]['stops']['loss'] for sym in data_cache.keys())
    total_trail_stops = sum(results[trail_pct][sym]['stops']['trail'] for sym in data_cache.keys())

    avg_sharpe = np.mean(sharpes)
    avg_dd = np.mean(dds)
    avg_return = np.mean(returns)

    summary_table.add_row([f"{trail_pct*100:.0f}%", avg_sharpe, avg_dd, avg_return, total_loss_stops, total_trail_stops])

    if avg_sharpe > best_sharpe['value']:
        best_sharpe = {'trail': trail_pct, 'value': avg_sharpe}
    if avg_dd < best_dd['value']:
        best_dd = {'trail': trail_pct, 'value': avg_dd}
    if avg_return > best_return['value']:
        best_return = {'trail': trail_pct, 'value': avg_return}

print("\n" + summary_table.render())

# Per-symbol breakdown
print("\n" + "=" * 80)
print("PER-SYMBOL SHARPE COMPARISON")
print("=" * 80)

sharpe_by_symbol = TableFormatter(title="SHARPE BY SYMBOL AND TRAIL WIDTH")
sharpe_by_symbol.add_column('Symbol', width=10, align='left')
sharpe_by_symbol.add_column('5%', width=10, align='right', format_spec='.3f')
sharpe_by_symbol.add_column('7%', width=10, align='right', format_spec='.3f')
sharpe_by_symbol.add_column('10%', width=10, align='right', format_spec='.3f')
sharpe_by_symbol.add_column('12%', width=10, align='right', format_spec='.3f')
sharpe_by_symbol.add_column('Best', width=8, align='center')

for sym in data_cache.keys():
    sharpes = [results[trail_pct][sym]['sharpe'] for trail_pct in TRAIL_STOPS]
    best_idx = np.argmax(sharpes)
    best_trail = TRAIL_STOPS[best_idx]

    row_data = [sym] + sharpes + [f"{best_trail*100:.0f}%"]
    sharpe_by_symbol.add_row(row_data)

print("\n" + sharpe_by_symbol.render())

print("\n" + "=" * 80)
print("PER-SYMBOL RETURN COMPARISON")
print("=" * 80)

return_by_symbol = TableFormatter(title="RETURN BY SYMBOL AND TRAIL WIDTH")
return_by_symbol.add_column('Symbol', width=10, align='left')
return_by_symbol.add_column('5%', width=11, align='right', format_spec='.2f')
return_by_symbol.add_column('7%', width=11, align='right', format_spec='.2f')
return_by_symbol.add_column('10%', width=11, align='right', format_spec='.2f')
return_by_symbol.add_column('12%', width=11, align='right', format_spec='.2f')
return_by_symbol.add_column('Best', width=8, align='center')

for sym in data_cache.keys():
    returns = [results[trail_pct][sym]['return'] for trail_pct in TRAIL_STOPS]
    best_idx = np.argmax(returns)
    best_trail = TRAIL_STOPS[best_idx]

    row_data = [sym] + returns + [f"{best_trail*100:.0f}%"]
    return_by_symbol.add_row(row_data)

print("\n" + return_by_symbol.render())

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print(f"\n[1] BEST SHARPE:  {best_sharpe['trail']*100:.0f}% trail (Sharpe: {best_sharpe['value']:.3f})")
print(f"[2] LOWEST DD:    {best_dd['trail']*100:.0f}% trail (DD: {best_dd['value']:.2f}%)")
print(f"[3] BEST RETURN:  {best_return['trail']*100:.0f}% trail (Return: {best_return['value']:.2f}%)")

# Count how many symbols prefer each trail width
symbol_preferences = {}
for trail_pct in TRAIL_STOPS:
    count = 0
    for sym in data_cache.keys():
        sharpes = [results[t][sym]['sharpe'] for t in TRAIL_STOPS]
        if results[trail_pct][sym]['sharpe'] == max(sharpes):
            count += 1
    symbol_preferences[trail_pct] = count

print(f"\n[4] SYMBOL PREFERENCES (by Sharpe):")
for trail_pct in TRAIL_STOPS:
    count = symbol_preferences[trail_pct]
    pct = count / len(data_cache) * 100
    print(f"    {trail_pct*100:>3.0f}% trail: {count}/{len(data_cache)} symbols ({pct:.0f}%)")

# Final recommendation
most_preferred = max(symbol_preferences, key=symbol_preferences.get)

print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

print(f"\n[OPTIMAL] Trail stop: {most_preferred*100:.0f}%")
print(f"          Loss stop:  {LOSS_STOP*100:.1f}% (unchanged)")

avg_sharpe_optimal = np.mean([results[most_preferred][sym]['sharpe'] for sym in data_cache.keys()])
avg_dd_optimal = np.mean([results[most_preferred][sym]['dd'] for sym in data_cache.keys()])
avg_return_optimal = np.mean([results[most_preferred][sym]['return'] for sym in data_cache.keys()])

print(f"\nExpected performance:")
print(f"  Average Sharpe:  {avg_sharpe_optimal:.3f}")
print(f"  Average DD:      {avg_dd_optimal:.2f}%")
print(f"  Average Return:  {avg_return_optimal:+.2f}%")

# Compare to 5% baseline
baseline_sharpe = np.mean([results[0.05][sym]['sharpe'] for sym in data_cache.keys()])
baseline_dd = np.mean([results[0.05][sym]['dd'] for sym in data_cache.keys()])
baseline_return = np.mean([results[0.05][sym]['return'] for sym in data_cache.keys()])

print(f"\nImprovement vs 5% trail:")
print(f"  Sharpe:  {avg_sharpe_optimal - baseline_sharpe:+.3f}")
print(f"  DD:      {avg_dd_optimal - baseline_dd:+.2f}%")
print(f"  Return:  {avg_return_optimal - baseline_return:+.2f}%")

print("\n" + "=" * 80)
print("[TEST COMPLETE]")
print("=" * 80)

# Update config recommendation
print(f"\n[CONFIGURATION]")
print(f"Update your Config with these optimal values:")
print(f"")
print(f"cfg.loss_stop_pct = {LOSS_STOP}")
print(f"cfg.profit_trail_pct = {most_preferred}")
print("=" * 80)
