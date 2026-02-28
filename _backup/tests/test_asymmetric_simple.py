"""
Simplified Multi-Symbol Test for Asymmetric Stop Loss
======================================================

Tests asymmetric stops on 5 symbols with simplified approach.
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
print("ASYMMETRIC STOP LOSS - 5 SYMBOL PERFORMANCE TEST")
print("=" * 80)

# Symbols to test
SYMBOLS = ['MSFT', 'AAPL', 'SPY', 'GOOGL', 'META']

# Download data
print("\n[1/3] Downloading market data (2020-2024)...")
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

# Simulate trading with/without asymmetric stops
print("\n[2/3] Running simulations...")
print("-" * 80)

def simulate_trading(prices, use_stops=False, loss_pct=0.015, trail_pct=0.05):
    """
    Simple trading simulation with optional asymmetric stops.

    Returns: (final_equity, sharpe, max_dd, trades, stops_hit)
    """
    starting_equity = 100000.0
    equity = starting_equity
    position = 0  # Shares
    entry_price = 0.0
    peak_pnl_pct = 0.0
    bars_in_trade = 0

    equity_curve = [float(equity)]
    trades = []
    stops_hit = {'loss': 0, 'trail': 0}

    for i in range(100, len(prices)):
        current_price = prices[i]

        # Check stop loss if in position
        if position != 0 and use_stops:
            # Calculate P&L %
            pnl_pct = (current_price - entry_price) / entry_price

            # Update peak
            if pnl_pct > peak_pnl_pct:
                peak_pnl_pct = pnl_pct

            # Check stops
            stop_triggered = False
            stop_type = None

            if pnl_pct < 0:
                # Losing position - tight stop
                if abs(pnl_pct) >= loss_pct:
                    stop_triggered = True
                    stop_type = 'loss'
            else:
                # Winning position - trailing stop
                drawdown_from_peak = peak_pnl_pct - pnl_pct
                if drawdown_from_peak >= trail_pct:
                    stop_triggered = True
                    stop_type = 'trail'

            if stop_triggered:
                # Close position - update equity
                exit_value = position * current_price
                equity = exit_value

                trades.append({
                    'entry': entry_price,
                    'exit': current_price,
                    'pnl': exit_value - (position * entry_price),
                    'pnl_pct': pnl_pct,
                    'type': stop_type
                })

                if stop_type:
                    stops_hit[stop_type] += 1

                position = 0
                entry_price = 0
                peak_pnl_pct = 0
                bars_in_trade = 0

        # Simple trading logic: Enter every 20 bars if flat
        if position == 0 and i % 20 == 0:
            # Go long with all equity
            shares_to_buy = int(equity / current_price)
            if shares_to_buy > 0:
                position = shares_to_buy
                entry_price = current_price
                peak_pnl_pct = 0
                bars_in_trade = 0

        # Calculate portfolio value
        if position != 0:
            portfolio_value = position * current_price
            bars_in_trade += 1
        else:
            portfolio_value = equity

        equity_curve.append(float(portfolio_value))

    # Close any open position at end
    if position != 0:
        exit_value = position * prices[-1]
        equity = exit_value

        trades.append({
            'entry': entry_price,
            'exit': prices[-1],
            'pnl': exit_value - (position * entry_price),
            'pnl_pct': (prices[-1] - entry_price) / entry_price
        })

    # Calculate metrics
    equity_arr = np.array(equity_curve, dtype=np.float64)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)) if len(returns) > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_dd = float(abs(np.min(drawdowns)) * 100) if len(drawdowns) > 0 else 0.0

    final_return_pct = float((equity - starting_equity) / starting_equity * 100)

    return float(equity), float(sharpe), float(max_dd), trades, stops_hit, float(final_return_pct)

results = {}

for sym, df in data_cache.items():
    print(f"\n{sym}:")
    print("  " + "-" * 76)

    prices = df['Close'].values

    # Baseline: No stops
    print("  [1/2] Baseline (no stops)...", end=" ")
    equity_base, sharpe_base, dd_base, trades_base, _, ret_base = simulate_trading(
        prices, use_stops=False
    )
    print(f"Sharpe: {sharpe_base:.3f}, DD: {dd_base:.2f}%, Return: {ret_base:+.2f}%")

    # Enhanced: With asymmetric stops
    print("  [2/2] With asymmetric stops...", end=" ")
    equity_enh, sharpe_enh, dd_enh, trades_enh, stops_hit, ret_enh = simulate_trading(
        prices, use_stops=True, loss_pct=0.015, trail_pct=0.05
    )
    print(f"Sharpe: {sharpe_enh:.3f}, DD: {dd_enh:.2f}%, Return: {ret_enh:+.2f}%")
    print(f"       Stops: {stops_hit['loss']} loss, {stops_hit['trail']} trail")

    results[sym] = {
        'baseline': {'sharpe': sharpe_base, 'dd': dd_base, 'return': ret_base, 'trades': len(trades_base)},
        'enhanced': {'sharpe': sharpe_enh, 'dd': dd_enh, 'return': ret_enh, 'trades': len(trades_enh)},
        'stops': stops_hit
    }

# Summary
print("\n" + "=" * 80)
print("[3/3] RESULTS SUMMARY")
print("=" * 80)

# Sharpe comparison table
sharpe_table = TableFormatter(title="SHARPE RATIO COMPARISON")
sharpe_table.add_column('Symbol', width=10, align='left')
sharpe_table.add_column('Baseline', width=12, align='right', format_spec='.3f')
sharpe_table.add_column('Enhanced', width=12, align='right', format_spec='.3f')
sharpe_table.add_column('Improvement', width=15, align='right', format_spec='+.3f')

sharpe_improvements = []
for sym, res in results.items():
    base = res['baseline']['sharpe']
    enh = res['enhanced']['sharpe']
    improvement = enh - base
    sharpe_improvements.append(improvement)
    sharpe_table.add_row([sym, base, enh, improvement])

print("\n" + sharpe_table.render())

# Max DD comparison table
dd_table = TableFormatter(title="MAX DRAWDOWN COMPARISON")
dd_table.add_column('Symbol', width=10, align='left')
dd_table.add_column('Baseline', width=12, align='right', format_spec='.2f')
dd_table.add_column('Enhanced', width=12, align='right', format_spec='.2f')
dd_table.add_column('Improvement', width=15, align='right', format_spec='+.2f')

dd_improvements = []
for sym, res in results.items():
    base = res['baseline']['dd']
    enh = res['enhanced']['dd']
    improvement = base - enh  # Lower DD is better
    dd_improvements.append(improvement)
    dd_table.add_row([sym, base, enh, improvement])

print("\n" + dd_table.render())

# Return comparison table
return_table = TableFormatter(title="TOTAL RETURN COMPARISON")
return_table.add_column('Symbol', width=10, align='left')
return_table.add_column('Baseline', width=12, align='right', format_spec='.2f')
return_table.add_column('Enhanced', width=12, align='right', format_spec='.2f')
return_table.add_column('Improvement', width=15, align='right', format_spec='+.2f')

return_improvements = []
for sym, res in results.items():
    base = res['baseline']['return']
    enh = res['enhanced']['return']
    improvement = enh - base
    return_improvements.append(improvement)
    return_table.add_row([sym, base, enh, improvement])

print("\n" + return_table.render())

# Aggregate stats
print("\n" + "=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80)

avg_sharpe_imp = np.mean(sharpe_improvements)
avg_dd_imp = np.mean(dd_improvements)
avg_return_imp = np.mean(return_improvements)

symbols_better_sharpe = sum(1 for x in sharpe_improvements if x > 0)
symbols_better_dd = sum(1 for x in dd_improvements if x > 0)
symbols_better_return = sum(1 for x in return_improvements if x > 0)

print(f"\nAverage Sharpe improvement:      {avg_sharpe_imp:+.3f}")
print(f"Average Max DD improvement:      {avg_dd_imp:+.2f}%")
print(f"Average Return improvement:      {avg_return_imp:+.2f}%")

print(f"\nSymbols with better Sharpe:      {symbols_better_sharpe}/{len(results)} ({symbols_better_sharpe/len(results)*100:.0f}%)")
print(f"Symbols with lower DD:           {symbols_better_dd}/{len(results)} ({symbols_better_dd/len(results)*100:.0f}%)")
print(f"Symbols with higher return:      {symbols_better_return}/{len(results)} ({symbols_better_return/len(results)*100:.0f}%)")

# Stop statistics
print("\n" + "=" * 80)
print("STOP LOSS STATISTICS")
print("=" * 80)

stop_table = TableFormatter(title="STOP EVENTS BY SYMBOL")
stop_table.add_column('Symbol', width=10, align='left')
stop_table.add_column('Loss Stops', width=12, align='right')
stop_table.add_column('Trail Stops', width=13, align='right')
stop_table.add_column('Total', width=10, align='right')

total_loss_stops = 0
total_trail_stops = 0

for sym, res in results.items():
    loss = res['stops']['loss']
    trail = res['stops']['trail']
    total_loss_stops += loss
    total_trail_stops += trail
    stop_table.add_row([sym, loss, trail, loss + trail])

stop_table.add_row(['TOTAL', total_loss_stops, total_trail_stops, total_loss_stops + total_trail_stops])

print("\n" + stop_table.render())

if total_loss_stops + total_trail_stops > 0:
    loss_pct = total_loss_stops / (total_loss_stops + total_trail_stops) * 100
    trail_pct = total_trail_stops / (total_loss_stops + total_trail_stops) * 100
    print(f"\nStop Distribution: {loss_pct:.1f}% loss stops, {trail_pct:.1f}% trail stops")

# Assessment
print("\n" + "=" * 80)
print("ASSESSMENT")
print("=" * 80)

if avg_sharpe_imp > 0.2:
    verdict = "[SUCCESS] STRONG POSITIVE IMPACT"
elif avg_sharpe_imp > 0:
    verdict = "[SUCCESS] POSITIVE IMPACT"
elif avg_sharpe_imp > -0.2:
    verdict = "[NEUTRAL] NEUTRAL IMPACT"
else:
    verdict = "[WARNING] NEGATIVE IMPACT"

print(f"\n{verdict}")
print(f"   Average Sharpe improvement: {avg_sharpe_imp:+.3f}")

if avg_dd_imp > 2:
    print(f"[OK] Drawdowns significantly reduced ({avg_dd_imp:+.2f}%)")
elif avg_dd_imp > 0:
    print(f"[OK] Drawdowns slightly reduced ({avg_dd_imp:+.2f}%)")
else:
    print(f"[WARNING] Drawdowns increased ({avg_dd_imp:+.2f}%)")

print("\n" + "=" * 80)
print("[TEST COMPLETE]")
print("=" * 80)
