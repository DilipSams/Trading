"""
Multi-Symbol Validation: Test Trailing Stops on MSFT and NFLX

Validates that the trailing stop improvement seen on SPY (13% WR -> 33.3% WR, -25.87% -> -2.38% P&L)
generalizes to individual stocks with different volatility profiles.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def compute_macd(closes, fast=12, slow=26, signal=9):
    """Standard MACD indicator"""
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    return macd_line, signal_line

def run_strategy(df, symbol, use_stops=False):
    """
    Run baseline MACD strategy with optional trailing stops

    Args:
        df: DataFrame with OHLCV data
        symbol: Ticker symbol for reporting
        use_stops: If True, use 5-bar swing trailing stops

    Returns:
        dict with performance metrics
    """
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values

    # Compute MACD
    macd_line, signal_line = compute_macd(closes)

    # Detect crossovers
    cross_up = np.zeros(len(closes), dtype=bool)
    cross_down = np.zeros(len(closes), dtype=bool)

    for i in range(1, len(closes)):
        if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
            cross_up[i] = True
        elif macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
            cross_down[i] = True

    # Trading simulation
    pos = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    stop = 0.0
    trades = []

    start_i = 50  # Wait for MACD to stabilize

    for i in range(start_i, len(closes)):
        current_price = closes[i]

        # Update trailing stop if enabled
        if use_stops and i >= 5:
            swing_low = lows[i-5:i].min()
            swing_high = highs[i-5:i].max()

            # Check stop hits BEFORE processing new signals
            if pos == 1:  # Long position
                stop = max(stop, swing_low)  # Trail up
                if lows[i] <= stop:
                    # Stopped out
                    exit_price = stop
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'LONG',
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'STOP'
                    })
                    pos = 0
                    continue  # Don't process signals this bar

            elif pos == -1:  # Short position
                stop = min(stop, swing_high)  # Trail down
                if highs[i] >= stop:
                    # Stopped out
                    exit_price = stop
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'SHORT',
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'STOP'
                    })
                    pos = 0
                    continue  # Don't process signals this bar

        # Process signals
        if pos == 0:  # Flat - look for entry
            if cross_up[i]:
                pos = 1
                entry_price = current_price
                entry_bar = i
                if use_stops and i >= 5:
                    stop = lows[i-5:i].min()  # Initial stop
            elif cross_down[i]:
                pos = -1
                entry_price = current_price
                entry_bar = i
                if use_stops and i >= 5:
                    stop = highs[i-5:i].max()  # Initial stop

        elif pos == 1:  # Long - look for exit
            if cross_down[i]:
                exit_price = current_price
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'LONG',
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'SIGNAL'
                })
                pos = 0

        elif pos == -1:  # Short - look for exit
            if cross_up[i]:
                exit_price = current_price
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': 'SHORT',
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'SIGNAL'
                })
                pos = 0

    # Calculate metrics
    if len(trades) == 0:
        return {
            'symbol': symbol,
            'num_trades': 0,
            'win_rate': 0.0,
            'total_pnl_pct': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'winners': 0,
            'losers': 0
        }

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df['pnl_pct'] > 0]
    losers = trades_df[trades_df['pnl_pct'] <= 0]

    return {
        'symbol': symbol,
        'num_trades': len(trades),
        'win_rate': len(winners) / len(trades) * 100,
        'total_pnl_pct': trades_df['pnl_pct'].sum(),
        'avg_win': winners['pnl_pct'].mean() if len(winners) > 0 else 0.0,
        'avg_loss': losers['pnl_pct'].mean() if len(losers) > 0 else 0.0,
        'winners': len(winners),
        'losers': len(losers),
        'stop_exits': len(trades_df[trades_df['exit_reason'] == 'STOP']) if use_stops else 0
    }

def main():
    print("=" * 80)
    print("MULTI-SYMBOL VALIDATION: Trailing Stops on MSFT and NFLX")
    print("=" * 80)
    print()
    print("Testing whether the SPY improvement generalizes to individual stocks:")
    print("  SPY Baseline:      13.0% WR, -25.87% P&L")
    print("  SPY w/ Stops:      33.3% WR,  -2.38% P&L  (+20.3pp WR, +23.49% P&L)")
    print()
    print("Downloading 2 years of data for MSFT and NFLX...")
    print()

    # Download data for both symbols
    symbols = ['MSFT', 'NFLX']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years

    results = []

    for symbol in symbols:
        print(f"\n{'=' * 80}")
        print(f"Testing {symbol}")
        print('=' * 80)

        try:
            # Download data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            if len(df) < 100:
                print(f"[ERROR] Insufficient data for {symbol}")
                continue

            print(f"Downloaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
            print()

            # Run baseline
            baseline = run_strategy(df, symbol, use_stops=False)

            # Run with trailing stops
            with_stops = run_strategy(df, symbol, use_stops=True)

            # Print results
            print(f"\n{symbol} BASELINE (MACD only):")
            print(f"  Trades:        {baseline['num_trades']}")
            print(f"  Win Rate:      {baseline['win_rate']:.1f}%")
            print(f"  Total P&L:     {baseline['total_pnl_pct']:+.2f}%")
            print(f"  Winners:       {baseline['winners']}")
            print(f"  Losers:        {baseline['losers']}")
            if baseline['winners'] > 0:
                print(f"  Avg Win:       {baseline['avg_win']:+.2f}%")
            if baseline['losers'] > 0:
                print(f"  Avg Loss:      {baseline['avg_loss']:+.2f}%")

            print(f"\n{symbol} WITH TRAILING STOPS (5-bar swing):")
            print(f"  Trades:        {with_stops['num_trades']}")
            print(f"  Win Rate:      {with_stops['win_rate']:.1f}%")
            print(f"  Total P&L:     {with_stops['total_pnl_pct']:+.2f}%")
            print(f"  Winners:       {with_stops['winners']}")
            print(f"  Losers:        {with_stops['losers']}")
            print(f"  Stop Exits:    {with_stops['stop_exits']}")
            if with_stops['winners'] > 0:
                print(f"  Avg Win:       {with_stops['avg_win']:+.2f}%")
            if with_stops['losers'] > 0:
                print(f"  Avg Loss:      {with_stops['avg_loss']:+.2f}%")

            # Calculate improvement
            wr_improvement = with_stops['win_rate'] - baseline['win_rate']
            pnl_improvement = with_stops['total_pnl_pct'] - baseline['total_pnl_pct']

            print(f"\n{symbol} IMPROVEMENT:")
            print(f"  Win Rate:      {wr_improvement:+.1f}pp")
            print(f"  Total P&L:     {pnl_improvement:+.2f}%")

            # Store results
            results.append({
                'symbol': symbol,
                'baseline_wr': baseline['win_rate'],
                'stops_wr': with_stops['win_rate'],
                'wr_improvement': wr_improvement,
                'baseline_pnl': baseline['total_pnl_pct'],
                'stops_pnl': with_stops['total_pnl_pct'],
                'pnl_improvement': pnl_improvement,
                'baseline_trades': baseline['num_trades'],
                'stops_trades': with_stops['num_trades']
            })

        except Exception as e:
            print(f"[ERROR] Failed to test {symbol}: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Trailing Stops Validation Across 3 Symbols")
    print("=" * 80)
    print()
    print(f"{'Symbol':<8} {'Base WR':<10} {'Stops WR':<10} {'WR Change':<10} {'Base P&L':<12} {'Stops P&L':<12} {'P&L Change':<12}")
    print("-" * 80)

    # Add SPY results from previous test
    print(f"{'SPY':<8} {'13.0%':<10} {'33.3%':<10} {'+20.3pp':<10} {'-25.87%':<12} {'-2.38%':<12} {'+23.49%':<12}")

    # Print new results
    for r in results:
        print(f"{r['symbol']:<8} "
              f"{r['baseline_wr']:<10.1f}% "
              f"{r['stops_wr']:<10.1f}% "
              f"{r['wr_improvement']:+.1f}pp   "
              f"{r['baseline_pnl']:<12.2f}% "
              f"{r['stops_pnl']:<12.2f}% "
              f"{r['pnl_improvement']:+.2f}%")

    print()
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if len(results) >= 2:
        avg_wr_improvement = sum(r['wr_improvement'] for r in results) / len(results)
        avg_pnl_improvement = sum(r['pnl_improvement'] for r in results) / len(results)

        all_positive_wr = all(r['wr_improvement'] > 0 for r in results)
        all_positive_pnl = all(r['pnl_improvement'] > 0 for r in results)

        print()
        print(f"Tested on {len(results) + 1} symbols (SPY + {len(results)} stocks)")
        print(f"Average WR improvement:  {avg_wr_improvement:+.1f}pp")
        print(f"Average P&L improvement: {avg_pnl_improvement:+.2f}%")
        print()

        if all_positive_wr and all_positive_pnl:
            print("[VALIDATION PASSED]")
            print("Trailing stops improve performance across all tested symbols.")
            print()
            print("RECOMMENDATION: IMPLEMENT trailing stops in production system")
            print()
            print("Implementation:")
            print("  1. Modify ExecutionEngine to track 5-bar swing highs/lows")
            print("  2. Update position management to trail stops each bar")
            print("  3. Add stop_loss field to Trade objects")
            print("  4. Log stop exits separately for analysis")
            print()
            print("Expected Impact:")
            print(f"  - Win Rate: +{avg_wr_improvement:.0f}pp improvement")
            print(f"  - P&L: +{avg_pnl_improvement:.1f}% improvement")
            print("  - Better protection of winners")
            print("  - Faster exit from losers")
        else:
            print("[VALIDATION MIXED]")
            print("Results vary across symbols - proceed with caution")
            print()
            print("RECOMMENDATION: Implement with adaptive parameters")
            print("Consider symbol-specific lookback periods or volatility-based stops")
    else:
        print("[INSUFFICIENT DATA]")
        print("Need at least 2 additional symbols for validation")

    print()

if __name__ == "__main__":
    main()
