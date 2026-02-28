"""
Test Intrabar Range Ratio Hypothesis
=====================================

Hypothesis: Range Ratio = (High - Low) / ATR provides useful information
about intrabar volatility and improves RL observation quality.

High range = Volatile bar (uncertainty, reduce confidence)
Low range = Calm bar (confidence, normal trading)

Decision Gate: If abs(IC) > +0.003, IMPLEMENT (30 min effort)
"""

import numpy as np
import pandas as pd
import yfinance as yf


def compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range."""
    if len(closes) < period + 1:
        return np.mean(highs - lows)  # Fallback

    tr_list = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        tr_list.append(tr)

    atr = np.mean(tr_list[-period:])
    return float(atr)


def compute_range_ratio(highs, lows, closes, atr_period=14):
    """
    Compute Range Ratio = (High - Low) / ATR

    > 1.0: Wide range (volatile)
    < 1.0: Narrow range (calm)
    """
    if len(closes) < atr_period + 1:
        return 1.0

    current_range = highs[-1] - lows[-1]
    atr = compute_atr(highs, lows, closes, period=atr_period)

    if atr < 1e-10:
        return 1.0

    range_ratio = current_range / atr
    return float(range_ratio)


def test_range_ratio_alpha(symbol='SPY', period='10y'):
    """
    Test if Range Ratio has predictive power.

    Hypothesis: Wide ranges predict mean reversion (overreaction)
    """
    print(f"\n{'='*70}")
    print(f"Testing Range Ratio on {symbol}")
    print(f"{'='*70}")

    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty or len(df) < 500:
        print(f"ERROR: Insufficient data")
        return None

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    print(f"Bars: {len(closes)}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Compute range ratio and forward returns
    range_ratios = []
    forward_returns = []

    for i in range(50, len(closes) - 5):
        rr = compute_range_ratio(highs[:i+1], lows[:i+1], closes[:i+1],
                                 atr_period=14)

        # Forward 5-bar return (short horizon for range effects)
        fwd_ret = np.log(closes[min(i+5, len(closes)-1)] / closes[i])

        range_ratios.append(rr)
        forward_returns.append(fwd_ret)

    range_ratios = np.array(range_ratios)
    forward_returns = np.array(forward_returns)

    # Compute IC
    ic = np.corrcoef(range_ratios, forward_returns)[0, 1]

    # T-statistic
    n = len(range_ratios)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

    print(f"\nIC (Range Ratio vs Forward Returns): {ic:+.4f}")
    print(f"t-stat: {t_stat:+.2f}")
    print(f"N: {n}")

    # Range Ratio stats
    print(f"\nRange Ratio Statistics:")
    print(f"  Mean: {np.mean(range_ratios):.2f}")
    print(f"  Std:  {np.std(range_ratios):.2f}")
    print(f"  Median: {np.median(range_ratios):.2f}")

    return {
        'symbol': symbol,
        'ic': ic,
        't_stat': t_stat,
        'n': n,
    }


def main():
    print("\n" + "="*70)
    print("INTRABAR RANGE RATIO HYPOTHESIS TEST")
    print("="*70)

    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    results = []

    for symbol in symbols:
        result = test_range_ratio_alpha(symbol, period='10y')
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: RANGE RATIO PREDICTIVE POWER")
    print(f"{'='*70}")
    print(f"{'Symbol':<8} {'IC':>10} {'t-stat':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['symbol']:<8} {r['ic']:>10.4f} {r['t_stat']:>10.2f}")

    avg_abs_ic = np.mean([abs(r['ic']) for r in results])
    print("-" * 70)
    print(f"{'AVG ABS':<8} {avg_abs_ic:>10.4f}")

    # Decision gate
    print(f"\n{'='*70}")
    print("DECISION GATE")
    print(f"{'='*70}")
    print(f"Target IC:       +0.003 (absolute)")
    print(f"Actual IC:       {avg_abs_ic:+.4f}")

    if avg_abs_ic >= 0.003:
        print(f"[PASS] Hypothesis confirmed!")
        print(f"   Recommendation: IMPLEMENT Range Ratio feature")
        print(f"   Effort: 30 minutes (add to compute_indicators)")
    elif avg_abs_ic >= 0.001:
        print(f"[PARTIAL] Weak signal")
        print(f"   Recommendation: Low priority, consider for RL obs")
    else:
        print(f"[FAIL] No significant signal")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
