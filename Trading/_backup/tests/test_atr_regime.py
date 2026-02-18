"""
Test ATR Regime Detector Hypothesis
====================================

Hypothesis: Adding ATR z-score as a feature improves regime detection
and RL observation quality, leading to IC improvement of +0.005.

ATR Regime = (ATR_20 - mean_ATR_60) / std_ATR_60

High ATR = High volatility regime (reduce exposure)
Low ATR = Low volatility regime (increase exposure)

Decision Gate: If IC improvement > +0.003, IMPLEMENT (30 min effort)
"""

import numpy as np
import pandas as pd
import yfinance as yf


def compute_atr(highs, lows, closes, period=14):
    """Compute Average True Range."""
    if len(closes) < period + 1:
        return 0.0

    tr_list = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])

        tr = max(high_low, high_close, low_close)
        tr_list.append(tr)

    atr = np.mean(tr_list[-period:])
    return float(atr)


def compute_atr_regime(highs, lows, closes, atr_period=20, lookback=60):
    """
    Compute ATR regime z-score.

    ATR_regime = (ATR_20 - mean_ATR_60) / std_ATR_60
    """
    if len(closes) < lookback + atr_period:
        return 0.0

    # Compute rolling ATR
    atr_values = []
    for i in range(atr_period, len(closes)):
        atr = compute_atr(highs[:i+1], lows[:i+1], closes[:i+1], period=atr_period)
        atr_values.append(atr)

    if len(atr_values) < lookback:
        return 0.0

    # Current ATR
    current_atr = atr_values[-1]

    # ATR statistics over lookback
    recent_atrs = atr_values[-lookback:]
    mean_atr = np.mean(recent_atrs)
    std_atr = np.std(recent_atrs)

    if std_atr < 1e-10:
        return 0.0

    atr_regime_z = (current_atr - mean_atr) / std_atr
    return float(atr_regime_z)


def test_atr_regime_alpha(symbol='SPY', period='10y'):
    """
    Test if ATR regime z-score has predictive power.

    Method:
    1. Compute ATR regime at each bar
    2. Test if high ATR predicts lower returns (risk-off)
    3. Test if low ATR predicts higher returns (risk-on)
    """
    print(f"\n{'='*70}")
    print(f"Testing ATR Regime on {symbol}")
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

    # Compute ATR regime and forward returns
    atr_regimes = []
    forward_returns = []

    for i in range(100, len(closes) - 15):
        atr_z = compute_atr_regime(highs[:i+1], lows[:i+1], closes[:i+1],
                                   atr_period=20, lookback=60)

        # Forward 15-bar return
        fwd_ret = np.log(closes[min(i+15, len(closes)-1)] / closes[i])

        atr_regimes.append(atr_z)
        forward_returns.append(fwd_ret)

    atr_regimes = np.array(atr_regimes)
    forward_returns = np.array(forward_returns)

    # Compute IC (correlation between ATR regime and returns)
    # Hypothesis: NEGATIVE IC (high ATR = low future returns)
    ic = np.corrcoef(atr_regimes, forward_returns)[0, 1]

    # T-statistic
    n = len(atr_regimes)
    t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

    print(f"\nIC (ATR Regime vs Forward Returns): {ic:+.4f}")
    print(f"t-stat: {t_stat:+.2f}")
    print(f"N: {n}")

    # ATR regime stats
    print(f"\nATR Regime Statistics:")
    print(f"  Mean: {np.mean(atr_regimes):+.2f}")
    print(f"  Std:  {np.std(atr_regimes):.2f}")
    print(f"  Min:  {np.min(atr_regimes):+.2f}")
    print(f"  Max:  {np.max(atr_regimes):+.2f}")

    # Test signal: inverse ATR regime (low ATR = long, high ATR = short)
    inverse_ic = -ic
    print(f"\nInverse Signal IC (use -ATR_regime as signal): {inverse_ic:+.4f}")

    return {
        'symbol': symbol,
        'ic': ic,
        'inverse_ic': inverse_ic,
        't_stat': t_stat,
        'n': n,
    }


def main():
    print("\n" + "="*70)
    print("ATR REGIME DETECTOR HYPOTHESIS TEST")
    print("="*70)

    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    results = []

    for symbol in symbols:
        result = test_atr_regime_alpha(symbol, period='10y')
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: ATR REGIME PREDICTIVE POWER")
    print(f"{'='*70}")
    print(f"{'Symbol':<8} {'Raw IC':>10} {'Inv IC':>10} {'t-stat':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r['symbol']:<8} {r['ic']:>10.4f} {r['inverse_ic']:>10.4f} {r['t_stat']:>10.2f}")

    # Use absolute IC (works either way with inversion)
    avg_abs_ic = np.mean([abs(r['ic']) for r in results])
    print("-" * 70)
    print(f"{'AVG ABS':<8} {'':<10} {avg_abs_ic:>10.4f}")

    # Decision gate
    print(f"\n{'='*70}")
    print("DECISION GATE")
    print(f"{'='*70}")
    print(f"Target IC:       +0.003 (absolute value)")
    print(f"Actual IC:       {avg_abs_ic:+.4f}")

    if avg_abs_ic >= 0.003:
        print(f"[PASS] Hypothesis confirmed!")
        print(f"   Recommendation: IMPLEMENT ATR Regime feature")
        print(f"   Effort: 30 minutes (add to compute_indicators)")
    elif avg_abs_ic >= 0.001:
        print(f"[PARTIAL] Weak signal but easy implementation")
        print(f"   Recommendation: Consider implementing (low risk)")
    else:
        print(f"[FAIL] Signal too weak")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
