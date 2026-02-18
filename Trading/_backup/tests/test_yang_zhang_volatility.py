"""
Test Yang-Zhang Volatility Estimator Hypothesis
================================================

Hypothesis: Yang-Zhang volatility (using OHLC) is 14x more efficient than
close-to-close volatility, leading to:
1. Lower noise in volatility estimates
2. Better volatility regime detection
3. Improved performance of vol-dependent alphas

Expected: Vol estimation error reduced by 50%+

Decision Gate: If estimation error reduction > 30%, IMPLEMENT
"""

import numpy as np
import pandas as pd
import yfinance as yf


def close_to_close_vol(closes, lookback=20, annualize=True):
    """Traditional close-to-close volatility."""
    if len(closes) < lookback + 1:
        return 0.15

    log_rets = np.diff(np.log(closes[-lookback-1:] + 1e-12))
    vol = float(np.std(log_rets))

    return vol * np.sqrt(252) if annualize else vol


def yang_zhang_vol(opens, highs, lows, closes, lookback=20, annualize=True):
    """
    Yang-Zhang volatility estimator (OHLC-based).

    Combines:
    - Overnight volatility: ln(O_t / C_{t-1})
    - Rogers-Satchell: (ln H/C)(ln H/O) + (ln L/C)(ln L/O)
    - Close-to-close component

    ~14x more efficient than close-to-close for intraday data.
    """
    if len(closes) < lookback + 1:
        return 0.15

    opens = opens[-lookback-1:]
    highs = highs[-lookback-1:]
    lows = lows[-lookback-1:]
    closes = closes[-lookback-1:]

    # Overnight volatility: ln(O_t / C_{t-1})
    overnight_rets = np.log(opens[1:] / (closes[:-1] + 1e-12))
    var_overnight = np.var(overnight_rets)

    # Rogers-Satchell (intraday without drift)
    log_ho = np.log(highs[1:] / (opens[1:] + 1e-12))
    log_hc = np.log(highs[1:] / (closes[1:] + 1e-12))
    log_lo = np.log(lows[1:] / (opens[1:] + 1e-12))
    log_lc = np.log(lows[1:] / (closes[1:] + 1e-12))

    rs = log_ho * log_hc + log_lo * log_lc
    var_rs = np.mean(rs)

    # Close-to-close
    log_cc = np.log(closes[1:] / (closes[:-1] + 1e-12))
    var_cc = np.var(log_cc)

    # Yang-Zhang formula (k=0.34 empirically optimal)
    k = 0.34
    var_yz = var_overnight + k * var_cc + (1 - k) * var_rs
    var_yz = max(var_yz, 1e-12)

    vol = np.sqrt(var_yz)
    return float(vol * np.sqrt(252)) if annualize else float(vol)


def compute_realized_volatility(returns, window=20):
    """Compute realized volatility as ground truth."""
    if len(returns) < window:
        return 0.15

    # Use next 20 bars' volatility as "realized"
    actual_vol = float(np.std(returns[-window:]))
    return actual_vol * np.sqrt(252)


def test_volatility_estimation(symbol='SPY', period='5y'):
    """
    Test Yang-Zhang vs Close-to-Close volatility estimation.

    Method:
    1. Estimate volatility using both methods at time t
    2. Measure realized volatility over next 20 bars
    3. Compare estimation errors
    """
    print(f"\n{'='*70}")
    print(f"Testing Volatility Estimators on {symbol}")
    print(f"{'='*70}")

    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty or len(df) < 500:
        print(f"ERROR: Insufficient data")
        return None

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values

    print(f"Bars: {len(closes)}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Compute returns for realized vol
    returns = np.diff(np.log(closes + 1e-12))

    # Rolling estimation errors
    cc_errors = []
    yz_errors = []

    lookback = 20
    forward_window = 20

    for i in range(lookback + 50, len(closes) - forward_window):
        # Estimate volatility at time i
        cc_est = close_to_close_vol(closes[:i+1], lookback=lookback)
        yz_est = yang_zhang_vol(opens[:i+1], highs[:i+1], lows[:i+1],
                                closes[:i+1], lookback=lookback)

        # Realized volatility over next 20 bars
        realized_vol = float(np.std(returns[i:i+forward_window])) * np.sqrt(252)

        # Errors (absolute percentage error)
        cc_error = abs(cc_est - realized_vol) / (realized_vol + 1e-10)
        yz_error = abs(yz_est - realized_vol) / (realized_vol + 1e-10)

        cc_errors.append(cc_error)
        yz_errors.append(yz_error)

    cc_errors = np.array(cc_errors)
    yz_errors = np.array(yz_errors)

    # Statistics
    cc_mean_error = np.mean(cc_errors) * 100
    yz_mean_error = np.mean(yz_errors) * 100

    cc_median_error = np.median(cc_errors) * 100
    yz_median_error = np.median(yz_errors) * 100

    error_reduction = ((cc_mean_error - yz_mean_error) / cc_mean_error) * 100

    print(f"\nClose-to-Close Volatility:")
    print(f"  Mean Error:    {cc_mean_error:.2f}%")
    print(f"  Median Error:  {cc_median_error:.2f}%")

    print(f"\nYang-Zhang Volatility:")
    print(f"  Mean Error:    {yz_mean_error:.2f}%")
    print(f"  Median Error:  {yz_median_error:.2f}%")

    print(f"\nError Reduction: {error_reduction:.1f}%")

    return {
        'symbol': symbol,
        'cc_mean_error': cc_mean_error,
        'yz_mean_error': yz_mean_error,
        'error_reduction_pct': error_reduction,
    }


def main():
    print("\n" + "="*70)
    print("YANG-ZHANG VOLATILITY HYPOTHESIS TEST")
    print("="*70)

    symbols = ['SPY', 'AAPL', 'MSFT']
    results = []

    for symbol in symbols:
        result = test_volatility_estimation(symbol, period='5y')
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: VOLATILITY ESTIMATION ERROR REDUCTION")
    print(f"{'='*70}")
    print(f"{'Symbol':<8} {'C2C Error':>12} {'YZ Error':>12} {'Reduction':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r['symbol']:<8} {r['cc_mean_error']:>11.2f}% {r['yz_mean_error']:>11.2f}% "
              f"{r['error_reduction_pct']:>11.1f}%")

    avg_reduction = np.mean([r['error_reduction_pct'] for r in results])
    print("-" * 70)
    print(f"{'AVERAGE':<8} {'':<12} {'':<12} {avg_reduction:>11.1f}%")

    # Decision gate
    print(f"\n{'='*70}")
    print("DECISION GATE")
    print(f"{'='*70}")
    print(f"Target Error Reduction:  30%")
    print(f"Actual Error Reduction:  {avg_reduction:.1f}%")

    if avg_reduction >= 30:
        print(f"[PASS] Hypothesis confirmed!")
        print(f"   Recommendation: IMPLEMENT Yang-Zhang volatility")
        print(f"   Replace np.std(log_rets) in 13 locations")
    elif avg_reduction >= 15:
        print(f"[PARTIAL] Improvement positive but below target")
        print(f"   Recommendation: Consider implementing for key alphas")
    else:
        print(f"[FAIL] Improvement insufficient")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
