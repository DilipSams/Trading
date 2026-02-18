"""
Test CalendarAlpha Hypothesis
==============================

Hypothesis: Calendar effects (turn-of-month, day-of-week) have
predictive power and will outperform current SeasonalityAlpha.

Effects tested:
1. Turn-of-month (first/last 5 trading days)
2. Monday effect (reversion after weekend)
3. Friday effect (positioning before weekend)

Expected: IC > +0.010 (vs SeasonalityAlpha IC ~0.007)

Decision Gate: If IC > +0.007, IMPLEMENT to replace SeasonalityAlpha
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def is_turn_of_month(date, df_index):
    """
    Check if date is in turn-of-month window (first/last 5 trading days).
    """
    # Find position in month
    month_start = pd.Timestamp(date.year, date.month, 1)
    if date.month == 12:
        month_end = pd.Timestamp(date.year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = pd.Timestamp(date.year, date.month + 1, 1) - timedelta(days=1)

    # Get trading days in this month
    month_mask = (df_index >= month_start) & (df_index <= month_end)
    month_days = df_index[month_mask]

    if len(month_days) < 5:
        return False

    # First/last 5 trading days
    first_5 = month_days[:5]
    last_5 = month_days[-5:]

    return date in first_5.values or date in last_5.values


def compute_calendar_features(df):
    """
    Compute calendar effect features:
    1. turn_of_month: 1 if first/last 5 trading days, else 0
    2. day_of_week: 0-4 (Monday-Friday)
    3. month: 1-12
    """
    features = []

    for i, date in enumerate(df.index):
        tom = 1.0 if is_turn_of_month(date, df.index) else 0.0
        dow = date.weekday()  # 0=Monday, 4=Friday
        month = date.month

        features.append({
            'turn_of_month': tom,
            'monday': 1.0 if dow == 0 else 0.0,
            'friday': 1.0 if dow == 4 else 0.0,
            'month': month,
        })

    return pd.DataFrame(features, index=df.index)


def test_calendar_alpha(symbol='SPY', period='10y'):
    """
    Test calendar effects on returns.
    """
    print(f"\n{'='*70}")
    print(f"Testing Calendar Effects on {symbol}")
    print(f"{'='*70}")

    # Download data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty or len(df) < 500:
        print(f"ERROR: Insufficient data")
        return None

    print(f"Bars: {len(df)}")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")

    # Compute calendar features
    calendar_df = compute_calendar_features(df)

    # Compute forward returns
    df['fwd_ret_5'] = np.log(df['Close'].shift(-5) / df['Close'])
    df['fwd_ret_15'] = np.log(df['Close'].shift(-15) / df['Close'])

    # Merge
    df = pd.concat([df, calendar_df], axis=1)
    df = df.dropna(subset=['fwd_ret_5', 'fwd_ret_15'])

    # Test 1: Turn-of-month effect
    tom_returns = df[df['turn_of_month'] == 1]['fwd_ret_5']
    non_tom_returns = df[df['turn_of_month'] == 0]['fwd_ret_5']

    tom_mean = tom_returns.mean() * 252  # Annualized
    non_tom_mean = non_tom_returns.mean() * 252

    print(f"\nTurn-of-Month Effect (5-day forward returns):")
    print(f"  TOM mean return:     {tom_mean:+.2%} annualized")
    print(f"  Non-TOM mean return: {non_tom_mean:+.2%} annualized")
    print(f"  Difference:          {(tom_mean - non_tom_mean):+.2%}")

    # Test 2: Monday effect
    monday_returns = df[df['monday'] == 1]['fwd_ret_5']
    other_returns = df[df['monday'] == 0]['fwd_ret_5']

    monday_mean = monday_returns.mean() * 252
    other_mean = other_returns.mean() * 252

    print(f"\nMonday Effect (5-day forward returns):")
    print(f"  Monday mean return:  {monday_mean:+.2%} annualized")
    print(f"  Other days mean:     {other_mean:+.2%} annualized")
    print(f"  Difference:          {(monday_mean - other_mean):+.2%}")

    # Test 3: Friday effect
    friday_returns = df[df['friday'] == 1]['fwd_ret_5']
    other_friday = df[df['friday'] == 0]['fwd_ret_5']

    friday_mean = friday_returns.mean() * 252
    other_friday_mean = other_friday.mean() * 252

    print(f"\nFriday Effect (5-day forward returns):")
    print(f"  Friday mean return:  {friday_mean:+.2%} annualized")
    print(f"  Other days mean:     {other_friday_mean:+.2%} annualized")
    print(f"  Difference:          {(friday_mean - other_friday_mean):+.2%}")

    # Composite calendar signal
    # Simple: TOM = +1, Monday = -1 (contrarian), Friday = +0.5
    df['calendar_signal'] = (
        df['turn_of_month'] * 1.0 +
        df['monday'] * -0.5 +
        df['friday'] * 0.3
    )

    # IC test
    ic_5 = df['calendar_signal'].corr(df['fwd_ret_5'])
    ic_15 = df['calendar_signal'].corr(df['fwd_ret_15'])

    n = len(df)
    t_stat_5 = ic_5 * np.sqrt(n - 2) / np.sqrt(1 - ic_5**2 + 1e-10)
    t_stat_15 = ic_15 * np.sqrt(n - 2) / np.sqrt(1 - ic_15**2 + 1e-10)

    print(f"\nComposite Calendar Signal:")
    print(f"  IC (5-bar):   {ic_5:+.4f}, t-stat: {t_stat_5:+.2f}")
    print(f"  IC (15-bar):  {ic_15:+.4f}, t-stat: {t_stat_15:+.2f}")

    return {
        'symbol': symbol,
        'tom_effect': (tom_mean - non_tom_mean),
        'monday_effect': (monday_mean - other_mean),
        'friday_effect': (friday_mean - other_friday_mean),
        'ic_5': ic_5,
        'ic_15': ic_15,
        't_stat_5': t_stat_5,
    }


def main():
    print("\n" + "="*70)
    print("CALENDAR ALPHA HYPOTHESIS TEST")
    print("="*70)

    symbols = ['SPY', 'QQQ', 'DIA']
    results = []

    for symbol in symbols:
        result = test_calendar_alpha(symbol, period='10y')
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: CALENDAR EFFECTS")
    print(f"{'='*70}")
    print(f"{'Symbol':<8} {'TOM':>8} {'Monday':>8} {'Friday':>8} {'IC':>8}")
    print("-" * 70)

    for r in results:
        print(f"{r['symbol']:<8} {r['tom_effect']:>7.2%} {r['monday_effect']:>7.2%} "
              f"{r['friday_effect']:>7.2%} {r['ic_5']:>8.4f}")

    avg_ic = np.mean([abs(r['ic_5']) for r in results])
    print("-" * 70)
    print(f"{'AVG ABS IC':<8} {'':<24} {avg_ic:>8.4f}")

    # Decision gate
    print(f"\n{'='*70}")
    print("DECISION GATE")
    print(f"{'='*70}")
    print(f"Target IC:       +0.007 (to beat SeasonalityAlpha)")
    print(f"Actual IC:       {avg_ic:+.4f}")

    if avg_ic >= 0.007:
        print(f"[PASS] Hypothesis confirmed!")
        print(f"   Recommendation: IMPLEMENT CalendarAlpha")
        print(f"   Replace SeasonalityAlpha (lines 878-1080)")
    elif avg_ic >= 0.003:
        print(f"[PARTIAL] Weak improvement over SeasonalityAlpha")
        print(f"   Recommendation: Consider implementing if time allows")
    else:
        print(f"[FAIL] Does not beat SeasonalityAlpha")
        print(f"   Recommendation: DO NOT implement")


if __name__ == "__main__":
    main()
