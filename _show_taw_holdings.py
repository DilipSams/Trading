"""
Show Tactical All-Weather monthly allocations and BearVol leverage.
Assets: SPY, TLT, GLD, DBC
Rules: positive 6m momentum -> allocate, weight by inverse 63d vol
"""
import numpy as np
import pandas as pd
import os, sys

_NORGATE_ROOT = os.environ.get("NORGATE_ROOT", r"D:\Experiments\norgate_data")
_TRADING_ROOT = os.environ.get("TRADING_ROOT", r"D:\Experiments\Trading")
sys.path.insert(0, _TRADING_ROOT)

NORGATE_EQ   = os.path.join(_NORGATE_ROOT, "US_Equities")
NORGATE_IDX  = os.path.join(_NORGATE_ROOT, "US_Indices")
NORGATE_ECON = os.path.join(_NORGATE_ROOT, "Economic")

def load_series(symbol, col="Adj Close", db="eq"):
    if db == "eq":
        path = os.path.join(NORGATE_EQ, f"{symbol}.parquet")
    elif db == "idx":
        path = os.path.join(NORGATE_IDX, f"{symbol}.parquet")
    elif db == "econ":
        path = os.path.join(NORGATE_ECON, f"{symbol}.parquet")
    else:
        path = os.path.join(NORGATE_EQ, f"{symbol}.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if col in df.columns:
        return df[col].dropna()
    elif "Close" in df.columns:
        return df["Close"].dropna()
    return df.iloc[:, 3].dropna()

# Load assets
assets = ["SPY", "TLT", "GLD", "DBC"]
asset_names = {"SPY": "US Stocks", "TLT": "Long Bonds", "GLD": "Gold", "DBC": "Commodities"}
asset_prices = {a: load_series(a) for a in assets}
prices = pd.DataFrame(asset_prices).dropna()
daily_rets = prices.pct_change()

# Load risk-free rate
rf_annual = load_series("%3MTCM", col="Close", db="econ")
spy = asset_prices["SPY"]
rf_daily = (rf_annual / 100 / 252).reindex(spy.index).ffill().fillna(0)

# Compute signals
mom_6m = prices.pct_change(126).shift(1)
vol_63d = daily_rets.rolling(63).std().shift(1) * np.sqrt(252)

# SPY SMA200 for BearVol overlay
spy_sma200 = spy.rolling(200).mean()
spy_bear = spy < spy_sma200  # True = bear

# Monthly rebalance dates
warmup = prices.index[max(252, 126) + 5]
dates = prices.index[prices.index >= warmup]

monthly_records = []

for i, date in enumerate(dates):
    if i % 21 != 0:
        continue

    mom = mom_6m.loc[date].dropna()
    vol = vol_63d.loc[date].dropna()

    # Positive momentum assets
    positive = mom[mom > 0]
    if len(positive) > 0:
        inv_vol = 1.0 / vol.reindex(positive.index).replace(0, np.nan).dropna()
        weights = inv_vol / inv_vol.sum()
        weights = weights.reindex(prices.columns).fillna(0)
    else:
        weights = pd.Series(0.0, index=prices.columns)

    # BearVol leverage computation
    is_bear = bool(spy_bear.reindex([date]).ffill().iloc[-1]) if date in spy_bear.index else False
    rv = daily_rets.loc[:date].tail(21).std().mean() * np.sqrt(252)
    if rv > 0.01:
        vol_lev = min(0.25 / rv * 3.0, 3.0)
    else:
        vol_lev = 3.0
    if is_bear:
        vol_lev = min(vol_lev, 0.5)
    vol_lev = max(0.25, min(vol_lev, 3.0))

    rec = {
        "date": date,
        "year": date.year,
        "month": date.month,
    }
    for a in assets:
        rec[f"{a}_mom6m"] = mom.get(a, np.nan)
        rec[f"{a}_wt"] = weights.get(a, 0)
    rec["n_assets"] = int((weights > 0.01).sum())
    rec["cash"] = weights.sum() < 0.01
    rec["spy_bear"] = is_bear
    rec["bear_vol_lev"] = round(vol_lev, 2)
    monthly_records.append(rec)

df = pd.DataFrame(monthly_records)

# ============================================================
# YEARLY SUMMARY TABLE
# ============================================================
print("=" * 120)
print("  TACTICAL ALL-WEATHER: YEARLY ALLOCATION SUMMARY")
print("  Assets: SPY (US Stocks), TLT (Long Bonds), GLD (Gold), DBC (Commodities)")
print("  Rule: Hold assets with positive 6-month momentum, weight by inverse volatility")
print("  BearVol3x: leverage up to 3x, cut to 0.5x when SPY < SMA200")
print("=" * 120)

print(f"\n  {'Year':<6s} {'#Assets':>7s} {'SPY':>6s} {'TLT':>6s} {'GLD':>6s} {'DBC':>6s} {'Cash':>6s} {'Bear':>6s} {'Avg':>6s} {'Allocations Held (month: assets @ weights)'}")
print(f"  {'-'*110}")

for year in sorted(df['year'].unique()):
    ydf = df[df['year'] == year]

    # Average weights for the year
    avg_spy = ydf['SPY_wt'].mean() * 100
    avg_tlt = ydf['TLT_wt'].mean() * 100
    avg_gld = ydf['GLD_wt'].mean() * 100
    avg_dbc = ydf['DBC_wt'].mean() * 100
    n_cash = ydf['cash'].sum()
    n_bear = ydf['spy_bear'].sum()
    avg_n = ydf['n_assets'].mean()

    # Monthly allocation summary
    monthly_allocs = []
    for _, row in ydf.iterrows():
        m = int(row['month'])
        held = []
        for a in assets:
            w = row[f'{a}_wt']
            if w > 0.01:
                held.append(f"{a}:{w*100:.0f}%")
        if not held:
            held = ["CASH"]
        monthly_allocs.append(f"{m:02d}={'+'.join(held)}")

    alloc_str = ", ".join(monthly_allocs)
    print(f"  {year:<6d} {avg_n:>5.1f}   {avg_spy:>5.1f} {avg_tlt:>5.1f} {avg_gld:>5.1f} {avg_dbc:>5.1f} {n_cash:>4d}mo {n_bear:>4d}mo {alloc_str}")

# ============================================================
# MONTH-BY-MONTH DETAIL (recent years)
# ============================================================
print(f"\n\n{'='*120}")
print(f"  MONTH-BY-MONTH DETAIL (2007-2026)")
print(f"  Showing: assets held, weights, 6m momentum, BearVol leverage")
print(f"{'='*120}")
print(f"  {'Date':<10s} {'SPY':>5s} {'TLT':>5s} {'GLD':>5s} {'DBC':>5s} {'#':>2s} {'Bear':>5s} {'Lev':>5s}  Holdings & Momentum")
print(f"  {'-'*100}")

for _, row in df.iterrows():
    date = row['date']
    n = int(row['n_assets'])
    bear = "BEAR" if row['spy_bear'] else ""
    lev = row['bear_vol_lev']

    held = []
    for a in assets:
        w = row[f'{a}_wt']
        mom = row[f'{a}_mom6m']
        if w > 0.01:
            held.append(f"{a}({w*100:.0f}%, 6m:{mom*100:+.1f}%)")
        elif not pd.isna(mom) and mom > 0:
            held.append(f"[{a} mom+{mom*100:.0f}% but filtered]")

    if not held:
        held = ["ALL CASH (all 4 assets have negative 6m momentum)"]

    print(f"  {date.strftime('%Y-%m'):<10s} "
          f"{row['SPY_wt']*100:>5.0f} {row['TLT_wt']*100:>5.0f} {row['GLD_wt']*100:>5.0f} {row['DBC_wt']*100:>5.0f} "
          f"{n:>2d} {bear:>5s} {lev:>5.2f}  {', '.join(held)}")

# ============================================================
# REGIME SUMMARY
# ============================================================
print(f"\n\n{'='*120}")
print(f"  REGIME BEHAVIOR SUMMARY")
print(f"{'='*120}")

regimes = [
    ("GFC (2008)", 2008, 2008),
    ("Recovery (2009-2010)", 2009, 2010),
    ("Bull (2013-2019)", 2013, 2019),
    ("COVID Crash (2020)", 2020, 2020),
    ("2022 Bear", 2022, 2022),
    ("AI Rally (2023-2025)", 2023, 2025),
]

for name, y1, y2 in regimes:
    rdf = df[(df['year'] >= y1) & (df['year'] <= y2)]
    if len(rdf) == 0:
        continue
    avg_spy = rdf['SPY_wt'].mean() * 100
    avg_tlt = rdf['TLT_wt'].mean() * 100
    avg_gld = rdf['GLD_wt'].mean() * 100
    avg_dbc = rdf['DBC_wt'].mean() * 100
    n_cash = rdf['cash'].sum()
    n_bear = rdf['spy_bear'].sum()
    n_total = len(rdf)
    avg_lev = rdf['bear_vol_lev'].mean()

    print(f"\n  {name}:")
    print(f"    Avg allocation: SPY {avg_spy:.0f}%, TLT {avg_tlt:.0f}%, GLD {avg_gld:.0f}%, DBC {avg_dbc:.0f}%")
    print(f"    Cash months: {n_cash}/{n_total}, Bear months: {n_bear}/{n_total}")
    print(f"    Avg BearVol leverage: {avg_lev:.2f}x")
