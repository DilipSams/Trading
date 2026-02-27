"""
Quick script to show which stocks the Concentrated Momentum strategy picks each year.
Replicates the dual_momentum ranking logic and dumps yearly holdings.
"""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, r"D:\Experiments\Trading")

# Load cached universe data
CACHE_DIR = r"D:\Experiments\Trading\data_cache"
rankings = pd.read_parquet(os.path.join(CACHE_DIR, "universe_rankings_top150.parquet"))
prices_df = pd.read_parquet(os.path.join(CACHE_DIR, "universe_prices_top150.parquet"))

# Build membership mask (top-100 by dollar volume)
TOP_N = 100
membership = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
for _, row in rankings.iterrows():
    date = row["date"]
    uid = row["uid"]
    rank = row["rank"]
    if rank <= TOP_N and uid in membership.columns:
        # Apply from this date until next ranking date
        mask = membership.index >= date
        membership.loc[mask, uid] = 1

# More precise: rebuild month-by-month
membership = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
rank_dates = sorted(rankings["date"].unique())
for i, rdate in enumerate(rank_dates):
    top = rankings[(rankings["date"] == rdate) & (rankings["rank"] <= TOP_N)]["uid"].values
    if i + 1 < len(rank_dates):
        next_date = rank_dates[i + 1]
        mask = (membership.index >= rdate) & (membership.index < next_date)
    else:
        mask = membership.index >= rdate
    for uid in top:
        if uid in membership.columns:
            membership.loc[mask, uid] = 1

all_prices = prices_df.copy()

# Carhart momentum: 12m return minus 1m return
mom_12m = all_prices.pct_change(252).shift(1)
mom_1m = all_prices.pct_change(21).shift(1)
rel_momentum = mom_12m - mom_1m

# Absolute momentum filter: 12m return > 0
abs_positive = (mom_12m > 0).shift(1).fillna(False)

# Composite score
composite = rel_momentum.copy()
composite[~abs_positive] = np.nan
composite[membership == 0] = np.nan

# Extract holdings at each monthly rebalance
warmup_idx = max(252, 200) + 5
dates = all_prices.index[warmup_idx:]

for top_n in [5, 10]:
    print(f"\n{'='*100}")
    print(f"  CONCENTRATED MOMENTUM TOP-{top_n}: YEARLY HOLDINGS")
    print(f"  Selection: Carhart momentum (12m-1m), absolute mom > 0, top-{top_n} from dynamic universe")
    print(f"{'='*100}")

    # Track holdings by year
    yearly_holdings = {}  # year -> dict of {symbol: count_of_months_held}
    yearly_snapshots = {}  # year -> list of (month, [symbols])

    rebal_count = 0
    for i, date in enumerate(dates):
        if i % 21 == 0:  # monthly rebalance
            scores = composite.loc[date].dropna().sort_values(ascending=False)
            top = scores.head(top_n)
            if len(top) == 0:
                continue

            year = date.year
            month = date.month
            symbols = list(top.index)

            # Clean up UIDs: strip delisted suffix for readability
            clean_syms = []
            for s in symbols:
                if '-' in s and s.split('-')[-1].isdigit():
                    clean_syms.append(s.split('-')[0] + '*')  # * = delisted
                else:
                    clean_syms.append(s)

            if year not in yearly_snapshots:
                yearly_snapshots[year] = []
            yearly_snapshots[year].append((month, clean_syms))

            if year not in yearly_holdings:
                yearly_holdings[year] = {}
            for s in clean_syms:
                yearly_holdings[year][s] = yearly_holdings[year].get(s, 0) + 1

    # Print year-by-year summary
    print(f"\n  {'Year':<6s} {'Months':>6s}  Stocks Held (sorted by months held)")
    print(f"  {'-'*90}")

    for year in sorted(yearly_holdings.keys()):
        holdings = yearly_holdings[year]
        n_months = len(yearly_snapshots[year])
        # Sort by months held descending
        sorted_h = sorted(holdings.items(), key=lambda x: -x[1])
        # Format: SYMBOL(N) where N = months held that year
        parts = [f"{sym}({cnt})" for sym, cnt in sorted_h]
        line = ", ".join(parts)
        print(f"  {year:<6d} {n_months:>4d}mo  {line}")

    # Also print month-by-month for recent years
    print(f"\n  MONTH-BY-MONTH DETAIL (2020-2026):")
    print(f"  {'Date':<12s}  Stocks")
    print(f"  {'-'*85}")
    for year in sorted(yearly_snapshots.keys()):
        if year < 2020:
            continue
        for month, syms in yearly_snapshots[year]:
            date_str = f"{year}-{month:02d}"
            print(f"  {date_str:<12s}  {', '.join(syms)}")

print("\n  * = delisted stock (identified by UID suffix)")
