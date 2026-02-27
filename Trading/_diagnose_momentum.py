"""
Diagnose why momentum strategy misses steady compounders like NVDA, MSFT, AMD.
"""
import numpy as np
import pandas as pd
import os

CACHE_DIR = r"D:\Experiments\Trading\data_cache"
rankings = pd.read_parquet(os.path.join(CACHE_DIR, "universe_rankings_top150.parquet"))
prices_df = pd.read_parquet(os.path.join(CACHE_DIR, "universe_prices_top150.parquet"))

# Build membership mask (top-100)
TOP_N = 100
rank_dates = sorted(rankings["date"].unique())
membership = pd.DataFrame(0, index=prices_df.index, columns=prices_df.columns)
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

# Carhart momentum
mom_12m = all_prices.pct_change(252).shift(1)
mom_1m = all_prices.pct_change(21).shift(1)
rel_momentum = mom_12m - mom_1m
abs_positive = (mom_12m > 0).shift(1).fillna(False)

composite = rel_momentum.copy()
composite[~abs_positive] = np.nan
composite[membership == 0] = np.nan

# Monthly dates
monthly_dates = all_prices.resample('M').last().index
monthly_dates = [d for d in monthly_dates if d in all_prices.index and d >= all_prices.index[260]]

# ============================================================
# PART 1: Diagnose NVDA, MSFT, AMD, AAPL ranking over time
# ============================================================
targets = ["NVDA", "MSFT", "AMD", "AAPL", "AMZN", "GOOGL", "META"]

print("=" * 120)
print("  MOMENTUM RANKING DIAGNOSIS: WHY DO WE MISS STEADY COMPOUNDERS?")
print("=" * 120)

for sym in targets:
    if sym not in all_prices.columns:
        print(f"\n  {sym}: not found in universe")
        continue

    print(f"\n  {sym} -- Monthly Momentum Rank & Status (2017-2025)")
    print(f"  {'Date':<10s} {'Price':>8s} {'12mRet%':>8s} {'Carhart':>8s} {'Rank':>6s} {'InUniv':>7s} {'AbsMom':>7s} {'Status':<30s}")
    print(f"  {'-'*90}")

    for date in monthly_dates:
        yr = date.year
        if yr < 2017 or yr > 2025:
            continue

        price = all_prices.loc[date, sym] if not pd.isna(all_prices.loc[date, sym]) else 0
        ret_12m = mom_12m.loc[date, sym] if date in mom_12m.index else np.nan
        carhart = rel_momentum.loc[date, sym] if date in rel_momentum.index else np.nan
        in_univ = membership.loc[date, sym] if date in membership.index else 0
        abs_mom = abs_positive.loc[date, sym] if date in abs_positive.index else False

        # Get rank among all qualifying stocks
        if date in composite.index:
            scores = composite.loc[date].dropna().sort_values(ascending=False)
            if sym in scores.index:
                rank = list(scores.index).index(sym) + 1
                rank_str = f"#{rank}"
            else:
                rank_str = "n/a"
        else:
            rank_str = "n/a"

        # Status
        status = ""
        if not in_univ:
            status = "NOT IN TOP-100 UNIVERSE"
        elif pd.isna(ret_12m) or not abs_mom:
            status = "NEGATIVE 12m return (filtered)"
        elif rank_str != "n/a":
            r = int(rank_str[1:])
            if r <= 5:
                status = "*** IN TOP-5 ***"
            elif r <= 10:
                status = "** IN TOP-10 **"
            elif r <= 20:
                status = "* IN TOP-20 *"
            else:
                status = f"(ranked but not top-20)"

        ret_str = f"{ret_12m*100:>+7.1f}%" if not pd.isna(ret_12m) else "    n/a"
        car_str = f"{carhart*100:>+7.1f}%" if not pd.isna(carhart) else "    n/a"

        print(f"  {date.strftime('%Y-%m'):<10s} {price:>8.1f} {ret_str:>8s} {car_str:>8s} {rank_str:>6s} {'Yes' if in_univ else 'NO':>7s} {'Yes' if abs_mom else 'No':>7s} {status}")

# ============================================================
# PART 2: What DID rank top-5 instead of NVDA during 2018-2022?
# ============================================================
print(f"\n\n{'='*120}")
print(f"  WHAT RANKED TOP-5 INSTEAD? (2018-2022)")
print(f"{'='*120}")
print(f"  {'Date':<10s} {'#1':<12s} {'#2':<12s} {'#3':<12s} {'#4':<12s} {'#5':<12s}  | NVDA rank")
print(f"  {'-'*100}")

for date in monthly_dates:
    yr = date.year
    if yr < 2018 or yr > 2022:
        continue

    scores = composite.loc[date].dropna().sort_values(ascending=False)
    top5 = list(scores.head(5).index)
    while len(top5) < 5:
        top5.append("--")

    # Clean names
    def clean(s):
        if '-' in s and s.split('-')[-1].isdigit():
            return s.split('-')[0] + '*'
        return s

    nvda_rank = "n/a"
    if "NVDA" in scores.index:
        nvda_rank = f"#{list(scores.index).index('NVDA') + 1}"
    elif "NVDA" not in membership.columns or membership.loc[date, "NVDA"] == 0:
        nvda_rank = "not in univ"
    elif not abs_positive.loc[date, "NVDA"]:
        nvda_rank = "neg 12m"

    print(f"  {date.strftime('%Y-%m'):<10s} {clean(top5[0]):<12s} {clean(top5[1]):<12s} {clean(top5[2]):<12s} {clean(top5[3]):<12s} {clean(top5[4]):<12s}  | {nvda_rank}")

# ============================================================
# PART 3: What if we used QQQ/XLK instead?
# ============================================================
print(f"\n\n{'='*120}")
print(f"  COMPARISON: QQQ vs INDIVIDUAL STOCK PICKING (2018-2025)")
print(f"{'='*120}")

# Load QQQ
NORGATE_EQ = r"D:\Experiments\norgate_data\US_Equities"
qqq_path = os.path.join(NORGATE_EQ, "QQQ.parquet")
spy_path = os.path.join(NORGATE_EQ, "SPY.parquet")
xlk_path = os.path.join(NORGATE_EQ, "XLK.parquet")

for etf_name, etf_path in [("QQQ", qqq_path), ("SPY", spy_path), ("XLK", xlk_path)]:
    if os.path.exists(etf_path):
        df = pd.read_parquet(etf_path)
        df.index = pd.to_datetime(df.index)
        if "Adj Close" in df.columns:
            p = df["Adj Close"]
        elif "Close" in df.columns:
            p = df["Close"]
        else:
            p = df.iloc[:, 3]

        # 2018-2025 return
        start = p.loc["2018-01-01":].iloc[0]
        end = p.iloc[-1]
        years = (p.index[-1] - p.loc["2018-01-01":].index[0]).days / 365.25
        total_ret = end / start - 1
        cagr = (end / start) ** (1/years) - 1
        print(f"  {etf_name}: 2018-2025 CAGR = {cagr*100:+.1f}%, Total = {total_ret*100:+.0f}%")

# Individual stock performance 2018-2025
for sym in ["NVDA", "MSFT", "AMD", "AAPL", "AMZN", "GOOGL", "META"]:
    if sym in all_prices.columns:
        p = all_prices[sym].dropna()
        p_2018 = p.loc["2018-01-01":]
        if len(p_2018) > 252:
            start = p_2018.iloc[0]
            end = p_2018.iloc[-1]
            years = (p_2018.index[-1] - p_2018.index[0]).days / 365.25
            total_ret = end / start - 1
            cagr = (end / start) ** (1/years) - 1
            print(f"  {sym}: 2018-2025 CAGR = {cagr*100:+.1f}%, Total = {total_ret*100:+.0f}%")

print(f"\n  KEY INSIGHT: These mega-caps compound at 20-50% CAGR for years.")
print(f"  A momentum strategy misses them because flashier names (meme stocks,")
print(f"  crypto proxies) have higher SHORT-TERM momentum even though mega-cap")
print(f"  tech has better LONG-TERM compounding.")
print(f"\n  SOLUTION OPTIONS:")
print(f"  1. QQQ/XLK core + momentum satellite (get compounders + alpha)")
print(f"  2. Hysteresis band: enter top-5, exit only when rank > 20")
print(f"  3. Multi-timeframe: blend 6m + 12m + 24m momentum (rewards persistence)")
print(f"  4. Minimum holding period: 3-6 months once entered")
