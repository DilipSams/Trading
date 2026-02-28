"""
ABBV Diagnostics Script
- 5-year and OOS (Aug 2022 - Feb 2026) price return
- SMA50/100/200 counts per bar
- Momentum ranking vs 50 tech stocks
- Simulated P&L at 0.9x position when bull_aligned
"""
import sys, os
import numpy as np
import pandas as pd

# ── Load ABBV ──────────────────────────────────────────────────────────────────
NORGATE_DIR = os.environ.get("NORGATE_ROOT", r"D:\Experiments\norgate_data")
abbv_path = os.path.join(NORGATE_DIR, "US_Equities", "ABBV.parquet")
df = pd.read_parquet(abbv_path)

# Normalize column names
col_map = {}
for c in df.columns:
    cl = c.lower().strip()
    if cl == "open":   col_map[c] = "Open"
    elif cl == "high": col_map[c] = "High"
    elif cl == "low":  col_map[c] = "Low"
    elif cl == "close" and "unadj" not in cl.replace(" ",""): col_map[c] = "Close"
    elif cl in ("volume","vol"): col_map[c] = "Volume"
df = df.rename(columns=col_map).dropna(subset=["Close"])

print(f"ABBV full history: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} bars)")

# ── 1. 5-Year return ───────────────────────────────────────────────────────────
end_date   = df.index[-1]
start_5y   = end_date - pd.DateOffset(years=5)
df_5y      = df[df.index >= start_5y]
ret_5y     = df_5y["Close"].iloc[-1] / df_5y["Close"].iloc[0] - 1.0
print(f"\n[1] ABBV 5-Year Return:")
print(f"    Period : {df_5y.index[0].date()} → {df_5y.index[-1].date()}")
print(f"    Price  : ${df_5y['Close'].iloc[0]:.2f} → ${df_5y['Close'].iloc[-1]:.2f}")
print(f"    Return : {ret_5y*100:.2f}%")

# ── OOS period ─────────────────────────────────────────────────────────────────
OOS_START = pd.Timestamp("2022-08-01")
OOS_END   = pd.Timestamp("2026-02-28")
df_oos = df[(df.index >= OOS_START) & (df.index <= OOS_END)].copy()
ret_oos = df_oos["Close"].iloc[-1] / df_oos["Close"].iloc[0] - 1.0
print(f"\n    OOS Period : {df_oos.index[0].date()} → {df_oos.index[-1].date()} ({len(df_oos)} bars)")
print(f"    OOS Return : {ret_oos*100:.2f}%")

# ── 2. SMA computations per OOS bar ───────────────────────────────────────────
closes_all = df["Close"].values
dates_all  = df.index

# For each bar in OOS, compute SMAs using all history up to that bar
oos_mask = (df.index >= OOS_START) & (df.index <= OOS_END)
oos_indices = np.where(oos_mask)[0]

sma50_vals  = []
sma100_vals = []
sma200_vals = []
prices      = []

for idx in oos_indices:
    p = closes_all[idx]
    prices.append(p)
    sma50_vals.append(np.mean(closes_all[max(0,idx-49):idx+1]))
    sma100_vals.append(np.mean(closes_all[max(0,idx-99):idx+1]) if idx >= 99 else np.nan)
    sma200_vals.append(np.mean(closes_all[max(0,idx-199):idx+1]) if idx >= 199 else np.nan)

prices    = np.array(prices)
sma50     = np.array(sma50_vals)
sma100    = np.array(sma100_vals)
sma200    = np.array(sma200_vals)

above_sma200  = np.sum((prices > sma200) & ~np.isnan(sma200))
below_sma200  = np.sum((prices < sma200) & ~np.isnan(sma200))
bull_aligned  = np.sum(
    (prices > sma50) & (sma50 > sma100) & (sma100 > sma200)
    & ~np.isnan(sma100) & ~np.isnan(sma200)
)
total_valid   = np.sum(~np.isnan(sma200))

print(f"\n[2] SMA Analysis over OOS ({len(oos_indices)} bars, {total_valid} with SMA200):")
print(f"    Above SMA200     : {above_sma200} bars ({above_sma200/total_valid*100:.1f}% of SMA200-eligible)")
print(f"    Below SMA200     : {below_sma200} bars ({below_sma200/total_valid*100:.1f}% of SMA200-eligible)")
print(f"    Bull-aligned     : {bull_aligned} bars ({bull_aligned/total_valid*100:.1f}% of SMA200-eligible)")

# ── 3. 12-Month Momentum vs 50 Tech Stocks ────────────────────────────────────
TECH_50 = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","ORCL","CRM","AMD",
    "PLTR","ADBE","CSCO","ACN","IBM","NOW","TXN","QCOM","KLAC","INTU",
    "AMAT","LRCX","APH","ANET","PANW","MU","INTC","ADI","NFLX","UBER",
    "APP","CRWD","WDC","SNPS","CDNS","DELL","MSI","GLW","DASH","MRVL",
    "NET","FTNT","SNOW","MPWR","ABBV","LLY","UNH","JNJ","MRK","TMO"
]

# Use last bar of OOS as reference for momentum
ref_date = df_oos.index[-1]
lookback_date = ref_date - pd.DateOffset(months=12)

momentum_scores = {}
equities_dir = os.path.join(NORGATE_DIR, "US_Equities")
for sym in TECH_50:
    fp = os.path.join(equities_dir, f"{sym}.parquet")
    if not os.path.exists(fp):
        continue
    try:
        d = pd.read_parquet(fp)
        cmap = {}
        for c in d.columns:
            cl = c.lower().strip()
            if cl == "close" and "unadj" not in cl.replace(" ",""): cmap[c] = "Close"
        d = d.rename(columns=cmap).dropna(subset=["Close"])
        d_lb = d[d.index >= lookback_date]
        d_ref = d[d.index <= ref_date]
        if len(d_lb) < 20 or len(d_ref) < 1:
            continue
        start_price = d_lb["Close"].iloc[0]
        end_price   = d_ref["Close"].iloc[-1]
        momentum_scores[sym] = end_price / start_price - 1.0
    except:
        pass

abbv_mom = momentum_scores.get("ABBV", np.nan)
sorted_mom = sorted(momentum_scores.items(), key=lambda x: -x[1])
abbv_rank = next((i+1 for i,(s,_) in enumerate(sorted_mom) if s=="ABBV"), None)

print(f"\n[3] 12-Month Momentum (to {ref_date.date()}):")
print(f"    ABBV momentum    : {abbv_mom*100:.2f}%")
print(f"    ABBV rank        : #{abbv_rank} of {len(sorted_mom)}")
print(f"    Top 10 tech momentum:")
for i,(sym,m) in enumerate(sorted_mom[:10]):
    marker = " ← ABBV" if sym == "ABBV" else ""
    print(f"      {i+1:2d}. {sym:<6s} {m*100:+7.2f}%{marker}")
if abbv_rank and abbv_rank > 10:
    for i,(sym,m) in enumerate(sorted_mom):
        if sym == "ABBV":
            print(f"      ...")
            print(f"      {i+1:2d}. {sym:<6s} {m*100:+7.2f}%  ← ABBV")
            break

# ── 4. Simulated P&L at 0.9x when bull_aligned ───────────────────────────────
# Build per-bar returns during OOS
oos_df = df_oos.copy()
close_arr = oos_df["Close"].values
bar_returns = np.diff(close_arr) / close_arr[:-1]  # len = N-1

# Build bull_aligned mask (same indices as oos_df, but we need bar-by-bar)
# We computed bull_aligned using oos_indices; rebuild as array
bull_mask = np.zeros(len(oos_indices), dtype=bool)
for k in range(len(oos_indices)):
    if not np.isnan(sma200[k]) and not np.isnan(sma100[k]):
        if prices[k] > sma50[k] > sma100[k] > sma200[k]:
            bull_mask[k] = True

# P&L: apply 0.9x position on bars where bull_aligned (use bar k return = close[k+1]/close[k]-1)
capital = 10_000.0
capital_arr = [capital]
position_taken = 0
for k in range(len(bar_returns)):
    if bull_mask[k]:
        pnl = capital * 0.9 * bar_returns[k]
        position_taken += 1
    else:
        pnl = 0.0
    capital += pnl
    capital_arr.append(capital)

final_capital = capital_arr[-1]
total_pnl     = final_capital - 10_000.0
hold_return   = (close_arr[-1] / close_arr[0] - 1.0)
hold_pnl      = 10_000.0 * hold_return

print(f"\n[4] Simulated P&L (0.9x on bull_aligned bars, $10,000 starting):")
print(f"    Bars in OOS             : {len(bar_returns)}")
print(f"    Bars with 0.9x position : {position_taken} ({position_taken/len(bar_returns)*100:.1f}%)")
print(f"    Final Capital           : ${final_capital:,.2f}")
print(f"    Total P&L               : ${total_pnl:+,.2f} ({total_pnl/10000*100:+.2f}%)")
print(f"    Buy-and-hold P&L        : ${hold_pnl:+,.2f} ({hold_return*100:+.2f}%)")
print(f"    Alpha over B&H          : ${total_pnl - hold_pnl:+,.2f}")

# Annualized
n_years = len(bar_returns) / 252
cagr = (final_capital / 10_000) ** (1/n_years) - 1.0
print(f"    CAGR (strategy)         : {cagr*100:.2f}%")
bah_cagr = (1 + hold_return) ** (1/n_years) - 1.0
print(f"    CAGR (buy-and-hold)     : {bah_cagr*100:.2f}%")

print("\nDone.")
