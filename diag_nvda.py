import sys, os
import numpy as np
import pandas as pd

NORGATE_DIR = os.environ.get("NORGATE_ROOT", r"D:\Experiments\norgate_data")
ACTION_TARGETS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.25, 1.5]
N_SELECTED = 50

nvda_path = os.path.join(NORGATE_DIR, "US_Equities", "NVDA.parquet")
df = pd.read_parquet(nvda_path)
col_map = {}
for c in df.columns:
    cl = c.lower().strip()
    if cl == "open":   col_map[c] = "Open"
    elif cl == "high": col_map[c] = "High"
    elif cl == "low":  col_map[c] = "Low"
    elif "close" in cl and "unadj" not in cl.replace(" ",""): col_map[c] = "Close"
    elif cl in ("volume","vol"): col_map[c] = "Volume"
df = df.rename(columns=col_map).dropna(subset=["Close"])
print("NVDA full history: %s to %s (%d bars)" % (df.index[0].date(), df.index[-1].date(), len(df)))

OOS_START = pd.Timestamp("2023-07-01")
OOS_END   = pd.Timestamp("2026-02-28")
oos_mask  = (df.index >= OOS_START) & (df.index <= OOS_END)
oos_idx   = list(__import__("numpy").where(oos_mask)[0])
df_oos    = df.iloc[oos_idx]
print("NVDA OOS: %s to %s (%d bars)" % (df_oos.index[0].date(), df_oos.index[-1].date(), len(df_oos)))

closes_all = df["Close"].values

def compute_v8_position(idx_in_full, rank_idx=0, n_selected=N_SELECTED):
    sma_closes = closes_all[:idx_in_full+1]
    n = len(sma_closes)
    current_price = float(sma_closes[-1])
    if n < 50:
        return 0.0, 0.0, 1.0, 1.0, "warmup"
    sma50  = float(np.mean(sma_closes[-50:]))
    sma100 = float(np.mean(sma_closes[-100:])) if n >= 100 else None
    sma200 = float(np.mean(sma_closes[-200:])) if n >= 200 else None
    rank_mult = 0.50 + 1.00 * (1.0 - rank_idx / n_selected)
    trend_boost = 1.0
    if sma200 is not None:
        dist_pct = (current_price / sma200) - 1.0
        trend_boost = 1.0 + min(max(dist_pct - 0.10, 0.0) * 1.5, 0.60)
        if current_price > sma200:
            if sma100 is not None and current_price > sma50 and sma50 > sma100 and sma100 > sma200:
                base_pos = 0.90; state = "bull_aligned"
            elif sma100 is not None and sma50 > sma100:
                base_pos = 0.80; state = "golden_cross"
            else:
                base_pos = 0.50; state = "above_200"
        else:
            base_pos = 0.0; trend_boost = 1.0; state = "below_200"
    elif sma100 is not None:
        trend_boost = 1.0
        if current_price > sma100:
            base_pos = 0.25; state = "above_100"
        else:
            base_pos = 0.0; state = "below_100"
    else:
        trend_boost = 1.0
        if current_price > sma50:
            base_pos = 0.25; state = "above_50"
        else:
            base_pos = 0.0; state = "below_50"
    raw = base_pos * trend_boost * rank_mult
    if (sma200 is not None and current_price > sma50 and sma100 is not None
            and sma50 > sma100 and sma100 > sma200 and n >= 64 and rank_mult >= 1.0):
        m3_return = float(sma_closes[-1] / sma_closes[-64]) - 1.0
        if m3_return > 0.20:
            para_mult = 1.0 + min((m3_return - 0.20) * 1.5, 0.50)
            raw = raw * para_mult
            state = state + ("(para*%.2f)" % para_mult)
    dist_pct_cap = 0.0
    if sma200 is not None and sma200 > 0:
        dist_pct_cap = (current_price / sma200) - 1.0
    dynamic_cap = 1.50 + min(max(dist_pct_cap - 0.50, 0.0) * 1.0, 1.50)
    raw = min(raw, dynamic_cap)
    return raw, base_pos, trend_boost, rank_mult, state

def discretize(pos, targets=ACTION_TARGETS):
    return min(targets, key=lambda t: abs(t - pos))

rows = []
for idx in oos_idx:
    raw, base, boost, rmult, state = compute_v8_position(idx, rank_idx=0)
    disc = discretize(raw)
    rows.append({"date": df.index[idx], "close": closes_all[idx], "raw_pos": raw,
                 "disc_pos": disc, "base_pos": base, "trend_boost": boost, "rank_mult": rmult, "state": state})

res = pd.DataFrame(rows).set_index("date")
print("Computed %d bars of v8.0 positions for NVDA OOS" % len(res))

print("\n[1] Position distribution:")
from collections import Counter
counts = Counter(res["disc_pos"].tolist())
for pos in sorted(counts.keys(), reverse=True):
    cnt = counts[pos]
    print("    %+.2f  : %4d bars  (%.1f%%)" % (pos, cnt, cnt/len(res)*100))

disc_arr = res["disc_pos"].values
n_125 = int(sum(1 for x in disc_arr if x == 1.25))
n_150 = int(sum(1 for x in disc_arr if x == 1.5))
n_below_100 = int(sum(1 for x in disc_arr if x < 1.0))
transitions_125_150 = sum(1 for i in range(1,len(disc_arr))
    if (disc_arr[i-1]==1.25 and disc_arr[i]==1.5) or (disc_arr[i-1]==1.5 and disc_arr[i]==1.25))

print("\n[2] Cycling analysis:")
print("    Bars at 1.50        : %4d (%.1f%%)" % (n_150, n_150/len(disc_arr)*100))
print("    Bars at 1.25        : %4d (%.1f%%)" % (n_125, n_125/len(disc_arr)*100))
print("    Bars below 1.00     : %4d (%.1f%%)" % (n_below_100, n_below_100/len(disc_arr)*100))
print("    1.25<->1.50 transitions: %d" % transitions_125_150)

close_arr = res["close"].values
bar_ret   = [close_arr[k+1]/close_arr[k] - 1.0 for k in range(len(close_arr)-1)]
cap_disc = 10000.0; cap_150  = 10000.0
for k in range(len(bar_ret)):
    cap_disc += cap_disc * disc_arr[k] * bar_ret[k]
    cap_150  += cap_150 * 1.5 * bar_ret[k]

n_years = len(bar_ret) / 252.0
cagr_disc = (cap_disc / 10000.0) ** (1.0/n_years) - 1.0
cagr_150  = (cap_150  / 10000.0) ** (1.0/n_years) - 1.0
print("\n[3] Lost return analysis (on $10,000):")
print("    Always 1.50x final capital  : $%12.2f  CAGR=%.2f%%" % (cap_150, cagr_150*100))
print("    Discretized pos final       : $%12.2f  CAGR=%.2f%%" % (cap_disc, cagr_disc*100))
print("    Lost dollar P&L             : $%+12.2f" % (cap_150 - cap_disc))
print("    Performance gap             : %.2f%%" % ((cap_150 / cap_disc - 1.0) * 100))

raw_arr = res["raw_pos"].values
dates_arr = list(res.index)
print("\n[4] Top 20 bars by raw_pos where disc_pos is 1.25 or 1.0:")
print("    %-12s %8s %-27s %8s %9s %6s %6s %6s" % ("Date","Close","State","raw_pos","disc_pos","base","boost","rank"))
print("    " + "-"*90)
sample_rows = [(i, dates_arr[i]) for i in range(len(disc_arr)) if disc_arr[i] in (1.0, 1.25)]
sample_rows.sort(key=lambda x: -raw_arr[x[0]])
for row_i, dt in sample_rows[:20]:
    r = rows[row_i]
    print("    %-12s %8.2f %-27s %8.4f %9.2f %6.2f %6.3f %6.3f" % (
        str(dt.date()), r["close"], r["state"][:27], r["raw_pos"], r["disc_pos"],
        r["base_pos"], r["trend_boost"], r["rank_mult"]))

raw_at_125 = [raw_arr[i] for i in range(len(disc_arr)) if disc_arr[i] == 1.25]
raw_at_100 = [raw_arr[i] for i in range(len(disc_arr)) if disc_arr[i] == 1.0]
print("\n[5] Midpoint analysis:")
print("    Midpoint(1.25, 1.5) = 1.375 -- raw must be >= 1.375 to snap to 1.5")
print("    Midpoint(1.0, 1.25) = 1.125 -- raw must be >= 1.125 to snap to 1.25")
if raw_at_125:
    print("    disc=1.25: raw range [%.4f, %.4f], mean=%.4f" % (min(raw_at_125), max(raw_at_125), sum(raw_at_125)/len(raw_at_125)))
if raw_at_100:
    print("    disc=1.0:  raw range [%.4f, %.4f], mean=%.4f" % (min(raw_at_100), max(raw_at_100), sum(raw_at_100)/len(raw_at_100)))
print("    Bars at disc=1.25 with raw_pos >= 1.30: %d" % sum(1 for x in raw_at_125 if x >= 1.30))
print("Done.")
