import os, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
warnings.filterwarnings("ignore")

DATA_DIR = Path(r"D:/Experiments/norgate_data/US_Equities")
START_DATE = "2010-03-11"
TC = 0.0005
INIT_CAPITAL = 100_000.0

LETF_CANDIDATES = [
    "TECL", "SOXL", "TQQQ", "UPRO", "FAS", "TNA", "UDOW",
    "ERX", "LABU", "NAIL", "CURE", "DPST", "HIBL", "WANT",
    "RETL", "NUGT", "JNUG", "MIDU",
]
BENCHMARKS = ["SPY", "QQQ"]

print("=" * 70)
print("STEP 1: FILE DISCOVERY")
print("=" * 70)
available = {}
missing = []
for sym in LETF_CANDIDATES + BENCHMARKS:
    fp = DATA_DIR / f"{sym}.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        available[sym] = df
        print(f"  FOUND  {sym:6s}  {df.index.min().date()} -> {df.index.max().date()}  ({len(df):,} rows)")
    else:
        missing.append(sym)
        print(f"  MISSING {sym}")
print("")
print(f"Available: {list(available.keys())}")
print(f"Missing  : {missing}")

print(chr(10) + "=" * 70)
print("STEP 2+3: BUILDING UNIFIED PRICE MATRIX")
print("=" * 70)
letf_syms = [s for s in LETF_CANDIDATES if s in available]
bench_syms = [s for s in BENCHMARKS if s in available]
spy_dates = available["SPY"].index
unified_dates = spy_dates[spy_dates >= START_DATE]
price_dict = {}
for sym, df in available.items():
    s = df["Close"].reindex(unified_dates)
    s = s.ffill()
    price_dict[sym] = s
prices = pd.DataFrame(price_dict, index=unified_dates)
print(f"Unified date range: {unified_dates[0].date()} -> {unified_dates[-1].date()}")
print(f"Trading days: {len(unified_dates):,}")
print(f"ETFs in pool: {letf_syms}")
print("")
print("First non-NaN date per ETF:")
for sym in letf_syms:
    first = prices[sym].first_valid_index()
    label = str(first.date()) if first is not None else "ALL NaN"
    print(f"  {sym:6s}: {label}")
vol_dict = {}
for sym in letf_syms:
    df = available[sym]
    dv = (df["Close"] * df["Volume"]).reindex(unified_dates).ffill()
    vol_dict[sym] = dv
dollar_vol = pd.DataFrame(vol_dict, index=unified_dates)

print(chr(10) + "=" * 70)
print("STEP 4: RUNNING STRATEGY ENGINE")
print("=" * 70)

spy_close = prices["SPY"]
spy_sma200 = spy_close.rolling(200, min_periods=150).mean()

def compute_momentum(prices_df, syms, date_idx):
    scores = {}
    for sym in syms:
        p = prices_df[sym]
        c = p.iloc[date_idx]
        if pd.isna(c):
            scores[sym] = np.nan
            continue
        def ret(lookback, _p=p, _c=c, _d=date_idx):
            if _d < lookback: return np.nan
            past = _p.iloc[_d - lookback]
            if pd.isna(past) or past <= 0: return np.nan
            return (_c - past) / past
        r1 = ret(21)
        r3 = ret(63)
        r6 = ret(126)
        if any(pd.isna(x) for x in [r1, r3, r6]):
            scores[sym] = np.nan
        else:
            scores[sym] = r1 * 0.40 + r3 * 0.35 + r6 * 0.25
    return pd.Series(scores)

def backtest(rebal_freq, top_n, bear_gate, mom_weights, liq_filter, strategy_name, use_only_1m=False):
    n = len(unified_dates)
    port_val = np.zeros(n)
    port_val[0] = INIT_CAPITAL
    daily_rets = np.zeros(n)
    holdings_log = []
    current_hold = []
    last_rebal_idx = -rebal_freq
    for i in range(1, n):
        dt = unified_dates[i]
        if (i - last_rebal_idx) >= rebal_freq or i == 1:
            signal_idx = i - 1
            spy_val = spy_close.iloc[signal_idx]
            spy_sma200_val = spy_sma200.iloc[signal_idx]
            spy_bearish = (not pd.isna(spy_sma200_val)) and (spy_val < spy_sma200_val)
            candidates = []
            for sym in letf_syms:
                p_val = prices[sym].iloc[signal_idx]
                if pd.isna(p_val): continue
                if liq_filter:
                    avg_dv = dollar_vol[sym].iloc[max(0, signal_idx-20):signal_idx+1].mean()
                    if avg_dv < 50_000_000: continue
                candidates.append(sym)
            if use_only_1m:
                scores_raw = {}
                for sym in candidates:
                    p = prices[sym]
                    c = p.iloc[signal_idx]
                    past = p.iloc[signal_idx - 21] if signal_idx >= 21 else np.nan
                    if pd.isna(c) or pd.isna(past) or past <= 0: scores_raw[sym] = np.nan
                    else: scores_raw[sym] = (c - past) / past
                scores = pd.Series(scores_raw).dropna()
            else:
                scores = compute_momentum(prices, candidates, signal_idx).dropna()
            if len(scores) == 0:
                new_hold = []
            else:
                ranked = scores.sort_values(ascending=False)
                selected = list(ranked.head(top_n).index)
                if bear_gate:
                    safe = []
                    for sym in selected:
                        p = prices[sym]
                        sma50_win = p.iloc[max(0, signal_idx-50):signal_idx+1]
                        sma50 = sma50_win.mean()
                        own_bearish = p.iloc[signal_idx] < sma50
                        if own_bearish and spy_bearish: pass
                        else: safe.append(sym)
                    selected = safe
                new_hold = [(sym, 1.0 / len(selected)) for sym in selected] if selected else []
            prev_syms = set(s for s, _ in current_hold)
            next_syms = set(s for s, _ in new_hold)
            changed = prev_syms.symmetric_difference(next_syms)
            tc_cost = len(changed) * TC * port_val[i-1]
            port_val[i-1] = max(port_val[i-1] - tc_cost, 0)
            current_hold = new_hold
            last_rebal_idx = i
            holdings_log.append((dt, [s for s, _ in current_hold]))
        if len(current_hold) == 0:
            daily_rets[i] = 0.0
            port_val[i] = port_val[i-1]
        else:
            day_ret = 0.0
            for sym, wt in current_hold:
                p_today = prices[sym].iloc[i]
                p_prev  = prices[sym].iloc[i-1]
                if pd.isna(p_today) or pd.isna(p_prev) or p_prev <= 0: r = 0.0
                else: r = (p_today - p_prev) / p_prev
                day_ret += wt * r
            daily_rets[i] = day_ret
            port_val[i] = port_val[i-1] * (1 + day_ret)
    return pd.Series(daily_rets[1:], index=unified_dates[1:]), holdings_log, port_val[-1]

def bnh_returns(sym):
    p = prices[sym]
    return p.pct_change().fillna(0)

def compute_metrics(daily_rets, final_val=None):
    r = daily_rets.copy()
    n_years = len(r) / 252
    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    cagr = cum.iloc[-1] ** (1 / n_years) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol > 0 else 0
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate = (r > 0).mean()
    if final_val is None: final_val = INIT_CAPITAL * (1 + total_ret)
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "Calmar": calmar,
            "FinalVal": final_val, "WinRate": win_rate, "TotalRet": total_ret}

print("Running R0: TQQQ Buy & Hold (baseline)...")
r0_rets = bnh_returns("TQQQ")
r0_final = INIT_CAPITAL * (1 + r0_rets).cumprod().iloc[-1]
print("Running R1: Top-1 rotation, no bear gate, monthly...")
r1_rets, r1_log, r1_final = backtest(21, 1, False, None, False, "R1")
print("Running R2: Top-1 rotation, bear gate, monthly...")
r2_rets, r2_log, r2_final = backtest(21, 1, True, None, False, "R2")
print("Running R3: Top-2 rotation, bear gate, monthly...")
r3_rets, r3_log, r3_final = backtest(21, 2, True, None, False, "R3")
print("Running R4: Top-1 rotation, bear gate, WEEKLY (5-day)...")
r4_rets, r4_log, r4_final = backtest(5, 1, True, None, False, "R4")
print("Running R5: Top-1 rotation, bear gate, 1m-only momentum...")
r5_rets, r5_log, r5_final = backtest(21, 1, True, None, False, "R5", use_only_1m=True)
print("Running R6: Top-1 rotation, bear gate, liquidity filter...")
r6_rets, r6_log, r6_final = backtest(21, 1, True, None, True, "R6")
print("Computing SPY / QQQ benchmarks...")
spy_rets = bnh_returns("SPY")
qqq_rets = bnh_returns("QQQ")

print(chr(10) + "=" * 70)
print("STEP 5: PERFORMANCE METRICS")
print("=" * 70)

strategies = {
    "R0_TQQQ_BnH": (r0_rets, r0_final),
    "R1_Top1_NoGate": (r1_rets, r1_final),
    "R2_Top1_BearGate": (r2_rets, r2_final),
    "R3_Top2_BearGate": (r3_rets, r3_final),
    "R4_Top1_Weekly": (r4_rets, r4_final),
    "R5_Top1_1mMom": (r5_rets, r5_final),
    "R6_Top1_LiqFilter": (r6_rets, r6_final),
    "SPY_BnH": (spy_rets, None),
    "QQQ_BnH": (qqq_rets, None),
}
metrics_rows = {}
for name, (rets, fv) in strategies.items():
    m = compute_metrics(rets, fv)
    metrics_rows[name] = m
metrics_df = pd.DataFrame(metrics_rows).T
cols = ["CAGR", "Sharpe", "MaxDD", "Calmar", "FinalVal", "WinRate"]
metrics_df = metrics_df[cols]

print(chr(10) + "=" * 70)
print("STEP 5: PERFORMANCE METRICS")
print("=" * 70)

strategies = {
    "R0_TQQQ_BnH": (r0_rets, r0_final),
    "R1_Top1_NoGate": (r1_rets, r1_final),
    "R2_Top1_BearGate": (r2_rets, r2_final),
    "R3_Top2_BearGate": (r3_rets, r3_final),
    "R4_Top1_Weekly": (r4_rets, r4_final),
    "R5_Top1_1mMom": (r5_rets, r5_final),
    "R6_Top1_LiqFilter": (r6_rets, r6_final),
    "SPY_BnH": (spy_rets, None),
    "QQQ_BnH": (qqq_rets, None),
}
metrics_rows = {}
for name, (rets, fv) in strategies.items():
    metrics_rows[name] = compute_metrics(rets, fv)
metrics_df = pd.DataFrame(metrics_rows).T
metrics_df = metrics_df[["CAGR","Sharpe","MaxDD","Calmar","FinalVal","WinRate"]]

hdr = ("Strategy".ljust(20) + " " + "CAGR".rjust(8) + " " + "Sharpe".rjust(8) +
        " " + "MaxDD".rjust(9) + " " + "Calmar".rjust(8) +
        " " + "Dollar_Final".rjust(14) + " " + "WinRate".rjust(8))
print(hdr)
print("-" * 82)
for name, row in metrics_df.iterrows():
    cagr_v = row["CAGR"]
    sharpe_v = row["Sharpe"]
    maxdd_v = row["MaxDD"]
    calmar_v = row["Calmar"]
    finalval_v = row["FinalVal"]
    winrate_v = row["WinRate"]
    fv_str = ("$" + f"{finalval_v:>13,.0f}") if not pd.isna(finalval_v) else "      N/A"
    parts = [name.ljust(20), f"{cagr_v:>7.1%}", f"{sharpe_v:>8.2f}",
             f"{maxdd_v:>8.1%}", f"{calmar_v:>8.2f}", fv_str, f"{winrate_v:>7.1%}"]
    print(" ".join(parts))

print(chr(10) + "=" * 70)
print("STEP 6: ANNUAL HOLDINGS (MOST COMMON ETF PER YEAR)")
print("=" * 70)

def build_annual_holdings(log):
    year_holdings = {}
    for dt, syms in log:
        y = dt.year
        year_holdings.setdefault(y, [])
        if syms:
            year_holdings[y].extend(syms)
        else:
            year_holdings[y].append("CASH")
    result = {}
    for y, lst in year_holdings.items():
        cnt = Counter(lst)
        top = cnt.most_common(3)
        result[y] = ", ".join(f"{s}({c})" for s, c in top)
    return result

h1 = build_annual_holdings(r1_log)
h2 = build_annual_holdings(r2_log)
h3 = build_annual_holdings(r3_log)
h4 = build_annual_holdings(r4_log)
h5 = build_annual_holdings(r5_log)
h6 = build_annual_holdings(r6_log)

years = sorted(set(list(h1.keys()) + list(h2.keys())))
hdr2 = "Year".ljust(6) + " " + "R1 (No Gate)".ljust(28) + " " + "R2 (Bear Gate)".ljust(28) + " " + "R3 (Top2)".ljust(28)
print(hdr2)
print("-" * 92)
for y in years:
    v1 = h1.get(y, "")
    v2 = h2.get(y, "")
    v3 = h3.get(y, "")
    print(str(y).ljust(6) + " " + v1.ljust(28) + " " + v2.ljust(28) + " " + v3.ljust(28))

print("")
hdr3 = "Year".ljust(6) + " " + "R4 (Weekly)".ljust(28) + " " + "R5 (1m Mom)".ljust(28) + " " + "R6 (Liq Filter)".ljust(28)
print(hdr3)
print("-" * 92)
for y in years:
    v4 = h4.get(y, "")
    v5 = h5.get(y, "")
    v6 = h6.get(y, "")
    print(str(y).ljust(6) + " " + v4.ljust(28) + " " + v5.ljust(28) + " " + v6.ljust(28))

print(chr(10) + "=" * 70)
print("STEP 7: YEAR-BY-YEAR RETURNS")
print("=" * 70)

all_rets_dict = {
    "TQQQ_BnH": r0_rets,
    "R1": r1_rets,
    "R2": r2_rets,
    "R3": r3_rets,
    "R4": r4_rets,
    "R5": r5_rets,
    "R6": r6_rets,
    "SPY": spy_rets,
    "QQQ": qqq_rets,
}

def annual_returns(rets):
    cum = (1 + rets).cumprod()
    ann = cum.resample("YE").last()
    first_yr = ann.iloc[0] - 1.0
    ann_ret = ann.pct_change()
    result = {}
    for i, (dt, v) in enumerate(ann_ret.items()):
        result[dt.year] = v if i > 0 else first_yr
    return result

ann_data = {name: annual_returns(r) for name, r in all_rets_dict.items()}
all_years = sorted(set(y for d in ann_data.values() for y in d.keys()))

col_names = list(all_rets_dict.keys())
hdr_yr = "Year".ljust(6) + " " + " ".join(n.rjust(12) for n in col_names)
print("")
print(hdr_yr)
print("-" * len(hdr_yr))
for y in all_years:
    row_parts = [str(y).ljust(6)]
    for name in col_names:
        v = ann_data[name].get(y, float("nan"))
        if pd.isna(v):
            row_parts.append("N/A".rjust(12))
        else:
            row_parts.append(f"{v:>11.1%} ")
    print(" ".join(row_parts))

print(chr(10) + "=" * 70)
print("STEP 8: BEAR MARKET PROTECTION ANALYSIS")
print("=" * 70)

def period_return(rets, start, end):
    mask = (rets.index >= start) & (rets.index <= end)
    sub = rets[mask]
    if len(sub) == 0: return float("nan")
    return (1 + sub).prod() - 1

def period_holdings(log, start, end):
    s_dt, e_dt = pd.Timestamp(start), pd.Timestamp(end)
    h = []
    for dt, syms in log:
        if s_dt <= dt <= e_dt:
            h.extend(syms if syms else ["CASH"])
    if not h: return "N/A"
    cnt = Counter(h)
    return ", ".join(f"{sym}({c})" for sym, c in cnt.most_common(3))

periods = [
    ("2022 Full Year",     "2022-01-03", "2022-12-30"),
    ("2020 COVID Crash",   "2020-02-19", "2020-03-23"),
    ("2018 Q4",            "2018-10-01", "2018-12-31"),
]

logs = {"R1": r1_log, "R2": r2_log, "R3": r3_log,
        "R4": r4_log, "R5": r5_log, "R6": r6_log}
ret_map = {"R1": r1_rets, "R2": r2_rets, "R3": r3_rets,
           "R4": r4_rets, "R5": r5_rets, "R6": r6_rets,
           "TQQQ": r0_rets, "SPY": spy_rets, "QQQ": qqq_rets}

for period_name, start, end in periods:
    print(f"")
    print(f"--- {period_name} ({start} to {end}) ---")
    print("  " + "Strategy".ljust(18) + " " + "Return".rjust(8) + "   Holdings")
    print("  " + "-" * 60)
    for name, rets in ret_map.items():
        pr = period_return(rets, start, end)
        hold_str = period_holdings(logs[name], start, end) if name in logs else "B&H"
        pr_fmt = f"{pr:>7.1%}" if not pd.isna(pr) else "    N/A"
        print(f"  {name.ljust(18)} {pr_fmt}   {hold_str}")

print(chr(10) + "=" * 70)
print("STEP 9: FINAL SUMMARY TABLE")
print("=" * 70)

print("")
print("Strategy Descriptions:")
print("  R0  TQQQ Buy & Hold           - Baseline, pure 3x Nasdaq, no rotation")
print("  R1  Top-1, No Bear Gate       - Best momentum ETF monthly, no protection")
print("  R2  Top-1, Bear Gate          - Best momentum ETF monthly + cash gate")
print("  R3  Top-2, Bear Gate          - Equal-weight top-2 monthly + cash gate")
print("  R4  Top-1, Weekly Rebal       - Best momentum ETF weekly + cash gate")
print("  R5  Top-1, 1m-Mom Only        - 1-month momentum only (simpler signal)")
print("  R6  Top-1, Liq Filter         - Exclude illiquid ETFs (<0M avg dv)")
print("  SPY Buy & Hold                - S&P 500 benchmark")
print("  QQQ Buy & Hold                - Nasdaq 100 benchmark")
print("")

summary_hdr = ("Strategy".ljust(22) + " " + "CAGR".rjust(8) + " " + "Sharpe".rjust(8) +
               " " + "MaxDD".rjust(9) + " " + "Calmar".rjust(8) + " " +
               (chr(36)+"100K -> "+chr(36)+"?").rjust(16) + " " + "WinRate".rjust(8))
print("=" * 88)

order = ["R0_TQQQ_BnH","R1_Top1_NoGate","R2_Top1_BearGate","R3_Top2_BearGate",
         "R4_Top1_Weekly","R5_Top1_1mMom","R6_Top1_LiqFilter","SPY_BnH","QQQ_BnH"]
for name in order:
    row = metrics_df.loc[name]
    cagr_v = row["CAGR"]
    sharpe_v = row["Sharpe"]
    maxdd_v = row["MaxDD"]
    calmar_v = row["Calmar"]
    finalval_v = row["FinalVal"]
    winrate_v = row["WinRate"]
    fv_str = ("$" + f"{finalval_v:>15,.0f}") if not pd.isna(finalval_v) else "            N/A"
    parts = [name.ljust(22), f"{cagr_v:>7.1%}", f"{sharpe_v:>8.2f}",
             f"{maxdd_v:>8.1%}", f"{calmar_v:>8.2f}", fv_str.rjust(16), f"{winrate_v:>7.1%}"]
    print(" ".join(parts))

print(chr(10) + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

rotation_order = ["R1_Top1_NoGate","R2_Top1_BearGate",
                  "R3_Top2_BearGate","R4_Top1_Weekly","R5_Top1_1mMom","R6_Top1_LiqFilter"]
sub_df = metrics_df.loc[rotation_order]
best_cagr_name = sub_df["CAGR"].idxmax()
best_sharpe_name = sub_df["Sharpe"].idxmax()
best_calmar_name = sub_df["Calmar"].idxmax()
print(f"  Best CAGR   : {best_cagr_name} ({sub_df.loc[best_cagr_name, chr(67)+chr(65)+chr(71)+chr(82)]:.1%})")
print(f"  Best Sharpe : {best_sharpe_name} ({sub_df.loc[best_sharpe_name, chr(83)+chr(104)+chr(97)+chr(114)+chr(112)+chr(101)]:.2f})")
print(f"  Best Calmar : {best_calmar_name} ({sub_df.loc[best_calmar_name, chr(67)+chr(97)+chr(108)+chr(109)+chr(97)+chr(114)]:.2f})")
r2_cagr = metrics_df.loc["R2_Top1_BearGate", "CAGR"]
r0_cagr = metrics_df.loc["R0_TQQQ_BnH", "CAGR"]
r2_dd   = metrics_df.loc["R2_Top1_BearGate", "MaxDD"]
r0_dd   = metrics_df.loc["R0_TQQQ_BnH", "MaxDD"]
diff_cagr = r2_cagr - r0_cagr
diff_dd   = r2_dd - r0_dd
print(f"  R2 vs TQQQ  : CAGR {diff_cagr:+.1%}, MaxDD {diff_dd:+.1%}")
print("")
print("Done.")
