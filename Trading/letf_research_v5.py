import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "d:/Experiments/norgate_data/US_Equities/"
START_FULL  = "2010-02-11"
START_TECL  = "2008-12-17"
START_TQQQ  = "2010-02-11"
CAPITAL     = 100_000
TC          = 0.0005
SPY_SMA_PERIOD        = 200
SPY_BELOW_SMA_DAYS    = 15
QQQ_RETURN_21D_THRESH = -0.12
QQQ_RETURN_63D_THRESH = -0.05
SPY_SMA_EXIT          = 50
QQQ_5D_EXIT           = 0.05


def load(ticker):
    path = DATA_DIR + ticker + ".parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df["Close"].rename(ticker)

print("Loading data...")
spy  = load("SPY")
qqq  = load("QQQ")
tqqq = load("TQQQ")
sqqq = load("SQQQ")
tecl = load("TECL")
tecs = load("TECS")
for tk, s in [("SPY",spy),("QQQ",qqq),("TQQQ",tqqq),("SQQQ",sqqq),("TECL",tecl),("TECS",tecs)]:
    print(f"  {tk:<5}: {s.index[0].date()} -> {s.index[-1].date()}  ({len(s)} trading days)")

all_idx = spy.index.union(qqq.index).union(tqqq.index).union(sqqq.index).union(tecl.index).union(tecs.index)

def align(series, idx):
    return series.reindex(idx).ffill()

spy_a  = align(spy,  all_idx)
qqq_a  = align(qqq,  all_idx)
tqqq_a = align(tqqq, all_idx)
sqqq_a = align(sqqq, all_idx)
tecl_a = align(tecl, all_idx)
tecs_a = align(tecs, all_idx)

spy_r  = spy_a.pct_change()
qqq_r  = qqq_a.pct_change()
tqqq_r = tqqq_a.pct_change()
sqqq_r = sqqq_a.pct_change()
tecl_r = tecl_a.pct_change()
tecs_r = tecs_a.pct_change()

print("\nBuilding TD bear signals...")
spy_sma200       = spy_a.rolling(SPY_SMA_PERIOD).mean()
spy_below_sma200 = (spy_a < spy_sma200).astype(int)

streak_arr = spy_below_sma200.values.copy().astype(float)
for i in range(1, len(streak_arr)):
    if spy_below_sma200.iloc[i] == 1:
        streak_arr[i] = streak_arr[i-1] + 1
    else:
        streak_arr[i] = 0.0
streak = pd.Series(streak_arr, index=all_idx)

qqq_21d_ret = qqq_a.pct_change(21)
qqq_63d_ret = qqq_a.pct_change(63)
qqq_5d_ret  = qqq_a.pct_change(5)

bear_entry_raw = (
    (streak >= SPY_BELOW_SMA_DAYS) &
    (qqq_21d_ret < QQQ_RETURN_21D_THRESH) &
    (qqq_63d_ret < QQQ_RETURN_63D_THRESH)
)

spy_sma50     = spy_a.rolling(SPY_SMA_EXIT).mean()
bear_exit_raw = ((spy_a > spy_sma50) | (qqq_5d_ret > QQQ_5D_EXIT))

in_bear_arr = np.zeros(len(all_idx), dtype=float)
state = 0
ent   = bear_entry_raw.values
ext   = bear_exit_raw.values
for i in range(len(all_idx)):
    if state == 0:
        if ent[i]:
            state = 1
    else:
        if ext[i]:
            state = 0
    in_bear_arr[i] = state
in_bear     = pd.Series(in_bear_arr, index=all_idx)
in_bear_lag = in_bear.shift(1).fillna(0)

print(f"  Bear days (total): {int(in_bear.sum())}")
print(f"  Bear days from {START_FULL}: {int(in_bear_lag[START_FULL:].sum())}")

qqq_sma200    = qqq_a.rolling(200).mean()
qqq_sma100    = qqq_a.rolling(100).mean()
bull_sma200   = (qqq_a >= qqq_sma200).astype(float).shift(1).fillna(0)
bull_dual_sma = ((qqq_a >= qqq_sma200) & (qqq_a >= qqq_sma100)).astype(float).shift(1).fillna(0)

tecl_60d_high = tecl_a.rolling(60).max()
stop_breach   = (tecl_a < 0.75 * tecl_60d_high)
tecl_sma40    = tecl_a.rolling(40).mean()
recovered_sig = (tecl_a > tecl_sma40)

stop_arr = np.zeros(len(all_idx), dtype=float)
state = 0
sb = stop_breach.values
rc = recovered_sig.values
for i in range(len(all_idx)):
    if state == 0:
        if sb[i]:
            state = 1
    else:
        if rc[i]:
            state = 0
    stop_arr[i] = state
stop_state    = pd.Series(stop_arr, index=all_idx)
in_stop_lag   = stop_state.shift(1).fillna(0)
bull_trailing = 1 - in_stop_lag


def simple_backtest(weights_series, daily_ret_series, start_date, capital=CAPITAL, tc=TC):
    combined = pd.concat([weights_series.rename("w"), daily_ret_series.rename("r")], axis=1).dropna()
    combined = combined[combined.index >= pd.Timestamp(start_date)]
    if len(combined) < 2:
        return pd.Series(dtype=float)
    port    = np.empty(len(combined))
    port[0] = float(capital)
    prev_w  = 0.0
    w_arr   = combined["w"].values
    r_arr   = combined["r"].values
    for i in range(1, len(combined)):
        w        = max(0.0, min(1.0, float(w_arr[i-1])))
        turnover = abs(w - prev_w)
        pv       = port[i-1] * (1 - turnover * tc)
        port[i]  = pv * (1 + w * float(r_arr[i]))
        prev_w   = w
    return pd.Series(port, index=combined.index)


def multi_backtest(asset_dict, start_date, capital=CAPITAL, tc=TC):
    names = list(asset_dict.keys())
    dfs = []
    for n in names:
        w, r = asset_dict[n]
        dfs.append(w.rename(f"w_{n}"))
        dfs.append(r.rename(f"r_{n}"))
    combined = pd.concat(dfs, axis=1).dropna()
    combined = combined[combined.index >= pd.Timestamp(start_date)]
    if len(combined) < 2:
        return pd.Series(dtype=float)
    port    = np.empty(len(combined))
    port[0] = float(capital)
    prev_w  = {n: 0.0 for n in names}
    for i in range(1, len(combined)):
        row_prev = combined.iloc[i-1]
        row_cur  = combined.iloc[i]
        raw      = {n: max(0.0, float(row_prev[f"w_{n}"])) for n in names}
        total    = sum(raw.values())
        if total > 1e-9:
            tgt = {n: raw[n]/total for n in names}
        else:
            tgt = {n: 0.0 for n in names}
        turnover = sum(abs(tgt[n] - prev_w[n]) for n in names)
        pv       = port[i-1] * (1 - turnover * tc)
        gain     = sum(tgt[n] * float(row_cur[f"r_{n}"]) for n in names)
        port[i]  = pv * (1 + gain)
        prev_w   = tgt.copy()
    return pd.Series(port, index=combined.index)


def metrics(equity, bear_mask=None):
    eq = equity.dropna()
    if len(eq) < 2:
        return {}
    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr    = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    daily_r = eq.pct_change().dropna()
    sharpe  = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else 0
    max_dd  = (eq / eq.cummax() - 1).min()
    final_v = eq.iloc[-1]
    bear_d  = int(bear_mask.reindex(eq.index, fill_value=0).sum()) if bear_mask is not None else None
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "final": final_v, "bear_days": bear_d}


def annual_returns(equity):
    eq  = equity.dropna()
    yrs = {}
    for yr in sorted(eq.index.year.unique()):
        yr_data = eq[eq.index.year == yr]
        prev    = eq[eq.index < pd.Timestamp(f"{yr}-01-01")]
        if len(prev) > 0:
            yrs[yr] = yr_data.iloc[-1] / prev.iloc[-1] - 1
        elif len(yr_data) >= 2:
            yrs[yr] = yr_data.iloc[-1] / yr_data.iloc[0] - 1
    return yrs


def period_return(equity, start, end):
    eq  = equity.dropna()
    pre = eq[eq.index < pd.Timestamp(start)]
    sub = eq[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    if len(sub) == 0:
        return None
    base = pre.iloc[-1] if len(pre) > 0 else sub.iloc[0]
    return sub.iloc[-1] / base - 1


print("\nRunning backtests...")
ones    = pd.Series(1.0, index=all_idx)
bull_td = 1 - in_bear_lag

results = {}
print("  B0: TQQQ B&H")
results["B0"]         = simple_backtest(ones, tqqq_r, START_TQQQ)
print("  B1: TECL B&H")
results["B1"]         = simple_backtest(ones, tecl_r, START_TECL)
print("  B0_aligned: TQQQ B&H (2010)")
results["B0_aligned"] = simple_backtest(ones, tqqq_r, START_FULL)
print("  B1_aligned: TECL B&H (2010)")
results["B1_aligned"] = simple_backtest(ones, tecl_r, START_FULL)
print("  S1: TECL + TD Cash")
results["S1"] = simple_backtest(bull_td, tecl_r, START_FULL)
print("  S2: TECL + TD TECS")
results["S2"] = multi_backtest({"TECL": (1-in_bear_lag, tecl_r), "TECS": (in_bear_lag, tecs_r)}, START_FULL)
print("  S3: TQQQ + TD Cash")
results["S3"] = simple_backtest(bull_td, tqqq_r, START_FULL)
print("  S4: TECL/TQQQ Adaptive + TD Cash")
tecl_21d      = tecl_a.pct_change(21)
tqqq_21d      = tqqq_a.pct_change(21)
tecl_wins_lag = (tecl_21d >= tqqq_21d).astype(float).shift(1).fillna(0.5)
results["S4"] = multi_backtest({
    "TECL": (bull_td * tecl_wins_lag, tecl_r),
    "TQQQ": (bull_td * (1-tecl_wins_lag), tqqq_r)
}, START_FULL)
print("  S5: TECL 0.75 + TQQQ 0.25 + TD Cash")
results["S5"] = multi_backtest({
    "TECL": (bull_td * 0.75, tecl_r),
    "TQQQ": (bull_td * 0.25, tqqq_r)
}, START_FULL)
print("  S6: TECL + QQQ SMA200 Gate")
results["S6"] = simple_backtest(bull_sma200, tecl_r, START_FULL)
print("  S7: TECL + Dual SMA Gate")
results["S7"] = simple_backtest(bull_dual_sma, tecl_r, START_FULL)
print("  S8: TECL + Trailing Stop")
results["S8"] = simple_backtest(bull_trailing, tecl_r, START_FULL)
print("  S9: TQQQ + SQQQ Bear (TD)")
results["S9"] = multi_backtest({"TQQQ": (bull_td, tqqq_r), "SQQQ": (in_bear_lag, sqqq_r)}, START_FULL)
print("  All backtests complete.")

strategy_labels = {
    "B0":         "TQQQ B&H (from 2010-02-11)",
    "B1":         "TECL B&H (from 2008-12-17)",
    "B0_aligned": "TQQQ B&H (2010, aligned)",
    "B1_aligned": "TECL B&H (2010, aligned)",
    "S1":         "S1: TECL + TD Cash",
    "S2":         "S2: TECL + TD TECS",
    "S3":         "S3: TQQQ + TD Cash",
    "S4":         "S4: TECL/TQQQ Adaptive+TD",
    "S5":         "S5: TECL75+TQQQ25+TD Cash",
    "S6":         "S6: TECL + QQQ SMA200 Gate",
    "S7":         "S7: TECL + Dual SMA Gate",
    "S8":         "S8: TECL + Trailing Stop",
    "S9":         "S9: TQQQ+SQQQ Bear (TD)",
}
ordered_keys   = ["B0","B1","B0_aligned","B1_aligned","S1","S2","S3","S4","S5","S6","S7","S8","S9"]
bear_mask_full = in_bear_lag.reindex(all_idx, fill_value=0)
mdict          = {k: metrics(results[k], bear_mask=bear_mask_full) for k in ordered_keys}

W = 42

W = 42

print()
print("="*100)
print("LETF RESEARCH V5 -- STRATEGY COMPARISON SUMMARY")
print("="*100)
col_hdr = ("%-42s | %7s | %7s | %8s | %9s | %13s" %
           ("Strategy", "CAGR", "Sharpe", "MaxDD", "Bear Days", "Final $"))
print(col_hdr)
print("-"*100)
for key in ordered_keys:
    m = mdict[key]
    if not m:
        continue
    bd_val = m.get("bear_days")
    bd = ("%9d" % bd_val) if bd_val is not None else "        -"
    lbl = strategy_labels[key]
    print("%-42s | %6.1f%% | %7.2f | %7.1f%%  | %s | $%12s" % (
        lbl, m["cagr"]*100, m["sharpe"], m["max_dd"]*100, bd,
        "{:,.0f}".format(m["final"])))
print("="*100)
print("TC=0.05%%/side | Capital=$100,000 | Data through: %s" % all_idx[-1].date())

all_years = list(range(2010,2026))
COL = 8
WTOTAL = W + 3 + len(all_years)*(COL+3) + 5
print()
print("="*WTOTAL)
print("YEAR-BY-YEAR RETURNS (2010-2025)")
print("="*WTOTAL)
yr_hdr = "%-42s |" % "Strategy"
for yr in all_years:
    yr_hdr += " %7d |" % yr
print(yr_hdr)
print("-"*WTOTAL)
for key in ordered_keys:
    ann = annual_returns(results[key])
    row = "%-42s |" % strategy_labels[key]
    for yr in all_years:
        if yr in ann:
            row += " %+6.1f%% |" % (ann[yr]*100)
        else:
            row += "   N/A  |"
    print(row)
print("="*WTOTAL)

bear_periods = [
    ("2011 correction",    "2011-05-01","2011-12-31"),
    ("2015-16 correction", "2015-07-01","2016-02-29"),
    ("2018 Q4",            "2018-10-01","2018-12-31"),
    ("2020 COVID crash",   "2020-02-19","2020-03-23"),
    ("2020 full year",     "2020-01-01","2020-12-31"),
    ("2022 bear market",   "2022-01-01","2022-12-31"),
]
PC = 20
PTOTAL = W + 3 + len(bear_periods)*(PC+3)
print()
print("="*PTOTAL)
print("PROTECTION ANALYSIS -- KEY BEAR/CORRECTION PERIODS")
print("="*PTOTAL)
ph = "%-42s |" % "Strategy"
for name,_,_ in bear_periods:
    ph += (" %-20s |" % name[:20])
print(ph)
print("-"*PTOTAL)
for key in ordered_keys:
    row = "%-42s |" % strategy_labels[key]
    for name,ps,pe in bear_periods:
        r = period_return(results[key], ps, pe)
        if r is not None:
            row += " %+19.1f%% |" % (r*100)
        else:
            row += "        N/A         |"
    print(row)
print("="*PTOTAL)

print()
print("="*72)
print("TD BEAR SIGNAL -- TRIGGER DATES AND DURATIONS")
print("="*72)
transitions = in_bear.diff().fillna(0)
entry_dates = all_idx[transitions == 1]
exit_dates  = all_idx[transitions == -1]
print("  %-14s | %-14s | %9s | %s" % ("Entry", "Exit", "Trd Days", "Year"))
print("  " + "-"*55)
for i, entry in enumerate(entry_dates):
    if i < len(exit_dates):
        exit_d = exit_dates[i]
        dur    = int(in_bear.loc[entry:exit_d].sum())
        print("  %-14s | %-14s | %9d | %s" % (str(entry.date()), str(exit_d.date()), dur, entry.year))
    else:
        dur    = int(in_bear.loc[entry:].sum())
        print("  %-14s | %-14s | %9d | %s+" % (str(entry.date()), "still active", dur, entry.year))
print("="*72)

print()
print("="*82)
print("ADDITIONAL RISK METRICS")
print("="*82)
risk_hdr = "%-42s | %7s | %8s | %8s | %9s" % ("Strategy","Calmar","Sortino","Ann Vol","Win Rate")
print(risk_hdr)
print("-"*82)
for key in ordered_keys:
    eq = results[key].dropna()
    if len(eq) < 10:
        continue
    dr      = eq.pct_change().dropna()
    vol     = dr.std() * (252**0.5)
    m       = mdict[key]
    calmar  = m["cagr"] / abs(m["max_dd"]) if m["max_dd"] != 0 else 0
    neg_r   = dr[dr < 0]
    sortino = (dr.mean()*252)/(neg_r.std()*(252**0.5)) if len(neg_r)>1 else 0
    winrate = (dr > 0).sum() / len(dr)
    print("%-42s | %7.2f | %8.2f | %7.1f%%  | %8.1f%%" % (
        strategy_labels[key], calmar, sortino, vol*100, winrate*100))
print("="*82)

print()
print("="*65)
print("S4 ADAPTIVE SELECTION BREAKDOWN (2010-02-11 onward)")
print("="*65)
bd2        = bull_td[START_FULL:]
tw2        = tecl_wins_lag[START_FULL:]
total_bull  = (bd2 > 0).sum()
tecl_chosen = ((tw2 > 0.5)  & (bd2 > 0)).sum()
tqqq_chosen = ((tw2 <= 0.5) & (bd2 > 0)).sum()
if total_bull > 0:
    print("  Total bull days  : %d" % total_bull)
    print("  TECL chosen      : %d  (%.1f%%)" % (tecl_chosen, 100*tecl_chosen/total_bull))
    print("  TQQQ chosen      : %d  (%.1f%%)" % (tqqq_chosen, 100*tqqq_chosen/total_bull))
print("="*65)

print()
print("letf_research_v5.py -- COMPLETE")
