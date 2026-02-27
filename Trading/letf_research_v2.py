#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

DATA_DIR = "D:/Experiments/norgate_data/US_Equities"
COST = 0.0005

def load(t):
    df = pd.read_parquet(os.path.join(DATA_DIR, t + ".parquet"))
    col = next((c for c in df.columns if c.lower()=="close" and "unadj" not in c.lower()), None)
    if col is None:
        col = next((c for c in df.columns if "close" in c.lower() and "unadj" not in c.lower()), None)
    s = df[col].rename(t)
    s.index = pd.to_datetime(s.index)
    return s

raw = {t: load(t) for t in ["UPRO","SPXU","TQQQ","SQQQ","UDOW","SDOW","SPY","QQQ","DIA"]}
df = pd.concat(raw.values(), axis=1).dropna()
df = df[df.index >= "2010-02-11"]
print("Period: " + str(df.index[0].date()) + " - " + str(df.index[-1].date()) + "  (" + str(len(df)) + " days)")
def sma(s, n): return s.rolling(n, min_periods=n).mean()
def rvol(s, n=20): return s.pct_change().rolling(n, min_periods=n).std() * np.sqrt(252)
mom = {"SP": (df["SPY"]/df["SPY"].shift(21)-1).shift(1), "NDQ": (df["QQQ"]/df["QQQ"].shift(21)-1).shift(1), "DOW": (df["DIA"]/df["DIA"].shift(21)-1).shift(1)}
sma200 = {"SP": (df["SPY"]>sma(df["SPY"],200)).shift(1).fillna(False), "NDQ": (df["QQQ"]>sma(df["QQQ"],200)).shift(1).fillna(False), "DOW": (df["DIA"]>sma(df["DIA"],200)).shift(1).fillna(False)}
sma100 = {"SP": (df["SPY"]>sma(df["SPY"],100)).shift(1).fillna(False), "NDQ": (df["QQQ"]>sma(df["QQQ"],100)).shift(1).fillna(False), "DOW": (df["DIA"]>sma(df["DIA"],100)).shift(1).fillna(False)}
etf_vol = {t: rvol(df[t]).shift(1) for t in ["UPRO","SPXU","TQQQ","SQQQ","UDOW","SDOW"]}
LONG = {"SP": "UPRO", "NDQ": "TQQQ", "DOW": "UDOW"}
SHORT = {"SP": "SPXU", "NDQ": "SQQQ", "DOW": "SDOW"}
def best_long(idx, require_sma=True, sma_ref=sma200):
    eligible = [k for k in ["SP","NDQ","DOW"] if (not require_sma or sma_ref[k].get(idx, False)) and not pd.isna(mom["SP"].get(idx, np.nan))]
    if not eligible: return None
    return LONG[max(eligible, key=lambda k: mom[k].get(idx, -999))]
def worst_short(idx):
    keys = [k for k in ["SP","NDQ","DOW"] if not pd.isna(mom["SP"].get(idx, np.nan))]
    if not keys: return None
    return SHORT[min(keys, key=lambda k: mom[k].get(idx, 999))]
rets = {t: df[t].pct_change().fillna(0) for t in df.columns}
def run(pos_series, label):
    eq, prev = [1.0], "CASH"
    for dt in df.index[1:]:
        pos = pos_series.get(dt, "CASH")
        tc = COST if pos != prev else 0.0
        r = float(rets[pos].get(dt, 0)) if pos != "CASH" else 0.0
        eq.append(eq[-1] * (1 + r - tc))
        prev = pos
    return pd.Series(eq[1:], index=df.index[1:], name=label)
def metrics(eq, label):
    r = eq.pct_change().dropna()
    ny = len(eq) / 252
    cagr = (eq.iloc[-1] ** (1/ny) - 1) * 100
    sh = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    peak = eq.cummax(); dd = ((eq - peak)/peak).min() * 100
    calmar = cagr / abs(dd) if dd != 0 else 0
    ye = {}
    for i, d in enumerate(eq.index): ye[str(d)[:4]] = float(eq.iloc[i])
    ann, prev = {}, 1.0
    for yr in sorted(ye): ann[yr] = (ye[yr]/prev - 1)*100; prev = ye[yr]
    pw = sum(1 for v in ann.values() if v > 0) / len(ann) * 100 if ann else 0
    return dict(label=label, cagr=cagr, sharpe=sh, maxdd=dd, calmar=calmar, winpct=pw, ann=ann, final10=eq.iloc[-1]*10000)
signals = {}
signals["B0"] = pd.Series("SPY", index=df.index)
signals["B1"] = pd.Series("TQQQ", index=df.index)
sig = pd.Series("CASH", index=df.index)
ndq_sma200_lag = (df["QQQ"] > sma(df["QQQ"],200)).shift(1).fillna(False)
sig[ndq_sma200_lag] = "TQQQ"
signals["S2"] = sig
sig = pd.Series("CASH", index=df.index)
bull_spy = (df["SPY"] > sma(df["SPY"],200)).shift(1).fillna(False)
for dt in df.index:
    if bull_spy.get(dt, False):
        p = best_long(dt, require_sma=True, sma_ref=sma200)
        sig[dt] = p if p else "TQQQ"
    else:
        p = worst_short(dt)
        sig[dt] = p if p else "CASH"
signals["S3"] = sig
sig = pd.Series("CASH", index=df.index)
prev_p, hold = "CASH", 0
for dt in df.index:
    hold = max(0, hold-1)
    if hold > 0:
        sig[dt] = prev_p; continue
    if bull_spy.get(dt, False):
        p = best_long(dt, require_sma=True, sma_ref=sma200) or "TQQQ"
        v = float(etf_vol[p].get(dt, 0) or 0)
        if v > 0.85: p = "CASH"
    else:
        p = worst_short(dt) or "CASH"
        if p != "CASH":
            v = float(etf_vol[p].get(dt, 0) or 0)
            if v > 0.85: p = "CASH"
    if p != prev_p: hold = 5
    sig[dt] = p; prev_p = p
signals["S4"] = sig
sig = pd.Series("CASH", index=df.index)
for dt in df.index:
    if bull_spy.get(dt, False):
        p = best_long(dt, require_sma=True, sma_ref=sma200) or "TQQQ"
    else:
        p = worst_short(dt) or "SPXU"
    sig[dt] = p
signals["S5"] = sig
etf6_mom = {t: (df[t]/df[t].shift(21)-1).shift(1) for t in ["UPRO","TQQQ","UDOW","SPXU","SQQQ","SDOW"]}
SHORT_UNDERLYING = {"SPXU":"SP","SQQQ":"NDQ","SDOW":"DOW"}
sig = pd.Series("CASH", index=df.index)
for dt in df.index:
    candidates = {}
    for t in ["UPRO","TQQQ","UDOW"]:
        idx_k = next(k for k,v in LONG.items() if v==t)
        if sma200[idx_k].get(dt, False): candidates[t] = float(etf6_mom[t].get(dt, -999) or -999)
    for t in ["SPXU","SQQQ","SDOW"]:
        idx_k = SHORT_UNDERLYING[t]
        if not sma200[idx_k].get(dt, True): candidates[t] = float(etf6_mom[t].get(dt, -999) or -999)
    if candidates: sig[dt] = max(candidates, key=candidates.get)
    else:
        all_long = {t: float(etf6_mom[t].get(dt,-999) or -999) for t in ["UPRO","TQQQ","UDOW"]}
        sig[dt] = max(all_long, key=all_long.get)
signals["S6"] = sig
qqq_rvol_lag = rvol(df["QQQ"], 20).shift(1)
sig = pd.Series("CASH", index=df.index)
for dt in df.index:
    v = float(qqq_rvol_lag.get(dt, np.nan) or np.nan)
    if np.isnan(v): sig[dt] = "CASH"; continue
    if v < 0.18: sig[dt] = "TQQQ"
    elif v < 0.30: sig[dt] = "UPRO"
    else:
        p = worst_short(dt)
        sig[dt] = p if p else "SPXU"
signals["S7"] = sig
sig = pd.Series("CASH", index=df.index)
prev_p, hold = "CASH", 0
for dt in df.index:
    hold = max(0, hold-1)
    if hold > 0: sig[dt] = prev_p; continue
    bull = bull_spy.get(dt, False)
    v = float(qqq_rvol_lag.get(dt, np.nan) or 0)
    if np.isnan(v): v = 0
    if bull and v < 0.25: p = best_long(dt, require_sma=True, sma_ref=sma200) or "TQQQ"
    elif not bull and v > 0.20: p = worst_short(dt) or "SPXU"
    else: p = "CASH"
    if p != prev_p: hold = 3
    sig[dt] = p; prev_p = p
signals["S8"] = sig
mom21_raw = {k: (df[{"SP":"SPY","NDQ":"QQQ","DOW":"DIA"}[k]]/df[{"SP":"SPY","NDQ":"QQQ","DOW":"DIA"}[k]].shift(21)-1).shift(1) for k in ["SP","NDQ","DOW"]}
mom5_raw = {k: (df[{"SP":"SPY","NDQ":"QQQ","DOW":"DIA"}[k]]/df[{"SP":"SPY","NDQ":"QQQ","DOW":"DIA"}[k]].shift(5)-1).shift(1) for k in ["SP","NDQ","DOW"]}
sig = pd.Series("CASH", index=df.index)
for dt in df.index:
    bull = bull_spy.get(dt, False)
    accel = {k: float(mom5_raw[k].get(dt,0) or 0) - float(mom21_raw[k].get(dt,0) or 0) for k in ["SP","NDQ","DOW"]}
    if bull:
        elig = {k: accel[k] for k in ["SP","NDQ","DOW"] if sma200[k].get(dt,False)}
        if not elig: elig = {k: accel[k] for k in ["SP","NDQ","DOW"]}
        sig[dt] = LONG[max(elig, key=elig.get)]
    else:
        sig[dt] = SHORT[min(accel, key=accel.get)]
signals["S9"] = sig
NAMES = {"B0":"SPY B&H","B1":"TQQQ B&H *","S2":"S2 TQQQ/Cash","S3":"S3 Long/Short Basic","S4":"S4 L/S + VolCap + Hold","S5":"S5 L/S Always Short","S6":"S6 Rank-6 ETFs","S7":"S7 Vol-Regime","S8":"S8 Hybrid SMA+Vol","S9":"S9 Momentum Accel"}
results = {}
for k in NAMES:
    eq = run(signals[k], NAMES[k])
    results[k] = (eq, metrics(eq, NAMES[k]))
W = 105
print()
print("="*W)
print("STRATEGY COMPARISON  (target: beat TQQQ B&H > 40.6% CAGR with MaxDD < -82%)")
print("="*W)
tqqq_cagr = results["B1"][1]["cagr"]
tqqq_dd   = results["B1"][1]["maxdd"]
hdr = ("  " + "Strategy".ljust(28) + "  " + "CAGR".rjust(7) + "  " + "Sharpe".rjust(7) + "  " + "MaxDD".rjust(8) + "  " + "Calmar".rjust(7) + "  " + "WinYrs".rjust(7) + "  " + "$10k->".rjust(12) + "  " + "Beat TQQQ?".rjust(12))
print(hdr)
print("-"*W)
for k in NAMES:
    m = results[k][1]
    beats = "YES" if m["cagr"] > tqqq_cagr else "no"
    bdd = "better DD" if m["maxdd"] > tqqq_dd else ""
    row = ("  " + m["label"].ljust(28) + "  " + (str(round(m["cagr"],1))+"%").rjust(7) + "  " + str(round(m["sharpe"],2)).rjust(7) + "  " + (str(round(m["maxdd"],1))+"%").rjust(8) + "  " + str(round(m["calmar"],2)).rjust(7) + "  " + (str(round(m["winpct"]))+"%").rjust(7) + "  $" + "{:>10,.0f}".format(m["final10"]) + "  " + beats + " " + bdd)
    print(row)
W2 = 140
print()
print("="*W2)
print("YEAR-BY-YEAR (%)")
print("="*W2)
klist = list(NAMES.keys())
all_yrs = sorted({yr for k in klist for yr in results[k][1]["ann"]})
shorts = ["SPY","TQQQ*","S2-Cash","S3-LS","S4-LScap","S5-LSnoc","S6-Rank6","S7-VolR","S8-Hybrd","S9-Accel"]
print("  " + "Year".ljust(6) + "".join("  " + s.rjust(10) for s in shorts))
print("  " + "-"*6 + ("  "+"-"*10)*len(klist))
for yr in all_yrs:
    row = "  " + yr.ljust(6)
    for k in klist:
        v = results[k][1]["ann"].get(yr)
        if v is not None: row += "  " + (("+" if v>=0 else "")+str(round(v,1))+"%").rjust(10)
        else: row += "  " + "N/A".rjust(10)
    print(row)
print("  " + "-"*6 + ("  "+"-"*10)*len(klist))
for rname in ("Total","CAGR"):
    row = "  " + rname.ljust(6)
    for k in klist:
        vals = list(results[k][1]["ann"].values())
        comp = 1.0
        for v in vals: comp *= (1+v/100)
        res = (comp-1)*100 if rname=="Total" else ((comp**(1.0/len(vals)))-1)*100 if vals else 0
        row += "  " + (("+" if res>=0 else "")+str(round(res,1))+"%").rjust(10)
    print(row)
print()
print("  Trades per strategy:")
for k in klist[2:]:
    sig = signals[k]
    n = int((sig != sig.shift()).sum())
    print("    " + NAMES[k] + ": " + str(n) + " trades (~" + str(round(n/(len(df)/252),1)) + "/yr)")
print("="*W2)
