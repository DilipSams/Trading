#!/usr/bin/env python3
import pandas as pd, numpy as np, os

DATA_DIR = r"D:\Experiments\norgate_data\US_Equities"
COST = 0.0005

def load(t):
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{t}.parquet"))
    col = next((c for c in df.columns if "close" in c.lower() and "unadj" not in c.lower()), None)
    s = df[col].rename(t); s.index = pd.to_datetime(s.index); return s

raw = {t: load(t) for t in ["UPRO","SPXU","TQQQ","SQQQ","UDOW","SDOW","SPY","QQQ","DIA"]}
df = pd.concat(raw.values(), axis=1).dropna()
df = df[df.index >= "2010-02-11"]
print(f"Period: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} days)" + chr(10))

def sma(s,n): return s.rolling(n,min_periods=n).mean()
def rvol(s,n=20): return s.pct_change().rolling(n,min_periods=n).std()*np.sqrt(252)

LONG  = {"SP":"UPRO","NDQ":"TQQQ","DOW":"UDOW"}
SHORT = {"SP":"SPXU","NDQ":"SQQQ","DOW":"SDOW"}
UND   = {"SP":"SPY","NDQ":"QQQ","DOW":"DIA"}

mom21 = {k: (df[UND[k]]/df[UND[k]].shift(21)-1).shift(1) for k in ["SP","NDQ","DOW"]}
mom63 = {k: (df[UND[k]]/df[UND[k]].shift(63)-1).shift(1) for k in ["SP","NDQ","DOW"]}
mom5  = {k: (df[UND[k]]/df[UND[k]].shift(5)-1).shift(1)  for k in ["SP","NDQ","DOW"]}
ab200 = {k: (df[UND[k]]>sma(df[UND[k]],200)).shift(1).fillna(False) for k in ["SP","NDQ","DOW"]}
ab50  = {k: (df[UND[k]]>sma(df[UND[k]],50)).shift(1).fillna(False)  for k in ["SP","NDQ","DOW"]}

etfvol = {t: rvol(df[t],20).shift(1) for t in ["UPRO","TQQQ","UDOW","SPXU","SQQQ","SDOW"]}

spy_below_streak = pd.Series(0, index=df.index)
spy_sma200 = sma(df["SPY"],200)
streak = 0
for dt in df.index:
    val = float(df["SPY"].get(dt,np.nan))
    sma_val = float(spy_sma200.get(dt,np.nan))
    if not np.isnan(val) and not np.isnan(sma_val) and val < sma_val:
        streak += 1
    else:
        streak = 0
    spy_below_streak[dt] = streak
spy_below_streak = spy_below_streak.shift(1).fillna(0)

rets = {t: df[t].pct_change().fillna(0) for t in df.columns}

def run(pos_series):
    eq, prev = [1.0], "CASH"
    for dt in df.index[1:]:
        pos = pos_series.get(dt,"CASH")
        tc  = COST if pos!=prev else 0.0
        r   = float(rets[pos].get(dt,0)) if pos!="CASH" else 0.0
        eq.append(eq[-1]*(1+r-tc)); prev=pos
    return pd.Series(eq[1:], index=df.index[1:])

def metrics(eq):
    r=eq.pct_change().dropna(); ny=len(eq)/252
    cagr=(eq.iloc[-1]**(1/ny)-1)*100
    sh=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0
    pk=eq.cummax(); dd=((eq-pk)/pk).min()*100; cal=cagr/abs(dd) if dd else 0
    ye={}
    for i,d in enumerate(eq.index): ye[str(d)[:4]]=float(eq.iloc[i])
    ann,prev={},1.0
    for yr in sorted(ye): ann[yr]=(ye[yr]/prev-1)*100; prev=ye[yr]
    pw=sum(1 for v in ann.values() if v>0)/len(ann)*100 if ann else 0
    return dict(cagr=cagr,sharpe=sh,maxdd=dd,calmar=cal,winpct=pw,ann=ann,final=eq.iloc[-1])

signals={}
signals["B1"] = pd.Series("TQQQ", index=df.index)

def best_long_nosma(dt, lookback="21"):
    m = mom21 if lookback=="21" else mom63
    scores = {k: float(m[k].get(dt,-999) or -999) for k in ["SP","NDQ","DOW"]}
    return LONG[max(scores, key=scores.get)]

def worst_short_by(dt, lookback="21"):
    m = mom21 if lookback=="21" else mom63
    scores = {k: float(m[k].get(dt,999) or 999) for k in ["SP","NDQ","DOW"]}
    return SHORT[min(scores, key=scores.get)]

def bear_confirmed(dt, streak_thresh=10, mom21_thresh=-0.10, mom63_thresh=-0.03, volcap=0.90):
    if int(spy_below_streak.get(dt,0)) < streak_thresh: return False, None
    if float(mom21["NDQ"].get(dt,0) or 0) >= mom21_thresh: return False, None
    if float(mom63["NDQ"].get(dt,0) or 0) >= mom63_thresh: return False, None
    etf = worst_short_by(dt)
    v = float(etfvol[etf].get(dt,0) or 0)
    if v > volcap: return True, "CASH"
    return True, etf

def bear_exit(dt):
    return (ab50["SP"].get(dt,False) or float(mom5["NDQ"].get(dt,0) or 0) > 0.05)

# TURBO-A
sig = pd.Series("TQQQ", index=df.index)
in_short = False
for dt in df.index:
    if in_short and bear_exit(dt):
        in_short = False
    if not in_short:
        is_bear, short_etf = bear_confirmed(dt)
        if is_bear:
            in_short = True
    if in_short:
        _, short_etf = bear_confirmed(dt)
        sig[dt] = short_etf if short_etf else "CASH"
    else:
        sig[dt] = "TQQQ"
signals["TA"] = sig

# TURBO-B
sig = pd.Series("CASH", index=df.index)
in_short = False
for dt in df.index:
    if in_short and bear_exit(dt):
        in_short = False
    if not in_short:
        is_bear, short_etf = bear_confirmed(dt)
        if is_bear:
            in_short = True
    if in_short:
        _, short_etf = bear_confirmed(dt)
        sig[dt] = short_etf if short_etf else "CASH"
    else:
        sig[dt] = best_long_nosma(dt)
signals["TB"] = sig

# TURBO-C
sig = pd.Series("CASH", index=df.index)
in_short = False
for dt in df.index:
    if in_short and bear_exit(dt):
        in_short = False
    if not in_short:
        is_bear, short_etf = bear_confirmed(dt)
        if is_bear:
            in_short = True
    if in_short:
        _, short_etf = bear_confirmed(dt)
        sig[dt] = short_etf if short_etf else "CASH"
    else:
        ndq21 = float(mom21["NDQ"].get(dt,0) or 0)
        dow21 = float(mom21["DOW"].get(dt,0) or 0)
        sp21  = float(mom21["SP"].get(dt,0) or 0)
        if dow21 - ndq21 > 0.05:
            sig[dt] = "UDOW"
        elif sp21 - ndq21 > 0.05:
            sig[dt] = "UPRO"
        else:
            sig[dt] = "TQQQ"
signals["TC"] = sig

# TURBO-D
sig = pd.Series("TQQQ", index=df.index)
in_short = False
for dt in df.index:
    if in_short and bear_exit(dt):
        in_short = False
    if not in_short:
        is_bear, short_etf = bear_confirmed(dt, streak_thresh=15, mom21_thresh=-0.12, mom63_thresh=-0.05)
        if is_bear:
            in_short = True
    if in_short:
        _, short_etf = bear_confirmed(dt, streak_thresh=15, mom21_thresh=-0.12, mom63_thresh=-0.05)
        sig[dt] = short_etf if short_etf else "CASH"
    else:
        sig[dt] = "TQQQ"
signals["TD"] = sig

# TURBO-E
sig = pd.Series("CASH", index=df.index)
in_short = False
exit_count = 0
for dt in df.index:
    if in_short:
        if ab50["SP"].get(dt,False):
            exit_count += 1
        else:
            exit_count = 0
        if exit_count >= 5 or float(mom5["NDQ"].get(dt,0) or 0) > 0.07:
            in_short = False; exit_count = 0
    if not in_short:
        is_bear, short_etf = bear_confirmed(dt)
        if is_bear:
            in_short = True; exit_count = 0
    if in_short:
        _, short_etf = bear_confirmed(dt)
        sig[dt] = short_etf if short_etf else "CASH"
    else:
        sig[dt] = best_long_nosma(dt)
signals["TE"] = sig

# TURBO-F
sig = pd.Series("TQQQ", index=df.index)
in_short = False
spy_ab200 = (df["SPY"] > sma(df["SPY"],200)).shift(1).fillna(False)
for dt in df.index:
    if in_short and bear_exit(dt):
        in_short = False
    if not in_short:
        is_bear, short_etf = bear_confirmed(dt)
        if is_bear:
            in_short = True
    if in_short:
        _, short_etf = bear_confirmed(dt)
        sig[dt] = short_etf if short_etf else best_long_nosma(dt)
    else:
        sig[dt] = best_long_nosma(dt)
signals["TF"] = sig

NAMES = {
    "B1":"TQQQ B&H (TARGET)",
    "TA":"TA: TQQQ+SQQQ (dual-confirm+volcap)",
    "TB":"TB: BestLong+Short (dual-confirm)",
    "TC":"TC: TQQQ+CrossIdx+Short",
    "TD":"TD: TQQQ+SQQQ (15d strict)",
    "TE":"TE: BestLong+Short (5d exit)",
    "TF":"TF: Always Invested (BestL/BestS)",
}

results={}
for k in NAMES:
    eq=run(signals[k]); results[k]=(eq,metrics(eq))

print("="*115)
print(f"  {chr(83)+chr(116)+chr(114)+chr(97)+chr(116)+chr(101)+chr(103)+chr(121):<40}  {chr(67)+chr(65)+chr(71)+chr(82):>7}  {chr(83)+chr(104)+chr(97)+chr(114)+chr(112)+chr(101):>7}  {chr(77)+chr(97)+chr(120)+chr(68)+chr(68):>8}  {chr(67)+chr(97)+chr(108)+chr(109)+chr(97)+chr(114):>7}  {chr(87)+chr(105)+chr(110)+chr(89)+chr(114)+chr(115):>7}  {chr(36)+chr(49)+chr(48)+chr(107)+chr(45)+chr(62):>12}  BEATS?")
print("-"*115)
tgt=results["B1"][1]["cagr"]
for k in NAMES:
    m=results[k][1]
    beat="*** YES ***" if m["cagr"]>tgt else "no"
    print(f"  {NAMES[k]:<38}  {m[chr(99)+chr(97)+chr(103)+chr(114)]:>6.1f}%  {m[chr(115)+chr(104)+chr(97)+chr(114)+chr(112)+chr(101)]:>7.2f}  {m[chr(109)+chr(97)+chr(120)+chr(100)+chr(100)]:>7.1f}%  {m[chr(99)+chr(97)+chr(108)+chr(109)+chr(97)+chr(114)]:>7.2f}  {m[chr(119)+chr(105)+chr(110)+chr(112)+chr(99)+chr(116)]:>6.0f}%  {chr(36)}{m[chr(102)+chr(105)+chr(110)+chr(97)+chr(108)]*10000:>10,.0f}  {beat}")

klist=list(NAMES.keys())
shorts=["TQQQ*","TA","TB","TC","TD","TE","TF"]
all_yrs=sorted({yr for k in klist for yr in results[k][1]["ann"]})
print(chr(10)+"="*115)
print("  "+"Year  "+"".join(f"  {s:>10}" for s in shorts))
print("  ------"+"  ----------"*len(klist))
for yr in all_yrs:
    row=f"  {yr:<6}"
    for k in klist:
        v=results[k][1]["ann"].get(yr)
        row+=f"  {v:>+9.1f}%" if v is not None else "       N/A"
    print(row)
print("  ------"+"  ----------"*len(klist))
row="  CAGR  "
for k in klist:
    vals=list(results[k][1]["ann"].values())
    comp=1.0
    for v in vals: comp*=(1+v/100)
    res=((comp**(1/len(vals)))-1)*100 if vals else 0
    row+=f"  {res:>+9.1f}%"
print(row)

print(chr(10)+"BEAR REGIME ANALYSIS:")
for k in ["TA","TB","TC","TD","TE","TF"]:
    sig=signals[k]; short_tickers={"SPXU","SQQQ","SDOW"}
    sd=sig[sig.isin(short_tickers)]
    if len(sd)==0: print(f"  {NAMES[k]}: NO shorts"); continue
    by_yr={}
    for dt,pos in sd.items(): by_yr.setdefault(str(dt)[:4],[]).append(pos)
    print(f"  {NAMES[k]}: {len(sd)} short days: "+", ".join(f"{yr}({len(v)}d {v[0]})" for yr,v in sorted(by_yr.items())))
print("="*115)
