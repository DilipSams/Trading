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
print(f"Period: {df.index[0].date()} to {df.index[-1].date()}  ({len(df)} days)
")

def sma(s,n): return s.rolling(n,min_periods=n).mean()
def rvol(s,n=20): return s.pct_change().rolling(n,min_periods=n).std()*np.sqrt(252)

LONG  = {"SP":"UPRO","NDQ":"TQQQ","DOW":"UDOW"}
SHORT = {"SP":"SPXU","NDQ":"SQQQ","DOW":"SDOW"}
UND   = {"SP":"SPY","NDQ":"QQQ","DOW":"DIA"}

spy_sma200 = sma(df["SPY"],200)
spy_sma100 = sma(df["SPY"],100)
spy_sma50  = sma(df["SPY"],50)
qqq_sma200 = sma(df["QQQ"],200)

mom21 = {k: (df[UND[k]]/df[UND[k]].shift(21)-1).shift(1) for k in ["SP","NDQ","DOW"]}
mom5  = {k: (df[UND[k]]/df[UND[k]].shift(5)-1).shift(1)  for k in ["SP","NDQ","DOW"]}
mom63 = {k: (df[UND[k]]/df[UND[k]].shift(63)-1).shift(1) for k in ["SP","NDQ","DOW"]}

ab200 = {k: (df[UND[k]]>sma(df[UND[k]],200)).shift(1).fillna(False) for k in ["SP","NDQ","DOW"]}
ab100 = {k: (df[UND[k]]>sma(df[UND[k]],100)).shift(1).fillna(False) for k in ["SP","NDQ","DOW"]}
ab50  = {k: (df[UND[k]]>sma(df[UND[k]],50)).shift(1).fillna(False)  for k in ["SP","NDQ","DOW"]}

etfvol = {t: rvol(df[t],20).shift(1) for t in ["UPRO","TQQQ","UDOW","SPXU","SQQQ","SDOW"]}

spy_below_streak = pd.Series(0, index=df.index)
streak = 0
for dt in df.index:
    if float(df["SPY"][dt]) < float(spy_sma200[dt]) if not pd.isna(spy_sma200[dt]) else False:
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

def metrics(eq, label):
    r=eq.pct_change().dropna(); ny=len(eq)/252
    cagr=(eq.iloc[-1]**(1/ny)-1)*100
    sh=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0
    pk=eq.cummax(); dd=((eq-pk)/pk).min()*100; cal=cagr/abs(dd) if dd else 0
    ye={}
    for i,d in enumerate(eq.index): ye[str(d)[:4]]=float(eq.iloc[i])
    ann,prev={},1.0
    for yr in sorted(ye): ann[yr]=(ye[yr]/prev-1)*100; prev=ye[yr]
    pw=sum(1 for v in ann.values() if v>0)/len(ann)*100 if ann else 0
    return dict(label=label,cagr=cagr,sharpe=sh,maxdd=dd,calmar=cal,winpct=pw,ann=ann,final=eq.iloc[-1])

signals={}

signals["B1"]=pd.Series("TQQQ",index=df.index)

