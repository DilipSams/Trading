#!/usr/bin/env python3
import pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

DATA = r"D:\Experiments\norgate_data\US_Equities"
COST = 0.0005

def load(t):
    fp = os.path.join(DATA, f"{t}.parquet")
    if not os.path.exists(fp): return None
    df = pd.read_parquet(fp)
    col = next((c for c in df.columns if "close" in c.lower() and "unadj" not in c.lower()), None)
    if not col: return None
    s = df[col].rename(t); s.index = pd.to_datetime(s.index); return s

def sma(s,n): return s.rolling(n,min_periods=n).mean()
def rvol(s,n=20): return s.pct_change().rolling(n,min_periods=n).std()*np.sqrt(252)

def run_positions(pos_s, price_df, extra_rets=None):
    base_rets = {t: price_df[t].pct_change().fillna(0) for t in price_df.columns}
    if extra_rets: base_rets.update(extra_rets)
    eq=[1.0]; prev="CASH"
    for i,dt in enumerate(pos_s.index):
        if i==0: continue
        pos=str(pos_s.iloc[i])
        tc = COST if pos!=prev else 0.0
        r = float(base_rets[pos].get(dt,0)) if pos!="CASH" and pos in base_rets else 0.0
        eq.append(eq[-1]*(1+r-tc)); prev=pos
    return pd.Series(eq, index=pos_s.index)

def met(eq, lbl=""):
    r=eq.pct_change().dropna(); ny=len(eq)/252
    if ny<0.5: return None
    cagr=(float(eq.iloc[-1])**(1/ny)-1)*100
    sh=float(r.mean()/r.std()*np.sqrt(252)) if r.std()>0 else 0
    pk=eq.cummax(); dd=((eq-pk)/pk).min()*100
    cal=cagr/abs(dd) if dd else 0
    ye={}
    for i,d in enumerate(eq.index): ye[str(d)[:4]]=float(eq.iloc[i])
    ann,pv={},1.0
    for yr in sorted(ye): ann[yr]=(ye[yr]/pv-1)*100; pv=ye[yr]
    pw=sum(1 for v in ann.values() if v>0)/len(ann)*100 if ann else 0
    return dict(lbl=lbl,cagr=cagr,sh=sh,dd=dd,cal=cal,pw=pw,ann=ann,fin=float(eq.iloc[-1]),
                s=str(eq.index[0])[:10],e=str(eq.index[-1])[:10])

def ptbl(rows, title, ref_cagr=40.6):
    if not rows: return
    print(f"\n{chr(61)*118}\n  {title}\n{chr(61)*118}")
    print(f"  {'Strategy':<40}  {'Period':>12}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>8}  {'Calmar':>7}  {'Win%':>6}  {'$10k->':>12}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*12}")
    for m in rows:
        if m is None: continue
        beat=" *BEATS*" if m["cagr"]>ref_cagr else ""
        lbl=m["lbl"]; s4=m["s"][:4]; e4=m["e"][:4]
        print(f"  {lbl:<40}  {s4+chr(45)+e4:>12}  {m['cagr']:>6.1f}%  {m['sh']:>7.2f}  {m['dd']:>7.1f}%  {m['cal']:>7.2f}  {m['pw']:>5.0f}%  ${m['fin']*10000:>10,.0f}{beat}")

def atbl(mmap, title):
    all_yrs=sorted({yr for m in mmap.values() for yr in (m["ann"] if m else {})})
    keys=[k for k in mmap if mmap[k] is not None]
    if not keys or not all_yrs: return
    print(f"\n  {title}")
    cw=11
    print("  "+"Year  "+"".join(f"  {k[:cw]:>{cw}}" for k in keys))
    print("  ------"+(f"  {chr(45)*cw}")*len(keys))
    for yr in all_yrs:
        row=f"  {yr:<6}"
        for k in keys:
            v=mmap[k]["ann"].get(yr) if mmap[k] else None
            row+=f"  {v:>+10.1f}%" if v is not None else f"  {'N/A':>{cw}}"
        print(row)
    print("  ------"+(f"  {chr(45)*cw}")*len(keys))
    row="  CAGR  "
    for k in keys:
        vals=list(mmap[k]["ann"].values()) if mmap[k] else []
        comp=1.0
        for v in vals: comp*=(1+v/100)
        res=((comp**(1/len(vals)))-1)*100 if vals else 0
        row+=f"  {res:>+10.1f}%"
    print(row)

def bear_streak(spy_s):
    sm=sma(spy_s,200)
    st=pd.Series(0,index=spy_s.index,dtype=int); c=0
    for dt in spy_s.index:
        v=float(spy_s.get(dt,np.nan)); sv=float(sm.get(dt,np.nan))
        c = c+1 if (not np.isnan(v) and not np.isnan(sv) and v<sv) else 0
        st[dt]=c
    return st.shift(1).fillna(0)

def td_signal(spy_s, und_s, long_t, short_t, price_df,
              streak_thr=15, m21_thr=-0.12, m63_thr=-0.05, vcap=0.90):
    stk=bear_streak(spy_s)
    m21=(und_s/und_s.shift(21)-1).shift(1)
    m63=(und_s/und_s.shift(63)-1).shift(1)
    m5 =(und_s/und_s.shift(5)-1).shift(1)
    ab50=(spy_s>sma(spy_s,50)).shift(1).fillna(False)
    sv=rvol(price_df[short_t] if short_t in price_df.columns else und_s,20).shift(1)
    sig=pd.Series(long_t,index=spy_s.index); ins=False
    for dt in spy_s.index:
        if ins and (ab50.get(dt,False) or float(m5.get(dt,0) or 0)>0.05): ins=False
        if not ins and int(stk.get(dt,0))>=streak_thr and float(m21.get(dt,0) or 0)<m21_thr and float(m63.get(dt,0) or 0)<m63_thr: ins=True
        if ins:
            v=float(sv.get(dt,0) or 0)
            sig[dt]="CASH" if v>vcap else short_t
        else:
            sig[dt]=long_t
    return sig

spy=load("SPY"); qqq=load("QQQ"); dia=load("DIA")
tqqq=load("TQQQ"); sqqq=load("SQQQ")
tmf=load("TMF"); tmv=load("TMV")

# PATH 1: TQQQ / TMF BLEND
print("\n"+"="*118+"\n  PATH 1: TQQQ / TMF (3x Long Bonds) Blend  |  TMF inception 2009-04-16\n"+"="*118)
p1=pd.concat([tqqq,tmf,spy,qqq],axis=1).dropna()
p1=p1[p1.index>="2009-04-16"]
print(f"  Period: {str(p1.index[0])[:10]} to {str(p1.index[-1])[:10]}  ({len(p1)} bars)\n")

def blend_monthly(df, allocations):
    rets={t:df[t].pct_change().fillna(0) for t in allocations}
    eq=[1.0]; wts=dict(allocations); pm=df.index[0].month
    for i,dt in enumerate(df.index[1:],1):
        if dt.month!=pm: wts=dict(allocations); pm=dt.month
        r=sum(wts[t]*float(rets[t].iloc[i]) for t in wts)
        eq.append(eq[-1]*(1+r))
        tot=sum(wts[t]*(1+float(rets[t].iloc[i])) for t in wts)
        for t in wts: wts[t]=wts[t]*(1+float(rets[t].iloc[i]))/tot
    return pd.Series(eq[1:],index=df.index[1:])

p1r={}
p1r["TQQQ B&H"]=met(run_positions(pd.Series("TQQQ",index=p1.index),p1),"TQQQ B&H")
p1r["TMF B&H"]=met(run_positions(pd.Series("TMF",index=p1.index),p1),"TMF B&H")
p1r["55/45 TQQQ/TMF monthly"]=met(blend_monthly(p1,{"TQQQ":0.55,"TMF":0.45}),"55/45 TQQQ/TMF monthly")
p1r["60/40 TQQQ/TMF monthly"]=met(blend_monthly(p1,{"TQQQ":0.60,"TMF":0.40}),"60/40 TQQQ/TMF monthly")
p1r["70/30 TQQQ/TMF monthly"]=met(blend_monthly(p1,{"TQQQ":0.70,"TMF":0.30}),"70/30 TQQQ/TMF monthly")
p1r["80/20 TQQQ/TMF monthly"]=met(blend_monthly(p1,{"TQQQ":0.80,"TMF":0.20}),"80/20 TQQQ/TMF monthly")

spy_p1=p1["SPY"]; qqq_p1=p1["QQQ"]
stk_p1=bear_streak(spy_p1)
m21_p1=(qqq_p1/qqq_p1.shift(21)-1).shift(1)
m63_p1=(qqq_p1/qqq_p1.shift(63)-1).shift(1)
m5_p1=(qqq_p1/qqq_p1.shift(5)-1).shift(1)
ab50_p1=(spy_p1>sma(spy_p1,50)).shift(1).fillna(False)
sig=pd.Series("TQQQ",index=p1.index); ins=False
for dt in p1.index:
    if ins and (ab50_p1.get(dt,False) or float(m5_p1.get(dt,0) or 0)>0.05): ins=False
    if not ins and int(stk_p1.get(dt,0))>=15 and float(m21_p1.get(dt,0) or 0)<-0.12 and float(m63_p1.get(dt,0) or 0)<-0.05: ins=True
    sig[dt]="TMF" if ins else "TQQQ"
p1r["TD+TMF: bull=TQQQ, bear=TMF"]=met(run_positions(sig,p1),"TD+TMF: bull=TQQQ, bear=TMF")

rets_p1={t:p1[t].pct_change().fillna(0) for t in ["TQQQ","TMF"]}
sig_dyn=pd.Series("CASH",index=p1.index); ins=False
eq_dyn=[1.0]; wt_t,wt_m=0.70,0.30
for i,dt in enumerate(p1.index[1:],1):
    if ins and (ab50_p1.get(dt,False) or float(m5_p1.get(dt,0) or 0)>0.05): ins=False
    if not ins and int(stk_p1.get(dt,0))>=15 and float(m21_p1.get(dt,0) or 0)<-0.12 and float(m63_p1.get(dt,0) or 0)<-0.05: ins=True
    wt_t,wt_m=(0.20,0.80) if ins else (0.70,0.30)
    r=wt_t*float(rets_p1["TQQQ"].iloc[i])+wt_m*float(rets_p1["TMF"].iloc[i])
    eq_dyn.append(eq_dyn[-1]*(1+r))
p1r["Dynamic 70/30->20/80 on bear confirm"]=met(pd.Series(eq_dyn[1:],index=p1.index[1:]),"Dynamic 70/30->20/80 on bear confirm")

ptbl(list(p1r.values()),"PATH 1: TQQQ/TMF BLEND STRATEGIES",ref_cagr=p1r["TQQQ B&H"]["cagr"])
atbl(p1r,"PATH 1 -- YEAR BY YEAR")

# PATH 2: SYNTHETIC 3x
print("\n\n"+"="*118+"\n  PATH 2: SYNTHETIC 3x QQQ & SPY -- 1999 to 2026 (includes 2000-02 dot-com, 2008 GFC)\n"+"="*118)

qqq_full=load("QQQ"); spy_full=load("SPY")
tqqq_act=load("TQQQ"); sqqq_act=load("SQQQ")
upro_act=load("UPRO"); spxu_act=load("SPXU")

ER=0.0095/252
synth_tqqq=qqq_full.pct_change()*3-ER
synth_sqqq=-qqq_full.pct_change()*3-ER
synth_upro=spy_full.pct_change()*3-ER
synth_spxu=-spy_full.pct_change()*3-ER

splice=pd.Timestamp("2010-02-11")
splice_upro=pd.Timestamp("2009-06-25")

def splice_rets(synth, actual, splice_date):
    a_rets=actual.pct_change().fillna(0)
    return pd.concat([synth[synth.index<splice_date], a_rets[a_rets.index>=splice_date]]).sort_index()

tr=splice_rets(synth_tqqq,tqqq_act,splice)
sr=splice_rets(synth_sqqq,sqqq_act,splice)
ur=splice_rets(synth_upro,upro_act,splice_upro)
xr=splice_rets(synth_spxu,spxu_act,splice_upro)

idx=qqq_full.index.intersection(spy_full.index)
idx=idx[idx>="1999-03-10"]
qqq_p2=qqq_full.reindex(idx); spy_p2=spy_full.reindex(idx)
tr=tr.reindex(idx).fillna(0); sr=sr.reindex(idx).fillna(0)
ur=ur.reindex(idx).fillna(0); xr=xr.reindex(idx).fillna(0)
extra={"TQQQ":tr,"SQQQ":sr,"UPRO":ur,"SPXU":xr,"SPY":spy_p2.pct_change().fillna(0)}
dummy=pd.DataFrame({"SPY":spy_p2})

print(f"  Period: {str(idx[0])[:10]} to {str(idx[-1])[:10]}  ({len(idx)} bars)")
print(f"  Pre-actual synthetic: 1999-03-10 to 2010-02-10 (TQQQ/SQQQ), 1999-03-10 to 2009-06-24 (UPRO/SPXU)\n")

p2r={}
p2r["Synth TQQQ B&H 1999-2026"]=met(run_positions(pd.Series("TQQQ",index=idx),dummy,extra),"Synth TQQQ B&H 1999-2026")
p2r["Synth UPRO B&H 1999-2026"]=met(run_positions(pd.Series("UPRO",index=idx),dummy,extra),"Synth UPRO B&H 1999-2026")
p2r["SPY B&H 1999-2026"]=met(run_positions(pd.Series("SPY",index=idx),dummy,extra),"SPY B&H 1999-2026")

ndq_ab=(qqq_p2>sma(qqq_p2,200)).shift(1).fillna(False)
sig=pd.Series(ndq_ab.map({True:"TQQQ",False:"CASH"}))
p2r["S2 TQQQ/Cash SMA200 1999-2026"]=met(run_positions(sig,dummy,extra),"S2 TQQQ/Cash SMA200 1999-2026")

stk_p2=bear_streak(spy_p2)
m21_p2=(qqq_p2/qqq_p2.shift(21)-1).shift(1)
m63_p2=(qqq_p2/qqq_p2.shift(63)-1).shift(1)
m5_p2=(qqq_p2/qqq_p2.shift(5)-1).shift(1)
ab50_p2=(spy_p2>sma(spy_p2,50)).shift(1).fillna(False)
sig=pd.Series("TQQQ",index=idx); ins=False
for dt in idx:
    if ins and (ab50_p2.get(dt,False) or float(m5_p2.get(dt,0) or 0)>0.05): ins=False
    if not ins and int(stk_p2.get(dt,0))>=15 and float(m21_p2.get(dt,0) or 0)<-0.12 and float(m63_p2.get(dt,0) or 0)<-0.05: ins=True
    sig[dt]="SQQQ" if ins else "TQQQ"
p2r["TD TQQQ/SQQQ 1999-2026"]=met(run_positions(sig,dummy,extra),"TD TQQQ/SQQQ 1999-2026")

tmf_r=pd.concat([pd.Series(0,index=idx[idx<"2009-04-16"]),
                  tmf.pct_change().fillna(0).reindex(idx[idx>="2009-04-16"])]).reindex(idx).fillna(0)
extra["TMF"]=tmf_r
sig2=pd.Series("TQQQ",index=idx); ins=False
for dt in idx:
    if ins and (ab50_p2.get(dt,False) or float(m5_p2.get(dt,0) or 0)>0.05): ins=False
    if not ins and int(stk_p2.get(dt,0))>=15 and float(m21_p2.get(dt,0) or 0)<-0.12 and float(m63_p2.get(dt,0) or 0)<-0.05: ins=True
    sig2[dt]="TMF" if ins else "TQQQ"
p2r["TD+TMF bear 1999-2026"]=met(run_positions(sig2,dummy,extra),"TD+TMF bear 1999-2026")

print("  Bear regime triggers (where TD held SQQQ):")
bear_dates=sig[sig=="SQQQ"]
by_yr={}
for dt in bear_dates.index: by_yr.setdefault(str(dt)[:4],[]).append(dt)
for yr,dts in sorted(by_yr.items()):
    print(f"    {yr}: {len(dts)} days  ({str(dts[0])[:10]} to {str(dts[-1])[:10]})")

ptbl(list(p2r.values()),"PATH 2: SYNTHETIC HISTORY 1999-2026",ref_cagr=20.0)
atbl(p2r,"PATH 2 -- YEAR BY YEAR (synthetic pre-2010)")

# PATH 3: ALL AVAILABLE 3x SECTOR ETFs
print("\n\n"+"="*118+"\n  PATH 3: OTHER SECTOR 3x ETFs -- B&H comparison + TD strategy applied to each\n"+"="*118)

SECTORS=[
    ("TQQQ","SQQQ","QQQ", "Nasdaq-100 3x",    "2010-02-11"),
    ("SPXL","SPXS","SPY", "S&P500 3x (SPXL)", "2008-11-05"),
    ("TNA", "TZA", "IWM", "Russell2000 3x",   "2008-11-05"),
    ("TECL","TECS","QQQ", "Technology 3x",    "2008-12-17"),
    ("FAS", "FAZ", "XLF", "Financials 3x",   "2008-11-06"),
    ("ERX", "ERY", "XLE", "Energy 3x",        "2008-11-06"),
    ("SOXL","SOXS","SMH", "Semicon 3x",       "2010-03-11"),
    ("DRN", "DRV", "IYR", "Real Estate 3x",  "2009-07-16"),
    ("YINN","YANG","FXI", "China 3x",         "2009-12-03"),
    ("MIDU","SMDD","MDY", "Midcap400 3x",     "2010-02-11"),
    ("URTY","SRTY","IWM", "Russell2000 3x(b)","2010-02-11"),
]

und_cache={}
def get_und(sym):
    if sym not in und_cache:
        s=load(sym)
        if s is None and sym=="IWM": s=load("IWM")
        und_cache[sym]=s
    return und_cache[sym]

sec_bh=[]; sec_td=[]
for long_t,short_t,und_sym,lbl,start in SECTORS:
    ls=load(long_t); ss=load(short_t); us=get_und(und_sym)
    if ls is None: continue
    if us is None:
        if und_sym in ["SPY","QQQ","DIA"]: us={"SPY":spy,"QQQ":qqq,"DIA":dia}[und_sym]
        else: us=spy
    parts=[ls,spy]
    if ss is not None: parts.append(ss)
    sec_df=pd.concat(parts,axis=1).dropna(); sec_df=sec_df[sec_df.index>=start]
    if len(sec_df)<252: continue
    m=met(run_positions(pd.Series(long_t,index=sec_df.index),sec_df),f"{lbl} B&H")
    if m: sec_bh.append(m)
    if ss is not None and short_t in sec_df.columns:
        spy_sec=spy.reindex(sec_df.index).ffill()
        und_sec=us.reindex(sec_df.index).ffill()
        sig=td_signal(spy_sec,und_sec,long_t,short_t,sec_df)
        m2=met(run_positions(sig,sec_df),f"{lbl} TD")
        if m2: sec_td.append(m2)

sec_bh.sort(key=lambda x:-x["cagr"]); sec_td.sort(key=lambda x:-x["cagr"])
ptbl(sec_bh,"PATH 3: B&H COMPARISON -- All 3x Sector ETFs (sorted by CAGR)",ref_cagr=40.6)
ptbl(sec_td,"PATH 3: TD STRATEGY on Each Sector (sorted by CAGR)",ref_cagr=40.6)

# PATH 3b: SECTOR ROTATION
print("\n\n"+"="*118+"\n  PATH 3b: BEST-SECTOR 3x ROTATION -- hold top-momentum 3x long, TD bear short\n"+"="*118)

sec_universe=["TQQQ","SPXL","TNA","TECL","FAS","SOXL"]
sec_loaded={t:load(t) for t in sec_universe}
sec_avail={t:s for t,s in sec_loaded.items() if s is not None}
rot_df=pd.concat(list(sec_avail.values()),axis=1).dropna()
rot_df=rot_df[rot_df.index>="2010-03-11"]
print(f"  Rotation period: {str(rot_df.index[0])[:10]} to {str(rot_df.index[-1])[:10]}  ({len(rot_df)} bars)")
print(f"  Universe: {list(sec_avail.keys())}\n")

moms={t:(rot_df[t]/rot_df[t].shift(21)-1).shift(1) for t in sec_avail}

sig=pd.Series("TQQQ",index=rot_df.index)
for dt in rot_df.index:
    scores={t:float(moms[t].get(dt,-999) or -999) for t in sec_avail}
    sig[dt]=max(scores,key=scores.get)
r1=met(run_positions(sig,rot_df),"Sector Rotation (best 21d, no filter)")

spy_ab200=(spy>sma(spy,200)).shift(1).fillna(False)
rot_df2=rot_df.copy(); rot_df2["CASH"]=0
sig2=pd.Series("CASH",index=rot_df.index)
for dt in rot_df.index:
    if spy_ab200.get(dt,False):
        scores={t:float(moms[t].get(dt,-999) or -999) for t in sec_avail}
        sig2[dt]=max(scores,key=scores.get)
r2=met(run_positions(sig2,rot_df2),"Sector Rotation + SPY SMA200 gate")

spy_rot=spy.reindex(rot_df.index).ffill()
qqq_rot=qqq.reindex(rot_df.index).ffill()
stk_rot=bear_streak(spy_rot)
m21_rot=(qqq_rot/qqq_rot.shift(21)-1).shift(1)
m63_rot=(qqq_rot/qqq_rot.shift(63)-1).shift(1)
m5_rot=(qqq_rot/qqq_rot.shift(5)-1).shift(1)
ab50_rot=(spy_rot>sma(spy_rot,50)).shift(1).fillna(False)
sqqq_vol=rvol(sqqq.reindex(rot_df.index).ffill(),20).shift(1)
rot_df3=rot_df.copy()
for t in ["SQQQ","CASH"]:
    if t not in rot_df3.columns:
        if t=="SQQQ" and sqqq is not None: rot_df3["SQQQ"]=sqqq.reindex(rot_df.index).ffill()
        else: rot_df3["CASH"]=0
sig3=pd.Series("TQQQ",index=rot_df.index); ins=False
for dt in rot_df.index:
    if ins and (ab50_rot.get(dt,False) or float(m5_rot.get(dt,0) or 0)>0.05): ins=False
    if not ins and int(stk_rot.get(dt,0))>=15 and float(m21_rot.get(dt,0) or 0)<-0.12 and float(m63_rot.get(dt,0) or 0)<-0.05: ins=True
    if ins:
        v=float(sqqq_vol.get(dt,0) or 0)
        sig3[dt]="CASH" if v>0.90 else "SQQQ"
    else:
        scores={t:float(moms[t].get(dt,-999) or -999) for t in sec_avail}
        sig3[dt]=max(scores,key=scores.get)
r3=met(run_positions(sig3,rot_df3),"Sector Rotation + TD Bear SQQQ")

r0=met(run_positions(pd.Series("TQQQ",index=rot_df.index),rot_df),"TQQQ B&H (2010-2026)")
ptbl([r0,r1,r2,r3],"PATH 3b: SECTOR ROTATION STRATEGIES",ref_cagr=r0["cagr"])
atbl({"TQQQ B&H":r0,"Rotation":r1,"Rot+SMA":r2,"Rot+TD":r3},"PATH 3b ANNUAL RETURNS")

print("\n\n=== ALL PATHS COMPLETE ===")
