#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

DATA_DIR=r'D:/Experiments/norgate_data/US_Equities'
COST_BPS=0.0005

def load(ticker):
    p=os.path.join(DATA_DIR,ticker+'.parquet')
    df=pd.read_parquet(p)[['Close']].rename(columns={'Close':ticker})
    return df

frames=[load(t) for t in ['UPRO','SPXU','TQQQ','SQQQ','UDOW','SDOW','SPY','QQQ','DIA']]
data=pd.concat(frames,axis=1).dropna()
data=data[data.index>='2010-02-11']
print(f'Data: {data.index[0].date()} to {data.index[-1].date()}  ({len(data)} trading days)')

def sma(series,n): return series.rolling(n,min_periods=n).mean()
def realized_vol(r,n=20): return r.rolling(n,min_periods=n).std()*np.sqrt(252)

def run_strategy(ss,prices,label,cost=COST_BPS):
    equity=1.0; prev='CASH'; eq=[]; dtl=[]
    for dt,pos in ss.items():
        if dt not in prices.index: continue
        tc=cost if pos!=prev else 0.0
        if pos=='CASH': ret=0.0
        else:
            idx=prices.index.get_loc(dt)
            if idx==0: ret=0.0
            else: ret=(prices[pos].iloc[idx]/prices[pos].iloc[idx-1])-1.0
        equity*=(1+ret-tc); prev=pos; eq.append(equity); dtl.append(dt)
    return pd.Series(eq,index=dtl,name=label)

def metrics(eq_curve,label):
    rets=eq_curve.pct_change().dropna()
    ny=len(eq_curve)/252
    cagr=(eq_curve.iloc[-1]**(1/ny))-1
    sh=(rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0
    rm=eq_curve.cummax(); dd=(eq_curve-rm)/rm; mdd=dd.min()
    calmar=cagr/abs(mdd) if mdd!=0 else 0
    ann=eq_curve.resample('YE').last().pct_change().dropna()*100
    ann.iloc[0]=(eq_curve.resample('YE').last().iloc[0]-1.0)*100
    pp=(ann>0).sum()/len(ann)*100 if len(ann)>0 else 0
    return {'label':label,'cagr':cagr*100,'sharpe':sh,'max_dd':mdd*100,'calmar':calmar,'pct_win_yrs':pp,'ann':ann}

close=data.copy()
spy_sma200=sma(close['SPY'],200)
spy_sma100=sma(close['SPY'],100)
qqq_sma200=sma(close['QQQ'],200)
qqq_sma100=sma(close['QQQ'],100)
dia_sma200=sma(close['DIA'],200)
dia_sma100=sma(close['DIA'],100)
mom_spy=close['SPY']/close['SPY'].shift(21)-1
mom_qqq=close['QQQ']/close['QQQ'].shift(21)-1
mom_dia=close['DIA']/close['DIA'].shift(21)-1
rv=realized_vol
vol_upro=rv(close['UPRO'].pct_change())
vol_tqqq=rv(close['TQQQ'].pct_change())
vol_udow=rv(close['UDOW'].pct_change())
vol_spxu=rv(close['SPXU'].pct_change())
vol_sqqq=rv(close['SQQQ'].pct_change())
vol_sdow=rv(close['SDOW'].pct_change())
IM={'SP':('UPRO','SPXU',vol_upro,vol_spxu),'NDQ':('TQQQ','SQQQ',vol_tqqq,vol_sqqq),'DOW':('UDOW','SDOW',vol_udow,vol_sdow)}

vd=close.index[close.index>=close.dropna().index[0]]
signals={k:pd.Series('CASH',index=vd) for k in ['S0','S1','S2','S3','S4','S5']}
signals['S0'][::] = 'SPY'
signals['S1'][::] = 'TQQQ'
ph4,ph5=0,0; MH=5; pp4,pp5='CASH','CASH'

for i,dt in enumerate(vd):
    if dt not in close.index: continue
    if pd.isna(spy_sma200[dt]) or pd.isna(mom_spy[dt]): continue
    msp=float(mom_spy[dt])
    mndq=float(mom_qqq[dt]) if not pd.isna(mom_qqq[dt]) else -999
    mdow=float(mom_dia[dt]) if not pd.isna(mom_dia[dt]) else -999
    s200=close['SPY'][dt]>spy_sma200[dt]
    n200=close['QQQ'][dt]>qqq_sma200[dt] if not pd.isna(qqq_sma200[dt]) else False
    d200=close['DIA'][dt]>dia_sma200[dt] if not pd.isna(dia_sma200[dt]) else False
    s100=close['SPY'][dt]>spy_sma100[dt] if not pd.isna(spy_sma100[dt]) else False
    n100=close['QQQ'][dt]>qqq_sma100[dt] if not pd.isna(qqq_sma100[dt]) else False
    d100=close['DIA'][dt]>dia_sma100[dt] if not pd.isna(dia_sma100[dt]) else False
    mo={'SP':msp,'NDQ':mndq,'DOW':mdow}
    u200={'SP':s200,'NDQ':n200,'DOW':d200}
    u100={'SP':s100,'NDQ':n100,'DOW':d100}
    signals['S2'][dt]='TQQQ' if n200 else 'CASH'
    if s200 or n200 or d200:
        el={k:v for k,v in mo.items() if u200[k]}
        signals['S3'][dt]=IM[max(el,key=el.get)][0] if el else 'CASH'
    else: signals['S3'][dt]='CASH'
    ph4=max(0,ph4-1)
    if ph4>0: signals['S4'][dt]=pp4
    else:
        if s200:
            el={k:v for k,v in mo.items() if u200[k]}
            if el:
                b=max(el,key=el.get); etfl,_,vl,_=IM[b]
                v=float(vl[dt]) if not pd.isna(vl[dt]) else 0
                ns4='CASH' if v>0.85 else etfl
            else: ns4='CASH'
        else:
            w=min(mo,key=mo.get); _,etfs,_,vs=IM[w]
            v=float(vs[dt]) if not pd.isna(vs[dt]) else 0
            ns4='CASH' if v>0.85 else etfs
        if ns4!=pp4: ph4=MH
        signals['S4'][dt]=ns4; pp4=ns4
    ph5=max(0,ph5-1)
    if ph5>0: signals['S5'][dt]=pp5
    else:
        nt=sum([s100,n100,d100]); VC=0.80
        if nt>=2:
            el={k:v for k,v in mo.items() if u100[k]}
            b=max(el,key=el.get); etfl,_,vl,_=IM[b]
            v=float(vl[dt]) if not pd.isna(vl[dt]) else 0
            ns5='CASH' if v>VC else etfl
        elif nt==1:
            el={k:v for k,v in mo.items() if u100[k]}
            b=max(el,key=el.get); etfl,_,vl,_=IM[b]
            v=float(vl[dt]) if not pd.isna(vl[dt]) else 0
            if u200.get(b,False): ns5='CASH' if v>VC else etfl
            else: ns5='CASH'
        else:
            w=min(mo,key=mo.get); _,etfs,_,vs=IM[w]
            v=float(vs[dt]) if not pd.isna(vs[dt]) else 0
            ns5='CASH' if v>VC else etfs
        if ns5!=pp5: ph5=MH
        signals['S5'][dt]=ns5; pp5=ns5

results={}
for key,sig in signals.items():
    LBL={'S0':'SPY B&H','S1':'TQQQ B&H','S2':'TQQQ/Cash (QQQ>200)','S3':'3-Way Long Rotation','S4':'Long/Short Rotation','S5':'Full ALR (v9 LETF)'}
    eq=run_strategy(sig,close,LBL[key])
    results[key]=(eq,metrics(eq,LBL[key]))

SEP=chr(61)*100
print(chr(10)+SEP)
print('STRATEGY COMPARISON — AlphaGo Leveraged Rotation (ALR)')
print('Period: '+str(data.index[0].date())+' — '+str(data.index[-1].date())+'   |   Costs: '+str(int(COST_BPS*10000))+'bps/trade one-way')
print(SEP)
print('%-28s  %7s  %7s  %8s  %7s  %7s  %s' % ('Strategy','CAGR','Sharpe','MaxDD','Calmar','WinYrs',chr(36)+'10k->'))
print(chr(45)*100)
for key in ['S0','S1','S2','S3','S4','S5']:
    eq,m=results[key]
    final=eq.iloc[-1]*10_000
    lbl=m['label']; cg=m['cagr']; sh=m['sharpe']; md=m['max_dd']; cl=m['calmar']; pw=m['pct_win_yrs']
    row='  %-26s  %6.1f%%  %7.2f  %7.1f%%  %7.2f  %6.0f%%' % (lbl,cg,sh,md,cl,pw)
    row+='  '+chr(36)+'%-10s' % '{:,.0f}'.format(final)
    print(row)

print(chr(10)+SEP)
print('YEAR-BY-YEAR RETURNS (%)')
print(SEP)
all_years=sorted(set().union(*[set(results[k][1]['ann'].index.year) for k in results]))
sk=['S0','S1','S2','S3','S4','S5']
ss=['SPY','TQQQ B&H','TQQQ/Cash','3-Way-L','L/S-Rot','ALR-Full']
print('  %-6s' % 'Year' + ''.join('  %10s' % x for x in ss))
print('  ' + '-'*6 + ('  ' + '-'*10) * len(sk))
for yr in all_years:
    row='  %-6s' % yr
    for key in sk:
        ann=results[key][1]['ann']
        ya=ann[ann.index.year==yr]
        if len(ya)>0: row+='  %+9.1f%%' % ya.iloc[0]
        else: row+='  %10s' % 'N/A'
    print(row)
print('  ' + '-'*6 + ('  ' + '-'*10) * len(sk))
for rname in ['Total','CAGR/yr']:
    row='  %-6s' % rname
    for key in sk:
        ann=results[key][1]['ann']
        vals=ann.values/100; comp=1.0
        for v in vals: comp*=(1+v)
        if rname=='Total': res=(comp-1.0)*100
        else: res=((comp**(1.0/len(vals)))-1.0)*100 if len(vals)>0 else 0
        row+='  %+9.1f%%' % res
    print(row)

print(chr(10)+'  TRADE COUNTS (signal changes):')
for key in sk[2:]:
    sig=signals[key]
    trades=(sig!=sig.shift()).sum()
    lbl=results[key][1]['label']
    print('    '+lbl+': '+str(trades)+' signal changes')

print(chr(10)+'  NOTE: Costs = '+str(int(COST_BPS*10000))+'bps per ONE-WAY trade (buy or sell).')
print('  TQQQ/UPRO/UDOW: launched Feb 2010. Backtest period: ~16 years.')
print('  Cash = 0% return (no T-bill credit). This understates real returns slightly.')
print(SEP)
