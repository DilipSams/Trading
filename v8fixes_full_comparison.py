"""
V7 vs V8+Fixes A/B/C  -  Full sector comparison with ALL parameters.

V7  : results_v7.json   per_symbol  (score=14.502, 202 symbols, confirmed)
V8+ : step_decisions.csv            (Feb-23 17:16 run, freshly trained V8+fixes)
      Confirmed V8+fixes: EOG 29->9 trades, PnL -$2,430->+$715

Parameters shown:
  TotalPnL  AvgPnL  Sharpe  MaxDD  AvgTrades  Win%

PnL   : portfolio[-1] - portfolio[0]
Sharpe: annualised from daily portfolio returns; clipped to [-10,10] to
        avoid near-zero-variance artefacts when symbol stays FLAT for long stretches.
MaxDD : max peak-to-trough on portfolio NAV (%).
"""

import pandas as pd
import numpy as np
import json

SECTOR = {
    'COP':'Energy','EOG':'Energy','CVX':'Energy','XOM':'Energy','OXY':'Energy',
    'BKR':'Energy','SLB':'Energy','PSX':'Energy','MPC':'Energy','VLO':'Energy',
    'WMB':'Energy','ET':'Energy','EPD':'Energy','MPLX':'Energy',
    'JNJ':'Healthcare','UNH':'Healthcare','ABBV':'Healthcare','MRK':'Healthcare',
    'PFE':'Healthcare','BMY':'Healthcare','LLY':'Healthcare','AMGN':'Healthcare',
    'GILD':'Healthcare','TMO':'Healthcare','DHR':'Healthcare','ABT':'Healthcare',
    'SYK':'Healthcare','BDX':'Healthcare','ELV':'Healthcare','VRTX':'Healthcare',
    'ZTS':'Healthcare','REGN':'Healthcare','MCK':'Healthcare','ISRG':'Healthcare',
    'COR':'Healthcare','BSX':'Healthcare','CI':'Healthcare','HCA':'Healthcare',
    'PEP':'Staples','KO':'Staples','MCD':'Staples','MDLZ':'Staples','MNST':'Staples',
    'WMT':'Staples','COST':'Staples','PG':'Staples','CL':'Staples','MO':'Staples',
    'SBUX':'Staples',
    'VZ':'Telecom','T':'Telecom','TMUS':'Telecom',
    'NEE':'Utilities','DUK':'Utilities','SO':'Utilities','AEP':'Utilities',
    'SRE':'Utilities','D':'Utilities',
    'JPM':'Financials','BAC':'Financials','WFC':'Financials','GS':'Financials',
    'MS':'Financials','BK':'Financials','USB':'Financials','TRV':'Financials',
    'AXP':'Financials','BLK':'Financials','SCHW':'Financials','CB':'Financials',
    'APO':'Financials','AFL':'Financials','PNC':'Financials','TFC':'Financials',
    'COF':'Financials','C':'Financials','MCO':'Financials','SPGI':'Financials',
    'ICE':'Financials','CME':'Financials','KKR':'Financials','BX':'Financials',
    'IBKR':'Financials',
    'CAT':'Industrials','HON':'Industrials','UNP':'Industrials','UPS':'Industrials',
    'CSX':'Industrials','NSC':'Industrials','GE':'Industrials','MMM':'Industrials',
    'ITW':'Industrials','EMR':'Industrials','WM':'Industrials','CMI':'Industrials',
    'URI':'Industrials','CTAS':'Industrials','HWM':'Industrials','AJG':'Industrials',
    'GD':'Industrials','LMT':'Industrials','RTX':'Industrials','NOC':'Industrials',
    'LHX':'Industrials','FDX':'Industrials','PH':'Industrials','PCAR':'Industrials',
    'RSG':'Industrials','PWR':'Industrials','GEV':'Industrials',
    'AMZN':'ConsDisc','TSLA':'ConsDisc','NKE':'ConsDisc','TJX':'ConsDisc',
    'HD':'ConsDisc','LOW':'ConsDisc','BKNG':'ConsDisc','DIS':'ConsDisc',
    'NFLX':'ConsDisc','AZO':'ConsDisc','ORLY':'ConsDisc','F':'ConsDisc',
    'GM':'ConsDisc','RCL':'ConsDisc','MAR':'ConsDisc','HLT':'ConsDisc',
    'CVNA':'ConsDisc',
    'AAPL':'Tech','MSFT':'Tech','NVDA':'Tech','GOOGL':'Tech','META':'Tech',
    'AVGO':'Tech','ORCL':'Tech','CRM':'Tech','AMD':'Tech','QCOM':'Tech',
    'TXN':'Tech','ADBE':'Tech','AMAT':'Tech','MU':'Tech','LRCX':'Tech',
    'KLAC':'Tech','ADI':'Tech','INTC':'Tech','IBM':'Tech','ACN':'Tech',
    'NOW':'Tech','INTU':'Tech','PANW':'Tech','ANET':'Tech','FTNT':'Tech',
    'CSCO':'Tech','CDNS':'Tech','SNPS':'Tech','MPWR':'Tech','DELL':'Tech',
    'WDC':'Tech','MSI':'Tech','MRVL':'Tech','CRWD':'Tech','NET':'Tech',
    'APP':'Tech','PLTR':'Tech','SNOW':'Tech',
    'FCX':'Materials','NEM':'Materials','APD':'Materials','LIN':'Materials',
    'SCCO':'Materials','GLW':'Materials','ECL':'Materials',
    'AMT':'REITs','PLD':'REITs','WELL':'REITs','O':'REITs','SPG':'REITs',
    'EQIX':'REITs','DLR':'REITs',
    'V':'Payments','MA':'Payments','PYPL':'Payments',
}

SECTOR_ORDER = ['Industrials','Materials','Financials','Tech','ConsDisc',
                'Healthcare','Utilities','REITs','Energy','Staples','Telecom','Payments','Other']

# ── helpers ──────────────────────────────────────────────────────────────────

def sharpe_from_portfolio(port: np.ndarray, clip: float = 10.0) -> float:
    """Annualised Sharpe from daily portfolio values. Clipped to [-clip, clip]."""
    if len(port) < 2:
        return 0.0
    rets = np.diff(port) / np.where(port[:-1] != 0, port[:-1], 1.0)
    std = rets.std()
    if std < 1e-8:
        return 0.0
    sh = float(rets.mean() / std * np.sqrt(252))
    return float(np.clip(sh, -clip, clip))

def maxdd_from_portfolio(port: np.ndarray) -> float:
    """Max drawdown (%) from portfolio value series."""
    if len(port) < 2:
        return 0.0
    peak = port[0]; mdd = 0.0
    for v in port:
        if v > peak:
            peak = v
        dd = (v - peak) / peak * 100.0
        if dd < mdd:
            mdd = dd
    return round(mdd, 1)

# ── Load V7 ──────────────────────────────────────────────────────────────────
print("Loading V7 …")
with open(r'C:\Users\dilip\alpha_trade_v2_results\results_v7.json',
          encoding='utf-8', errors='replace') as f:
    d = json.load(f)

v7_rows = []
for sym_key, data in d['per_symbol'].items():
    sym = sym_key.replace('_1d', '')
    pnl = data.get('pnl', 0.0)
    # results_v7.json stores max_dd as a negative % (e.g. -12.8); abs() for display
    raw_mdd = data.get('max_dd', 0.0)
    mdd = abs(raw_mdd) if raw_mdd != 0.0 else 0.0
    v7_rows.append({
        'symbol':  sym,
        'sector':  SECTOR.get(sym, 'Other'),
        'pnl':     pnl,
        'trades':  data.get('trades', 0),
        'sharpe':  float(np.clip(data.get('sharpe', 0.0), -10, 10)),
        'mdd':     mdd,
        'win':     pnl > 0,
    })
v7 = pd.DataFrame(v7_rows)
print(f"  V7 : {len(v7)} symbols | PnL ${v7.pnl.sum():,.0f} | "
      f"Win {100*v7.win.mean():.1f}% | Sharpe {v7.sharpe.mean():.3f}")

# ── Load V8+Fixes ─────────────────────────────────────────────────────────────
print("Loading V8+Fixes …")
sd = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\step_decisions.csv',
                 low_memory=False)

v8_rows = []
for sym_key in sd.symbol.unique():
    sub  = sd[sd.symbol == sym_key].reset_index(drop=True)
    sym  = sym_key.replace('_1d', '')
    port = sub.portfolio.values

    pnl    = port[-1] - port[0]
    trades = int((sub.position != sub.position.shift(1)).sum())
    sh     = sharpe_from_portfolio(port)
    mdd    = abs(maxdd_from_portfolio(port))   # store as positive %

    v8_rows.append({
        'symbol': sym,
        'sector': SECTOR.get(sym, 'Other'),
        'pnl':    pnl,
        'trades': trades,
        'sharpe': sh,
        'mdd':    mdd,
        'win':    pnl > 0,
    })
v8 = pd.DataFrame(v8_rows)
print(f"  V8+: {len(v8)} symbols | PnL ${v8.pnl.sum():,.0f} | "
      f"Win {100*v8.win.mean():.1f}% | Sharpe {v8.sharpe.mean():.3f}")

# ── Merge on common symbols ───────────────────────────────────────────────────
m = v7.merge(v8, on='symbol', suffixes=('_v7','_v8'))
m['sector'] = m['sector_v7']
print(f"\n  Merged: {len(m)} common symbols\n")

# ── Sector aggregation helper ─────────────────────────────────────────────────
def sec_stats(df, sfx):
    g = df.groupby('sector')
    return pd.DataFrame({
        'n':       g['symbol'].count(),
        'total':   g[f'pnl{sfx}'].sum().round(0),
        'avg_pnl': g[f'pnl{sfx}'].mean().round(0),
        'sharpe':  g[f'sharpe{sfx}'].mean().round(2),
        'mdd':     g[f'mdd{sfx}'].mean().round(1),
        'trades':  g[f'trades{sfx}'].mean().round(1),
        'win':    (g[f'win{sfx}'].mean() * 100).round(0),
    }).reindex([s for s in SECTOR_ORDER if s in g.groups])

v7s = sec_stats(m, '_v7')
v8s = sec_stats(m, '_v8')

# ── Printing helpers ──────────────────────────────────────────────────────────
W  = 105
HDR = f"{'Sector':<14} {'#':>2}  {'TotalPnL':>11}  {'AvgPnL':>8}  " \
      f"{'Sharpe':>7}  {'MaxDD':>6}  {'Trades':>7}  {'Win%':>5}"
SEP = '-' * W

def print_row(name, n, total, avg, sh, mdd, tr, win):
    print(f"{name:<14} {int(n):>2}  ${total:>+10,.0f}  ${avg:>+7,.0f}  "
          f"{sh:>7.2f}  {mdd:>5.1f}%  {tr:>7.1f}  {win:>4.0f}%")

def print_table(sec_df, label, sym_df, sfx):
    """sfx is '_v7' or '_v8'."""
    print('=' * W)
    print(f"  {label}  |  OOS Aug 2022 - Feb 2026 (3.5 yrs)  |  $10k / symbol")
    print('=' * W)
    print(HDR)
    print(SEP)
    for sec_name in SECTOR_ORDER:
        if sec_name not in sec_df.index:
            continue
        r = sec_df.loc[sec_name]
        print_row(sec_name, r['n'], r['total'], r['avg_pnl'],
                  r['sharpe'], r['mdd'], r['trades'], r['win'])
    print(SEP)
    print_row('OVERALL', len(sym_df),
              sym_df[f'pnl{sfx}'].sum(),
              sym_df[f'pnl{sfx}'].mean(),
              sym_df[f'sharpe{sfx}'].mean(),
              sym_df[f'mdd{sfx}'].mean(),
              sym_df[f'trades{sfx}'].mean(),
              sym_df[f'win{sfx}'].mean() * 100)
    print('=' * W)
    print()

m2 = m.copy()

print_table(v7s, 'V7 BASELINE SECTOR PERFORMANCE',      m2, '_v7')
print_table(v8s, 'V8+FIXES A/B/C SECTOR PERFORMANCE',   m2, '_v8')

# ── Delta table ───────────────────────────────────────────────────────────────
print('=' * W)
print('  DELTA  (V8+Fixes minus V7)  |  ^ = improvement  v = degradation')
print('=' * W)
print(f"{'Sector':<14} {'#':>2}  {'dTotalPnL':>11}  {'dAvgPnL':>8}  "
      f"{'dSharpe':>7}  {'dMaxDD':>6}  {'dTrades':>7}  {'dWin%':>5}")
print(SEP)
for sec_name in SECTOR_ORDER:
    if sec_name not in v7s.index or sec_name not in v8s.index:
        continue
    r7 = v7s.loc[sec_name]
    r8 = v8s.loc[sec_name]
    n  = int(r7['n'])
    dt = r8['total']   - r7['total']
    da = r8['avg_pnl'] - r7['avg_pnl']
    ds = r8['sharpe']  - r7['sharpe']
    dm = r8['mdd']     - r7['mdd']
    dtr= r8['trades']  - r7['trades']
    dw = r8['win']     - r7['win']
    pa = '^' if da > 50  else ('v' if da < -50  else ' ')
    sa = '^' if ds > 0.05 else ('v' if ds < -0.05 else ' ')
    ta = 'v' if dtr < -2 else ('^' if dtr > 2 else ' ')
    print(f"{sec_name:<14} {n:>2}  {pa}${dt:>+9,.0f}  {pa}${da:>+6,.0f}  "
          f"{sa}{ds:>+6.2f}  {dm:>+5.1f}%  {ta}{dtr:>+6.1f}  {dw:>+4.0f}%")
print(SEP)
dp_tot = m2.pnl_v8.sum()  - m2.pnl_v7.sum()
da_tot = m2.pnl_v8.mean() - m2.pnl_v7.mean()
ds_tot = m2.sharpe_v8.mean() - m2.sharpe_v7.mean()
dm_tot = m2.mdd_v8.mean()    - m2.mdd_v7.mean()
dtr_tot= m2.trades_v8.mean() - m2.trades_v7.mean()
dw_tot = (m2.win_v8.mean()   - m2.win_v7.mean()) * 100
print(f"{'OVERALL':<14} {len(m2):>2}  ${dp_tot:>+10,.0f}  ${da_tot:>+6,.0f}  "
      f"{ds_tot:>+6.2f}  {dm_tot:>+5.1f}%  {dtr_tot:>+6.1f}  {dw_tot:>+4.0f}%")
print('=' * W)

# ── Headline summary ──────────────────────────────────────────────────────────
print()
print("HEADLINE SUMMARY")
print(f"  {'':12}  {'Score':>8}  {'TotalPnL':>12}  {'AvgPnL':>8}  "
      f"{'Sharpe':>7}  {'MaxDD':>6}  {'Trades':>7}  {'Win%':>5}")
print(f"  {'V7 Baseline':<12}  {'14.50':>8}  ${m2.pnl_v7.sum():>+11,.0f}  "
      f"${m2.pnl_v7.mean():>+7,.0f}  {m2.sharpe_v7.mean():>7.3f}  "
      f"{m2.mdd_v7.mean():>5.1f}%  {m2.trades_v7.mean():>7.1f}  "
      f"{100*m2.win_v7.mean():>4.0f}%")
print(f"  {'V8+Fixes':<12}  {'183.56':>8}  ${m2.pnl_v8.sum():>+11,.0f}  "
      f"${m2.pnl_v8.mean():>+7,.0f}  {m2.sharpe_v8.mean():>7.3f}  "
      f"{m2.mdd_v8.mean():>5.1f}%  {m2.trades_v8.mean():>7.1f}  "
      f"{100*m2.win_v8.mean():>4.0f}%")
print(f"  {'Delta':<12}  {'+12.7x':>8}  ${dp_tot:>+11,.0f}  "
      f"${da_tot:>+7,.0f}  {ds_tot:>+7.3f}  "
      f"{dm_tot:>+5.1f}%  {dtr_tot:>+6.1f}  {dw_tot:>+4.0f}%")
print()

# ── Top/bottom movers ────────────────────────────────────────────────────────
m2['d_pnl'] = m2.pnl_v8 - m2.pnl_v7
print("TOP 5 SYMBOLS IMPROVED (dPnL):")
for _, r in m2.nlargest(5,'d_pnl').iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12})  "
          f"V7 ${r['pnl_v7']:>+7,.0f} Sh{r['sharpe_v7']:>+.2f} Tr{r['trades_v7']:.0f}  ->  "
          f"V8 ${r['pnl_v8']:>+7,.0f} Sh{r['sharpe_v8']:>+.2f} Tr{r['trades_v8']:.0f}  "
          f"d${r['d_pnl']:>+7,.0f}")
print()
print("TOP 5 SYMBOLS HURT (dPnL):")
for _, r in m2.nsmallest(5,'d_pnl').iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12})  "
          f"V7 ${r['pnl_v7']:>+7,.0f} Sh{r['sharpe_v7']:>+.2f} Tr{r['trades_v7']:.0f}  ->  "
          f"V8 ${r['pnl_v8']:>+7,.0f} Sh{r['sharpe_v8']:>+.2f} Tr{r['trades_v8']:.0f}  "
          f"d${r['d_pnl']:>+7,.0f}")
print()

# ── Fix C anti-churn impact on over-trading sectors ──────────────────────────
print("FIX C ANTI-CHURN - OVER-TRADING SECTOR DETAIL:")
print(f"  {'Sector':<14}  {'V7 Tr':>6}  {'V8 Tr':>6}  {'dTr':>5}  "
      f"{'V7 PnL':>9}  {'V8 PnL':>9}  {'dPnL':>8}  {'V7 Win':>7}  {'V8 Win':>7}")
for sec in ['Energy','Utilities','Staples','Healthcare','Industrials']:
    sub = m2[m2.sector == sec]
    if len(sub) == 0:
        continue
    print(f"  {sec:<14}  {sub.trades_v7.mean():>6.1f}  {sub.trades_v8.mean():>6.1f}  "
          f"{(sub.trades_v8-sub.trades_v7).mean():>+5.1f}  "
          f"${sub.pnl_v7.mean():>+8,.0f}  ${sub.pnl_v8.mean():>+8,.0f}  "
          f"${(sub.pnl_v8-sub.pnl_v7).mean():>+7,.0f}  "
          f"{100*(sub.pnl_v7>0).mean():>6.0f}%  {100*(sub.pnl_v8>0).mean():>6.0f}%")
