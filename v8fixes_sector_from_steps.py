"""
Compute V8+Fixes A/B/C per-symbol sector results from step_decisions.csv.
The step_decisions.csv is from the V8+Fixes run (confirmed: EOG final=$10,716 vs V7's $7,570).
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

# Load step_decisions.csv
print("Loading step_decisions.csv...")
sd = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\step_decisions.csv', low_memory=False)
print(f"  {len(sd):,} rows, {sd.symbol.nunique()} symbols")

# Compute per-symbol stats
rows = []
for sym_key in sorted(sd.symbol.unique()):
    sub = sd[sd.symbol == sym_key].copy().reset_index(drop=True)
    sym = sym_key.replace('_1d', '')

    # PnL = final portfolio - initial ($10,000)
    pnl = sub.portfolio.iloc[-1] - 10_000.0

    # Daily returns from portfolio
    port = sub.portfolio.values
    rets = np.diff(port) / port[:-1]

    # Sharpe
    if len(rets) > 1 and rets.std() > 0:
        sh = float(rets.mean() / rets.std() * np.sqrt(252))
    else:
        sh = 0.0

    # Max drawdown
    nav = 1.0; peak = 1.0; mdd = 0.0
    for r in rets:
        nav *= (1 + r)
        if nav > peak: peak = nav
        mdd = min(mdd, (nav - peak) / peak * 100)

    # Position changes (trades)
    pos_changes = int((sub.position != sub.position.shift(1)).sum())

    rows.append({
        'symbol': sym,
        'sector': SECTOR.get(sym, 'Other'),
        'pnl': pnl,
        'trades': pos_changes,
        'sharpe': round(sh, 2),
        'mdd': round(mdd, 1),
        'pos': pnl > 0,
    })

df = pd.DataFrame(rows)
print(f"Computed stats for {len(df)} symbols")
print(f"Total PnL: ${df.pnl.sum():,.0f}")
print(f"Avg Sharpe: {df.sharpe.mean():.3f}")
print(f"Win rate: {100*df.pos.mean():.1f}%")
print()

# Also load V7 for comparison
with open(r'C:\Users\dilip\alpha_trade_v2_results\results_v7.json', encoding='utf-8', errors='replace') as f:
    d = json.load(f)
v7_per_sym = d['per_symbol']
v7_rows = []
for sym_key, data in v7_per_sym.items():
    sym = sym_key.replace('_1d', '')
    v7_rows.append({
        'symbol': sym,
        'v7_pnl': data.get('pnl', 0),
        'v7_sharpe': data.get('sharpe', 0),
        'v7_trades': data.get('trades', 0),
        'v7_mdd': data.get('max_dd', 0),
    })
v7df = pd.DataFrame(v7_rows)

# Merge
merged = df.merge(v7df, on='symbol', how='inner')
merged['d_pnl'] = merged['pnl'] - merged['v7_pnl']
merged['d_sharpe'] = merged['sharpe'] - merged['v7_sharpe']
merged['d_trades'] = merged['trades'] - merged['v7_trades']

# Sector comparison
SECTOR_ORDER = ['Industrials','Materials','Financials','Tech','ConsDisc',
                'Healthcare','Utilities','REITs','Energy','Staples','Telecom','Payments','Other']

g_v8 = df.groupby('sector')
g_m  = merged.groupby('sector')

sec = pd.DataFrame({
    'n':           g_v8['symbol'].count(),
    'v8_pnl':      g_v8['pnl'].sum().round(0),
    'v8_avg_pnl':  g_v8['pnl'].mean().round(0),
    'v8_sharpe':   g_v8['sharpe'].mean().round(2),
    'v8_mdd':      g_v8['mdd'].mean().round(1),
    'v8_trades':   g_v8['trades'].mean().round(1),
    'v8_win':      (g_v8['pos'].mean()*100).round(0),
    'd_pnl':       g_m['d_pnl'].mean().round(0),
    'd_sharpe':    g_m['d_sharpe'].mean().round(2),
    'd_trades':    g_m['d_trades'].mean().round(1),
}).reindex([s for s in SECTOR_ORDER if s in g_v8.groups])

# V7 sector for comparison
g_v7 = merged.groupby('sector')
v7_sec = pd.DataFrame({
    'v7_pnl':     g_v7['v7_pnl'].sum().round(0),
    'v7_avg_pnl': g_v7['v7_pnl'].mean().round(0),
    'v7_sharpe':  g_v7['v7_sharpe'].mean().round(2),
    'v7_trades':  g_v7['v7_trades'].mean().round(1),
    'v7_win':     (g_v7['v7_pnl'].apply(lambda x: 100*(x>0).mean())).round(0),
}).reindex([s for s in SECTOR_ORDER if s in g_v7.groups])

combined = v7_sec.join(sec)

W = 130
print('='*W)
print('  V7 BASELINE  vs  V8+Fixes A/B/C  |  OOS Aug 2022-Feb 2026  |  $10k/symbol')
print('  Fix A: golden_cross 0.80→0.92 | Fix B: sticky-max lock | Fix C: 21-bar anti-churn')
print('='*W)
print(f"{'Sector':<14} {'#':>2}"
      f" | {'V7 PnL':>9} {'V7 Sh':>6} {'V7 Tr':>5} {'V7 Win':>6}"
      f" | {'V8 PnL':>9} {'V8 Sh':>6} {'V8 Tr':>5} {'V8 Win':>6}"
      f" | {'ΔPnL':>8} {'ΔSh':>6} {'ΔTr':>5}")
print('-'*W)

for sec_name in SECTOR_ORDER:
    if sec_name not in combined.index:
        continue
    r = combined.loc[sec_name]
    n = int(r.get('n', 0))
    pa = '▲' if r['d_pnl'] > 50 else ('▼' if r['d_pnl'] < -50 else ' ')
    sa = '▲' if r['d_sharpe'] > 0.03 else ('▼' if r['d_sharpe'] < -0.03 else ' ')
    print(f"{sec_name:<14} {n:>2}"
          f" | ${r['v7_avg_pnl']:>+7,.0f} {r['v7_sharpe']:>6.2f} {r['v7_trades']:>5.1f} {r['v7_win']:>5.0f}%"
          f" | ${r['v8_avg_pnl']:>+7,.0f} {r['v8_sharpe']:>6.2f} {r['v8_trades']:>5.1f} {r['v8_win']:>5.0f}%"
          f" | {pa}${r['d_pnl']:>+6,.0f} {sa}{r['d_sharpe']:>+5.2f} {r['d_trades']:>+5.1f}")

print('-'*W)
n_tot = len(merged)
v7_tot_avg = merged.v7_pnl.mean()
v8_tot_avg = df.pnl.mean()
print(f"{'OVERALL':<14} {n_tot:>2}"
      f" | ${v7_tot_avg:>+7,.0f} {merged.v7_sharpe.mean():>6.2f} {merged.v7_trades.mean():>5.1f} {100*(merged.v7_pnl>0).mean():>5.0f}%"
      f" | ${v8_tot_avg:>+7,.0f} {df.sharpe.mean():>6.2f} {df.trades.mean():>5.1f} {100*df.pos.mean():>5.0f}%"
      f" | ${(v8_tot_avg - v7_tot_avg):>+6,.0f} {(df.sharpe.mean()-merged.v7_sharpe.mean()):>+5.2f} {(df.trades.mean()-merged.v7_trades.mean()):>+5.1f}")
print('='*W)

# Total PnL comparison
print()
print(f"  V7  Total PnL: ${merged.v7_pnl.sum():>+10,.0f}  |  Avg Sharpe: {merged.v7_sharpe.mean():.3f}  |  Win Rate: {100*(merged.v7_pnl>0).mean():.1f}%")
print(f"  V8+ Total PnL: ${df.pnl.sum():>+10,.0f}  |  Avg Sharpe: {df.sharpe.mean():.3f}  |  Win Rate: {100*df.pos.mean():.1f}%")
print(f"  Delta:         ${(df.pnl.sum()-merged.v7_pnl.sum()):>+10,.0f}  |  ΔSharpe: {(df.sharpe.mean()-merged.v7_sharpe.mean()):>+.3f}  |  ΔWin: {100*(df.pos.mean()-(merged.v7_pnl>0).mean()):>+.1f}%")

# Stock-level movers
print()
print("TOP 5 STOCKS IMPROVED BY V8+FIXES (ΔPnL):")
top5 = merged.nlargest(5, 'd_pnl')[['symbol','sector','v7_pnl','pnl','d_pnl','v7_sharpe','sharpe','v7_trades','trades']]
for _, r in top5.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) PnL ${r['v7_pnl']:>+6,.0f}→${r['pnl']:>+6,.0f}  Δ${r['d_pnl']:>+6,.0f}"
          f"  Sh {r['v7_sharpe']:>+.2f}→{r['sharpe']:>+.2f}"
          f"  Trades {r['v7_trades']:.0f}→{r['trades']:.0f}")

print()
print("TOP 5 STOCKS HURT BY V8+FIXES (ΔPnL):")
bot5 = merged.nsmallest(5, 'd_pnl')[['symbol','sector','v7_pnl','pnl','d_pnl','v7_sharpe','sharpe','v7_trades','trades']]
for _, r in bot5.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) PnL ${r['v7_pnl']:>+6,.0f}→${r['pnl']:>+6,.0f}  Δ${r['d_pnl']:>+6,.0f}"
          f"  Sh {r['v7_sharpe']:>+.2f}→{r['sharpe']:>+.2f}"
          f"  Trades {r['v7_trades']:.0f}→{r['trades']:.0f}")

# Sector-specific analysis of Fix C impact
print()
print("FIX C (ANTI-CHURN) IMPACT — OVER-TRADING SECTORS:")
for sec_name in ['Energy', 'Utilities', 'Staples', 'Healthcare']:
    sub_m = merged[merged.sector == sec_name]
    sub_d = df[df.sector == sec_name]
    if len(sub_m) == 0:
        continue
    avg_trade_reduction = sub_m.d_trades.mean()
    avg_pnl_delta = sub_m.d_pnl.mean()
    print(f"  {sec_name:<12}: avg trades {sub_m.v7_trades.mean():.1f}→{sub_d.trades.mean():.1f} ({avg_trade_reduction:>+.1f})"
          f"  avg PnL {sub_m.v7_pnl.mean():>+.0f}→{sub_d.pnl.mean():>+.0f} ({avg_pnl_delta:>+.0f})")
