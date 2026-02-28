"""V7 baseline sector summary from per_symbol data in results_v7.json."""
import json, pandas as pd

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

with open(r'C:\Users\dilip\alpha_trade_v2_results\results_v7.json', encoding='utf-8', errors='replace') as f:
    d = json.load(f)

print(f"Source: {d.get('label')} | Score: {d.get('score'):.2f} | Total PnL: ${d.get('total_pnl'):,.0f} | Trades: {d.get('total_trades')}")

per_sym = d['per_symbol']
rows = []
for sym_key, data in per_sym.items():
    sym = sym_key.replace('_1d', '')
    pnl = data.get('pnl', 0)
    trades = data.get('trades', 0)
    sh = data.get('sharpe', 0)
    dd = data.get('max_dd', 0)
    rows.append({
        'symbol': sym,
        'sector': SECTOR.get(sym, 'Other'),
        'pnl': pnl,
        'trades': trades,
        'sharpe': sh,
        'mdd': dd,
        'pos': pnl > 0
    })

df = pd.DataFrame(rows)
print(f'Symbols: {len(df)}  |  Win rate: {100*df.pos.mean():.1f}%  |  Avg Sharpe: {df.sharpe.mean():.3f}')
print()

SECTOR_ORDER = ['Industrials','Materials','Financials','Tech','ConsDisc',
                'Healthcare','Utilities','REITs','Energy','Staples','Telecom','Payments','Other']
g = df.groupby('sector')
sec = pd.DataFrame({
    'n':       g['symbol'].count(),
    'pnl':     g['pnl'].sum().round(0),
    'avg_pnl': g['pnl'].mean().round(0),
    'sharpe':  g['sharpe'].mean().round(2),
    'mdd':     g['mdd'].mean().round(1),
    'trades':  g['trades'].mean().round(1),
    'win_pct': (g['pos'].mean()*100).round(0),
}).reindex([s for s in SECTOR_ORDER if s in g.groups])

W = 95
print('='*W)
print('  V7 BASELINE SECTOR RESULTS  |  OOS Aug 2022 – Feb 2026 (3.5 yrs)  |  $10k / symbol')
print('='*W)
print(f"{'Sector':<14} {'#':>2}  {'TotalPnL':>11}  {'AvgPnL':>8}  {'Sharpe':>7}  {'MaxDD':>6}  {'Trades':>7}  {'Win%':>5}")
print('-'*W)
for sec_name in SECTOR_ORDER:
    if sec_name not in sec.index:
        continue
    r = sec.loc[sec_name]
    print(f"{sec_name:<14} {int(r['n']):>2}  ${r['pnl']:>+10,.0f}  ${r['avg_pnl']:>+7,.0f}  {r['sharpe']:>7.2f}  {r['mdd']:>5.1f}%  {r['trades']:>7.1f}  {r['win_pct']:>4.0f}%")
print('-'*W)
print(f"{'OVERALL':<14} {len(df):>2}  ${df.pnl.sum():>+10,.0f}  ${df.pnl.mean():>+7,.0f}  {df.sharpe.mean():>7.2f}  {df.mdd.mean():>5.1f}%  {df.trades.mean():>7.1f}  {100*df.pos.mean():>4.0f}%")
print('='*W)

# Best/worst stocks per sector
print()
print("TOP 3 STOCKS PER SECTOR (by PnL):")
for sec_name in SECTOR_ORDER:
    sub = df[df['sector'] == sec_name].nlargest(3, 'pnl')
    if len(sub) == 0: continue
    stocks = ', '.join(f"{r['symbol']}(${r['pnl']:+,.0f})" for _, r in sub.iterrows())
    print(f"  {sec_name:<14}: {stocks}")

print()
print("BOTTOM 3 STOCKS PER SECTOR (by PnL):")
for sec_name in SECTOR_ORDER:
    sub = df[df['sector'] == sec_name].nsmallest(3, 'pnl')
    if len(sub) == 0: continue
    stocks = ', '.join(f"{r['symbol']}(${r['pnl']:+,.0f})" for _, r in sub.iterrows())
    print(f"  {sec_name:<14}: {stocks}")
