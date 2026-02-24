"""
V7 vs V8+Fixes A/B/C sector comparison.

V7 data: results_v7.json per_symbol (confirmed: score=14.502, 202 symbols)
V8+Fixes data: step_decisions.csv (Feb 23 17:16 run, freshly trained V8+fixes model)
  - Confirmed V8+fixes: EOG trades 29→9, PnL -$2,430→+$715
  - Higher win rate (85%) with more consistent per-symbol returns
  - Total PnL lower due to fresh training vs optimized V7 model

PnL method: portfolio[-1] - portfolio[0] (not portfolio[-1] - 10000)
Sharpe excluded from step_decisions (near-zero variance when FLAT creates artifacts).
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

# ── Load V7 per-symbol data ──────────────────────────────────────────────────
print("Loading V7 data from results_v7.json...")
with open(r'C:\Users\dilip\alpha_trade_v2_results\results_v7.json', encoding='utf-8', errors='replace') as f:
    d = json.load(f)

v7_rows = []
for sym_key, data in d['per_symbol'].items():
    sym = sym_key.replace('_1d', '')
    v7_rows.append({
        'symbol': sym,
        'sector': SECTOR.get(sym, 'Other'),
        'v7_pnl': data.get('pnl', 0),
        'v7_trades': data.get('trades', 0),
        'v7_sharpe': data.get('sharpe', 0),
        'v7_mdd': data.get('max_dd', 0),
        'v7_pos': data.get('pnl', 0) > 0,
    })
v7df = pd.DataFrame(v7_rows)
print(f"  V7: {len(v7df)} symbols | Total PnL: ${v7df.v7_pnl.sum():,.0f} | Win: {100*v7df.v7_pos.mean():.1f}% | Sharpe: {v7df.v7_sharpe.mean():.3f}")

# ── Load V8+Fixes data from step_decisions.csv ──────────────────────────────
print("\nLoading V8+Fixes data from step_decisions.csv...")
sd = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\step_decisions.csv', low_memory=False)

v8_rows = []
for sym_key in sd.symbol.unique():
    sub = sd[sd.symbol == sym_key].reset_index(drop=True)
    sym = sym_key.replace('_1d', '')

    # PnL = final portfolio - initial portfolio (not -10000)
    pnl = sub.portfolio.iloc[-1] - sub.portfolio.iloc[0]

    # Trade count: position changes
    trades = int((sub.position != sub.position.shift(1)).sum())

    v8_rows.append({
        'symbol': sym,
        'sector': SECTOR.get(sym, 'Other'),
        'v8_pnl': pnl,
        'v8_trades': trades,
        'v8_pos': pnl > 0,
    })
v8df = pd.DataFrame(v8_rows)
print(f"  V8+: {len(v8df)} symbols | Total PnL: ${v8df.v8_pnl.sum():,.0f} | Win: {100*v8df.v8_pos.mean():.1f}%")

# ── Merge ────────────────────────────────────────────────────────────────────
merged = v7df.merge(v8df, on='symbol', how='inner')
merged['d_pnl']   = merged['v8_pnl'] - merged['v7_pnl']
merged['d_trades'] = merged['v8_trades'] - merged['v7_trades']
# Resolve sector column after merge (may become sector_x/sector_y)
if 'sector_x' in merged.columns:
    merged['sector'] = merged['sector_x']
print(f"\nMerged: {len(merged)} symbols")

# ── Sector aggregation ───────────────────────────────────────────────────────
SECTOR_ORDER = ['Industrials','Materials','Financials','Tech','ConsDisc',
                'Healthcare','Utilities','REITs','Energy','Staples','Telecom','Payments','Other']

def sec_agg(df, pnl_col, trades_col, pos_col):
    g = df.groupby('sector')
    return pd.DataFrame({
        'n':        g['symbol'].count(),
        'total_pnl': g[pnl_col].sum().round(0),
        'avg_pnl':  g[pnl_col].mean().round(0),
        'trades':   g[trades_col].mean().round(1),
        'win_pct':  (g[pos_col].mean()*100).round(0),
    })

v7_sec = sec_agg(merged, 'v7_pnl', 'v7_trades', 'v7_pos').reindex(
    [s for s in SECTOR_ORDER if s in merged.groupby('sector').groups])
v8_sec = sec_agg(merged, 'v8_pnl', 'v8_trades', 'v8_pos').reindex(
    [s for s in SECTOR_ORDER if s in merged.groupby('sector').groups])

# ── Print table ──────────────────────────────────────────────────────────────
W = 120
print()
print('='*W)
print('  V7 BASELINE  vs  V8+Fixes A/B/C  |  OOS Aug 2022-Feb 2026  |  $10k/symbol')
print('  Fix A: golden_cross base_pos 0.80→0.92  |  Fix B: sticky-max position lock')
print('  Fix C: 21-bar anti-churn minimum hold  |  V8+: fresh training + fixes')
print('='*W)
print(f"{'Sector':<14} {'#':>2}"
      f"  {'V7 AvgPnL':>10} {'V7 Tr':>5} {'V7 Win':>6}"
      f"  {'V8 AvgPnL':>10} {'V8 Tr':>5} {'V8 Win':>6}"
      f"  {'ΔAvgPnL':>9} {'ΔTrades':>7}")
print('-'*W)

for sec_name in SECTOR_ORDER:
    if sec_name not in v7_sec.index:
        continue
    v7 = v7_sec.loc[sec_name]
    v8 = v8_sec.loc[sec_name]
    n  = int(v7['n'])
    dp = v8['avg_pnl'] - v7['avg_pnl']
    dt = v8['trades'] - v7['trades']
    pa = '▲' if dp > 100 else ('▼' if dp < -100 else ' ')
    ta = '▼' if dt < -2 else ('▲' if dt > 2 else ' ')
    print(f"{sec_name:<14} {n:>2}"
          f"  ${v7['avg_pnl']:>+8,.0f} {v7['trades']:>5.1f} {v7['win_pct']:>5.0f}%"
          f"  ${v8['avg_pnl']:>+8,.0f} {v8['trades']:>5.1f} {v8['win_pct']:>5.0f}%"
          f"  {pa}${dp:>+7,.0f} {ta}{dt:>+6.1f}")

print('-'*W)
v7_tot = merged.v7_pnl.mean()
v8_tot = merged.v8_pnl.mean()
v7_tr  = merged.v7_trades.mean()
v8_tr  = merged.v8_trades.mean()
print(f"{'OVERALL':<14} {len(merged):>2}"
      f"  ${v7_tot:>+8,.0f} {v7_tr:>5.1f} {100*(merged.v7_pnl>0).mean():>5.0f}%"
      f"  ${v8_tot:>+8,.0f} {v8_tr:>5.1f} {100*(merged.v8_pnl>0).mean():>5.0f}%"
      f"  ${(v8_tot-v7_tot):>+7,.0f} {(v8_tr-v7_tr):>+6.1f}")
print('='*W)

# Summary headline
print()
print(f"  V7  Total PnL: ${merged.v7_pnl.sum():>+11,.0f}  |  Win Rate: {100*(merged.v7_pnl>0).mean():.1f}%  |  Avg Trades/sym: {v7_tr:.1f}")
print(f"  V8+ Total PnL: ${merged.v8_pnl.sum():>+11,.0f}  |  Win Rate: {100*(merged.v8_pnl>0).mean():.1f}%  |  Avg Trades/sym: {v8_tr:.1f}")
print(f"  Delta:         ${(merged.v8_pnl-merged.v7_pnl).sum():>+11,.0f}  |  ΔWin: {100*(merged.v8_pos.mean()-(merged.v7_pnl>0).mean()):>+.1f}%  |  ΔTrades: {(v8_tr-v7_tr):>+.1f}")

# Stock movers
print()
print("TOP 5 STOCKS IMPROVED BY V8+FIXES (ΔPnL):")
top5 = merged.nlargest(5, 'd_pnl')
for _, r in top5.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) V7 ${r['v7_pnl']:>+7,.0f} → V8+ ${r['v8_pnl']:>+7,.0f}"
          f"  Δ${r['d_pnl']:>+7,.0f}  Trades {r['v7_trades']:.0f}→{r['v8_trades']:.0f}")

print()
print("TOP 5 STOCKS HURT BY V8+FIXES (ΔPnL):")
bot5 = merged.nsmallest(5, 'd_pnl')
for _, r in bot5.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) V7 ${r['v7_pnl']:>+7,.0f} → V8+ ${r['v8_pnl']:>+7,.0f}"
          f"  Δ${r['d_pnl']:>+7,.0f}  Trades {r['v7_trades']:.0f}→{r['v8_trades']:.0f}")

# Anti-churn Fix C impact analysis
print()
print("FIX C (21-BAR ANTI-CHURN) — OVER-TRADING SECTOR IMPACT:")
print(f"  {'Sector':<14} {'V7 Trades':>10} {'V8 Trades':>10} {'ΔTrades':>8} {'V7 AvgPnL':>10} {'V8 AvgPnL':>10} {'ΔAvgPnL':>9} {'V7 Win':>7} {'V8 Win':>7}")
for sec_name in ['Energy', 'Utilities', 'Staples', 'Healthcare', 'Industrials']:
    sub = merged[merged['sector_x' if 'sector_x' in merged.columns else 'sector'] == sec_name] if 'sector_x' in merged.columns else merged[merged.sector == sec_name]
    if len(sub) == 0:
        # try alternative sector column
        sub = merged[merged.get('sector', pd.Series(dtype=str)) == sec_name]
    if len(sub) == 0: continue
    print(f"  {sec_name:<14} {sub.v7_trades.mean():>10.1f} {sub.v8_trades.mean():>10.1f} {(sub.v8_trades-sub.v7_trades).mean():>+8.1f}"
          f" ${sub.v7_pnl.mean():>9,.0f} ${sub.v8_pnl.mean():>9,.0f} ${(sub.v8_pnl-sub.v7_pnl).mean():>+8,.0f}"
          f" {100*(sub.v7_pnl>0).mean():>6.0f}% {100*(sub.v8_pnl>0).mean():>6.0f}%")
