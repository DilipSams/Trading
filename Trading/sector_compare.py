"""V7 vs V8 sector-level comparison."""
import re, json
import numpy as np
import pandas as pd

# ── SECTOR MAP ─────────────────────────────────────────────────────────────
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

def parse_arr(s):
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?', str(s))
    return np.array([float(x) for x in nums])

def returns_to_row(sym, arr):
    if len(arr) == 0:
        return None
    n_years = max(len(arr) / 252, 0.01)
    cum = float(np.prod(1 + arr) - 1)
    cagr = float((1 + cum) ** (1 / n_years) - 1) * 100
    sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0
    peak, nav, mdd = 1.0, 1.0, 0.0
    for r in arr:
        nav *= (1 + r)
        if nav > peak:
            peak = nav
        mdd = min(mdd, (nav - peak) / peak * 100)
    return {
        'symbol': sym,
        'sector': SECTOR.get(sym, 'Other'),
        'cagr': round(cagr, 1),
        'cum_ret': round(cum * 100, 1),
        'sharpe': round(sharpe, 2),
        'max_dd': round(mdd, 1),
        'bars': len(arr),
    }

# ── V7 data from results_v7.json ────────────────────────────────────────────
with open("C:/Users/dilip/alpha_trade_v2_results/results_v7.json",
          encoding='utf-8', errors='replace') as f:
    v7_res = json.load(f)

v7_rows = []
for sym_key, ret_str in v7_res['per_symbol_returns_net'].items():
    sym = sym_key.replace('_1d', '')
    row = returns_to_row(sym, parse_arr(ret_str))
    if row:
        v7_rows.append(row)
v7_df = pd.DataFrame(v7_rows)

# ── V8 data from the full-eval log ──────────────────────────────────────────
LOG = '../run_output/v8_full_202_eval.log'
with open(LOG, encoding='utf-8', errors='replace') as f:
    content = f.read()
ansi = re.compile(r'\x1b\[[0-9;]*m')
clean = ansi.sub('', content)
lines = clean.split('\n')
pnl_lines = [l for l in lines if '_1d  PnL:' in l or '_1d PnL:' in l]

pnl_re = re.compile(
    r'(\w+)_1d\s+PnL:\$\s*([\+\-][\d,\.]+)\s+'
    r'\(trade:\$\s*([\+\-][\d,\.]+)\s+cash:\$\s*([\+\-][\d,\.]+)\)\s+'
    r'Sh:([\+\-][\d\.]+)\s+DD:([\d\.]+)%\s+Trades:(\d+)\s+Suppressed:(\d+)%'
)

v8_rows = []
seen = set()
for l in pnl_lines:
    m = pnl_re.search(l)
    if m:
        sym = m.group(1)
        key = sym
        # Take only first occurrence per symbol (the primary OOS period)
        if key in seen:
            continue
        seen.add(key)
        v8_rows.append({
            'symbol': sym,
            'sector': SECTOR.get(sym, 'Other'),
            'pnl': float(m.group(2).replace(',', '')),
            'trade_pnl': float(m.group(3).replace(',', '')),
            'sharpe_v8': float(m.group(5)),
            'max_dd_v8': float(m.group(6)),
            'trades_v8': int(m.group(7)),
        })

v8_df = pd.DataFrame(v8_rows)

# ── Merge v7 and v8 ─────────────────────────────────────────────────────────
merged = v7_df.merge(v8_df, on=['symbol', 'sector'], how='inner')
merged['cagr_delta'] = merged['sharpe_v8'] - merged['sharpe']  # used below

# ── Sector aggregation ──────────────────────────────────────────────────────
def sector_agg(df, prefix):
    g = df.groupby('sector')
    r = pd.DataFrame({
        'n': g['symbol'].count(),
        f'{prefix}_cagr': g['cagr'].mean().round(1),
        f'{prefix}_sharpe': g['sharpe'].mean().round(2),
        f'{prefix}_pct_pos': g['cagr'].apply(lambda x: round(100*(x>0).mean(), 0)),
        f'{prefix}_dd': g['max_dd'].mean().round(1),
    })
    return r

v7_sec = sector_agg(v7_df, 'v7')

v8_grp = merged.groupby('sector')
v8_sec = pd.DataFrame({
    'v8_sharpe': v8_grp['sharpe_v8'].mean().round(2),
    'v8_pnl_avg': v8_grp['pnl'].mean().round(0),
    'v8_trades_avg': v8_grp['trades_v8'].mean().round(1),
    'v8_pct_pos': v8_grp['pnl'].apply(lambda x: round(100*(x>0).mean(), 0)),
})

# Scale v7 pnl: cagr to approximate $10k pnl
v7_pnl = v7_df.copy()
v7_pnl['pnl_approx'] = v7_pnl['cum_ret'] * 100  # $10k * cum_ret%
v7_pnl_sec = v7_pnl.groupby('sector')['pnl_approx'].mean().round(0)

summary = v7_sec.join(v8_sec).join(v7_pnl_sec.rename('v7_pnl_avg'))
summary['sharpe_delta'] = (summary['v8_sharpe'] - summary['v7_sharpe']).round(2)
summary['pnl_delta'] = (summary['v8_pnl_avg'] - summary['v7_pnl_avg']).round(0)
summary = summary.sort_values('v8_sharpe', ascending=False)

SECTOR_ORDER = [
    'Industrials','Materials','Financials','Tech','ConsDisc',
    'Healthcare','Utilities','REITs','Staples','Energy','Telecom','Payments','Other'
]
summary = summary.reindex([s for s in SECTOR_ORDER if s in summary.index])

# ── Print table ─────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("  V7 vs V8 SECTOR COMPARISON  |  OOS Aug 2022 – Feb 2026  |  202 Symbols")
print("=" * 100)
print(f"{'':14}  {'── V7 (Base) ──':^30}  {'── V8 (SMA Overlay) ──':^30}  {'── Delta ──':^16}")
print(f"{'Sector':<14}  {'#':>2}  {'CAGR':>6}  {'Sharpe':>6}  {'%Pos':>5}  {'AvgDD':>6}"
      f"  {'Sharpe':>6}  {'AvgPnL':>8}  {'%Pos':>5}  {'Trades':>6}"
      f"  {'ΔSharpe':>7}  {'ΔPnL':>7}")
print("-" * 100)

for sec in summary.index:
    r = summary.loc[sec]
    sh_arrow = '▲' if r['sharpe_delta'] > 0 else ('▼' if r['sharpe_delta'] < 0 else '─')
    pnl_arrow = '▲' if r['pnl_delta'] > 0 else ('▼' if r['pnl_delta'] < 0 else '─')
    print(
        f"{sec:<14}  {int(r['n']):>2}  {r['v7_cagr']:>5.1f}%  {r['v7_sharpe']:>6.2f}  "
        f"{r['v7_pct_pos']:>4.0f}%  {r['v7_dd']:>5.1f}%"
        f"  {r['v8_sharpe']:>6.2f}  ${r['v8_pnl_avg']:>+7,.0f}  {r['v8_pct_pos']:>4.0f}%  "
        f"{r['v8_trades_avg']:>6.1f}"
        f"  {sh_arrow}{r['sharpe_delta']:>+5.2f}  {pnl_arrow}${r['pnl_delta']:>+5,.0f}"
    )

print("-" * 100)
# Totals
print(
    f"{'TOTAL/AVG':<14}  {len(merged):>2}  {v7_df['cagr'].mean():>5.1f}%  "
    f"{v7_df['sharpe'].mean():>6.2f}  "
    f"{100*(v7_df['cagr']>0).mean():>4.0f}%  {v7_df['max_dd'].mean():>5.1f}%"
    f"  {merged['sharpe_v8'].mean():>6.2f}  ${merged['pnl'].mean():>+7,.0f}  "
    f"{100*(merged['pnl']>0).mean():>4.0f}%  {merged['trades_v8'].mean():>6.1f}"
    f"  {'▲':>1}{(merged['sharpe_v8']-v7_df.set_index('symbol').loc[merged['symbol'],'sharpe'].values).mean():>+5.2f}"
    f"  {'':>8}"
)
print()

# ── Top movers ──────────────────────────────────────────────────────────────
print("TOP 5 STOCKS IMPROVED BY SMA OVERLAY (V7→V8 Sharpe gain):")
merged['sh_delta'] = merged['sharpe_v8'] - merged['sharpe']
top_imp = merged.nlargest(5, 'sh_delta')[['symbol','sector','sharpe','sharpe_v8','sh_delta','pnl']]
for _, r in top_imp.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) Sharpe {r['sharpe']:+.2f} → {r['sharpe_v8']:+.2f}"
          f"  (Δ{r['sh_delta']:+.2f})  V8 PnL ${r['pnl']:>+,.0f}")

print()
print("TOP 5 STOCKS HURT BY SMA OVERLAY (V7→V8 Sharpe loss):")
top_hurt = merged.nsmallest(5, 'sh_delta')[['symbol','sector','sharpe','sharpe_v8','sh_delta','pnl']]
for _, r in top_hurt.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) Sharpe {r['sharpe']:+.2f} → {r['sharpe_v8']:+.2f}"
          f"  (Δ{r['sh_delta']:+.2f})  V8 PnL ${r['pnl']:>+,.0f}")
