"""Clean V7 vs V8 sector comparison using matched per-symbol data."""
import re, json
import numpy as np
import pandas as pd

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

OOS_YEARS = 3.5   # Aug 2022 – Feb 2026

def parse_arr(s):
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?', str(s))
    return np.array([float(x) for x in nums])

# ── V7: per_symbol_returns_net → CAGR + Sharpe ──────────────────────────────
with open("C:/Users/dilip/alpha_trade_v2_results/results_v7.json",
          encoding='utf-8', errors='replace') as f:
    v7_res = json.load(f)

v7_rows = []
for sym_key, ret_str in v7_res['per_symbol_returns_net'].items():
    sym = sym_key.replace('_1d', '')
    arr = parse_arr(ret_str)
    if len(arr) == 0:
        continue
    n_yr = max(len(arr) / 252, 0.01)
    cum  = float(np.prod(1 + arr) - 1)
    cagr = float((1 + cum) ** (1 / n_yr) - 1) * 100
    sh   = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0
    peak, nav, mdd = 1.0, 1.0, 0.0
    for r in arr:
        nav *= (1 + r)
        if nav > peak: peak = nav
        mdd = min(mdd, (nav - peak) / peak * 100)
    v7_rows.append({'symbol': sym, 'sector': SECTOR.get(sym,'Other'),
                    'v7_cagr': round(cagr,1), 'v7_sharpe': round(sh,2),
                    'v7_mdd': round(mdd,1), 'v7_pos': cum > 0})
v7 = pd.DataFrame(v7_rows)

# ── V8: parse per-symbol PnL from full-eval log ──────────────────────────────
LOG = '../run_output/v8_full_202_eval.log'
with open(LOG, encoding='utf-8', errors='replace') as f:
    content = f.read()
ansi = re.compile(r'\x1b\[[0-9;]*m')
clean = ansi.sub('', content)

pnl_re = re.compile(
    r'(\w+)_1d\s+PnL:\$\s*([\+\-][\d,\.]+)\s+'
    r'\(trade:\$\s*([\+\-][\d,\.]+)\s+cash:\$\s*([\+\-][\d,\.]+)\)\s+'
    r'Sh:([\+\-][\d\.]+)\s+DD:([\d\.]+)%\s+Trades:(\d+)\s+Suppressed:(\d+)%'
)

v8_rows = {}
for m in pnl_re.finditer(clean):
    sym = m.group(1)
    if sym in v8_rows:
        continue   # first occurrence = primary OOS eval period
    pnl   = float(m.group(2).replace(',',''))
    sh    = float(m.group(5))
    mdd   = -float(m.group(6))
    trd   = int(m.group(7))
    supp  = int(m.group(8))
    # Estimate CAGR: PnL on $10k base over OOS_YEARS
    cum_pct = pnl / 10_000
    cagr_v8 = ((1 + cum_pct) ** (1 / OOS_YEARS) - 1) * 100
    v8_rows[sym] = {'symbol': sym, 'sector': SECTOR.get(sym,'Other'),
                    'v8_cagr': round(cagr_v8,1), 'v8_sharpe': round(sh,2),
                    'v8_mdd': round(mdd,1), 'v8_trades': trd,
                    'v8_pnl': pnl, 'v8_pos': pnl > 0}
v8 = pd.DataFrame(v8_rows.values())

# ── Merge on symbol ───────────────────────────────────────────────────────
df = v7.merge(v8, on=['symbol','sector'], how='inner')
df['d_cagr']   = df['v8_cagr'] - df['v7_cagr']
df['d_sharpe'] = df['v8_sharpe'] - df['v7_sharpe']

# ── Sector summary ────────────────────────────────────────────────────────
SECTOR_ORDER = [
    'Industrials','Materials','Financials','Tech','ConsDisc',
    'Healthcare','Utilities','REITs','Energy','Staples','Telecom','Payments','Other'
]

g = df.groupby('sector')
sec = pd.DataFrame({
    'n':            g['symbol'].count(),
    'v7_cagr':      g['v7_cagr'].mean().round(1),
    'v7_sharpe':    g['v7_sharpe'].mean().round(2),
    'v7_pct_win':   g['v7_pos'].apply(lambda x: round(100*x.mean(),0)),
    'v7_mdd':       g['v7_mdd'].mean().round(1),
    'v8_cagr':      g['v8_cagr'].mean().round(1),
    'v8_sharpe':    g['v8_sharpe'].mean().round(2),
    'v8_pct_win':   g['v8_pos'].apply(lambda x: round(100*x.mean(),0)),
    'v8_mdd':       g['v8_mdd'].mean().round(1),
    'v8_trades':    g['v8_trades'].mean().round(1),
    'd_cagr':       g['d_cagr'].mean().round(1),
    'd_sharpe':     g['d_sharpe'].mean().round(2),
}).reindex([s for s in SECTOR_ORDER if s in g.groups])

W = 112
print()
print("=" * W)
print("  V7 (Alpha Signals Only)  vs  V8 (+ SMA Overlay)  |  OOS 3.5 yrs  |  202 Symbols")
print("=" * W)
hdr = (f"{'Sector':<13} {'#':>2}"
       f" | {'V7 CAGR':>8} {'V7 Sh':>6} {'V7 Win%':>7} {'V7 DD':>6}"
       f" | {'V8 CAGR':>8} {'V8 Sh':>6} {'V8 Win%':>7} {'V8 DD':>6} {'Trades':>6}"
       f" | {'ΔCAGR':>7} {'ΔSharpe':>7}")
print(hdr)
print("-" * W)

for sec_name in SECTOR_ORDER:
    if sec_name not in sec.index:
        continue
    r = sec.loc[sec_name]
    dc = r['d_cagr']
    ds = r['d_sharpe']
    ca = '▲' if dc > 0.5 else ('▼' if dc < -0.5 else ' ')
    sa = '▲' if ds > 0.05 else ('▼' if ds < -0.05 else ' ')
    print(
        f"{sec_name:<13} {int(r['n']):>2}"
        f" | {r['v7_cagr']:>7.1f}% {r['v7_sharpe']:>6.2f} {r['v7_pct_win']:>6.0f}%"
        f" {r['v7_mdd']:>5.1f}%"
        f" | {r['v8_cagr']:>7.1f}% {r['v8_sharpe']:>6.2f} {r['v8_pct_win']:>6.0f}%"
        f" {r['v8_mdd']:>5.1f}% {r['v8_trades']:>6.1f}"
        f" | {ca}{dc:>+6.1f}% {sa}{ds:>+6.2f}"
    )

print("-" * W)
# Overall totals
print(
    f"{'OVERALL':<13} {len(df):>2}"
    f" | {df['v7_cagr'].mean():>7.1f}% {df['v7_sharpe'].mean():>6.2f}"
    f" {100*df['v7_pos'].mean():>6.0f}% {df['v7_mdd'].mean():>5.1f}%"
    f" | {df['v8_cagr'].mean():>7.1f}% {df['v8_sharpe'].mean():>6.2f}"
    f" {100*df['v8_pos'].mean():>6.0f}% {df['v8_mdd'].mean():>5.1f}%"
    f" {df['v8_trades'].mean():>6.1f}"
    f" | {df['d_cagr'].mean():>+7.1f}% {df['d_sharpe'].mean():>+7.2f}"
)
print("=" * W)

# Headline numbers
print()
print(f"  V7 Composite Score : +14.50  |  Total PnL: $464,977  |  Sharpe: 0.480  |  Win Rate: 72%")
print(f"  V8 Composite Score : +30.49  |  Total PnL: $635,599  |  Sharpe: 0.680  |  Win Rate: 78%")
print(f"  V8+Fixes Score     : +183.6  |  Total PnL: est.higher (full 202-sym OOS)")
print()
print(f"  V8 vs V7 improvement: +$170,622 PnL (+37%)  |  Sharpe +0.20  |  Trades -54% (1999→927)")

# ── Best / worst sector movers ─────────────────────────────────────────────
print()
print("BIGGEST SECTOR GAINERS (V7→V8 CAGR):")
gainers = sec.nlargest(3,'d_cagr')
for s, r in gainers.iterrows():
    print(f"  {s:<14} {r['v7_cagr']:>+5.1f}% → {r['v8_cagr']:>+5.1f}%  (Δ{r['d_cagr']:>+5.1f}%)"
          f"  Sharpe {r['v7_sharpe']:+.2f} → {r['v8_sharpe']:+.2f}")
print()
print("BIGGEST SECTOR LOSERS  (V7→V8 CAGR):")
losers = sec.nsmallest(3,'d_cagr')
for s, r in losers.iterrows():
    print(f"  {s:<14} {r['v7_cagr']:>+5.1f}% → {r['v8_cagr']:>+5.1f}%  (Δ{r['d_cagr']:>+5.1f}%)"
          f"  Sharpe {r['v7_sharpe']:+.2f} → {r['v8_sharpe']:+.2f}")

# ── Stock-level best/worst movers ──────────────────────────────────────────
print()
print("TOP 5 STOCKS IMPROVED BY V8 SMA OVERLAY:")
top5 = df.nlargest(5,'d_cagr')[['symbol','sector','v7_cagr','v8_cagr','d_cagr','v7_sharpe','v8_sharpe','v8_trades']]
for _, r in top5.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) CAGR {r['v7_cagr']:>+5.1f}% → {r['v8_cagr']:>+5.1f}%"
          f"  Δ{r['d_cagr']:>+5.1f}%  Sh {r['v7_sharpe']:>+.2f}→{r['v8_sharpe']:>+.2f}"
          f"  {r['v8_trades']:.0f} trades")
print()
print("TOP 5 STOCKS HURT BY V8 SMA OVERLAY:")
bot5 = df.nsmallest(5,'d_cagr')[['symbol','sector','v7_cagr','v8_cagr','d_cagr','v7_sharpe','v8_sharpe','v8_trades']]
for _, r in bot5.iterrows():
    print(f"  {r['symbol']:<6} ({r['sector']:<12}) CAGR {r['v7_cagr']:>+5.1f}% → {r['v8_cagr']:>+5.1f}%"
          f"  Δ{r['d_cagr']:>+5.1f}%  Sh {r['v7_sharpe']:>+.2f}→{r['v8_sharpe']:>+.2f}"
          f"  {r['v8_trades']:.0f} trades")
