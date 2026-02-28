"""META diagnosis and sector-wise CAGR analysis."""
import json, pandas as pd, numpy as np, re

# ── META diagnosis ─────────────────────────────────────────────────────────
sd = pd.read_csv("C:/Users/dilip/alpha_trade_v2_results/step_decisions.csv", low_memory=False)
meta = sd[sd['symbol'] == 'META_1d'].copy().reset_index(drop=True)

print("=== META DIAGNOSIS ===")
print(f"OOS period : {meta['timestamp'].iloc[0]}  to  {meta['timestamp'].iloc[-1]}  ({len(meta)} bars)")
print(f"Price start: ${meta['C'].iloc[0]:.2f}")
print(f"Price end  : ${meta['C'].iloc[-1]:.2f}  ({(meta['C'].iloc[-1]/meta['C'].iloc[0]-1)*100:.1f}% actual stock return)")
print(f"Peak price : ${meta['C'].max():.2f}  on {meta.loc[meta['C'].idxmax(),'timestamp']}")
print(f"Trough     : ${meta['C'].min():.2f}  on {meta.loc[meta['C'].idxmin(),'timestamp']}")
print()
print(f"Position split: {meta['position'].value_counts().to_dict()}")
print(f"Action split  : {meta['action'].value_counts().to_dict()}")
print()

phases = [
    ("Jan-Mar 2025  META $674→$509",   0,   50),
    ("Apr-Aug 2025  META $509→$775",  50,  150),
    ("Sep-Nov 2025  META $779→$630", 150,  210),
    ("Dec25-Feb26   META recovers",  210,  266),
]
print(f"{'Phase':<36} {'METARet':>8} {'PortRet':>8} {'DomAction':>12} {'DomPos':>8}")
print("-" * 78)
for label, s, e in phases:
    e = min(e, len(meta))
    seg = meta.iloc[s:e]
    if len(seg) == 0:
        continue
    price_ret = (seg['C'].iloc[-1] / seg['C'].iloc[0] - 1) * 100
    port_ret  = (seg['portfolio'].iloc[-1] - seg['portfolio'].iloc[0]) / 10000 * 100
    dom_act   = seg['action'].value_counts().index[0]
    dom_pos   = seg['position'].value_counts().index[0]
    print(f"{label:<36} {price_ret:>+7.1f}%  {port_ret:>+7.1f}%  {dom_act:>12}  {dom_pos:>8}")

print()
print("ROOT CAUSE:")
print("  * META OOS starts Jan 2025 at $674 - already near ATH after 2023-24 bull run")
print("  * VALUE alpha flags it overvalued (negative value_est) every single bar")
print("  * RL confidence is ~0.20 (uniform random) - model has no edge here")
print("  * Strategy mostly FLAT (92.5%) - misses META peak at $779 (+15.5%)")
print("  * But also misses META crash $779→$630 (-19%) by staying flat")
print("  * Net: strategy underperforms vs META's actual -4.4% return by being too cautious")
print()

# ── SECTOR CAGR ────────────────────────────────────────────────────────────
print("=" * 90)
print("=== SECTOR-WISE CAGR TABLE ===")

with open("C:/Users/dilip/alpha_trade_v2_results/results_v7.json",
          encoding='utf-8', errors='replace') as f:
    results = json.load(f)

def parse_arr(s):
    nums = re.findall(r'[-+]?\d*\.?\d+e?[+-]?\d*', str(s))
    return np.array([float(x) for x in nums])

SECTOR = {
    'COP': 'Energy', 'EOG': 'Energy', 'CVX': 'Energy', 'XOM': 'Energy',
    'OXY': 'Energy', 'BKR': 'Energy', 'SLB': 'Energy', 'PSX': 'Energy',
    'MPC': 'Energy', 'VLO': 'Energy', 'WMB': 'Energy', 'ET': 'Energy',
    'EPD': 'Energy', 'MPLX': 'Energy',
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'PFE': 'Healthcare', 'BMY': 'Healthcare',
    'LLY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'TMO': 'Healthcare', 'DHR': 'Healthcare', 'ABT': 'Healthcare',
    'SYK': 'Healthcare', 'BDX': 'Healthcare', 'ELV': 'Healthcare',
    'VRTX': 'Healthcare', 'ZTS': 'Healthcare', 'REGN': 'Healthcare',
    'MCK': 'Healthcare', 'ISRG': 'Healthcare', 'COR': 'Healthcare',
    'BSX': 'Healthcare', 'CI': 'Healthcare', 'HCA': 'Healthcare',
    'PEP': 'Staples', 'KO': 'Staples', 'MCD': 'Staples', 'MDLZ': 'Staples',
    'MNST': 'Staples', 'WMT': 'Staples', 'COST': 'Staples', 'PG': 'Staples',
    'CL': 'Staples', 'MO': 'Staples', 'SBUX': 'Staples',
    'VZ': 'Telecom', 'T': 'Telecom', 'TMUS': 'Telecom',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'AEP': 'Utilities', 'SRE': 'Utilities', 'D': 'Utilities',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials', 'MS': 'Financials', 'BK': 'Financials',
    'USB': 'Financials', 'TRV': 'Financials', 'AXP': 'Financials',
    'BLK': 'Financials', 'SCHW': 'Financials', 'CB': 'Financials',
    'APO': 'Financials', 'AFL': 'Financials', 'PNC': 'Financials',
    'TFC': 'Financials', 'COF': 'Financials', 'C': 'Financials',
    'MCO': 'Financials', 'SPGI': 'Financials', 'ICE': 'Financials',
    'CME': 'Financials', 'KKR': 'Financials', 'BX': 'Financials',
    'IBKR': 'Financials',
    'CAT': 'Industrials', 'HON': 'Industrials', 'UNP': 'Industrials',
    'UPS': 'Industrials', 'CSX': 'Industrials', 'NSC': 'Industrials',
    'GE': 'Industrials', 'MMM': 'Industrials', 'ITW': 'Industrials',
    'EMR': 'Industrials', 'WM': 'Industrials', 'CMI': 'Industrials',
    'URI': 'Industrials', 'CTAS': 'Industrials', 'HWM': 'Industrials',
    'AJG': 'Industrials', 'GD': 'Industrials', 'LMT': 'Industrials',
    'RTX': 'Industrials', 'NOC': 'Industrials', 'LHX': 'Industrials',
    'FDX': 'Industrials', 'PH': 'Industrials', 'PCAR': 'Industrials',
    'RSG': 'Industrials', 'PWR': 'Industrials', 'GEV': 'Industrials',
    'AMZN': 'ConsDisc', 'TSLA': 'ConsDisc', 'NKE': 'ConsDisc',
    'TJX': 'ConsDisc', 'HD': 'ConsDisc', 'LOW': 'ConsDisc',
    'BKNG': 'ConsDisc', 'DIS': 'ConsDisc', 'NFLX': 'ConsDisc',
    'AZO': 'ConsDisc', 'ORLY': 'ConsDisc', 'F': 'ConsDisc',
    'GM': 'ConsDisc', 'RCL': 'ConsDisc', 'MAR': 'ConsDisc',
    'HLT': 'ConsDisc', 'CVNA': 'ConsDisc',
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech',
    'META': 'Tech', 'AVGO': 'Tech', 'ORCL': 'Tech', 'CRM': 'Tech',
    'AMD': 'Tech', 'QCOM': 'Tech', 'TXN': 'Tech', 'ADBE': 'Tech',
    'AMAT': 'Tech', 'MU': 'Tech', 'LRCX': 'Tech', 'KLAC': 'Tech',
    'ADI': 'Tech', 'INTC': 'Tech', 'IBM': 'Tech', 'ACN': 'Tech',
    'NOW': 'Tech', 'INTU': 'Tech', 'PANW': 'Tech', 'ANET': 'Tech',
    'FTNT': 'Tech', 'CSCO': 'Tech', 'CDNS': 'Tech', 'SNPS': 'Tech',
    'MPWR': 'Tech', 'DELL': 'Tech', 'WDC': 'Tech', 'MSI': 'Tech',
    'MRVL': 'Tech', 'CRWD': 'Tech', 'NET': 'Tech', 'APP': 'Tech',
    'PLTR': 'Tech', 'SNOW': 'Tech',
    'FCX': 'Materials', 'NEM': 'Materials', 'APD': 'Materials',
    'LIN': 'Materials', 'SCCO': 'Materials', 'GLW': 'Materials',
    'ECL': 'Materials',
    'AMT': 'REITs', 'PLD': 'REITs', 'WELL': 'REITs', 'O': 'REITs',
    'SPG': 'REITs', 'EQIX': 'REITs', 'DLR': 'REITs',
    'V': 'Payments', 'MA': 'Payments', 'PYPL': 'Payments',
}

rows = []
per_sym_net = results['per_symbol_returns_net']
for sym, ret_str in per_sym_net.items():
    arr = parse_arr(ret_str)
    if len(arr) == 0:
        continue
    sym_clean = sym.replace('_1d', '')
    n_years = len(arr) / 252
    cum_ret = float(np.prod(1 + arr) - 1)
    cagr = float((1 + cum_ret) ** (1 / max(n_years, 0.01)) - 1) * 100
    sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0
    peak = 1.0; nav = 1.0; mdd = 0.0
    for r in arr:
        nav *= (1 + r)
        if nav > peak:
            peak = nav
        mdd = min(mdd, (nav - peak) / peak * 100)
    rows.append({
        'symbol': sym_clean,
        'sector': SECTOR.get(sym_clean, 'Other'),
        'cagr': round(cagr, 1),
        'cum_ret': round(cum_ret * 100, 1),
        'sharpe': round(sharpe, 2),
        'max_dd': round(mdd, 1),
        'bars': len(arr),
    })

df = pd.DataFrame(rows)

sec = df.groupby('sector').agg(
    symbols=('symbol', 'count'),
    avg_cagr=('cagr', 'mean'),
    median_cagr=('cagr', 'median'),
    best_cagr=('cagr', 'max'),
    worst_cagr=('cagr', 'min'),
    avg_sharpe=('sharpe', 'mean'),
    avg_dd=('max_dd', 'mean'),
    pct_pos=('cagr', lambda x: round(100 * (x > 0).mean(), 0)),
).reset_index().sort_values('avg_cagr', ascending=False)

hdr = (f"{'Sector':<14} {'#':>3}  {'AvgCAGR':>8}  {'Median':>8}  "
       f"{'Best':>8}  {'Worst':>9}  {'AvgSharpe':>10}  {'AvgDD':>7}  {'%Profit':>8}")
print(hdr)
print("-" * len(hdr))
for _, r in sec.iterrows():
    print(f"{r['sector']:<14} {r['symbols']:>3}  {r['avg_cagr']:>7.1f}%  "
          f"{r['median_cagr']:>7.1f}%  {r['best_cagr']:>7.1f}%  "
          f"{r['worst_cagr']:>8.1f}%  {r['avg_sharpe']:>9.2f}  "
          f"{r['avg_dd']:>6.1f}%  {r['pct_pos']:>7.0f}%")

print()
# Top 3 per sector
print("=== TOP 3 PERFORMERS PER SECTOR ===")
for sector in sec['sector'].tolist():
    top3 = df[df['sector'] == sector].nlargest(3, 'cagr')[['symbol', 'cagr', 'sharpe']]
    names = ', '.join(f"{r['symbol']}({r['cagr']:+.0f}%)" for _, r in top3.iterrows())
    print(f"  {sector:<14}: {names}")
