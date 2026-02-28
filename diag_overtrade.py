"""Diagnose over-trading patterns from v8.0 run logs."""
import re, sys
import pandas as pd
import numpy as np

LOG_PATH = '../run_output/v8_full_202_eval.log'

with open(LOG_PATH, encoding='utf-8', errors='replace') as f:
    content = f.read()

ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
clean = ansi_escape.sub('', content)
lines = clean.split('\n')
pnl_lines = [l for l in lines if '_1d  PnL:' in l or '_1d PnL:' in l]

pnl_re = re.compile(
    r'(\w+)_1d\s+PnL:\$\s*([\+\-][\d,\.]+)\s+'
    r'\(trade:\$\s*([\+\-][\d,\.]+)\s+cash:\$\s*([\+\-][\d,\.]+)\)\s+'
    r'Sh:([\+\-][\d\.]+)\s+DD:([\d\.]+)%\s+Trades:(\d+)\s+Suppressed:(\d+)%'
)

rows = []
for l in pnl_lines:
    m = pnl_re.search(l)
    if m:
        rows.append({
            'symbol': m.group(1),
            'pnl': float(m.group(2).replace(',', '')),
            'trade_pnl': float(m.group(3).replace(',', '')),
            'sharpe': float(m.group(5)),
            'max_dd': float(m.group(6)),
            'trades': int(m.group(7)),
            'suppressed_pct': int(m.group(8)),
        })

df = pd.DataFrame(rows)

SECTOR = {
    'COP': 'Energy', 'EOG': 'Energy', 'CVX': 'Energy', 'XOM': 'Energy',
    'OXY': 'Energy', 'BKR': 'Energy', 'HAL': 'Energy', 'SLB': 'Energy',
    'PSX': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy', 'WMB': 'Energy',
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'PFE': 'Healthcare', 'BMY': 'Healthcare',
    'LLY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'TMO': 'Healthcare', 'DHR': 'Healthcare', 'ABT': 'Healthcare',
    'SYK': 'Healthcare', 'BDX': 'Healthcare', 'ELV': 'Healthcare',
    'VRTX': 'Healthcare', 'ZTS': 'Healthcare',
    'PEP': 'Staples', 'KO': 'Staples', 'MCD': 'Staples', 'MDLZ': 'Staples',
    'MNST': 'Staples', 'WMT': 'Staples', 'COST': 'Staples', 'PG': 'Staples',
    'CL': 'Staples', 'KMB': 'Staples', 'SYY': 'Staples',
    'VZ': 'Telecom', 'T': 'Telecom', 'TMUS': 'Telecom',
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'D': 'Utilities',
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
    'GS': 'Financials', 'MS': 'Financials', 'BK': 'Financials',
    'USB': 'Financials', 'TRV': 'Financials', 'AXP': 'Financials',
    'BLK': 'Financials', 'SCHW': 'Financials', 'CB': 'Financials',
    'APO': 'Financials', 'AFL': 'Financials',
    'CAT': 'Industrials', 'HON': 'Industrials', 'UNP': 'Industrials',
    'UPS': 'Industrials', 'CSX': 'Industrials', 'NSC': 'Industrials',
    'GE': 'Industrials', 'MMM': 'Industrials', 'ITW': 'Industrials',
    'EMR': 'Industrials', 'ETN': 'Industrials', 'WM': 'Industrials',
    'CMI': 'Industrials', 'URI': 'Industrials', 'CTAS': 'Industrials',
    'HWM': 'Industrials', 'AJG': 'Industrials', 'VRSK': 'Industrials',
    'AMZN': 'ConsDisc', 'TSLA': 'ConsDisc', 'NKE': 'ConsDisc',
    'TJX': 'ConsDisc', 'HD': 'ConsDisc', 'LOW': 'ConsDisc',
    'BKNG': 'ConsDisc', 'DIS': 'ConsDisc', 'NFLX': 'ConsDisc',
    'SBUX': 'ConsDisc', 'AZO': 'ConsDisc', 'ORLY': 'ConsDisc',
    'F': 'ConsDisc', 'GM': 'ConsDisc',
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'GOOGL': 'Tech',
    'META': 'Tech', 'AVGO': 'Tech', 'ORCL': 'Tech', 'CRM': 'Tech',
    'AMD': 'Tech', 'QCOM': 'Tech', 'TXN': 'Tech', 'ADBE': 'Tech',
    'AMAT': 'Tech', 'MU': 'Tech', 'LRCX': 'Tech', 'KLAC': 'Tech',
    'ADI': 'Tech', 'INTC': 'Tech', 'IBM': 'Tech', 'ACN': 'Tech',
    'NOW': 'Tech', 'INTU': 'Tech', 'PANW': 'Tech', 'ANET': 'Tech',
    'FTNT': 'Tech',
    'FCX': 'Materials', 'NEM': 'Materials', 'APD': 'Materials',
    'LIN': 'Materials', 'NUE': 'Materials',
    'AMT': 'REITs', 'PLD': 'REITs', 'WELL': 'REITs', 'O': 'REITs',
}

df['sector'] = df['symbol'].map(SECTOR).fillna('Other')

# --- Sector analysis ---
sec_agg = df.groupby('sector').agg(
    count=('symbol', 'count'),
    pnl_sum=('pnl', 'sum'),
    pnl_mean=('pnl', 'mean'),
    trades_mean=('trades', 'mean'),
    neg_count=('pnl', lambda x: (x < 0).sum()),
    sharpe_mean=('sharpe', 'mean'),
).reset_index()
sec_agg['neg_pct'] = 100 * sec_agg['neg_count'] / sec_agg['count']
sec_agg = sec_agg.sort_values('pnl_sum', ascending=True)

print('=== SECTOR-LEVEL ANALYSIS (all eval periods) ===')
print(f'{"Sector":<16} {"Syms":>5} {"NegPct":>8} {"AvgTrades":>10}'
      f' {"AvgPnL":>10} {"TotalPnL":>12} {"AvgSharpe":>10}')
print('-' * 75)
for _, r in sec_agg.iterrows():
    print(f'{r["sector"]:<16} {r["count"]:>5}  {r["neg_pct"]:>6.0f}%'
          f'  {r["trades_mean"]:>9.1f}  ${r["pnl_mean"]:>+8,.0f}'
          f'  ${r["pnl_sum"]:>+10,.0f}  {r["sharpe_mean"]:>+9.2f}')

print()
# --- Trade bucket analysis ---
print('=== TRADE BUCKET vs PnL (all symbols) ===')
bins = [0, 3, 6, 10, 15, 100]
labels = ['1-3', '4-6', '7-10', '11-15', '16+']
df['bucket'] = pd.cut(df['trades'], bins=bins, labels=labels)
bucket_agg = df.groupby('bucket', observed=True).agg(
    count=('symbol', 'count'),
    pnl_mean=('pnl', 'mean'),
    pnl_sum=('pnl', 'sum'),
    neg_pct=('pnl', lambda x: 100 * (x < 0).sum() / len(x)),
).reset_index()

print(f'{"Trades":>8} {"Count":>6} {"AvgPnL":>10} {"TotalPnL":>12} {"NegPct":>8}')
print('-' * 50)
for _, r in bucket_agg.iterrows():
    print(f'{str(r["bucket"]):>8} {r["count"]:>6}  ${r["pnl_mean"]:>+8,.0f}'
          f'  ${r["pnl_sum"]:>+10,.0f}  {r["neg_pct"]:>6.0f}%')

print()
corr = df['trades'].corr(df['pnl'])
print(f'Pearson corr(trades, pnl): {corr:.3f}')

# --- Worst losers detail (sectors of interest) ---
bad_sectors = ['Energy', 'Healthcare', 'Staples', 'Telecom', 'Utilities']
bad_df = df[df['sector'].isin(bad_sectors) & (df['pnl'] < 0)].sort_values('pnl')
print()
print('=== OVER-TRADING LOSERS in defensive/cyclical sectors ===')
print(f'{"Symbol":<8} {"Sector":<12} {"PnL":>10} {"TradePnL":>11}'
      f' {"Trades":>7} {"Supp%":>7} {"Sharpe":>7}')
print('-' * 70)
for _, r in bad_df.head(30).iterrows():
    print(f'{r["symbol"]:<8} {r["sector"]:<12} ${r["pnl"]:>+8,.0f}'
          f'  ${r["trade_pnl"]:>+8,.0f}  {r["trades"]:>6}'
          f'   {r["suppressed_pct"]:>4}%  {r["sharpe"]:>+6.2f}')

print()
print(f'Total loss from these sectors (neg only): ${bad_df["pnl"].sum():+,.0f}')
print(f'Avg trades per loser in these sectors:    {bad_df["trades"].mean():.1f}')
