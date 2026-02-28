"""Check step_decisions.csv to identify run source and compute per-symbol stats."""
import pandas as pd, numpy as np, re

sd = pd.read_csv(r'C:\Users\dilip\alpha_trade_v2_results\step_decisions.csv', low_memory=False)
print(f'Total rows: {len(sd):,}')
print(f'Unique symbols: {sd.symbol.nunique()}')
print(f'Date range: {sd.timestamp.min()} to {sd.timestamp.max()}')
print()

# Check EOG trades (should be 29 in V7, fewer in V8+fixes)
eog = sd[sd.symbol == 'EOG_1d'].copy().reset_index(drop=True)
print(f'EOG bars: {len(eog)}')
pos_changes = (eog.position != eog.position.shift(1)).sum()
print(f'EOG position changes: {pos_changes}')
print(f'EOG final portfolio: {eog.portfolio.iloc[-1]:.2f}' if len(eog) > 0 else 'N/A')

# Check SMA-related reasons
if 'reason' in eog.columns:
    sma_rows = eog[eog.reason.str.contains('sma|SMA|sticky|churn', na=False, case=False)]
    print(f'EOG SMA/sticky/churn actions: {len(sma_rows)}')
    if len(sma_rows) > 0:
        print('Sample:', sma_rows.reason.iloc[0])

print()
# Check action distribution for EOG
print('EOG action counts:', dict(eog.action.value_counts()))
