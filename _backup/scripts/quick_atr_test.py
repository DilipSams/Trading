"""Quick ATR Regime IC Test on MSFT"""
import numpy as np
import yfinance as yf

# Download MSFT
print("Downloading MSFT 10-year data...")
df = yf.Ticker("MSFT").history(period="10y")
closes = df['Close'].values
highs = df['High'].values
lows = df['Low'].values

# Compute ATR
def compute_atr(highs, lows, closes, period=20):
    tr = np.maximum(highs[1:] - lows[1:],
                   np.maximum(abs(highs[1:] - closes[:-1]),
                             abs(lows[1:] - closes[:-1])))
    atr_list = []
    for i in range(len(tr)):
        if i < period - 1:
            atr_list.append(np.mean(tr[:i+1]))
        else:
            atr_list.append(np.mean(tr[i-period+1:i+1]))
    return np.array([0] + atr_list)

atr = compute_atr(highs, lows, closes, period=20)

# Compute ATR regime z-score
atr_regime_z = []
for i in range(len(atr)):
    if i < 60:
        atr_regime_z.append(0.0)
    else:
        atr_hist = atr[i-60:i]
        z = (atr[i] - np.mean(atr_hist)) / (np.std(atr_hist) + 1e-10)
        atr_regime_z.append(z)

atr_regime_z = np.array(atr_regime_z)

# Compute forward returns (5-bar)
horizon = 5
forward_rets = []
for i in range(len(closes) - horizon):
    fwd_ret = np.log(closes[i + horizon] / closes[i])
    forward_rets.append(fwd_ret)

# Align arrays
atr_signals = atr_regime_z[:-horizon]
forward_rets = np.array(forward_rets)

# Filter valid signals (after warmup)
valid_idx = atr_signals != 0
atr_signals = atr_signals[valid_idx]
forward_rets = forward_rets[valid_idx]

# Compute IC
ic = np.corrcoef(atr_signals, forward_rets)[0, 1]
n = len(atr_signals)
t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

print(f"\n{'='*60}")
print(f"ATR REGIME IC TEST - MSFT")
print(f"{'='*60}")
print(f"Samples:     {n}")
print(f"IC:          {ic:+.4f}")
print(f"t-stat:      {t_stat:+.2f}")
print(f"Significant: {'YES' if abs(t_stat) > 1.96 else 'NO'}")
print(f"\nDecision: {'PASS' if abs(ic) >= 0.003 else 'FAIL'} (threshold: +0.003)")
print(f"{'='*60}\n")
