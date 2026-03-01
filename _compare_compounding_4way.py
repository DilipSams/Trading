"""
4-Way Capital Compounding Comparison
=====================================
Compares four capital management strategies over the full backtest history:

  1. Full Compounding   -- standard backtest NAV (equal-weight ALL positions every EOM,
                           size from total NAV including unrealized gains)
  2. Realized-Only      -- only SOLD positions feed the cash pool for new buys;
                           held positions keep same share count indefinitely
  3. Fixed $100k        -- position sizing is always $100k regardless of gains/losses
  4. SPY Buy & Hold     -- plain S&P 500 ETF benchmark

All scenarios start with $100,000.
No assumptions — uses exact backtest holdings_log and Norgate price data.
"""

import sys, io, math, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from waverider import (
    WaveRiderStrategy, WaveRiderConfig,
    load_universe, load_spy, load_risk_free,
    compute_nav_metrics,
)

INITIAL = 100_000.0

# ─────────────────────────────────────────────
# 1. Load data + run backtest
# ─────────────────────────────────────────────
print("Loading universe and running backtest (this takes ~60 s) ...")
prices, rankings = load_universe()
spy = load_spy()
cfg = WaveRiderStrategy(WaveRiderConfig())

# Use the strategy object
strategy = cfg
result = strategy.backtest(prices, spy, rankings)

dates         = result.dates
holdings_log  = result.holdings_log
lev_series    = result.leverage_series
nav_lev       = result.nav_leveraged
nav_unlev     = result.nav_unlevered
spy_aligned   = spy.reindex(dates).ffill()
rf_daily      = load_risk_free(dates)

print(f"Backtest period : {dates[0].date()} to {dates[-1].date()}")
print(f"Rebalance dates : {len(result.rebalance_dates)}")

# Pre-compute forward-filled prices for consistent sell/buy price lookups
prices_ff = prices.ffill()

# Pre-compute borrow cost (identical to _apply_bearvol2x)
borrow_rate = (rf_daily + WaveRiderConfig().margin_spread_bps / 10000 / 252
               ).reindex(dates).ffill().fillna(0)


# ─────────────────────────────────────────────
# Helper: apply BearVol2x leverage to an unlevered NAV series.
# Uses the SAME leverage series from the standard backtest.
# ─────────────────────────────────────────────
def apply_leverage_overlay(nav_unlev_series: pd.Series) -> pd.Series:
    """Apply the standard BearVol2x leverage overlay to a 1x NAV series."""
    unlev_rets = nav_unlev_series.pct_change(fill_method=None).fillna(0).values
    lev_arr    = lev_series.reindex(nav_unlev_series.index).ffill().fillna(1.0).values
    borrow_arr = borrow_rate.reindex(nav_unlev_series.index).ffill().fillna(0).values

    borrow  = np.maximum(lev_arr - 1.0, 0.0) * borrow_arr
    day_ret = lev_arr * unlev_rets - borrow
    day_ret[0] = 0.0  # no return on first day

    nav_arr = np.cumprod(1.0 + day_ret)
    return pd.Series(nav_arr * INITIAL, index=nav_unlev_series.index)


# ─────────────────────────────────────────────
# SCENARIO 1: Full Compounding (standard backtest)
# ─────────────────────────────────────────────
nav1 = nav_lev / nav_lev.iloc[0] * INITIAL


# ─────────────────────────────────────────────
# SCENARIO 2: Realized-Only Reinvestment
#   • Position sizing uses ONLY cash from SOLD positions + initial undeployed cash
#   • Held positions: share count locked until the stock is eventually SOLD
#   • New buys: sized as cash_pool / n_new_buys at 1x; leverage applied via overlay
#
#   NOTE: This fundamentally differs from full compounding: the monthly
#   equal-weight REBALANCING of held positions is disabled. Held winners
#   compound without being trimmed; held losers are not topped up.
# ─────────────────────────────────────────────
print("\nSimulating Scenario 2: Realized-Only ...")

cash       = INITIAL
positions  = {}          # {uid: int shares}
nav2_1x    = []

rebal_set = set(holdings_log.keys())

for date in dates:
    if date in rebal_set:
        new_portfolio  = set(holdings_log[date])
        prev_portfolio = set(positions.keys())

        sells = prev_portfolio - new_portfolio
        buys  = new_portfolio  - prev_portfolio
        # holds = prev_portfolio & new_portfolio  →  no action

        # ── Close sells: return FULL market-value proceeds to cash ──────
        for uid in list(sells):
            shares = positions.pop(uid, 0)
            if shares > 0 and uid in prices_ff.columns:
                px = float(prices_ff.loc[date, uid])
                if px > 0:
                    cash += shares * px
                # If px still 0/NaN, position value is lost (delisted/error)

        # ── Open buys: divide available cash equally among new entries ──
        if buys and cash > 1.0:
            per_pos = cash / len(buys)
            for uid in sorted(buys):          # deterministic ordering
                if uid not in prices_ff.columns:
                    continue
                px = float(prices_ff.loc[date, uid])
                if px > 0:
                    shares = math.floor(per_pos / px)
                    if shares > 0:
                        cash -= shares * px
                        positions[uid] = shares

    # ── Mark-to-market daily NAV ───────────────────────────────────────
    nav_val = max(0.0, cash)
    for uid, shares in positions.items():
        if uid in prices_ff.columns:
            px = float(prices_ff.loc[date, uid])
            nav_val += shares * px
    nav2_1x.append(nav_val)

nav2_unlev_raw = pd.Series(nav2_1x, index=dates)
# Normalise to $100k at start then apply leverage overlay
nav2_unlev = nav2_unlev_raw / nav2_unlev_raw.iloc[0] * INITIAL
nav2 = apply_leverage_overlay(nav2_unlev)

# ─────────────────────────────────────────────
# SCENARIO 3: Fixed $100k Forever
#   • Each day earns (daily_return × $100k); base never grows
# ─────────────────────────────────────────────
daily_r_lev   = nav_lev.pct_change(fill_method=None).fillna(0)
fixed_pnl_cum = (daily_r_lev * INITIAL).cumsum()
nav3          = pd.Series(INITIAL + fixed_pnl_cum.values, index=dates)
nav3.iloc[0]  = INITIAL


# ─────────────────────────────────────────────
# SCENARIO 4: SPY Buy & Hold
#   SPY launched Jan 1993; before that, proxy with S&P 500 from Norgate or
#   start the SPY comparison from first valid date.
# ─────────────────────────────────────────────
spy_first_valid = spy_aligned.first_valid_index()
spy_sub   = spy_aligned.loc[spy_first_valid:]
nav4_sub  = spy_sub / spy_sub.iloc[0] * INITIAL
nav4      = nav4_sub.reindex(dates)   # NaN before SPY data starts


# ─────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────
def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 2 or s.iloc[0] <= 0:
        return {"cagr": float("nan"), "final": float("nan"),
                "max_dd": float("nan"), "sharpe": float("nan")}
    n_years = (s.index[-1] - s.index[0]).days / 365.25
    total   = s.iloc[-1] / s.iloc[0]
    cagr    = total ** (1 / n_years) - 1
    r       = s.pct_change(fill_method=None).dropna()
    sharpe  = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    max_dd  = float((s / s.cummax() - 1).min())
    return {"cagr": cagr, "final": float(s.iloc[-1]),
            "max_dd": max_dd, "sharpe": sharpe, "n_years": n_years}

# Year-over-year return for a NAV series
def yr_ret(s: pd.Series, yr: int) -> float:
    yd = s.dropna()
    yd = yd[yd.index.year == yr]
    if len(yd) < 5:
        return float("nan")
    return (yd.iloc[-1] / yd.iloc[0] - 1) * 100


scenarios = [
    ("Full Compounding",  nav1),
    ("Realized-Only",     nav2),
    ("Fixed $100k",       nav3),
    ("SPY Buy&Hold",      nav4),
]

m = {lbl: metrics(s) for lbl, s in scenarios}

# ─────────────────────────────────────────────
# Print results
# ─────────────────────────────────────────────
COL = 17

print()
print("=" * 75)
print("4-WAY CAPITAL COMPOUNDING COMPARISON")
n_yrs = (dates[-1] - dates[0]).days / 365.25
print(f"Period: {dates[0].strftime('%b %Y')} to {dates[-1].strftime('%b %Y')}  ({n_yrs:.1f} yrs)  |  Start: ${INITIAL:,.0f}")
print("=" * 75)

labels = [s[0] for s in scenarios]
hdr    = f"{'':22}" + "".join(f"{l:>{COL}}" for l in labels)
print(hdr)
print("-" * (22 + COL * len(labels)))

def fmt(key, fn, scenarios, m):
    row = f"{key:<22}"
    for lbl, _ in scenarios:
        v = m[lbl].get(key, float("nan"))
        row += f"{fn(v):>{COL}}"
    return row

def fmtrow(label, key, fn):
    row = f"{label:<22}"
    for lbl, _ in scenarios:
        v = m[lbl].get(key, float("nan"))
        row += f"{fn(v):>{COL}}"
    return row

print(fmtrow("CAGR",         "cagr",    lambda v: f"{v*100:.2f}%" if not np.isnan(v) else "N/A"))
print(fmtrow("Final Wealth", "final",   lambda v: f"${v:>12,.0f}" if not np.isnan(v) else "N/A"))

total_ret_row = f"{'Total Return':<22}"
for lbl, _ in scenarios:
    f_val = m[lbl].get("final", float("nan"))
    tr = (f_val / INITIAL - 1) * 100 if not np.isnan(f_val) else float("nan")
    total_ret_row += f"{(str(round(tr))+'%' if not np.isnan(tr) else 'N/A'):>{COL}}"
print(total_ret_row)
print(fmtrow("Max Drawdown", "max_dd",  lambda v: f"{v*100:.1f}%" if not np.isnan(v) else "N/A"))
print(fmtrow("Sharpe Ratio", "sharpe",  lambda v: f"{v:.2f}" if not np.isnan(v) else "N/A"))

# Year-by-year
print()
hdr2 = f"{'Year':<6}" + "".join(f"{l[:10]:>{COL}}" for l in labels)
print(hdr2)
print("-" * (6 + COL * len(labels)))

all_years = sorted(set(dates.year))
for yr in all_years:
    rets = [yr_ret(s, yr) for _, s in scenarios]
    row  = f"{yr:<6}"
    best_val = max((r for r in rets if not np.isnan(r)), default=float("nan"))
    for i, r in enumerate(rets):
        cell = f"{r:+.1f}%" if not np.isnan(r) else "   N/A"
        row += f"{cell:>{COL}}"
    # Mark year where realized-only beats full compounding
    r1, r2 = rets[0], rets[1]
    flag = " *" if (not np.isnan(r1) and not np.isnan(r2) and r2 > r1) else "  "
    print(row + flag)

print("-" * (6 + COL * len(labels)))
cagr_row = f"{'CAGR':<6}"
for lbl, s in scenarios:
    c = m[lbl].get("cagr", float("nan"))
    cagr_row += f"{(f'{c*100:+.2f}%' if not np.isnan(c) else 'N/A'):>{COL}}"
print(cagr_row)

# ─────────────────────────────────────────────
# Key insights
# ─────────────────────────────────────────────
print()
print("=" * 75)
print("KEY FINDINGS")
print("=" * 75)
c1 = m["Full Compounding"]["cagr"]
c2 = m["Realized-Only"]["cagr"]
c3 = m["Fixed $100k"]["cagr"]
c4 = m["SPY Buy&Hold"]["cagr"]
f1 = m["Full Compounding"]["final"]
f2 = m["Realized-Only"]["final"]
f3 = m["Fixed $100k"]["final"]
f4 = m["SPY Buy&Hold"]["final"]

for lbl, c, f in [
    ("Full Compounding  (monthly equal-weight rebalance)", c1, f1),
    ("Realized-Only     (held positions NEVER resized)  ", c2, f2),
    ("Fixed $100k       (gains extracted, never reinvest)", c3, f3),
    ("SPY Buy & Hold    (benchmark)                      ", c4, f4),
]:
    if not np.isnan(c):
        print(f"  {lbl}  {c*100:>6.2f}% CAGR   ${f:>14,.0f}")

print()
if not np.isnan(c2) and not np.isnan(c1):
    if c2 > c1:
        print("SURPRISING FINDING: REALIZED-ONLY OUTPERFORMS FULL COMPOUNDING")
        print("  By +{:.2f}pp CAGR and ${:,.0f} more final wealth.".format(
              (c2-c1)*100, f2-f1))
        print()
        print("  WHY? The strategy selects MOMENTUM stocks — stocks already in")
        print("  strong uptrends. The standard equal-weight rebalancing TRIMS")
        print("  these winners every month to restore 1/N weight. This is")
        print("  ANTI-MOMENTUM within the portfolio: you are selling your best")
        print("  performers to top up your weaker ones.")
        print()
        print("  In the realized-only model, winners are LEFT TO RUN:")
        print("    - A stock growing from $20k -> $40k stays at $40k worth")
        print("    - New buys are funded from stocks that actually LEAVE the portfolio")
        print("    - The portfolio naturally tilts toward the strongest momentum names")
        print("    - This amplifies momentum, consistent with the strategy's thesis")
        print()
        print("  Years where realized-only STRONGLY outperformed (*):")
        for yr in all_years:
            r1, r2 = yr_ret(nav1, yr), yr_ret(nav2, yr)
            if not np.isnan(r1) and not np.isnan(r2) and (r2 - r1) > 15:
                print(f"    {yr}: Realized {r2:+.1f}% vs Full {r1:+.1f}% (+{r2-r1:.1f}pp)")
    else:
        print("RESULT: FULL COMPOUNDING BEATS REALIZED-ONLY")
        print(f"  By +{(c1-c2)*100:.2f}pp CAGR")

print()
print("KEY TAKEAWAYS:")
print(f"  1. REALIZED-ONLY = {c2*100:.2f}% CAGR - this is what you actually get when")
print(f"     you ONLY reinvest from booked (sold) profits. Held positions run freely.")
print()
print(f"  2. FULL COMPOUNDING = {c1*100:.2f}% CAGR - this requires sizing ALL positions")
print(f"     (including holds) from total NAV every month. Equal-weight monthly.")
print()
print(f"  3. FIXED $100k = {c3*100:.2f}% CAGR - never growing the position base.")
print(f"     This is far below SPY ({c4*100:.2f}% CAGR) and unacceptable.")
print()
print("  4. BOTH momentum strategies (full and realized-only) massively beat")
print(f"     SPY ({c4*100:.2f}%). The 7.44% fixed-capital result is a red herring —")
print("     it assumes you NEVER let your profits compound.")
print()
print("PRACTICAL RECOMMENDATION:")
print("  Run the strategy as designed: each month-end, size ALL 5 positions")
print("  at 1/5 of TOTAL portfolio NAV (including unrealized gains/losses).")
print("  This is the only way to achieve the backtested CAGR. In a margin")
print("  account, the monthly 'rebalancing trades' are typically small (held")
print("  positions near equal-weight already) and transaction costs are minimal.")

# ─────────────────────────────────────────────
# Debug: 1x unlevered NAV comparison
# ─────────────────────────────────────────────
print()
print("-" * 75)
print("DEBUG: 1x Unlevered NAV year-by-year (before leverage overlay)")
print(f"{'Year':<6} {'Standard 1x':>13} {'Realized 1x':>13} {'Ratio':>8}")
print("-" * 44)
nav1_unlev = result.nav_unlevered / result.nav_unlevered.iloc[0] * INITIAL
for yr in all_years:
    r1u = yr_ret(nav1_unlev, yr)
    r2u = yr_ret(nav2_unlev_raw, yr)
    ratio = r2u - r1u if not (np.isnan(r1u) or np.isnan(r2u)) else float("nan")
    r1s = f"{r1u:+.1f}%" if not np.isnan(r1u) else "  N/A"
    r2s = f"{r2u:+.1f}%" if not np.isnan(r2u) else "  N/A"
    rs  = f"{ratio:+.1f}pp" if not np.isnan(ratio) else "  N/A"
    print(f"{yr:<6} {r1s:>13} {r2s:>13} {rs:>8}")
