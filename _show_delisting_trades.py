"""
_show_delisting_trades.py -- Sample calculation: 7-day NaN-streak delisting events.

Finds every mid-month sell+replace event triggered by the 7-day rule, then shows
the price-by-price streak trace and the full sell/buy cash-flow for each one.
"""
import re
import numpy as np
import pandas as pd

from waverider import (
    WaveRiderStrategy, WaveRiderConfig,
    load_universe, load_spy, clean_uid,
)

DATED_PAT = re.compile(r"^([A-Z\-]+)-(\d{6})$")

# ---------------------------------------------------------------------------
# Load and run
# ---------------------------------------------------------------------------
print("Loading data and running backtest …")
prices, rankings = load_universe()
spy_price = load_spy()

config  = WaveRiderConfig()
strategy = WaveRiderStrategy(config)
result  = strategy.backtest(prices, spy_price, rankings)

rebal_set = set(result.rebalance_dates)

# ---------------------------------------------------------------------------
# Step 1 -- identify mid-month trade dates (streak-triggered, NOT EOM)
# ---------------------------------------------------------------------------
mid_month_dates = sorted(d for d in result.trades_log if d not in rebal_set)
eom_trade_dates = sorted(d for d in result.trades_log if d in rebal_set)

print(f"\nTrades summary:")
print(f"  EOM rebalance dates with trades : {len(eom_trade_dates)}")
print(f"  Mid-month delisting events      : {len(mid_month_dates)}  <- 7-day rule triggered")
print()

# ---------------------------------------------------------------------------
# Step 2 -- for each dated UID ever held, simulate the 7-day streak
#           to find the exact confirm date and sell price
# ---------------------------------------------------------------------------
all_ever_held: set = set()
for syms in result.holdings_log.values():
    all_ever_held.update(syms)
held_dated = sorted(u for u in all_ever_held if DATED_PAT.match(u) and u in prices.columns)

delistings = []
for uid in held_dated:
    px = prices[uid]
    valid = px.dropna()
    if valid.empty:
        continue
    last_valid_date  = valid.index[-1]
    last_valid_price = float(valid.iloc[-1])

    # Walk forward from last valid date and count NaN trading days
    forward = px.loc[last_valid_date:]  # index[0] = last valid (price OK)
    streak = 0
    confirm_date = None
    for d, v in forward.iloc[1:].items():          # skip the last valid day itself
        if pd.isna(v) or v <= 0:
            streak += 1
            if streak >= 7:
                confirm_date = d
                break
        else:
            streak = 0    # stock came back (edge case)

    if confirm_date is None:
        continue   # delisted too close to end of data -- streak never hit 7

    # Find all rebalances where this UID was held
    held_at = sorted(rd for rd, syms in result.holdings_log.items() if uid in syms)
    if not held_at:
        continue

    # Rebalance immediately before the last valid date = entry context
    entry_rebals = [rd for rd in result.rebalance_dates if rd <= last_valid_date]
    entry_rebal  = entry_rebals[-1] if entry_rebals else held_at[0]
    is_mid_month = (confirm_date not in rebal_set)

    delistings.append(dict(
        uid=uid, sym=clean_uid(uid),
        last_valid_date=last_valid_date,
        last_valid_price=last_valid_price,
        confirm_date=confirm_date,
        is_mid_month=is_mid_month,
        entry_rebal=entry_rebal,
        held_at=held_at,
    ))

mid_dl = [d for d in delistings if d["is_mid_month"]]
eom_dl = [d for d in delistings if not d["is_mid_month"]]

print(f"Held dated UIDs with 7-day streak confirmed:")
print(f"  Mid-month confirms : {len(mid_dl)}")
print(f"  EOM-day confirms   : {len(eom_dl)}")
print(f"  Total              : {len(delistings)}")

# ---------------------------------------------------------------------------
# Step 3 -- print detailed examples
# ---------------------------------------------------------------------------
SHOW_N = min(5, len(mid_dl))
examples = mid_dl[:SHOW_N] if mid_dl else delistings[:SHOW_N]
if not examples:
    print("\nNo confirmed delisting events found in backtest period.")
else:
    print(f"\n{'='*72}")
    print(f"  DETAILED EXAMPLES -- MID-MONTH 7-DAY DELISTING CONFIRMATION")
    print(f"{'='*72}")

for i, dl in enumerate(examples, 1):
    uid          = dl["uid"]
    sym          = dl["sym"]
    last_dt      = dl["last_valid_date"]
    last_px      = dl["last_valid_price"]
    confirm_dt   = dl["confirm_date"]
    entry_rebal  = dl["entry_rebal"]
    sell_price   = float(prices[uid].ffill().reindex([confirm_dt]).iloc[0])

    # Window: 3 trading days before last valid through the confirm day
    all_dates = prices.index
    last_loc  = all_dates.get_loc(last_dt)
    conf_loc  = all_dates.get_loc(confirm_dt)
    win_start = max(0, last_loc - 3)
    window    = all_dates[win_start : conf_loc + 1]

    px_win = prices.loc[window, uid]

    print(f"\n  -- Example {i}: {sym}  ({uid}) --")
    print(f"  Last valid price : {last_dt.date()}  ${last_px:.4f}")
    print(f"  Confirmed (day7) : {confirm_dt.date()}  (sell at ffill = ${sell_price:.4f})")

    # ----- Price-series trace -----
    print()
    print(f"  {'Date':<14}  {'Raw Price':>12}  {'NaN Streak':>10}  Annotation")
    print(f"  {'-'*58}")
    streak = 0
    for d in window:
        v = float(px_win.get(d, float("nan")))
        if pd.isna(v) or v <= 0:
            streak += 1
            if streak == 7:
                ann = "<- streak = 7  CONFIRMED -- sell triggered"
            else:
                ann = f"streak = {streak}"
            print(f"  {str(d.date()):<14}  {'NaN':>12}  {streak:>10}  {ann}")
        else:
            streak = 0
            print(f"  {str(d.date()):<14}  ${v:>11.4f}  {'0':>10}")

    # ----- Sell calculation -----
    if entry_rebal and entry_rebal in result.holdings_log:
        holdings_at = result.holdings_log[entry_rebal]
        nav_at      = float(result.nav_unlevered.loc[entry_rebal])
        n_held      = len(holdings_at)
        pos_val     = nav_at / n_held            # equal-weight approx
        shares_est  = pos_val / last_px if last_px > 0 else 0.0
        proceeds    = shares_est * sell_price
        tc_bps      = config.transaction_cost_bps
        net_proceeds = proceeds * (1 - tc_bps / 10_000)

        print()
        print(f"  SELL CALCULATION (equal-weight approx)")
        print(f"  {'-'*50}")
        print(f"  Entry rebal date      : {entry_rebal.date()}")
        print(f"  NAV at entry rebal    : {nav_at:.6f}  (normalized; 1.0 = starting capital)")
        print(f"  Holdings count        : {n_held}  stocks")
        print(f"  Position value (~)    : {nav_at:.6f} / {n_held} = {pos_val:.6f}")
        print(f"  Entry price           : ${last_px:.4f}  (last valid price = ffill anchor)")
        print(f"  Shares (est.)         : {pos_val:.6f} / ${last_px:.4f} = {shares_est:.6f} shares")
        print()
        print(f"  Sell price (ffill)    : ${sell_price:.4f}  (frozen at last valid -- no loss)")
        print(f"  Gross proceeds        : {shares_est:.6f} x ${sell_price:.4f} = {proceeds:.6f}")
        print(f"  Transaction cost      : {tc_bps} bps = x{1 - tc_bps/10_000:.4f}")
        print(f"  Net cash to pool      : {net_proceeds:.6f}  <- available for replacement buy")

    # ----- Replacement bought -----
    # Look at the next rebalance's holdings to find what was newly added
    prev_rebals  = [rd for rd in result.rebalance_dates if rd <  confirm_dt]
    next_rebals  = [rd for rd in result.rebalance_dates if rd >= confirm_dt]
    prev_rebal   = prev_rebals[-1] if prev_rebals else None
    next_rebal   = next_rebals[0]  if next_rebals else None

    if prev_rebal and next_rebal:
        prev_h = set(result.holdings_log.get(prev_rebal, []))
        next_h = set(result.holdings_log.get(next_rebal, []))
        new_entries = sorted(next_h - prev_h, key=lambda u: clean_uid(u))

        print()
        print(f"  REPLACEMENT(S) -- added between {prev_rebal.date()} and {next_rebal.date()}")
        print(f"  {'-'*50}")
        if new_entries:
            for new_uid in new_entries:
                # Price on confirm day (the buy date for mid-month replacements)
                buy_px = float(prices.at[confirm_dt, new_uid]) if new_uid in prices.columns else float("nan")
                eom_px = float(prices.at[next_rebal,  new_uid]) if new_uid in prices.columns else float("nan")
                print(f"  + {clean_uid(new_uid):<10}  mid-month buy px: ${buy_px:.4f}"
                      f"   (EOM px: ${eom_px:.4f})" if pd.notna(buy_px) else
                      f"  + {clean_uid(new_uid):<10}  (price unavailable on confirm date)")
            if entry_rebal in result.holdings_log:
                nav_at   = float(result.nav_unlevered.loc[entry_rebal])
                pos_val  = nav_at / n_held
                proceeds = (pos_val / last_px) * sell_price if last_px > 0 else 0.0
                net      = proceeds * (1 - config.transaction_cost_bps / 10_000)
                n_buys   = len(new_entries)
                if n_buys > 0:
                    per_buy = net / n_buys
                    for new_uid in new_entries:
                        buy_px = float(prices.at[confirm_dt, new_uid]) \
                                 if (new_uid in prices.columns and
                                     pd.notna(prices.at[confirm_dt, new_uid])) else float("nan")
                        if pd.notna(buy_px) and buy_px > 0:
                            sh_bought = per_buy / buy_px * (1 - config.transaction_cost_bps / 10_000)
                            print(f"    Buy: {per_buy:.6f} / ${buy_px:.4f} x (1-{config.transaction_cost_bps}bps)"
                                  f" = {sh_bought:.6f} shares of {clean_uid(new_uid)}")
        else:
            print(f"  (all next-rebalance holdings were already in prev holdings -- "
                  f"may be a Scenario B tentative hold case)")

    print()

# ---------------------------------------------------------------------------
# Step 4 -- aggregate summary
# ---------------------------------------------------------------------------
print("=" * 72)
print("  AGGREGATE SUMMARY")
print("=" * 72)
print(f"  Total dated UIDs in universe : {len([c for c in prices.columns if DATED_PAT.match(c)])}")
print(f"  Dated UIDs ever held         : {len(held_dated)}")
print(f"  Delistings with 7-day streak : {len(delistings)}")
print(f"    Mid-month (streak < 7 at EOM, confirmed later) : {len(mid_dl)}")
print(f"    EOM-day confirms (streak hit 7 exactly on EOM) : {len(eom_dl)}")
print()
print(f"  Per event the logic is:")
print(f"    1. Day 1-6 NaN  -> tentative hold; NAV still frozen at last valid price via ffill")
print(f"    2. Day 7 NaN    -> confirmed; sell at ffill price (= acquisition price for cash deals)")
print(f"    3. Same day     -> replacement bought immediately using sell proceeds")
print(f"    4. EOM rebal    -> normal; confirmed slots already filled, only remaining slots rotated")
