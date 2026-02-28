"""
Cross-check: Wave Rider T5 MS BearVol2x vs actual stock performance.
Part 1: Did we capture the big winners? (1yr, 3yr, 5yr top performers)
Part 2: Did we avoid boom-then-bust stocks? (ran up then crashed/delisted)
Part 3: Delisted stock audit — every delisted stock we ever held.
Part 4: Year-by-year missed mega-winners.

Uses the canonical WaveRiderStrategy from waverider.py (single source of truth).
"""
import numpy as np
import pandas as pd
import re

from waverider import (
    WaveRiderStrategy, WaveRiderConfig, load_universe, load_spy, clean_uid,
)

# --- Load data & run backtest ---
print("  Loading data...", end="", flush=True)
prices_df, rankings = load_universe()
spy_price = load_spy()
print(" done.")

print("  Running backtest...", end="", flush=True)
config = WaveRiderConfig()
strategy = WaveRiderStrategy(config)
result = strategy.backtest(prices_df, spy_price, rankings)
print(" done.\n")

# --- Build cross-check variables from backtest result ---
membership = strategy.build_membership(prices_df, rankings)
all_prices = prices_df.copy()

# Convert holdings_log (date -> list) to strategy_holdings (date -> set)
strategy_holdings = {d: set(h) for d, h in result.holdings_log.items()}
rebal_dates = result.rebalance_dates


def is_delisted(uid):
    return bool(re.match(r"^(.+)-(\d{6})$", uid))

def get_base_symbol(uid):
    m = re.match(r"^(.+)-(\d{6})$", uid)
    if m:
        return m.group(1)
    return uid

# ============================================================
# PART 1: DID WE CAPTURE THE WINNERS?
# ============================================================
print("=" * 130)
print("  PART 1: DID WAVE RIDER T5 MS BearVol2x CAPTURE THE BIG WINNERS?")
print("  For each snapshot year, show top-20 performers and whether we held them")
print("=" * 130)

# Returns over various horizons
ret_1y = all_prices.pct_change(252)
ret_3y = all_prices.pct_change(756)
ret_5y = all_prices.pct_change(1260)

snapshot_years = [2005, 2008, 2010, 2013, 2016, 2019, 2022, 2025]

for snap_year in snapshot_years:
    # Find last trading day of that year (or latest available)
    year_dates = all_prices.index[all_prices.index.year == snap_year]
    if len(year_dates) == 0:
        continue
    snap_date = year_dates[-1]

    # Get strategy holdings for this year (all months)
    year_held = set()
    for rd in rebal_dates:
        if rd.year == snap_year:
            year_held |= strategy_holdings[rd]

    print(f"\n  {'='*120}")
    print(f"  YEAR-END {snap_year} (snapshot: {snap_date.strftime('%Y-%m-%d')})")
    print(f"  Strategy held {len(year_held)} unique stocks this year: {', '.join(sorted(clean_uid(s) for s in year_held))}")
    print(f"  {'='*120}")

    for horizon_name, ret_series, min_year_offset in [("1-Year", ret_1y, 1), ("3-Year", ret_3y, 3), ("5-Year", ret_5y, 5)]:
        if snap_date not in ret_series.index:
            continue
        rets_at_snap = ret_series.loc[snap_date].dropna()
        # Only include stocks that were in our universe at some point in the snapshot year
        in_univ = [uid for uid in rets_at_snap.index if uid in membership.columns and membership.loc[snap_date, uid] == 1]
        if len(in_univ) == 0:
            # Fallback: use all stocks with valid returns
            in_univ = list(rets_at_snap.index)
        rets_filtered = rets_at_snap.reindex(in_univ).dropna().sort_values(ascending=False)

        top20 = rets_filtered.head(20)
        if len(top20) == 0:
            continue

        captured = sum(1 for uid in top20.index if uid in year_held)
        capture_rate = captured / len(top20) * 100

        print(f"\n  {horizon_name} Return — Top 20 Performers (in universe)  [Capture Rate: {captured}/{len(top20)} = {capture_rate:.0f}%]")
        print(f"  {'Rank':<5s} {'Stock':<12s} {'Return':>10s} {'Held?':>8s}  Notes")
        print(f"  {'-'*80}")

        for rank, (uid, ret) in enumerate(top20.items(), 1):
            held = "YES" if uid in year_held else ""
            notes = ""
            if is_delisted(uid):
                last_trade = all_prices[uid].dropna().index[-1]
                notes = f"delisted (last: {last_trade.strftime('%Y-%m')})"
            elif held:
                # Find which months we held it
                months_held = []
                for rd in rebal_dates:
                    if rd.year == snap_year and uid in strategy_holdings[rd]:
                        months_held.append(rd.month)
                if months_held:
                    notes = f"held months: {','.join(str(m) for m in months_held)}"

            print(f"  {rank:<5d} {clean_uid(uid):<12s} {ret*100:>+9.1f}% {held:>8s}  {notes}")

# Overall capture summary
print(f"\n\n{'='*130}")
print(f"  CAPTURE RATE SUMMARY ACROSS ALL YEARS")
print(f"{'='*130}")
print(f"  {'Year':<6s} {'1yr Top20':>12s} {'3yr Top20':>12s} {'5yr Top20':>12s}  Unique Stocks Held")
print(f"  {'-'*80}")

for snap_year in snapshot_years:
    year_dates = all_prices.index[all_prices.index.year == snap_year]
    if len(year_dates) == 0:
        continue
    snap_date = year_dates[-1]
    year_held = set()
    for rd in rebal_dates:
        if rd.year == snap_year:
            year_held |= strategy_holdings[rd]

    rates = []
    for ret_series in [ret_1y, ret_3y, ret_5y]:
        if snap_date not in ret_series.index:
            rates.append("n/a")
            continue
        rets = ret_series.loc[snap_date].dropna()
        in_univ = [u for u in rets.index if u in membership.columns and membership.loc[snap_date, u] == 1]
        if not in_univ:
            in_univ = list(rets.index)
        top20 = rets.reindex(in_univ).dropna().nlargest(20)
        captured = sum(1 for u in top20.index if u in year_held)
        rates.append(f"{captured}/20={captured*5:.0f}%")

    n_stocks = len(year_held)
    print(f"  {snap_year:<6d} {rates[0]:>12s} {rates[1]:>12s} {rates[2]:>12s}  {n_stocks} stocks")


# ============================================================
# PART 2: DID WE AVOID THE BOOM-THEN-BUST STOCKS?
# ============================================================
print(f"\n\n{'='*130}")
print(f"  PART 2: BOOM-THEN-BUST STOCKS — DID WE RIDE THE WAVE AND EXIT IN TIME?")
print(f"  Criteria: >100% gain in trailing 12m, then >50% drop from peak within next 12m")
print(f"{'='*130}")

# For each stock, find boom-bust episodes
boom_bust_records = []

for uid in all_prices.columns:
    p = all_prices[uid].dropna()
    if len(p) < 504:  # Need at least 2 years
        continue

    r12m = p.pct_change(252)

    for date in p.index[252:]:
        r = r12m.get(date, np.nan)
        if pd.isna(r) or r < 1.0:  # Not a >100% boom
            continue

        # Found a boom. Now check if bust follows within 12 months
        peak_price = p.loc[date]
        future = p.loc[date:]
        if len(future) < 2:
            continue

        # Look at next 252 trading days
        future_12m = future.iloc[:min(252, len(future))]
        min_after = future_12m.min()
        drawdown = (min_after - peak_price) / peak_price

        if drawdown < -0.50:  # >50% crash
            crash_date = future_12m.idxmin()
            # Was stock delisted?
            delisted = is_delisted(uid)
            last_trade = p.index[-1]

            boom_bust_records.append({
                "uid": uid,
                "symbol": clean_uid(uid),
                "boom_date": date,
                "boom_12m_ret": r,
                "peak_price": peak_price,
                "crash_date": crash_date,
                "crash_drawdown": drawdown,
                "delisted": delisted,
                "last_trade": last_trade,
            })
            break  # One episode per stock (first one)

boom_bust_df = pd.DataFrame(boom_bust_records)

if len(boom_bust_df) > 0:
    # Sort by boom date
    boom_bust_df = boom_bust_df.sort_values("boom_date")

    print(f"\n  Found {len(boom_bust_df)} boom-then-bust episodes")
    print(f"\n  {'Symbol':<12s} {'Boom Date':<11s} {'12m Gain':>9s} {'Crash':>8s} {'Crash Date':<11s} {'Delisted':>9s} {'We Held?':>9s} {'Exit OK?':>9s}")
    print(f"  {'-'*100}")

    n_held_during_boom = 0
    n_exited_before_crash = 0
    n_held_through_crash = 0

    for _, row in boom_bust_df.iterrows():
        uid = row["uid"]
        boom_date = row["boom_date"]
        crash_date = row["crash_date"]

        # Check if we held this stock during the boom period (12m before boom_date)
        held_during_boom = False
        exited_before_crash = True
        held_months_boom = []
        held_months_crash = []

        for rd in rebal_dates:
            if uid in strategy_holdings.get(rd, set()):
                # Boom window: 12 months before boom_date to boom_date
                boom_start = boom_date - pd.Timedelta(days=365)
                if boom_start <= rd <= boom_date:
                    held_during_boom = True
                    held_months_boom.append(rd.strftime("%Y-%m"))
                # Crash window: boom_date to crash_date
                if boom_date < rd <= crash_date:
                    exited_before_crash = False
                    held_months_crash.append(rd.strftime("%Y-%m"))

        if held_during_boom:
            n_held_during_boom += 1
            if exited_before_crash:
                n_exited_before_crash += 1
            else:
                n_held_through_crash += 1

        held_str = "YES" if held_during_boom else ""
        exit_str = ""
        if held_during_boom:
            exit_str = "EXITED" if exited_before_crash else "CAUGHT!"

        print(f"  {row['symbol']:<12s} {boom_date.strftime('%Y-%m'):<11s} {row['boom_12m_ret']*100:>+8.0f}% {row['crash_drawdown']*100:>+7.0f}% {crash_date.strftime('%Y-%m'):<11s} {'YES' if row['delisted'] else '':>9s} {held_str:>9s} {exit_str:>9s}")

    print(f"\n  BOOM-BUST SUMMARY:")
    print(f"    Total boom-bust episodes found: {len(boom_bust_df)}")
    print(f"    Episodes where we held during the boom: {n_held_during_boom}")
    print(f"    - Exited before the crash (GOOD): {n_exited_before_crash}")
    print(f"    - Held through the crash (BAD):   {n_held_through_crash}")
    if n_held_during_boom > 0:
        dodge_rate = n_exited_before_crash / n_held_during_boom * 100
        print(f"    Dodge rate: {dodge_rate:.0f}% (% of boom-bust stocks we exited in time)")
else:
    print("\n  No boom-bust episodes found in the data.")


# ============================================================
# PART 3: DELISTED STOCK AUDIT
# ============================================================
print(f"\n\n{'='*130}")
print(f"  PART 3: DELISTED STOCK AUDIT — EVERY DELISTED STOCK WAVE RIDER T5 MS HELD")
print(f"{'='*130}")

# Find all delisted UIDs that were ever in our strategy holdings
all_held_uids = set()
for rd, holdings in strategy_holdings.items():
    all_held_uids |= holdings

delisted_held = [uid for uid in all_held_uids if is_delisted(uid)]
delisted_held.sort()

if len(delisted_held) > 0:
    print(f"\n  Found {len(delisted_held)} delisted stocks in our T5 holdings history")
    print(f"\n  {'Symbol':<15s} {'Base':<10s} {'First Held':<11s} {'Last Held':<11s} {'Last Trade':<11s} {'Months':>7s} {'Entry Px':>9s} {'Exit Px':>9s} {'P&L':>9s} {'Held@Delist?':>13s}")
    print(f"  {'-'*120}")

    for uid in delisted_held:
        base = get_base_symbol(uid)
        p = all_prices[uid].dropna()
        if len(p) == 0:
            continue
        last_trade = p.index[-1]

        # Find when we held this stock
        held_dates = [rd for rd in rebal_dates if uid in strategy_holdings.get(rd, set())]
        if not held_dates:
            continue

        first_held = min(held_dates)
        last_held = max(held_dates)
        months_held = len(held_dates)

        # Entry and exit prices (approximate)
        entry_price = p.asof(first_held) if first_held in p.index or first_held > p.index[0] else p.iloc[0]
        # Exit: price at last_held + ~21 days (next rebalance) or last available
        exit_idx = min(len(p) - 1, p.index.searchsorted(last_held) + 21)
        exit_price = p.iloc[min(exit_idx, len(p) - 1)]

        pnl = (exit_price / entry_price - 1) * 100 if entry_price > 0 else 0

        # Did we hold at delisting?
        held_at_delist = last_held >= last_trade - pd.Timedelta(days=30)

        print(f"  {clean_uid(uid):<15s} {base:<10s} {first_held.strftime('%Y-%m'):<11s} {last_held.strftime('%Y-%m'):<11s} {last_trade.strftime('%Y-%m'):<11s} {months_held:>5d}mo {entry_price:>9.2f} {exit_price:>9.2f} {pnl:>+8.1f}% {'!! YES !!' if held_at_delist else 'No':>13s}")

    # Summary
    n_held_at_delist = sum(1 for uid in delisted_held if max(rd for rd in rebal_dates if uid in strategy_holdings.get(rd, set())) >= all_prices[uid].dropna().index[-1] - pd.Timedelta(days=30))
    print(f"\n  DELISTED AUDIT SUMMARY:")
    print(f"    Total delisted stocks ever held: {len(delisted_held)}")
    print(f"    Held at/near delisting date:     {n_held_at_delist}")
    print(f"    Exited well before delisting:    {len(delisted_held) - n_held_at_delist}")
else:
    print("\n  No delisted stocks found in our T5 holdings history.")


# ============================================================
# PART 4: YEAR-BY-YEAR MISSED MEGA-WINNERS
# ============================================================
print(f"\n\n{'='*130}")
print(f"  PART 4: BIGGEST MISSED WINNERS — STOCKS WITH >50% 1YR RETURN THAT WE DIDN'T HOLD")
print(f"{'='*130}")

for snap_year in range(2005, 2026):
    year_dates = all_prices.index[all_prices.index.year == snap_year]
    if len(year_dates) == 0:
        continue
    snap_date = year_dates[-1]

    year_held = set()
    for rd in rebal_dates:
        if rd.year == snap_year:
            year_held |= strategy_holdings[rd]

    if snap_date not in ret_1y.index:
        continue
    rets = ret_1y.loc[snap_date].dropna()
    # Only in-universe stocks
    in_univ = [u for u in rets.index if u in membership.columns and membership.loc[snap_date, u] == 1]
    rets_univ = rets.reindex(in_univ).dropna().sort_values(ascending=False)

    # Big winners we missed
    big_winners = rets_univ[rets_univ > 0.50]
    missed = [(uid, ret) for uid, ret in big_winners.items() if uid not in year_held]
    captured = [(uid, ret) for uid, ret in big_winners.items() if uid in year_held]

    if len(big_winners) == 0:
        continue

    cap_rate = len(captured) / len(big_winners) * 100 if len(big_winners) > 0 else 0

    # Compact output
    missed_str = ", ".join(f"{clean_uid(u)}({r*100:+.0f}%)" for u, r in missed[:10])
    captured_str = ", ".join(f"{clean_uid(u)}({r*100:+.0f}%)" for u, r in captured[:10])

    print(f"\n  {snap_year}: {len(big_winners)} stocks >50% return | Captured {len(captured)}/{len(big_winners)} ({cap_rate:.0f}%)")
    if captured_str:
        print(f"    HELD:   {captured_str}")
    if missed_str:
        trunc = "..." if len(missed) > 10 else ""
        print(f"    MISSED: {missed_str}{trunc}")

print(f"\n\n{'='*130}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*130}")
