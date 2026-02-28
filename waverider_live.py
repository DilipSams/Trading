"""
Wave Rider T5 MS BearVol2x — Monthly Rebalance Signal Generator.

Run this script to get the current recommended portfolio, leverage level,
and buy/sell instructions with dollar amounts and share quantities.

Usage:
    python waverider_live.py                   # default $100k capital
    python waverider_live.py --capital 250000   # custom capital
"""
import argparse
import math
import numpy as np
import pandas as pd

from waverider import (
    WaveRiderStrategy, WaveRiderConfig, PortfolioSignal,
    load_universe, load_spy, clean_uid, compute_nav_metrics,
    SECTOR_MAP, SYM_TO_SECTOR,
)


def classify(score: float, cfg: WaveRiderConfig) -> str:
    if score <= cfg.meme_max2:
        return "Clean Growth"
    elif score <= cfg.meme_max1:
        return "Gray Zone"
    elif score <= cfg.meme_exclude:
        return "Meme-Adjacent"
    return "Pure Meme"


def get_current_price(prices: pd.DataFrame, uid: str) -> float:
    """Get last available price for a UID."""
    if uid in prices.columns:
        s = prices[uid].dropna()
        if len(s) > 0:
            return float(s.iloc[-1])
    return 0.0


def get_price_on_date(prices: pd.DataFrame, uid: str, date) -> float:
    """Get price on or closest before a specific date."""
    if uid in prices.columns:
        p = prices[uid]
        val = p.asof(date)
        if pd.notna(val):
            return float(val)
    return 0.0


def find_entry_date(uid: str, holdings_log: dict, rebalance_dates: list) -> pd.Timestamp:
    """Walk backwards through rebalance dates to find when this UID first entered
    (most recent continuous holding streak)."""
    entry_date = None
    for rd in reversed(rebalance_dates):
        if uid in holdings_log.get(rd, []):
            entry_date = rd
        else:
            break  # streak broken — the next one was the entry
    return entry_date


def main():
    parser = argparse.ArgumentParser(description="WR T5 MS BearVol2x — Monthly Signal")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Total portfolio capital in dollars (default: $100,000)")
    args = parser.parse_args()
    capital = args.capital

    print("  Loading data...", end="", flush=True)
    prices, rankings = load_universe()
    spy_price = load_spy()
    print(" done.")

    cfg = WaveRiderConfig()
    strategy = WaveRiderStrategy(cfg)

    print("  Computing signals and replaying history...", end="", flush=True)
    signal = strategy.current_portfolio(prices, spy_price, rankings)
    print(" done.\n")

    # Also get backtest for YTD/recent stats
    result = strategy.backtest(prices, spy_price, rankings)
    m_lev = compute_nav_metrics(result.nav_leveraged)

    # Build UID lookup: clean symbol -> UID (for price lookup)
    uid_map = {}
    for uid in signal.holdings:
        uid_map[clean_uid(uid)] = uid

    # Build entry date + entry price for each current holding
    entry_info = {}  # clean_sym -> (entry_date, entry_price)
    for uid in signal.holdings:
        sym = clean_uid(uid)
        entry_date = find_entry_date(uid, result.holdings_log, result.rebalance_dates)
        entry_price = get_price_on_date(prices, uid, entry_date) if entry_date else 0.0
        entry_info[sym] = (entry_date, entry_price)

    # ===================================================================
    # Header
    # ===================================================================
    print("=" * 110)
    print("  WAVE RIDER T5 MS BearVol2x -- MONTHLY REBALANCE SIGNAL")
    print(f"  Signal date: {signal.date.strftime('%Y-%m-%d')}")
    print(f"  Capital: ${capital:,.0f}")
    print("=" * 110)

    # ===================================================================
    # Leverage
    # ===================================================================
    vol_lev = cfg.target_vol / signal.realized_vol * cfg.target_leverage if signal.realized_vol > 0.01 else cfg.target_leverage
    vol_lev_capped = min(cfg.target_leverage, vol_lev)

    effective_capital = capital * signal.leverage
    n = len(signal.holdings_clean)
    per_stock = effective_capital / n if n > 0 else 0
    weight_pct = 100.0 / n if n > 0 else 0

    print(f"\n  LEVERAGE: {signal.leverage:.2f}x")
    print(f"    Realized vol (21d): {signal.realized_vol:.2f} ann.")
    print(f"    Vol-target leverage: {vol_lev_capped:.2f}x (= {cfg.target_vol}/{signal.realized_vol:.2f} * {cfg.target_leverage})")
    bear_str = "ON  (SPY < SMA200) -> capped at 0.5x" if signal.bear_regime else "OFF (SPY > SMA200)"
    print(f"    Bear gate: {bear_str}")
    print(f"    Effective exposure: {signal.leverage:.2f}x * ${capital:,.0f} = ${effective_capital:,.0f}")

    # ===================================================================
    # Current portfolio with entry dates, prices, and P&L since entry
    # ===================================================================
    print(f"\n  CURRENT PORTFOLIO ({n} stocks, equal weight {weight_pct:.0f}% each, ${per_stock:,.0f}/stock):")
    print(f"    {'#':<3s} {'Symbol':<7s} {'CurPrice':>9s} {'Shares':>7s} {'Alloc':>10s}  "
          f"{'EntryDate':>10s} {'EntryPx':>8s} {'Since':>7s}  {'MemeScr':>7s} {'Classification'}")
    print(f"    {'-' * 104}")

    for i, sym in enumerate(signal.holdings_clean, 1):
        uid = uid_map.get(sym, sym)
        price = get_current_price(prices, uid)
        shares = math.floor(per_stock / price) if price > 0 else 0
        alloc = shares * price
        ms = signal.meme_scores.get(sym, 0)
        cls = classify(ms, cfg)
        ed, ep = entry_info.get(sym, (None, 0))
        ed_str = ed.strftime('%Y-%m-%d') if ed else "n/a"
        ep_str = f"${ep:.2f}" if ep > 0 else "n/a"
        since_entry = f"{(price/ep - 1)*100:+.1f}%" if ep > 0 else "n/a"
        print(f"    {i:<3d} {sym:<7s} ${price:>8.2f} {shares:>6d}   ${alloc:>8,.0f}  "
              f"{ed_str:>10s} {ep_str:>8s} {since_entry:>7s}  {ms:>7.0f}   {cls}")

    cash_remainder = effective_capital - sum(
        math.floor(per_stock / get_current_price(prices, uid_map.get(s, s))) *
        get_current_price(prices, uid_map.get(s, s))
        for s in signal.holdings_clean
        if get_current_price(prices, uid_map.get(s, s)) > 0
    )
    print(f"    {'':3s} {'CASH':<7s} {'':>9s} {'':>7s}   ${cash_remainder:>8,.0f}")
    print(f"    {'-' * 104}")
    print(f"    {'':3s} {'TOTAL':<7s} {'':>9s} {'':>7s}   ${effective_capital:>8,.0f}  "
          f"(${capital:,.0f} x {signal.leverage:.2f}x leverage)")

    # ===================================================================
    # Trades with quantities, dates, and prices
    # ===================================================================
    rebal_date_str = signal.date.strftime('%Y-%m-%d')
    print(f"\n  TRADES THIS MONTH (recommendation date: {rebal_date_str}):")
    if signal.buys:
        for sym in signal.buys:
            uid = uid_map.get(sym, sym)
            price = get_current_price(prices, uid)
            shares = math.floor(per_stock / price) if price > 0 else 0
            alloc = shares * price
            print(f"    BUY:  {sym:<8s} {shares:>6d} shares @ ${price:.2f} = ${alloc:,.0f}  "
                  f"(rec. {rebal_date_str})")
    if signal.sells:
        for sym in signal.sells:
            print(f"    SELL: {sym:<8s} (exit entire position)  (rec. {rebal_date_str})")
    holds = [s for s in signal.holdings_clean if s not in signal.buys]
    if holds:
        hold_details = []
        for s in holds:
            uid = uid_map.get(s, s)
            p = get_current_price(prices, uid)
            sh = math.floor(per_stock / p) if p > 0 else 0
            hold_details.append(f"{s}({sh}sh)")
        print(f"    HOLD: {', '.join(hold_details)}")
    if not signal.buys and not signal.sells:
        print(f"    No changes this month.")

    # ===================================================================
    # Top candidates
    # ===================================================================
    print(f"\n  TOP 20 CANDIDATES (ranked by composite momentum score):")
    print(f"    {'Rank':>4s}  {'Symbol':<8s} {'Composite':>9s}  {'MemeScore':>9s}  {'Classification':<16s} {'Status'}")
    print(f"    {'-' * 68}")
    for j, c in enumerate(signal.candidates, 1):
        status = "HELD" if c["in_portfolio"] else ""
        print(f"    {j:>4d}  {c['symbol']:<8s} {c['composite']:>9.2f}  {c['meme_score']:>9.0f}  "
              f"{c['classification']:<16s} {status}")

    # ===================================================================
    # Multi-period performance comparison table
    # ===================================================================
    if len(result.dates) == 0:
        print("\n  No backtest dates available — skipping performance tables.")
        return
    end_date = result.dates[-1]
    start_date = result.dates[0]
    nav_l = result.nav_leveraged
    nav_u = result.nav_unlevered

    # Define periods: (label, start_date)
    periods = []
    # YTD
    ytd_start_dates = result.dates[result.dates.year == end_date.year]
    if len(ytd_start_dates) > 5:
        periods.append(("YTD", ytd_start_dates[0]))
    # 1Y, 3Y, 5Y, 10Y, 15Y, 20Y
    for yrs in [1, 3, 5, 10, 15, 20]:
        cutoff = end_date - pd.DateOffset(years=yrs)
        valid = result.dates[result.dates >= cutoff]
        if len(valid) > 20 and valid[0] < end_date - pd.Timedelta(days=yrs * 300):
            periods.append((f"{yrs}Y", valid[0]))
    # Full
    periods.append(("Full", start_date))

    def _fmt_ret(val: float, width: int = 9) -> str:
        """Format return% compactly: 1234.5% -> +1,235%, 1816349% -> +1.8M%."""
        if not np.isfinite(val):
            return f"{'n/a':>{width}s}"
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:>+{width-2}.1f}M%"
        if abs(val) >= 10_000:
            return f"{val/1_000:>+{width-2}.0f}k%"
        if abs(val) >= 1_000:
            return f"{val:>+{width-1},.0f}%"
        return f"{val:>+{width-1}.1f}%"

    print(f"\n  PERFORMANCE COMPARISON:")
    print(f"    {'Period':<8s} {'WR 2x':>9s} {'WR 1x':>9s} {'SPY':>9s} {'vs SPY':>9s}  "
          f"{'CAGR 2x':>8s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'AvgLev':>7s}")
    print(f"    {'-' * 98}")

    for label, p_start in periods:
        mask = (result.dates >= p_start) & (result.dates <= end_date)
        p_dates = result.dates[mask]
        if len(p_dates) < 5:
            continue

        # Strategy returns
        ret_l = (nav_l.loc[p_dates[-1]] / nav_l.loc[p_dates[0]] - 1) * 100
        ret_u = (nav_u.loc[p_dates[-1]] / nav_u.loc[p_dates[0]] - 1) * 100

        # SPY return
        spy_aligned = spy_price.reindex(p_dates).ffill().bfill().dropna()
        spy_ret = (spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1) * 100 if len(spy_aligned) > 5 and spy_aligned.iloc[0] > 0 else np.nan
        vs_spy = ret_l - spy_ret if not np.isnan(spy_ret) else np.nan

        # Period-specific CAGR, Sharpe, Sortino, MaxDD (from sub-NAV)
        sub_nav = nav_l.loc[p_dates]
        sub_nav_norm = sub_nav / sub_nav.iloc[0]  # normalize to 1.0 for metrics
        p_years = (p_dates[-1] - p_dates[0]).days / 365.25
        p_cagr = (sub_nav_norm.iloc[-1] ** (1 / p_years) - 1) * 100 if p_years > 0.1 else ret_l

        daily_r = sub_nav_norm.pct_change().dropna()
        p_sharpe = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else 0
        down = daily_r[daily_r < 0]
        p_sortino = daily_r.mean() * 252 / (down.std() * np.sqrt(252)) if len(down) > 0 and down.std() > 0 else 0
        p_maxdd = (sub_nav_norm / sub_nav_norm.cummax() - 1).min() * 100

        avg_lev = result.leverage_series.loc[p_dates].mean()

        vs_str = _fmt_ret(vs_spy) if not np.isnan(vs_spy) else "      n/a"
        spy_str = _fmt_ret(spy_ret) if not np.isnan(spy_ret) else "      n/a"
        print(f"    {label:<8s} {_fmt_ret(ret_l)} {_fmt_ret(ret_u)} {spy_str} {vs_str}  "
              f"{p_cagr:>+7.1f}% {p_sharpe:>7.2f} {p_sortino:>8.2f} {p_maxdd:>6.1f}% {avg_lev:>6.2f}x")

    # ===================================================================
    # Winner capture rate table
    # ===================================================================
    membership = strategy.build_membership(prices, rankings)
    strategy_holdings = {d: set(h) for d, h in result.holdings_log.items()}
    rebal_dates = result.rebalance_dates

    capture_horizons = [1, 3, 5, 10]  # years
    top_k = 10  # top performers to check

    print(f"\n  WINNER CAPTURE RATE (did we hold the top-{top_k} performers?):")
    header_periods = "  ".join(f"{h}Y" for h in capture_horizons)
    print(f"    {'Year':<6s}  " + "  ".join(f"{'%dY Cap' % h:>8s}" for h in capture_horizons)
          + f"  {'Stocks Held':>11s}")
    print(f"    {'-' * (8 + 10 * len(capture_horizons) + 13)}")

    # Precompute return series for each horizon
    ret_by_horizon = {}
    for h in capture_horizons:
        trading_days = h * 252
        if len(prices) > trading_days:
            ret_by_horizon[h] = prices.pct_change(trading_days)

    # Show for each year that has enough data
    snap_years = sorted(set(d.year for d in result.dates))
    yearly_captures = {h: [] for h in capture_horizons}

    for snap_year in snap_years:
        year_dates = prices.index[prices.index.year == snap_year]
        if len(year_dates) < 20:
            continue
        snap_date = year_dates[-1]

        # Stocks held this year across all rebalances
        year_held = set()
        for rd in rebal_dates:
            if rd.year == snap_year:
                year_held |= strategy_holdings.get(rd, set())
        n_held = len(year_held)

        rate_strs = []
        for h in capture_horizons:
            if h not in ret_by_horizon or snap_date not in ret_by_horizon[h].index:
                rate_strs.append(f"{'n/a':>8s}")
                continue
            rets = ret_by_horizon[h].loc[snap_date].dropna()
            # Filter to stocks in our universe at snapshot time
            in_univ = [u for u in rets.index
                       if u in membership.columns
                       and snap_date in membership.index
                       and membership.loc[snap_date, u] == 1]
            if not in_univ:
                in_univ = list(rets.index)
            top = rets.reindex(in_univ).dropna().nlargest(top_k)
            if len(top) < 3:  # not enough data for meaningful capture rate
                rate_strs.append(f"{'n/a':>8s}")
                continue
            captured = sum(1 for u in top.index if u in year_held)
            pct = captured / len(top) * 100
            rate_strs.append(f"{captured}/{len(top)}={pct:2.0f}%")
            yearly_captures[h].append(pct)

        print(f"    {snap_year:<6d}  " + "  ".join(f"{s:>8s}" for s in rate_strs)
              + f"  {n_held:>7d}")

    # Average capture rates
    print(f"    {'-' * (8 + 10 * len(capture_horizons) + 13)}")
    avg_strs = []
    for h in capture_horizons:
        if yearly_captures[h]:
            avg = np.mean(yearly_captures[h])
            avg_strs.append(f"avg {avg:2.0f}%")
        else:
            avg_strs.append("n/a")
    print(f"    {'Avg':<6s}  " + "  ".join(f"{s:>8s}" for s in avg_strs))

    # ===================================================================
    # Sector analysis — universe composition + historical performance
    # ===================================================================
    print(f"\n  SECTOR ANALYSIS (universe stocks by sector + historical holding performance):")

    # 1. Map every UID ever held to its sector
    def _sector(uid):
        base = clean_uid(uid).rstrip("*")
        return SYM_TO_SECTOR.get(base, "Other")

    # 2. Compute per-holding returns between rebalance dates
    from collections import defaultdict
    sector_stats = defaultdict(lambda: {"hold_months": 0, "returns": [], "stocks": set(),
                                         "current": []})

    rdates = result.rebalance_dates
    for idx in range(len(rdates)):
        rd = rdates[idx]
        next_rd = rdates[idx + 1] if idx + 1 < len(rdates) else result.dates[-1]
        holdings = result.holdings_log.get(rd, [])

        for uid in holdings:
            sec = _sector(uid)
            sym = clean_uid(uid)
            sector_stats[sec]["hold_months"] += 1
            sector_stats[sec]["stocks"].add(sym)

            # Return from rd to next_rd
            p0 = get_price_on_date(prices, uid, rd)
            p1 = get_price_on_date(prices, uid, next_rd)
            if p0 > 0 and p1 > 0:
                sector_stats[sec]["returns"].append(p1 / p0 - 1)

    # Mark current holdings
    for uid in signal.holdings:
        sec = _sector(uid)
        sector_stats[sec]["current"].append(clean_uid(uid))

    # 3. Current universe composition by sector
    current_universe = set(prices.columns)  # all UIDs in the price matrix
    sector_univ_count = defaultdict(int)
    for uid in current_universe:
        sec = _sector(uid)
        sector_univ_count[sec] += 1

    # 4. Print table sorted by total hold-months (most exposure first)
    print(f"    {'Sector':<22s} {'Univ':>5s} {'Held':>5s} {'HoldMo':>7s} {'AvgMoRet':>9s} "
          f"{'WinRate':>8s} {'BestMo':>8s} {'WorstMo':>9s}  Current Holdings")
    print(f"    {'-' * 115}")

    total_hold_months = 0
    total_returns = []

    for sec in sorted(sector_stats.keys(),
                      key=lambda s: sector_stats[s]["hold_months"], reverse=True):
        st = sector_stats[sec]
        n_univ = sector_univ_count.get(sec, 0)
        n_stocks = len(st["stocks"])
        hm = st["hold_months"]
        total_hold_months += hm
        rets = st["returns"]
        total_returns.extend(rets)

        if rets:
            avg_r = np.mean(rets) * 100
            win = sum(1 for r in rets if r > 0) / len(rets) * 100
            best = max(rets) * 100
            worst = min(rets) * 100
        else:
            avg_r = win = best = worst = 0

        cur = ", ".join(st["current"]) if st["current"] else ""
        print(f"    {sec:<22s} {n_univ:>5d} {n_stocks:>5d} {hm:>7d} {avg_r:>+8.1f}% "
              f"{win:>7.0f}% {best:>+7.1f}% {worst:>+8.1f}%  {cur}")

    print(f"    {'-' * 115}")
    if total_returns:
        t_avg = np.mean(total_returns) * 100
        t_win = sum(1 for r in total_returns if r > 0) / len(total_returns) * 100
        print(f"    {'TOTAL':<22s} {sum(sector_univ_count.values()):>5d} "
              f"{sum(len(sector_stats[s]['stocks']) for s in sector_stats):>5d} "
              f"{total_hold_months:>7d} {t_avg:>+8.1f}% {t_win:>7.0f}%")

    # ===================================================================
    # Backtest model note
    # ===================================================================
    print(f"\n  NOTE: Backtest uses fixed-dollar equal-weight allocation.")
    print(f"  Each month, all {cfg.top_n} positions rebalanced to equal $ amounts.")
    print(f"  CAGR reflects compounded percentage returns, not a specific $ amount.")
    print()


if __name__ == "__main__":
    main()
