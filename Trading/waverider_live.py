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
        return float(prices[uid].dropna().iloc[-1])
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
    # YTD performance
    # ===================================================================
    current_year = signal.date.year
    yr_dates = result.dates[result.dates.year == current_year]
    if len(yr_dates) > 5:
        ytd_ret = (result.nav_leveraged.loc[yr_dates[-1]] / result.nav_leveraged.loc[yr_dates[0]] - 1) * 100
        spy_yr = spy_price[spy_price.index.year == current_year]
        spy_ytd = (spy_yr.iloc[-1] / spy_yr.iloc[0] - 1) * 100 if len(spy_yr) > 5 else 0
        print(f"\n  YTD PERFORMANCE: {ytd_ret:+.1f}% (SPY: {spy_ytd:+.1f}%)")

    # ===================================================================
    # Lifetime summary
    # ===================================================================
    print(f"\n  LIFETIME ({m_lev['n_years']:.0f} years): "
          f"CAGR {m_lev['cagr']*100:+.1f}% | "
          f"Sharpe {m_lev['sharpe']:.2f} | "
          f"MaxDD {m_lev['max_dd']*100:.1f}%")

    # ===================================================================
    # Backtest model note
    # ===================================================================
    print(f"\n  NOTE: Backtest uses fixed-dollar equal-weight allocation.")
    print(f"  Each month, all {cfg.top_n} positions rebalanced to equal $ amounts.")
    print(f"  CAGR reflects compounded percentage returns, not a specific $ amount.")
    print()


if __name__ == "__main__":
    main()
