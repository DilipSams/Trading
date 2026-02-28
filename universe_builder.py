"""
Point-in-time dynamic stock universe builder.

Eliminates survivorship bias by ranking ALL stocks (active + delisted)
by trailing dollar volume at each monthly rebalance date, using only
information available on that date.

Cache-based: first run takes ~60-90s to process 34k files, subsequent
runs load from parquet cache in <1s.
"""

import os
import re
import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

# Default paths
NORGATE_ROOT = os.environ.get("NORGATE_ROOT", r"C:\ProgramData\NorgateData")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")

# ETFs / indices to exclude from the stock universe (they'd dominate by turnover)
UNIVERSE_EXCLUDE = {
    "SPY", "QQQ", "IVV", "VOO", "DIA", "IWM", "VTI", "VTV", "VUG", "VIG",
    "VXUS", "BND", "AGG", "TLT", "GLD", "SLV", "USO", "UNG", "HYG", "LQD",
    "EEM", "EFA", "VWO", "VEA", "IEMG", "IEFA", "SHY", "IEF", "TIP",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLI", "XLE", "XLY", "XLP", "XLB", "XLU", "XLRE", "XLC",
    # Leveraged ETFs
    "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SDS", "QLD", "SSO",
    "TECL", "ERX", "FAS", "FAZ", "LABU", "LABD", "SOXL", "SOXS",
}


def _parse_catalog(norgate_root: str) -> pd.DataFrame:
    """
    Load Norgate catalog and return filtered DataFrame with columns:
      symbol, database, first_date, last_date, bars, filepath, base_symbol, uid
    """
    cat_path = os.path.join(norgate_root, "_catalog.parquet")
    cat = pd.read_parquet(cat_path)

    # Filter to equities only
    eq_mask = cat["database"].isin(["US Equities", "US Equities Delisted"])
    cat = cat[eq_mask].copy()

    # Need minimum history for 63-day rolling average
    cat = cat[cat["bars"] >= 126].copy()

    # Build filepath and extract base_symbol / uid
    eq_dir = os.path.join(norgate_root, "US_Equities")
    de_dir = os.path.join(norgate_root, "US_Equities_Delisted")

    filepaths = []
    base_symbols = []
    uids = []

    for _, row in cat.iterrows():
        sym = row["symbol"]
        db = row["database"]

        if db == "US Equities":
            filepaths.append(os.path.join(eq_dir, f"{sym}.parquet"))
            base_symbols.append(sym)
            uids.append(sym)
        else:
            # Delisted: symbol is like "LEHMQ-201203"
            filepaths.append(os.path.join(de_dir, f"{sym}.parquet"))
            # Extract base symbol by stripping -YYYYMM suffix
            m = re.match(r"^(.+)-(\d{6})$", sym)
            if m:
                base_symbols.append(m.group(1))
            else:
                base_symbols.append(sym)
            uids.append(sym)  # full symbol as uid for uniqueness

    cat["filepath"] = filepaths
    cat["base_symbol"] = base_symbols
    cat["uid"] = uids

    # Filter out warrants, units, and other non-equity instruments
    exclude_patterns = [
        r"\.WS$", r"\.WSA$", r"\.WSB$",  # warrants
        r"\.U$", r"\.UN$",                 # units
        r"\.R$", r"\.RT$",                 # rights
        r"\.P$", r"\.PA$", r"\.PB$",       # preferred (some)
    ]
    combined_re = re.compile("|".join(exclude_patterns))
    keep = cat["base_symbol"].apply(lambda s: not combined_re.search(s))
    cat = cat[keep].copy()

    # Exclude ETFs/indices by base_symbol
    cat = cat[~cat["base_symbol"].isin(UNIVERSE_EXCLUDE)].copy()

    # Exclude single-character symbols (often index artifacts)
    cat = cat[cat["base_symbol"].str.len() >= 2].copy()

    cat = cat.reset_index(drop=True)
    return cat


def _compute_monthly_rankings(
    catalog: pd.DataFrame,
    top_n: int = 150,
    start_date: str = "1990-01-01",
) -> pd.DataFrame:
    """
    For each month-end, rank all alive stocks by trailing 63-day average
    Turnover. Return top-N rankings.
    """
    # Generate month-end dates
    today = pd.Timestamp.now().normalize()
    month_ends = pd.date_range(start_date, today, freq="ME")
    if len(month_ends) == 0:
        return pd.DataFrame()

    _n_turnover_errs = [0]  # mutable counter for error suppression

    # Pre-index catalog by date ranges for fast alive-stock lookup
    cat_first = catalog["first_date"].values.astype("datetime64[ns]")
    cat_last = catalog["last_date"].values.astype("datetime64[ns]")
    cat_uids = catalog["uid"].values
    cat_bases = catalog["base_symbol"].values
    cat_paths = catalog["filepath"].values
    n_stocks = len(catalog)

    # Cache of loaded Turnover series: uid -> pd.Series
    turnover_cache = {}

    all_records = []
    total_months = len(month_ends)
    t0 = time.time()

    for mi, me in enumerate(month_ends):
        if mi % 24 == 0:
            elapsed = time.time() - t0
            pct = mi / total_months * 100
            print(f"    Building rankings: {mi}/{total_months} months ({pct:.0f}%), "
                  f"{elapsed:.0f}s elapsed, cache={len(turnover_cache)} series", end="\r")

        me_np = np.datetime64(me)
        # 63 trading days ~ 90 calendar days lookback
        lookback_start = me_np - np.timedelta64(90, "D")

        # Find alive stocks: first_date <= me AND last_date >= lookback_start
        alive_mask = (cat_first <= me_np) & (cat_last >= lookback_start)
        alive_idx = np.where(alive_mask)[0]

        # Compute trailing 63-day average turnover for each alive stock
        turnover_vals = []
        alive_uids_for_month = []
        alive_bases_for_month = []

        for idx in alive_idx:
            uid = cat_uids[idx]
            base = cat_bases[idx]
            fpath = cat_paths[idx]

            # Lazy load into cache
            if uid not in turnover_cache:
                if os.path.exists(fpath):
                    try:
                        df = pd.read_parquet(fpath, columns=["Turnover"])
                        df.index = pd.to_datetime(df.index)
                        turnover_cache[uid] = df["Turnover"]
                    except Exception as e:
                        if _n_turnover_errs[0] < 5:
                            print(f"    WARNING: {uid}: {type(e).__name__}: {e}")
                        _n_turnover_errs[0] += 1
                        turnover_cache[uid] = pd.Series(dtype=float)
                else:
                    turnover_cache[uid] = pd.Series(dtype=float)

            ts = turnover_cache[uid]
            if len(ts) == 0:
                continue

            # Get trailing 63 trading days up to me
            mask = ts.index <= me
            recent = ts[mask].iloc[-63:] if mask.any() else pd.Series(dtype=float)

            if len(recent) >= 21:  # need at least 1 month of data
                avg_to = recent.mean()
                if avg_to > 0 and not np.isnan(avg_to):
                    turnover_vals.append(avg_to)
                    alive_uids_for_month.append(uid)
                    alive_bases_for_month.append(base)

        # Rank by turnover, take top-N
        if turnover_vals:
            arr = np.array(turnover_vals)
            top_idx = np.argsort(arr)[::-1][:top_n]

            for rank, ti in enumerate(top_idx, 1):
                all_records.append({
                    "date": me,
                    "rank": rank,
                    "uid": alive_uids_for_month[ti],
                    "base_symbol": alive_bases_for_month[ti],
                    "avg_turnover_63d": turnover_vals[ti],
                })

        # Periodically prune cache: remove stocks that delisted > 6 months ago
        if mi % 12 == 0 and mi > 0:
            prune_cutoff = me_np - np.timedelta64(180, "D")
            to_remove = []
            for ci in range(n_stocks):
                uid = cat_uids[ci]
                if uid in turnover_cache and cat_last[ci] < prune_cutoff:
                    to_remove.append(uid)
            for uid in to_remove:
                del turnover_cache[uid]

    elapsed = time.time() - t0
    print(f"    Building rankings: {total_months}/{total_months} months (100%), "
          f"{elapsed:.0f}s total, {len(all_records)} rank records          ")

    rankings = pd.DataFrame(all_records)
    return rankings


def _build_price_matrix(
    rankings: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load adjusted Close prices for all UIDs that ever appeared in rankings.
    Returns wide DataFrame: index=Date, columns=uid.
    """
    unique_uids = rankings["uid"].unique()
    print(f"    Loading prices for {len(unique_uids)} unique symbols...")

    # Map uid -> filepath from catalog
    uid_to_path = dict(zip(catalog["uid"], catalog["filepath"]))

    price_dict = {}
    loaded = 0
    n_errs = 0
    for uid in unique_uids:
        fpath = uid_to_path.get(uid)
        if fpath and os.path.exists(fpath):
            try:
                df = pd.read_parquet(fpath, columns=["Close"])
                df.index = pd.to_datetime(df.index)
                price_dict[uid] = df["Close"]
                loaded += 1
            except Exception as e:
                if n_errs < 5:
                    print(f"    WARNING: {uid}: {type(e).__name__}: {e}")
                n_errs += 1

    if n_errs:
        print(f"    WARNING: {n_errs} price files failed to load")
    print(f"    Loaded {loaded} price series")
    prices = pd.DataFrame(price_dict)
    prices = prices.sort_index()
    return prices


def build_universe_cache(
    norgate_root: str = NORGATE_ROOT,
    cache_dir: str = CACHE_DIR,
    top_n: int = 250,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build or load the cached dynamic universe.

    Returns:
      rankings: DataFrame [date, rank, uid, base_symbol, avg_turnover_63d]
      prices:   DataFrame [Date index, uid columns] of adjusted Close prices
    """
    os.makedirs(cache_dir, exist_ok=True)

    rank_path = os.path.join(cache_dir, f"universe_rankings_top{top_n}.parquet")
    price_path = os.path.join(cache_dir, f"universe_prices_top{top_n}.parquet")

    # Check if cache is fresh
    if not force_rebuild and os.path.exists(rank_path) and os.path.exists(price_path):
        cat_path = os.path.join(norgate_root, "_catalog.parquet")
        cat_mtime = os.path.getmtime(cat_path)
        cache_mtime = os.path.getmtime(rank_path)

        if cache_mtime >= cat_mtime:
            print("  Loading cached universe...")
            t0 = time.time()
            rankings = pd.read_parquet(rank_path)
            prices = pd.read_parquet(price_path)
            print(f"  Universe cache loaded in {time.time()-t0:.1f}s "
                  f"({len(rankings)} rankings, {prices.shape[1]} symbols)")
            return rankings, prices
        else:
            print("  Cache stale (catalog updated), rebuilding...")

    # Build from scratch
    print("  Building point-in-time universe (first run, ~60-90s)...")
    t_start = time.time()

    print("  Step 1/3: Parsing catalog...")
    catalog = _parse_catalog(norgate_root)
    print(f"    {len(catalog)} candidate stocks (active + delisted)")

    print("  Step 2/3: Computing monthly rankings...")
    rankings = _compute_monthly_rankings(catalog, top_n=top_n)

    print("  Step 3/3: Building price matrix...")
    prices = _build_price_matrix(rankings, catalog)

    # Save cache
    rankings.to_parquet(rank_path, index=False)
    prices.to_parquet(price_path)
    print(f"  Universe built in {time.time()-t_start:.0f}s, "
          f"cached to {cache_dir}")

    return rankings, prices


def get_universe_at_date(
    rankings: pd.DataFrame,
    target_date: pd.Timestamp,
    top_n: int = 100,
) -> List[str]:
    """
    Return the top_n UIDs as of the most recent month-end <= target_date.
    """
    valid = rankings[rankings["date"] <= target_date]
    if valid.empty:
        return []
    latest_date = valid["date"].max()
    month_data = valid[(valid["date"] == latest_date) & (valid["rank"] <= top_n)]
    return list(month_data["uid"].values)


def uid_to_ticker(uid: str) -> str:
    """Convert uid back to a display-friendly ticker."""
    m = re.match(r"^(.+)-(\d{6})$", uid)
    if m:
        return m.group(1)
    return uid


def get_full_universe_history(
    rankings: pd.DataFrame,
    top_n: int = 150,
) -> dict:
    """
    Return dict mapping each rebalance date to its list of UIDs.

    Useful for batch processing (year-by-year breakdown, regime analysis).

    Returns:
        dict[pd.Timestamp -> List[str]]
    """
    filtered = rankings[rankings["rank"] <= top_n]
    result = {}
    for dt in sorted(filtered["date"].unique()):
        mask = filtered["date"] == dt
        result[dt] = list(filtered.loc[mask, "uid"].values)
    return result


def get_universe_stats(rankings: pd.DataFrame, top_n: int = 150) -> dict:
    """
    Compute universe membership statistics.

    Returns:
        dict with keys: n_unique, n_entered_per_year, n_exited_per_year,
        turnover_rate, first_date, last_date, n_rebalances.
    """
    filtered = rankings[rankings["rank"] <= top_n]
    dates = sorted(filtered["date"].unique())
    if len(dates) < 2:
        return {}

    # Build per-date sets
    date_sets = {}
    for dt in dates:
        mask = filtered["date"] == dt
        date_sets[dt] = set(filtered.loc[mask, "uid"].values)

    # Count entries/exits per year
    all_uids = set()
    entries_per_year = {}
    exits_per_year = {}
    prev_set = set()
    for dt in dates:
        curr_set = date_sets[dt]
        entered = curr_set - prev_set
        exited = prev_set - curr_set
        yr = dt.year
        entries_per_year[yr] = entries_per_year.get(yr, 0) + len(entered)
        exits_per_year[yr] = exits_per_year.get(yr, 0) + len(exited)
        all_uids.update(curr_set)
        prev_set = curr_set

    years = sorted(set(entries_per_year.keys()) | set(exits_per_year.keys()))
    avg_entries = np.mean([entries_per_year.get(y, 0) for y in years]) if years else 0
    avg_exits = np.mean([exits_per_year.get(y, 0) for y in years]) if years else 0

    return {
        "n_unique": len(all_uids),
        "n_rebalances": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "avg_entries_per_year": avg_entries,
        "avg_exits_per_year": avg_exits,
        "turnover_rate": (avg_entries + avg_exits) / (2 * top_n),
        "entries_per_year": entries_per_year,
        "exits_per_year": exits_per_year,
    }


if __name__ == "__main__":
    # Standalone test: build cache and print summary
    rankings, prices = build_universe_cache(top_n=250, force_rebuild=True)

    print(f"\nRankings: {len(rankings)} records")
    print(f"Date range: {rankings['date'].min()} to {rankings['date'].max()}")
    print(f"Unique symbols: {rankings['uid'].nunique()}")
    print(f"Price matrix: {prices.shape}")

    # Show top 10 at a few dates
    for check_date in ["2000-01-31", "2008-01-31", "2020-01-31"]:
        dt = pd.Timestamp(check_date)
        top10 = get_universe_at_date(rankings, dt, top_n=10)
        tickers = [uid_to_ticker(u) for u in top10]
        print(f"\nTop 10 by dollar volume as of {check_date}:")
        print(f"  {', '.join(tickers)}")
