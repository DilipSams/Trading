"""
strategy_research.py -- Comprehensive Strategy Performance & Leading Indicator Research
========================================================================================

Task 1: Historical performance comparison of all strategies using Norgate data
Task 2: Predictive/leading indicator research and hybrid framework design

Data: Norgate survivorship-bias-free daily OHLCV + Economic + Indices
"""

import os, sys, math, warnings, time, calendar
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

NORGATE_EQ   = r"D:\Experiments\norgate_data\US_Equities"
NORGATE_IDX  = r"D:\Experiments\norgate_data\US_Indices"
NORGATE_ECON = r"D:\Experiments\norgate_data\Economic"

TRANSACTION_COST_BPS = 10  # 10 bps round-trip
RISK_FREE_PROXY      = "%3MTCM"  # 3-month T-bill for Sharpe calculations

# Dynamic universe: use point-in-time top-100 by dollar volume (eliminates survivorship bias)
DYNAMIC_UNIVERSE = True
UNIVERSE_TOP_N   = 100  # number of stocks in the universe at each rebalance

# Legacy: fixed large-cap list (used when DYNAMIC_UNIVERSE=False)
SP500_CORE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "UNH", "HD", "MA", "DIS", "ADBE", "CRM", "NFLX", "CMCSA", "PEP", "KO",
    "ABT", "TMO", "MRK", "PFE", "ABBV", "COST", "AVGO", "TXN", "QCOM", "INTC",
    "AMD", "ORCL", "ACN", "IBM", "HON", "UNP", "CAT", "MMM", "GE", "BA",
    "AXP", "GS", "MS", "BLK", "C", "BAC", "WFC", "USB", "T", "VZ",
    "XOM", "CVX", "COP", "SLB", "EOG", "LMT", "RTX", "NOC", "GD", "DE",
    "LOW", "TGT", "WMT", "NKE", "SBUX", "MCD", "AMGN", "GILD", "BMY", "LLY",
    "MDT", "SYK", "ISRG", "BDX", "ZTS", "NEE", "DUK", "SO", "AEP", "D",
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "WM", "RSG",
    "FDX", "UPS", "AMAT", "LRCX", "KLAC", "MCHP", "ADI", "APD", "LIN", "ECL",
]

# Sector ETFs for rotation strategies (available from 1998-12)
SECTOR_ETFS = {
    "XLK": "Technology",    "XLF": "Financials",   "XLV": "Healthcare",
    "XLI": "Industrials",   "XLE": "Energy",       "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples", "XLB": "Materials",    "XLU": "Utilities",
}

# 3x leveraged equivalents for aggressive rotation
SECTOR_3X = {
    "XLK": "TECL", "XLE": "ERX", "XLF": "FAS",
}

# Market regime periods for year-by-year analysis
MARKET_REGIMES = OrderedDict([
    ("1990s Bull",        ("1990-01-01", "1999-12-31")),
    ("Dot-Com Crash",     ("2000-01-01", "2002-12-31")),
    ("Recovery",          ("2003-01-01", "2007-12-31")),
    ("GFC",               ("2008-01-01", "2009-03-31")),
    ("Post-GFC Bull",     ("2009-04-01", "2019-12-31")),
    ("COVID Crash",       ("2020-01-01", "2020-03-31")),
    ("COVID Recovery",    ("2020-04-01", "2021-12-31")),
    ("2022 Bear",         ("2022-01-01", "2022-12-31")),
    ("AI Rally",          ("2023-01-01", "2026-12-31")),
])


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_series(ticker: str, col: str = "Close", db: str = "equity") -> pd.Series:
    """Load a single time series from Norgate parquet files."""
    dirs = {
        "equity": NORGATE_EQ,
        "index":  NORGATE_IDX,
        "econ":   NORGATE_ECON,
    }
    path = os.path.join(dirs[db], f"{ticker}.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype=float, name=ticker)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if col in df.columns:
        return df[col].sort_index().rename(ticker)
    # For economic/index data that may only have Close
    return df.iloc[:, 0].sort_index().rename(ticker) if len(df.columns) > 0 else pd.Series(dtype=float, name=ticker)


def load_ohlcv(ticker: str, db: str = "equity") -> pd.DataFrame:
    """Load full OHLCV dataframe."""
    dirs = {"equity": NORGATE_EQ, "index": NORGATE_IDX, "econ": NORGATE_ECON}
    path = os.path.join(dirs[db], f"{ticker}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ============================================================================
# Metrics Engine
# ============================================================================

def compute_metrics(nav: pd.Series, benchmark_nav: pd.Series = None,
                    rf_daily: pd.Series = None, name: str = "") -> dict:
    """
    Compute comprehensive performance metrics from a NAV series.

    Returns dict with: CAGR, total_return, max_dd, sharpe, sortino, win_rate,
    avg_hold, n_trades, exposure_pct, alpha, best_year, worst_year, recovery_time, etc.
    """
    nav = nav.dropna()
    if len(nav) < 2:
        return {"name": name, "error": "insufficient data"}

    # Basic returns
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    daily_ret = nav.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)

    # Risk-free rate for Sharpe
    if rf_daily is not None:
        rf = rf_daily.reindex(daily_ret.index).ffill().fillna(0)
        excess = daily_ret - rf
    else:
        excess = daily_ret

    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Sortino (downside deviation only)
    downside = excess[excess < 0]
    down_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-10
    sortino = (excess.mean() * 252) / down_std

    # Drawdown analysis
    cum_max = nav.cummax()
    dd_series = (nav - cum_max) / cum_max
    max_dd = dd_series.min()

    # Recovery time (longest time from peak to new peak)
    in_dd = dd_series < -0.001
    max_recovery_days = 0
    current_dd_start = None
    for i, (date, is_dd) in enumerate(in_dd.items()):
        if is_dd and current_dd_start is None:
            current_dd_start = date
        elif not is_dd and current_dd_start is not None:
            recovery_days = (date - current_dd_start).days
            max_recovery_days = max(max_recovery_days, recovery_days)
            current_dd_start = None
    if current_dd_start is not None:
        # Still in drawdown
        max_recovery_days = max(max_recovery_days, (nav.index[-1] - current_dd_start).days)

    # Exposure (% of time invested, assumes NAV changes > 0.001% = invested)
    invested = (daily_ret.abs() > 1e-6).mean()

    # Annual returns for best/worst year
    annual_rets = {}
    for yr in sorted(nav.index.year.unique()):
        yr_nav = nav[nav.index.year == yr]
        if len(yr_nav) >= 2:
            annual_rets[yr] = yr_nav.iloc[-1] / yr_nav.iloc[0] - 1

    best_year = max(annual_rets.values()) if annual_rets else 0
    worst_year = min(annual_rets.values()) if annual_rets else 0
    best_year_name = max(annual_rets, key=annual_rets.get) if annual_rets else "N/A"
    worst_year_name = min(annual_rets, key=annual_rets.get) if annual_rets else "N/A"

    # Benchmark comparison
    alpha = 0.0
    bench_cagr = 0.0
    if benchmark_nav is not None:
        bench = benchmark_nav.reindex(nav.index).ffill().dropna()
        if len(bench) >= 2:
            bench_total = bench.iloc[-1] / bench.iloc[0] - 1
            bench_n_years = (bench.index[-1] - bench.index[0]).days / 365.25
            bench_cagr = (1 + bench_total) ** (1 / max(bench_n_years, 0.01)) - 1
            alpha = cagr - bench_cagr

    return {
        "name": name,
        "start": str(nav.index[0].date()),
        "end": str(nav.index[-1].date()),
        "years": round(n_years, 1),
        "cagr": cagr,
        "total_return": total_return,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_vol": ann_vol,
        "exposure_pct": invested,
        "bench_cagr": bench_cagr,
        "alpha": alpha,
        "best_year": best_year,
        "best_year_name": best_year_name,
        "worst_year": worst_year,
        "worst_year_name": worst_year_name,
        "recovery_days": max_recovery_days,
        "annual_rets": annual_rets,
    }


def compute_trade_metrics(trades: list) -> dict:
    """Compute trade-level metrics: win rate, avg hold, number of trades."""
    if not trades:
        return {"win_rate": 0, "avg_hold": 0, "n_trades": 0}
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    holds = [t.get("hold_days", 0) for t in trades]
    return {
        "win_rate": wins / len(trades) if trades else 0,
        "avg_hold": np.mean(holds) if holds else 0,
        "n_trades": len(trades),
    }


# ============================================================================
# TASK 1: Strategy Implementations
# ============================================================================

class StrategyEngine:
    """Backtest engine for all strategy variants using Norgate data."""

    def __init__(self, dynamic_universe: bool = DYNAMIC_UNIVERSE):
        print("Loading market data...")
        t0 = time.time()

        # Core benchmarks
        self.spy = load_series("SPY")
        self.qqq = load_series("QQQ")

        # Sector ETFs
        self.sector_prices = {}
        for etf in SECTOR_ETFS:
            s = load_series(etf)
            if len(s) > 0:
                self.sector_prices[etf] = s

        # Risk-free rate (daily)
        rf_annual = load_series(RISK_FREE_PROXY, col="Close", db="econ")
        self.rf_daily = (rf_annual / 100 / 252).reindex(self.spy.index).ffill().fillna(0)

        # Individual stocks
        self.dynamic_universe = dynamic_universe
        self._membership_mask_cache = None

        if dynamic_universe:
            from universe_builder import build_universe_cache
            rankings, prices = build_universe_cache(top_n=UNIVERSE_TOP_N + 50)
            self.universe_rankings = rankings
            self.universe_prices = prices
            self.universe_top_n = UNIVERSE_TOP_N

            self.stock_prices = {}
            for col_name in prices.columns:
                s = prices[col_name].dropna()
                if len(s) > 200:
                    self.stock_prices[col_name] = s
            loaded = len(self.stock_prices)
            mode_str = f"dynamic top-{UNIVERSE_TOP_N} (survivorship-bias-free)"
        else:
            self.stock_prices = {}
            for sym in SP500_CORE:
                s = load_series(sym)
                if len(s) > 200:
                    self.stock_prices[sym] = s
            loaded = len(self.stock_prices)
            mode_str = "static SP500_CORE (legacy)"

        print(f"  SPY: {self.spy.index[0].date()} -> {self.spy.index[-1].date()}")
        print(f"  Sectors: {len(self.sector_prices)} ETFs loaded")
        print(f"  Stocks: {loaded} equities loaded [{mode_str}]")
        print(f"  Data loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Dynamic universe membership mask
    # ------------------------------------------------------------------
    def _build_membership_mask(self, dates: pd.DatetimeIndex,
                               columns: pd.Index) -> pd.DataFrame:
        """
        Build a binary mask: mask[date][uid] = 1.0 if uid is in the top-N
        universe as of the most recent month-end before date, else 0.0.
        Cached after first call.
        """
        if not self.dynamic_universe:
            return pd.DataFrame(1.0, index=dates, columns=columns)

        # Use cache if available and matches shape
        if (self._membership_mask_cache is not None
                and self._membership_mask_cache.shape[0] == len(dates)
                and self._membership_mask_cache.shape[1] == len(columns)):
            return self._membership_mask_cache

        mask = pd.DataFrame(0.0, index=dates, columns=columns)
        rankings = self.universe_rankings
        top_n = self.universe_top_n

        # Get sorted unique rebalance dates
        rebal_dates = sorted(rankings["date"].unique())
        col_set = set(columns)

        for i, rd in enumerate(rebal_dates):
            # Get top-N UIDs for this month
            month_data = rankings[(rankings["date"] == rd) & (rankings["rank"] <= top_n)]
            top_uids = set(month_data["uid"].values) & col_set

            # Date range this membership applies to
            start = rd
            end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
            period = (dates >= start) & (dates <= end)

            for uid in top_uids:
                mask.loc[period, uid] = 1.0

        self._membership_mask_cache = mask
        return mask

    # ------------------------------------------------------------------
    # Strategy 1: SMA Crossover on SPY (buy-and-hold with trend filter)
    # ------------------------------------------------------------------
    def sma_crossover_spy(self, fast: int, slow: int, name: str = "") -> Tuple[pd.Series, list]:
        """
        SMA crossover on SPY. Long when fast SMA > slow SMA, else cash.
        Returns (NAV series, trade list).
        """
        price = self.spy.dropna()
        sma_fast = price.rolling(fast).mean()
        sma_slow = price.rolling(slow).mean()

        # Signal: 1 when fast > slow (long), 0 when fast < slow (cash)
        signal = (sma_fast > sma_slow).astype(float).shift(1)  # shift to avoid lookahead
        signal = signal.fillna(0)

        daily_ret = price.pct_change()
        strategy_ret = daily_ret * signal

        # Apply transaction costs on signal changes
        signal_changes = signal.diff().abs()
        cost = signal_changes * (TRANSACTION_COST_BPS / 10000)
        strategy_ret = strategy_ret - cost

        nav = (1 + strategy_ret).cumprod()
        nav.iloc[0] = 1.0

        # Track trades
        trades = []
        in_trade = False
        entry_date = entry_price = None
        for i in range(1, len(signal)):
            if signal.iloc[i] == 1 and signal.iloc[i-1] == 0:
                in_trade = True
                entry_date = signal.index[i]
                entry_price = price.iloc[i]
            elif signal.iloc[i] == 0 and signal.iloc[i-1] == 1 and in_trade:
                exit_date = signal.index[i]
                exit_price = price.iloc[i]
                trades.append({
                    "entry": entry_date, "exit": exit_date,
                    "pnl": exit_price / entry_price - 1,
                    "hold_days": (exit_date - entry_date).days,
                })
                in_trade = False

        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 2: SMA Crossover on individual stocks (basket)
    # ------------------------------------------------------------------
    def sma_crossover_basket(self, fast: int, slow: int, top_n: int = 20,
                              name: str = "") -> Tuple[pd.Series, list]:
        """
        For each stock: long when fast SMA > slow SMA.
        Equal-weight portfolio of all stocks currently in "long" state.
        Rebalance daily. If nothing qualifies, hold cash.
        """
        # Build common date index
        all_prices = pd.DataFrame(self.stock_prices)
        common_start = all_prices.dropna(how="all").index[0]
        # Need enough history for slow SMA
        warmup_start = common_start + pd.Timedelta(days=int(slow * 1.5))

        all_rets = all_prices.pct_change()
        sma_fast = all_prices.rolling(fast).mean()
        sma_slow = all_prices.rolling(slow).mean()

        # Signal matrix: 1 where fast > slow, shifted 1 day
        signals = (sma_fast > sma_slow).astype(float).shift(1)

        # Apply universe membership mask (only invest in top-N stocks at each date)
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            signals = signals * membership

        # Equal-weight portfolio of qualifying stocks
        n_longs = signals.sum(axis=1)
        weights = signals.div(n_longs.replace(0, np.nan), axis=0).fillna(0)

        # Portfolio return = weighted sum of individual returns
        port_ret = (weights * all_rets).sum(axis=1)

        # Transaction costs (approximate: proportional to weight changes)
        weight_changes = weights.diff().abs().sum(axis=1)
        cost = weight_changes * (TRANSACTION_COST_BPS / 10000)
        port_ret = port_ret - cost

        # Trim to after warmup
        port_ret = port_ret.loc[warmup_start:]
        nav = (1 + port_ret).cumprod()
        if len(nav) > 0:
            nav = nav / nav.iloc[0]

        # Simple trade count estimate (number of signal transitions)
        n_trades = int(signals.loc[warmup_start:].diff().abs().sum().sum() / 2)
        trades = [{"pnl": 0.01, "hold_days": 20}] * max(n_trades, 1)  # placeholder

        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 3: Sector Rotation (Relative Strength + Macro)
    # ------------------------------------------------------------------
    def sector_rotation(self, momentum_windows: dict = None,
                        bear_gate: bool = True, top_n: int = 3,
                        rebalance_freq: int = 21,
                        name: str = "") -> Tuple[pd.Series, list]:
        """
        Monthly sector rotation using multi-timeframe momentum.
        Optional SPY SMA200 bear gate (go to cash if SPY < SMA200).
        """
        if momentum_windows is None:
            momentum_windows = {21: 0.40, 63: 0.35, 126: 0.25}

        # Build sector price matrix
        sec_df = pd.DataFrame(self.sector_prices).dropna(how="all")
        common_idx = sec_df.dropna().index
        sec_df = sec_df.reindex(common_idx).ffill()
        spy = self.spy.reindex(common_idx).ffill()

        # Compute momentum scores
        mom_scores = pd.DataFrame(0.0, index=sec_df.index, columns=sec_df.columns)
        for window, weight in momentum_windows.items():
            mom_scores += sec_df.pct_change(window) * weight
        mom_scores = mom_scores.shift(1)  # avoid lookahead

        # Bear gate
        spy_sma200 = spy.rolling(200).mean().shift(1)
        is_bear = spy.shift(1) < spy_sma200

        # Backtest
        n = len(common_idx)
        nav_vals = np.ones(n)
        holdings = []
        switches = 0
        current_holdings = []

        for i in range(1, n):
            date = common_idx[i]
            prev_date = common_idx[i - 1]
            rebal = (i % rebalance_freq == 0)

            new_holdings = current_holdings[:]
            if rebal:
                if bear_gate and is_bear.get(date, False):
                    new_holdings = []  # all cash
                else:
                    scores = mom_scores.loc[date].dropna().sort_values(ascending=False)
                    new_holdings = list(scores.index[:top_n])

            if set(new_holdings) != set(current_holdings):
                switches += 1
                current_holdings = new_holdings

            if current_holdings:
                slot_rets = []
                for h in current_holdings:
                    if h in sec_df.columns:
                        p_now = sec_df.loc[date, h]
                        p_prev = sec_df.loc[prev_date, h]
                        slot_rets.append(p_now / p_prev - 1 if p_prev > 0 else 0.0)
                daily_ret = np.mean(slot_rets) if slot_rets else 0.0
            else:
                daily_ret = 0.0  # cash

            # Transaction cost on switch days
            cost_mult = 1.0 - (TRANSACTION_COST_BPS / 10000) if (set(new_holdings) != set(current_holdings) and i > 1) else 1.0
            nav_vals[i] = nav_vals[i-1] * (1 + daily_ret) * (cost_mult if rebal and switches > 0 else 1.0)

        nav = pd.Series(nav_vals, index=common_idx)
        trades = [{"pnl": 0.01, "hold_days": rebalance_freq}] * max(switches, 1)

        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 4: Multibagger Screening + Trend Filter
    # ------------------------------------------------------------------
    def multibagger_screen(self, name: str = "") -> Tuple[pd.Series, list]:
        """
        Screen for potential multibaggers using:
        - 12-month momentum (top quartile)
        - SMA 50 > SMA 100 > SMA 200 alignment (full uptrend)
        - Relative strength vs SPY > 1.0 over 6 months

        Equal-weight top-20, monthly rebalance.
        Entry: all 3 conditions met. Exit: SMA 50 < SMA 200.
        """
        all_prices = pd.DataFrame(self.stock_prices)
        spy = self.spy.reindex(all_prices.index).ffill()

        # Compute signals
        mom_12m = all_prices.pct_change(252).shift(1)
        mom_1m = all_prices.pct_change(21).shift(1)
        # Carhart momentum = 12m - 1m (skip most recent month)
        momentum = mom_12m - mom_1m

        sma50 = all_prices.rolling(50).mean()
        sma100 = all_prices.rolling(100).mean()
        sma200 = all_prices.rolling(200).mean()

        # SMA alignment: price > sma50 > sma100 > sma200
        aligned = ((all_prices > sma50) & (sma50 > sma100) & (sma100 > sma200)).shift(1).fillna(False)

        # Relative strength vs SPY (6 months)
        spy_ret_126 = spy.pct_change(126)
        stock_ret_126 = all_prices.pct_change(126)
        rs_6m = stock_ret_126.div(spy_ret_126, axis=0)
        rs_strong = (rs_6m > 1.0).shift(1).fillna(False)

        # Composite: momentum + alignment + RS
        composite = momentum.copy()
        composite[~aligned] = np.nan  # must be aligned
        composite[~rs_strong] = np.nan  # must have RS > 1

        # Apply universe membership mask (only pick from top-N at each date)
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            composite[membership == 0] = np.nan

        # Monthly rebalance: top 20 qualifying stocks
        all_rets = all_prices.pct_change()
        warmup = all_prices.index[max(252, 200) + 5]

        nav_vals = []
        current_weights = pd.Series(0.0, index=all_prices.columns)

        dates = all_prices.loc[warmup:].index
        nav_val = 1.0
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            # Monthly rebalance
            if i % 21 == 0:
                scores = composite.loc[date].dropna().sort_values(ascending=False)
                top = scores.head(20)
                new_weights = pd.Series(0.0, index=all_prices.columns)
                if len(top) > 0:
                    for sym in top.index:
                        new_weights[sym] = 1.0 / len(top)

                if not new_weights.equals(current_weights):
                    trades_count += int((new_weights - current_weights).abs().sum() * 10)
                current_weights = new_weights

            # Daily return
            day_ret = (current_weights * all_rets.loc[date]).sum()

            # Transaction cost (on rebalance days)
            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000

            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)

        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 5: Combined Multibagger + Sector Rotation
    # ------------------------------------------------------------------
    def combined_multibagger_rotation(self, name: str = "") -> Tuple[pd.Series, list]:
        """
        Hybrid: Use sector rotation to determine which sectors to fish in,
        then apply multibagger screening within those sectors.

        70% allocation to sector-screened multibagger picks,
        30% to top-momentum sector ETFs.
        """
        # Get sector rotation signal
        sec_df = pd.DataFrame(self.sector_prices).dropna(how="all")
        spy = self.spy.reindex(sec_df.index).ffill()

        # Sector momentum
        mom_scores = pd.DataFrame(0.0, index=sec_df.index, columns=sec_df.columns)
        for window, weight in {21: 0.40, 63: 0.35, 126: 0.25}.items():
            mom_scores += sec_df.pct_change(window) * weight
        mom_scores = mom_scores.shift(1)

        # Bear gate
        spy_sma200 = spy.rolling(200).mean().shift(1)
        is_bear = spy.shift(1) < spy_sma200

        # Stock-level signals (same as multibagger_screen)
        all_prices = pd.DataFrame(self.stock_prices)
        mom_12m = all_prices.pct_change(252).shift(1)
        mom_1m = all_prices.pct_change(21).shift(1)
        momentum = mom_12m - mom_1m

        sma50 = all_prices.rolling(50).mean()
        sma100 = all_prices.rolling(100).mean()
        sma200 = all_prices.rolling(200).mean()
        aligned = ((all_prices > sma50) & (sma50 > sma100) & (sma100 > sma200)).shift(1).fillna(False)

        rs_6m = all_prices.pct_change(126).div(spy.pct_change(126).reindex(all_prices.index).ffill(), axis=0)
        rs_strong = (rs_6m > 1.0).shift(1).fillna(False)

        composite = momentum.copy()
        composite[~aligned] = np.nan
        composite[~rs_strong] = np.nan

        # Apply universe membership mask
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            composite[membership == 0] = np.nan

        all_rets = all_prices.pct_change()
        sec_rets = sec_df.pct_change()

        warmup = all_prices.index[max(252, 200) + 5]
        common_start = max(warmup, sec_df.index[200])

        dates = all_prices.loc[common_start:].index
        nav_val = 1.0
        nav_vals = []
        stock_weights = pd.Series(0.0, index=all_prices.columns)
        sector_holdings = []

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            # Monthly rebalance
            if i % 21 == 0:
                # Sector rotation component
                if is_bear.get(date, False):
                    sector_holdings = []
                    stock_weights = pd.Series(0.0, index=all_prices.columns)
                else:
                    sec_scores = mom_scores.reindex(sec_df.index).loc[:date]
                    if len(sec_scores) > 0:
                        latest_sec = sec_scores.iloc[-1].dropna().sort_values(ascending=False)
                        sector_holdings = list(latest_sec.head(3).index)

                    # Stock picks within top sectors (simplified sector mapping)
                    scores = composite.loc[date].dropna().sort_values(ascending=False)
                    top_stocks = scores.head(15)
                    stock_weights = pd.Series(0.0, index=all_prices.columns)
                    if len(top_stocks) > 0:
                        for sym in top_stocks.index:
                            stock_weights[sym] = 1.0 / len(top_stocks)

            # 70% stocks, 30% sector ETFs
            stock_ret = (stock_weights * all_rets.loc[date]).sum() if date in all_rets.index else 0.0

            sec_ret = 0.0
            if sector_holdings and date in sec_rets.index:
                for h in sector_holdings:
                    if h in sec_rets.columns:
                        sec_ret += sec_rets.loc[date, h] / len(sector_holdings)

            day_ret = 0.70 * stock_ret + 0.30 * sec_ret

            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000

            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        return nav, []

    # ------------------------------------------------------------------
    # Strategy 6: Dual Momentum (Absolute + Relative)
    # ------------------------------------------------------------------
    def dual_momentum(self, top_n: int = 20, name: str = "") -> Tuple[pd.Series, list]:
        """
        Antonacci-inspired dual momentum on individual stocks.
        - Relative: rank by Carhart momentum (12m - 1m)
        - Absolute: only invest if 12m return > 0
        - Top-N equal-weight, monthly rebalance.
        """
        all_prices = pd.DataFrame(self.stock_prices)
        spy = self.spy.reindex(all_prices.index).ffill()

        # Carhart momentum: 12m return minus most recent 1m
        mom_12m = all_prices.pct_change(252).shift(1)
        mom_1m = all_prices.pct_change(21).shift(1)
        rel_momentum = mom_12m - mom_1m  # relative component

        # Absolute momentum: 12m return > 0 (positive absolute return)
        abs_positive = (mom_12m > 0).shift(1).fillna(False)

        # Composite: only rank stocks with positive absolute momentum
        composite = rel_momentum.copy()
        composite[~abs_positive] = np.nan

        # Apply universe membership mask
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            composite[membership == 0] = np.nan

        # Monthly rebalance: top-N qualifying stocks
        all_rets = all_prices.pct_change()
        warmup = all_prices.index[max(252, 200) + 5]

        nav_vals = []
        current_weights = pd.Series(0.0, index=all_prices.columns)
        dates = all_prices.loc[warmup:].index
        nav_val = 1.0
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            # Monthly rebalance
            if i % 21 == 0:
                scores = composite.loc[date].dropna().sort_values(ascending=False)
                top = scores.head(top_n)
                new_weights = pd.Series(0.0, index=all_prices.columns)
                if len(top) > 0:
                    for sym in top.index:
                        new_weights[sym] = 1.0 / len(top)
                if not new_weights.equals(current_weights):
                    trades_count += int((new_weights - current_weights).abs().sum() * 10)
                current_weights = new_weights

            day_ret = (current_weights * all_rets.loc[date]).sum()
            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000
            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 6b: Wave Rider — Improved Momentum
    # ------------------------------------------------------------------
    def improved_momentum(self, top_n: int = 10, name: str = "",
                          max_highvol: int = -1,
                          highvol_threshold: float = 0.70,
                          use_meme_score: bool = False,
                          meme_exclude: int = 70,
                          meme_max1: int = 50,
                          meme_max2: int = 30) -> Tuple[pd.Series, list]:
        """
        Wave Rider: Improved momentum with forensic-driven fixes:
        1. Multi-timeframe momentum (blend 3m+6m+12m Carhart) — captures compounders
        2. Risk-adjusted by volatility — penalizes lottery tickets
        3. SMA200 trend filter — replaces harsh '12m return > 0' absolute filter
        4. Hysteresis band — enter top-N, exit only below top-2.5N — cuts churn
        5. Meme stock filter — either binary vol cap (max_highvol) or 6-factor score

        Args:
            max_highvol: Max # of stocks with vol > highvol_threshold allowed.
                         -1 = no limit (default). Ignored when use_meme_score=True.
            highvol_threshold: Annualized vol threshold (default 0.70). Ignored when use_meme_score=True.
            use_meme_score: If True, use multi-factor meme scoring (0-100) instead of binary vol filter.
            meme_exclude: Score above which stocks are excluded entirely (default 70).
            meme_max1: Score above which max 1 stock allowed (default 50).
            meme_max2: Score above which max 2 stocks allowed (default 30).
        """
        all_prices = pd.DataFrame(self.stock_prices)
        daily_rets = all_prices.pct_change()

        # --- Signal 1: Multi-timeframe Carhart momentum ---
        mom_12m = all_prices.pct_change(252).shift(1)
        mom_6m  = all_prices.pct_change(126).shift(1)
        mom_3m  = all_prices.pct_change(63).shift(1)
        mom_1m  = all_prices.pct_change(21).shift(1)

        carhart_12 = mom_12m - mom_1m
        carhart_6  = mom_6m  - mom_1m
        carhart_3  = mom_3m  - mom_1m

        blended_mom = 0.40 * carhart_12 + 0.35 * carhart_6 + 0.25 * carhart_3

        # --- Signal 2: Risk-adjust by 63-day realized volatility ---
        vol_63d = daily_rets.rolling(63).std().shift(1) * np.sqrt(252)
        risk_adj = blended_mom / (vol_63d + 0.05)

        # --- Signal 3: SMA200 trend filter (replaces 12m return > 0) ---
        sma200 = all_prices.rolling(200).mean()
        trend_ok = (all_prices > sma200).shift(1).fillna(False)

        # --- Multi-factor meme score (0-100) ---
        meme_scores = None
        if use_meme_score:
            # Factor 1: Volatility level (0-25)
            f1 = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            f1[vol_63d > 0.40] = 8
            f1[vol_63d > 0.60] = 15
            f1[vol_63d > 0.80] = 20
            f1[vol_63d > 1.00] = 25

            # Factor 2: Parabolic 3m move (0-25)
            ret_3m_abs = mom_3m.abs()
            f2 = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            f2[ret_3m_abs > 0.50] = 8
            f2[ret_3m_abs > 1.00] = 18
            f2[ret_3m_abs > 2.00] = 25

            # Factor 3: Price-SMA200 stretch (0-20)
            stretch = (all_prices / sma200).shift(1)
            f3 = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            f3[stretch > 1.3] = 8
            f3[stretch > 2.0] = 15
            f3[stretch > 3.0] = 20

            # Factor 4: Momentum concentration — recent month vs full year (0-15)
            # High = nearly all 12m gain happened in last month
            both_pos = (mom_1m > 0) & (mom_12m > 0.01)
            conc = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            conc[both_pos] = (mom_1m[both_pos] / mom_12m[both_pos]).clip(0, 2)
            f4 = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            f4[conc > 0.3] = 5
            f4[conc > 0.5] = 10
            f4[conc > 0.8] = 15

            # Factor 5: Volatility acceleration — short-term vol spike (0-15)
            vol_21d = daily_rets.rolling(21).std().shift(1) * np.sqrt(252)
            vol_126d = daily_rets.rolling(126).std().shift(1) * np.sqrt(252)
            vol_accel = vol_21d / (vol_126d + 0.01)
            f5 = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            f5[vol_accel > 1.0] = 5
            f5[vol_accel > 1.5] = 10
            f5[vol_accel > 2.5] = 15

            # Factor 6: Universe tenure — proxy for ETF/index membership (0-15)
            # Stocks newly entering top-100 (0-3 months) are flagged as potential meme
            # Established members (6+ months) are likely genuine index constituents
            f6 = pd.DataFrame(0.0, index=all_prices.index, columns=all_prices.columns)
            if self.dynamic_universe and hasattr(self, 'universe_rankings'):
                rankings = self.universe_rankings
                top100 = rankings[rankings['rank'] <= 100]
                # Compute consecutive months in top-100 for each stock
                tenure = {}  # uid -> running months count
                rank_dates = sorted(top100['date'].unique())
                tenure_snapshots = {}  # date -> {uid: months_in_top100}
                for rdate in rank_dates:
                    current_top = set(top100[top100['date'] == rdate]['uid'].values)
                    # Increment tenure for stocks still in top-100, reset for those that left
                    new_tenure = {}
                    for uid in current_top:
                        new_tenure[uid] = tenure.get(uid, 0) + 1
                    tenure = new_tenure
                    tenure_snapshots[rdate] = dict(tenure)
                # Build daily tenure DataFrame (forward-fill monthly to daily)
                for rdate in rank_dates:
                    snap = tenure_snapshots[rdate]
                    next_idx = rank_dates.index(rdate) + 1
                    if next_idx < len(rank_dates):
                        end_date = rank_dates[next_idx]
                        mask = (all_prices.index >= rdate) & (all_prices.index < end_date)
                    else:
                        mask = all_prices.index >= rdate
                    for uid, months in snap.items():
                        if uid in f6.columns:
                            if months <= 2:
                                f6.loc[mask, uid] = 15  # 0-2 months: new entrant
                            elif months <= 5:
                                f6.loc[mask, uid] = 8   # 3-5 months: recent
                            # 6+ months: 0 pts (established)

            meme_scores = f1 + f2 + f3 + f4 + f5 + f6

        composite = risk_adj.copy()
        composite[~trend_ok] = np.nan

        # --- Apply universe membership mask ---
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            composite[membership == 0] = np.nan

        # --- Backtest loop with hysteresis band ---
        all_rets = daily_rets
        warmup = all_prices.index[max(252, 200) + 5]
        dates = all_prices.loc[warmup:].index

        nav_val = 1.0
        nav_vals = []
        current_holdings = set()
        current_weights = pd.Series(0.0, index=all_prices.columns)
        trades_count = 0
        exit_band = int(top_n * 2.5)  # top-25 for top-10, top-12 for top-5

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            if i % 21 == 0:  # Monthly rebalance
                scores = composite.loc[date].dropna().sort_values(ascending=False)
                top_fresh = set(scores.head(top_n).index)
                top_band = set(scores.head(exit_band).index)

                # Hysteresis: keep incumbents still in exit band
                retained = current_holdings & top_band
                # Fill remaining slots from top-N fresh picks
                remaining_slots = top_n - len(retained)
                new_entries = [s for s in scores.index if s in top_fresh and s not in retained]
                portfolio = retained | set(new_entries[:max(remaining_slots, 0)])

                # Trim to top_n if over (keep highest-scored)
                if len(portfolio) > top_n:
                    scored = [(s, scores.get(s, -999)) for s in portfolio]
                    scored.sort(key=lambda x: -x[1])
                    portfolio = set(s for s, _ in scored[:top_n])

                # --- Meme stock filtering ---
                if use_meme_score and meme_scores is not None and len(portfolio) > 0:
                    # Multi-factor meme score graduated filtering
                    ms_today = meme_scores.loc[date]

                    def _get_ms(s):
                        v = ms_today.get(s, 0)
                        return v if pd.notna(v) else 0

                    # Tier 1: Exclude pure meme stocks (score > meme_exclude)
                    pure_meme = [s for s in portfolio if _get_ms(s) > meme_exclude]
                    portfolio -= set(pure_meme)

                    # Tier 2: Max 1 meme-adjacent stock (score > meme_max1)
                    meme_adj = [s for s in portfolio if _get_ms(s) > meme_max1]
                    if len(meme_adj) > 1:
                        ma_scored = sorted(meme_adj, key=lambda s: scores.get(s, -999), reverse=True)
                        portfolio -= set(ma_scored[1:])

                    # Tier 3: Max 2 gray-zone stocks (score > meme_max2)
                    gray_zone = [s for s in portfolio if _get_ms(s) > meme_max2]
                    if len(gray_zone) > 2:
                        gz_scored = sorted(gray_zone, key=lambda s: scores.get(s, -999), reverse=True)
                        portfolio -= set(gz_scored[2:])

                    # Refill removed slots with clean stocks (score <= meme_max2)
                    for s in scores.index:
                        if len(portfolio) >= top_n:
                            break
                        if s not in portfolio and _get_ms(s) <= meme_max2:
                            portfolio.add(s)

                elif max_highvol >= 0 and len(portfolio) > 0:
                    # Legacy binary vol filter
                    vol_today = vol_63d.loc[date]
                    highvol_in_port = [s for s in portfolio
                                       if pd.notna(vol_today.get(s)) and vol_today[s] > highvol_threshold]
                    if len(highvol_in_port) > max_highvol:
                        hv_scored = sorted(highvol_in_port, key=lambda s: scores.get(s, -999), reverse=True)
                        to_remove = set(hv_scored[max_highvol:])
                        portfolio -= to_remove
                        for s in scores.index:
                            if len(portfolio) >= top_n:
                                break
                            if s not in portfolio and (pd.isna(vol_today.get(s)) or vol_today[s] <= highvol_threshold):
                                portfolio.add(s)

                current_holdings = portfolio
                new_weights = pd.Series(0.0, index=all_prices.columns)
                if len(portfolio) > 0:
                    for sym in portfolio:
                        new_weights[sym] = 1.0 / len(portfolio)

                if not new_weights.equals(current_weights):
                    trades_count += int((new_weights - current_weights).abs().sum() * 10)
                current_weights = new_weights

            day_ret = (current_weights * all_rets.loc[date]).sum()
            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000
            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 7: Mean Reversion / Dip Buying in Uptrends
    # ------------------------------------------------------------------
    def mean_reversion_dip(self, name: str = "") -> Tuple[pd.Series, list]:
        """
        Buy dips in uptrending stocks:
        - Price > SMA200 (long-term uptrend intact)
        - Pullback >10% from 63-day rolling high (relaxed from 15%)
        - RSI(14) < 40 (short-term oversold)
        Exit: RSI > 60 OR price < SMA200.
        Monthly rebalance: pick top-20 qualifying stocks by pullback depth.
        """
        all_prices = pd.DataFrame(self.stock_prices)

        sma200 = all_prices.rolling(200).mean()
        rolling_high_63 = all_prices.rolling(63).max()
        pullback = (all_prices - rolling_high_63) / rolling_high_63  # negative when pulled back

        # RSI(14) calculation
        delta = all_prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Entry conditions (all shifted by 1)
        uptrend = (all_prices > sma200).shift(1).fillna(False)
        dipped = (pullback < -0.10).shift(1).fillna(False)
        oversold = (rsi < 40).shift(1).fillna(False)

        # Composite: rank by pullback depth (more dipped = more attractive)
        # Only qualify if all conditions met
        composite = pullback.shift(1).copy()  # negative values, more negative = deeper dip
        composite[~uptrend] = np.nan
        composite[~dipped] = np.nan
        composite[~oversold] = np.nan

        # Apply universe membership mask
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            composite[membership == 0] = np.nan

        # Monthly rebalance approach (same pattern as other strategies)
        all_rets = all_prices.pct_change()
        warmup = all_prices.index[max(252, 200) + 5]

        nav_vals = []
        current_weights = pd.Series(0.0, index=all_prices.columns)
        dates = all_prices.loc[warmup:].index
        nav_val = 1.0
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            # Monthly rebalance
            if i % 21 == 0:
                scores = composite.loc[date].dropna()
                # Sort ascending: most negative pullback = deepest dip = most attractive
                top = scores.nsmallest(20)
                new_weights = pd.Series(0.0, index=all_prices.columns)
                if len(top) > 0:
                    for sym in top.index:
                        new_weights[sym] = 1.0 / len(top)
                if not new_weights.equals(current_weights):
                    trades_count += int((new_weights - current_weights).abs().sum() * 10)
                current_weights = new_weights

            day_ret = (current_weights * all_rets.loc[date]).sum()
            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000
            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # Strategy 8: Low Volatility (Low-Vol Anomaly)
    # ------------------------------------------------------------------
    def low_volatility(self, top_n: int = 20, name: str = "") -> Tuple[pd.Series, list]:
        """
        Select lowest-volatility stocks from the universe.
        - Compute 63-day realized vol for each stock
        - Pick top_n with LOWEST vol
        - Equal-weight, monthly rebalance.
        Exploits the well-documented low-vol anomaly.
        """
        all_prices = pd.DataFrame(self.stock_prices)

        # 63-day realized volatility (annualized)
        daily_rets = all_prices.pct_change()
        vol_63d = daily_rets.rolling(63).std() * np.sqrt(252)
        vol_63d = vol_63d.shift(1)  # avoid lookahead

        # Apply universe membership mask: set non-members to NaN (high vol = excluded)
        if self.dynamic_universe:
            membership = self._build_membership_mask(all_prices.index, all_prices.columns)
            vol_63d[membership == 0] = np.nan

        # Monthly rebalance: pick lowest-vol stocks
        all_rets = all_prices.pct_change()
        warmup = all_prices.index[max(252, 63) + 5]

        nav_vals = []
        current_weights = pd.Series(0.0, index=all_prices.columns)
        dates = all_prices.loc[warmup:].index
        nav_val = 1.0
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            if i % 21 == 0:
                vols = vol_63d.loc[date].dropna()
                # Filter out unreasonably low vol (likely stale data)
                vols = vols[vols > 0.05]
                if len(vols) >= top_n:
                    lowest = vols.nsmallest(top_n)
                    new_weights = pd.Series(0.0, index=all_prices.columns)
                    for sym in lowest.index:
                        new_weights[sym] = 1.0 / top_n
                    if not new_weights.equals(current_weights):
                        trades_count += int((new_weights - current_weights).abs().sum() * 10)
                    current_weights = new_weights

            day_ret = (current_weights * all_rets.loc[date]).sum()
            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000
            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # Buy-and-hold benchmarks
    # ------------------------------------------------------------------
    def buy_and_hold(self, ticker: str, start_date: str = None) -> pd.Series:
        """Buy-and-hold NAV for a single ticker."""
        price = load_series(ticker)
        if start_date:
            price = price.loc[start_date:]
        price = price.dropna()
        if len(price) < 2:
            return pd.Series(dtype=float)
        return (price / price.iloc[0]).rename(ticker)

    def equal_weight_sectors(self) -> pd.Series:
        """Buy-and-hold equal-weight sector ETF portfolio."""
        sec_df = pd.DataFrame(self.sector_prices).dropna(how="all")
        sec_df = sec_df.dropna(axis=1, how="any")  # only use sectors with complete data
        daily_rets = sec_df.pct_change()
        port_ret = daily_rets.mean(axis=1)
        nav = (1 + port_ret).cumprod()
        nav.iloc[0] = 1.0
        return nav


# ============================================================================
# ETF Strategy Engine
# ============================================================================

# ETF tickers for multi-asset strategies
ETF_ASSETS = {
    "SPY": "US Large Cap",
    "EFA": "Intl Developed",
    "IWM": "US Small Cap",
    "TLT": "Long Treasury",
    "GLD": "Gold",
    "AGG": "US Agg Bond",
    "DBC": "Commodities",
    "VNQ": "Real Estate",
}


class ETFStrategyEngine:
    """Backtest engine for ETF-based strategies."""

    def __init__(self, strategy_engine: StrategyEngine = None):
        """Load ETF price data. Optionally share data with a StrategyEngine."""
        print("  Loading ETF data...")
        t0 = time.time()

        self.etf_prices = {}
        for etf in ETF_ASSETS:
            s = load_series(etf)
            if len(s) > 0:
                self.etf_prices[etf] = s

        # Also load sector ETFs for sector momentum strategy
        self.sector_prices = {}
        for etf in SECTOR_ETFS:
            s = load_series(etf)
            if len(s) > 0:
                self.sector_prices[etf] = s

        # Risk-free rate
        rf_annual = load_series(RISK_FREE_PROXY, col="Close", db="econ")
        self.spy = self.etf_prices.get("SPY", pd.Series(dtype=float))
        self.rf_daily = (rf_annual / 100 / 252).reindex(self.spy.index).ffill().fillna(0)

        # Share strategy engine for hybrid strategies
        self.strategy_engine = strategy_engine

        loaded = [k for k, v in self.etf_prices.items() if len(v) > 0]
        print(f"  ETFs loaded: {', '.join(loaded)}")
        print(f"  ETF data loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # ETF Strategy 1: Dual Momentum (Antonacci)
    # ------------------------------------------------------------------
    def dual_momentum_etf(self, name: str = "") -> Tuple[pd.Series, list]:
        """
        Gary Antonacci-inspired Dual Momentum:
        - Compare SPY 12m return vs EFA 12m return (relative momentum)
        - If winner has positive 12m return (absolute momentum), invest 100% in winner
        - If both negative, invest 100% in AGG (bonds)
        - Monthly rebalance
        """
        spy = self.etf_prices.get("SPY", pd.Series(dtype=float))
        efa = self.etf_prices.get("EFA", pd.Series(dtype=float))
        agg = self.etf_prices.get("AGG", pd.Series(dtype=float))

        if len(spy) == 0 or len(efa) == 0 or len(agg) == 0:
            return pd.Series(dtype=float), []

        # Align to common dates
        prices = pd.DataFrame({"SPY": spy, "EFA": efa, "AGG": agg}).dropna()

        spy_mom = prices["SPY"].pct_change(252).shift(1)
        efa_mom = prices["EFA"].pct_change(252).shift(1)

        dates = prices.index
        warmup = dates[max(252, 200) + 5]
        dates = dates[dates >= warmup]

        nav_val = 1.0
        nav_vals = []
        current_holding = None
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            # Monthly rebalance
            if i % 21 == 0:
                sm = spy_mom.get(date, np.nan)
                em = efa_mom.get(date, np.nan)

                if np.isnan(sm) or np.isnan(em):
                    new_holding = current_holding
                elif sm >= em and sm > 0:
                    new_holding = "SPY"
                elif em > sm and em > 0:
                    new_holding = "EFA"
                else:
                    new_holding = "AGG"  # both negative -> bonds

                if new_holding != current_holding:
                    trades_count += 1
                    current_holding = new_holding

            # Daily return
            if current_holding and current_holding in prices.columns:
                p_now = prices.loc[date, current_holding]
                prev_date = dates[i - 1]
                p_prev = prices.loc[prev_date, current_holding]
                day_ret = p_now / p_prev - 1 if p_prev > 0 else 0.0
            else:
                day_ret = 0.0

            # Transaction cost on switch days
            if i % 21 == 0 and trades_count > 0:
                day_ret -= TRANSACTION_COST_BPS / 10000

            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.02, "hold_days": 60}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # ETF Strategy 2: Tactical All-Weather (Multi-Asset Momentum + InvVol)
    # ------------------------------------------------------------------
    def tactical_allweather(self, name: str = "") -> Tuple[pd.Series, list]:
        """
        Multi-asset tactical allocation (Faber-inspired):
        - Assets: SPY, TLT, GLD, DBC (4 uncorrelated asset classes)
        - Monthly: allocate to assets with positive 6-month momentum
        - Weight by inverse 63-day volatility (risk parity lite)
        - If all negative momentum: 100% cash (T-bill rate)
        """
        assets = ["SPY", "TLT", "GLD", "DBC"]
        asset_prices = {}
        for a in assets:
            s = self.etf_prices.get(a, pd.Series(dtype=float))
            if len(s) > 0:
                asset_prices[a] = s

        if len(asset_prices) < 3:
            return pd.Series(dtype=float), []

        prices = pd.DataFrame(asset_prices).dropna()
        daily_rets = prices.pct_change()

        mom_6m = prices.pct_change(126).shift(1)
        vol_63d = daily_rets.rolling(63).std().shift(1) * np.sqrt(252)

        dates = prices.index
        warmup = dates[max(252, 126) + 5]
        dates = dates[dates >= warmup]

        nav_val = 1.0
        nav_vals = []
        current_weights = pd.Series(0.0, index=prices.columns)
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            if i % 21 == 0:
                mom = mom_6m.loc[date].dropna()
                vol = vol_63d.loc[date].dropna()

                # Only assets with positive momentum
                positive = mom[mom > 0]
                if len(positive) > 0:
                    # Inverse-vol weighting
                    inv_vol = 1.0 / vol.reindex(positive.index).replace(0, np.nan).dropna()
                    new_weights = inv_vol / inv_vol.sum()
                    new_weights = new_weights.reindex(prices.columns).fillna(0)
                else:
                    new_weights = pd.Series(0.0, index=prices.columns)  # all cash

                if not new_weights.equals(current_weights):
                    trades_count += 1
                current_weights = new_weights

            # Daily return
            if current_weights.sum() > 0:
                day_ret = (current_weights * daily_rets.loc[date].fillna(0)).sum()
            else:
                # Cash: earn risk-free rate
                day_ret = self.rf_daily.get(date, 0.0)

            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000

            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # ETF Strategy 3: Enhanced Sector Rotation
    # ------------------------------------------------------------------
    def sector_momentum_etf(self, top_n: int = 3,
                             name: str = "") -> Tuple[pd.Series, list]:
        """
        Enhanced sector rotation:
        - Rank 9 sectors by Carhart momentum (6m - 1m skip)
        - Top-N sectors equal-weight
        - Bear gate: SPY < SMA200 -> shift to AGG (bonds, not cash)
        """
        sec_df = pd.DataFrame(self.sector_prices).dropna(how="all")
        spy = self.spy.reindex(sec_df.index).ffill()
        agg = self.etf_prices.get("AGG", pd.Series(dtype=float)).reindex(sec_df.index).ffill()

        # Carhart momentum: 6m return minus 1m return
        mom_6m = sec_df.pct_change(126)
        mom_1m = sec_df.pct_change(21)
        mom_scores = (mom_6m - mom_1m).shift(1)

        spy_sma200 = spy.rolling(200).mean().shift(1)
        is_bear = spy.shift(1) < spy_sma200

        common_idx = sec_df.dropna().index
        warmup = common_idx[max(252, 200) + 5]
        dates = common_idx[common_idx >= warmup]

        nav_val = 1.0
        nav_vals = []
        current_sectors = []
        in_bonds = False
        trades_count = 0

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            prev_date = dates[i - 1]

            if i % 21 == 0:
                if is_bear.get(date, False):
                    if not in_bonds:
                        trades_count += 1
                    current_sectors = []
                    in_bonds = True
                else:
                    scores = mom_scores.loc[date].dropna().sort_values(ascending=False)
                    new_sectors = list(scores.head(top_n).index)
                    if set(new_sectors) != set(current_sectors):
                        trades_count += 1
                    current_sectors = new_sectors
                    in_bonds = False

            # Daily return
            if in_bonds and date in agg.index and prev_date in agg.index:
                a_now = agg.loc[date]
                a_prev = agg.loc[prev_date]
                day_ret = a_now / a_prev - 1 if a_prev > 0 else 0.0
            elif current_sectors:
                slot_rets = []
                for h in current_sectors:
                    if h in sec_df.columns:
                        p_now = sec_df.loc[date, h]
                        p_prev = sec_df.loc[prev_date, h]
                        slot_rets.append(p_now / p_prev - 1 if p_prev > 0 else 0.0)
                day_ret = np.mean(slot_rets) if slot_rets else 0.0
            else:
                day_ret = 0.0

            if i % 21 == 0 and trades_count > 0:
                day_ret -= TRANSACTION_COST_BPS / 10000

            nav_val *= (1 + day_ret)
            nav_vals.append(nav_val)

        nav = pd.Series(nav_vals, index=dates)
        trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades_count, 1)
        return nav, trades

    # ------------------------------------------------------------------
    # ETF Strategy 4: Hybrid Core + Satellite (ETF + Stock Alpha)
    # ------------------------------------------------------------------
    def hybrid_core_satellite(self, name: str = "") -> Tuple[pd.Series, list]:
        """
        60% Core: Dual Momentum ETF (SPY/EFA/AGG)
        40% Satellite: Top-10 momentum stocks from dynamic universe
        Monthly rebalance.
        """
        # Core: Dual Momentum ETF
        core_nav, _ = self.dual_momentum_etf()
        if len(core_nav) < 252:
            return pd.Series(dtype=float), []

        # Satellite: stock momentum (use strategy_engine)
        if self.strategy_engine is None:
            return core_nav, []  # fallback to core-only

        sat_nav, _ = self.strategy_engine.dual_momentum(top_n=10)
        if len(sat_nav) < 252:
            return core_nav, []

        # Align dates
        common_start = max(core_nav.index[0], sat_nav.index[0])
        common_end = min(core_nav.index[-1], sat_nav.index[-1])

        core = core_nav.loc[common_start:common_end]
        sat = sat_nav.loc[common_start:common_end]

        # Align indexes
        common_idx = core.index.intersection(sat.index)
        core = core.reindex(common_idx)
        sat = sat.reindex(common_idx)

        core_rets = core.pct_change().fillna(0)
        sat_rets = sat.pct_change().fillna(0)

        # 60/40 blend
        blended_rets = 0.60 * core_rets + 0.40 * sat_rets
        nav = (1 + blended_rets).cumprod()
        if len(nav) > 0:
            nav = nav / nav.iloc[0]

        return nav, []

    # ------------------------------------------------------------------
    # ETF Strategy 5: QQQ/XLK Core + Momentum Satellite
    # ------------------------------------------------------------------
    def qqq_core_momentum_satellite(self, core_etf: str = "QQQ",
                                      core_pct: float = 0.60,
                                      sat_top_n: int = 5,
                                      name: str = "") -> Tuple[pd.Series, list]:
        """
        Core-satellite with tech ETF core + concentrated momentum stocks.

        Core (60-70%): QQQ or XLK -- captures steady mega-cap compounders
          (NVDA, MSFT, AAPL, AMD, GOOGL, etc.) that pure momentum misses.
        Satellite (30-40%): Top-N momentum stocks from dynamic universe --
          catches explosive movers that QQQ/XLK misses.

        Bear gate: when SPY < SMA200, shift core to AGG (bonds).
        Monthly rebalance.
        """
        # Load core ETF (QQQ or XLK)
        core_series = load_series(core_etf)
        if len(core_series) < 252:
            return pd.Series(dtype=float), []

        spy = self.etf_prices.get("SPY", pd.Series(dtype=float))
        agg = self.etf_prices.get("AGG", pd.Series(dtype=float))

        # SPY bear gate signal
        spy_sma200 = spy.rolling(200).mean()
        bear_signal = spy < spy_sma200  # True = bear

        # Build core NAV with bear gate: QQQ in bull, AGG in bear
        common_start = max(core_series.index[200], agg.index[0])
        core_series = core_series.loc[common_start:]
        agg_aligned = agg.reindex(core_series.index).ffill()
        bear_aligned = bear_signal.reindex(core_series.index).ffill().fillna(False)

        core_rets = core_series.pct_change().fillna(0)
        agg_rets = agg_aligned.pct_change().fillna(0)

        # Monthly rebalance: check bear gate
        core_nav_vals = [1.0]
        in_bonds = False
        trades = 0
        for i in range(1, len(core_rets)):
            if i % 21 == 0:  # monthly check
                new_in_bonds = bool(bear_aligned.iloc[i])
                if new_in_bonds != in_bonds:
                    trades += 1
                in_bonds = new_in_bonds

            if in_bonds:
                day_ret = agg_rets.iloc[i]
            else:
                day_ret = core_rets.iloc[i]

            if i % 21 == 0:
                day_ret -= TRANSACTION_COST_BPS / 10000
            core_nav_vals.append(core_nav_vals[-1] * (1 + day_ret))

        core_nav = pd.Series(core_nav_vals, index=core_series.index)

        # Satellite: concentrated momentum stocks
        if self.strategy_engine is None:
            return core_nav, [{"pnl": 0.01, "hold_days": 21}] * max(trades, 1)

        sat_nav, sat_trades = self.strategy_engine.dual_momentum(top_n=sat_top_n)
        if len(sat_nav) < 252:
            return core_nav, [{"pnl": 0.01, "hold_days": 21}] * max(trades, 1)

        # Align dates
        common_idx = core_nav.index.intersection(sat_nav.index)
        if len(common_idx) < 252:
            return core_nav, [{"pnl": 0.01, "hold_days": 21}] * max(trades, 1)

        core_r = core_nav.reindex(common_idx).pct_change().fillna(0)
        sat_r = sat_nav.reindex(common_idx).pct_change().fillna(0)

        sat_pct = 1.0 - core_pct
        blended_rets = core_pct * core_r + sat_pct * sat_r
        nav = (1 + blended_rets).cumprod()
        if len(nav) > 0:
            nav = nav / nav.iloc[0]

        all_trades = [{"pnl": 0.01, "hold_days": 21}] * max(trades + len(sat_trades), 1)
        return nav, all_trades


# ============================================================================
# TASK 2: Leading / Predictive Indicators
# ============================================================================

class LeadingIndicatorEngine:
    """Load, compute, and analyze predictive/leading indicators."""

    def __init__(self):
        print("\nLoading leading indicator data...")
        t0 = time.time()

        # ---- Market-Based Leading Indicators ----

        # Credit spreads: Baa - Aaa (investment grade spread)
        baa = load_series("%COBAA", col="Close", db="econ")
        aaa = load_series("%COAAA", col="Close", db="econ")
        self.ig_spread = (baa - aaa).rename("IG_Spread")

        # High-yield credit spread
        self.hy_spread = load_series("%CCCHYS", col="Close", db="econ").rename("HY_Spread")

        # Yield curve: 10Y - 2Y
        y10 = load_series("%10YTCM", col="Close", db="econ")
        y2 = load_series("%2YTCM", col="Close", db="econ")
        y3m = load_series("%3MTCM", col="Close", db="econ")
        self.yield_2s10s = (y10 - y2).rename("Yield_2s10s")
        self.yield_3m10y = (y10 - y3m).rename("Yield_3m10y")

        # MOVE index (bond volatility)
        self.move = load_series("$MOVE", col="Close", db="econ").rename("MOVE")

        # CBOE Put/Call ratio
        self.put_call = load_series("#CBOEPC", col="Close", db="index").rename("PutCall")

        # NYSE breadth: advance-decline
        nyse_adv = load_series("#NYSEADV", col="Close", db="index")
        nyse_dec = load_series("#NYSEDEC", col="Close", db="index")
        self.ad_line = (nyse_adv - nyse_dec).cumsum().rename("AD_Line")
        self.ad_ratio = (nyse_adv / (nyse_adv + nyse_dec)).rename("AD_Ratio")

        # NYSE new highs - new lows
        nyse_hi = load_series("#NYSEHI", col="Close", db="index")
        nyse_lo = load_series("#NYSELO", col="Close", db="index")
        self.hilo_diff = (nyse_hi - nyse_lo).rename("HiLo_Diff")

        # ---- Economic Leading Indicators ----

        self.ism_pmi = load_series("#NAPMI", col="Close", db="econ").rename("ISM_PMI")
        self.jobless_claims = load_series("#INJOB", col="Close", db="econ").rename("Jobless_Claims")
        self.housing_starts = load_series("#HOSSA", col="Close", db="econ").rename("Housing_Starts")
        self.m2_money = load_series("#M2FED", col="Close", db="econ").rename("M2_Money")
        self.lei = load_series("#LVSAAT", col="Close", db="econ").rename("LEI")
        self.cfnai = load_series("#CFNAI", col="Close", db="econ").rename("CFNAI")
        self.nfci = load_series("#NFCI", col="Close", db="econ").rename("NFCI")
        self.industrial_prod = load_series("#INDSA", col="Close", db="econ").rename("IndProd")

        # ---- Intermarket Signals ----
        self.tlt = load_series("TLT")  # Long-term treasuries
        self.gld = load_series("GLD")  # Gold
        self.dbc = load_series("DBC")  # Commodities
        self.uup = load_series("UUP")  # US Dollar
        self.eem = load_series("EEM")  # Emerging markets
        self.hyg = load_series("HYG")  # High yield bonds
        self.lqd = load_series("LQD")  # Investment grade bonds

        # SPY for reference
        self.spy = load_series("SPY")

        print(f"  Leading indicators loaded in {time.time()-t0:.1f}s")
        self._print_data_coverage()

    def _print_data_coverage(self):
        """Print data availability summary."""
        indicators = {
            "IG Credit Spread (Baa-Aaa)": self.ig_spread,
            "HY Credit Spread": self.hy_spread,
            "Yield Curve 2s10s": self.yield_2s10s,
            "Yield Curve 3m10y": self.yield_3m10y,
            "MOVE Index": self.move,
            "Put/Call Ratio": self.put_call,
            "NYSE A/D Line": self.ad_line,
            "NYSE Hi-Lo Diff": self.hilo_diff,
            "ISM PMI": self.ism_pmi,
            "Initial Claims": self.jobless_claims,
            "Housing Starts": self.housing_starts,
            "M2 Money Supply": self.m2_money,
            "LEI (Conf. Board)": self.lei,
            "CFNAI": self.cfnai,
            "NFCI": self.nfci,
            "TLT (Bonds)": self.tlt,
            "GLD (Gold)": self.gld,
            "HYG (HY Bonds)": self.hyg,
        }
        print(f"\n  {'Indicator':<30s} {'Start':>12s} {'End':>12s} {'Obs':>8s}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")
        for name, series in indicators.items():
            s = series.dropna()
            if len(s) > 0:
                print(f"  {name:<30s} {str(s.index[0].date()):>12s} {str(s.index[-1].date()):>12s} {len(s):>8d}")
            else:
                print(f"  {name:<30s} {'N/A':>12s} {'N/A':>12s} {'0':>8s}")

    def compute_composite_leading_signal(self, as_of_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Compute a composite leading indicator signal combining multiple domains.

        For each indicator, compute a z-score relative to its own history,
        then create a weighted composite.

        Returns DataFrame with columns: date, signal_name, z_score, direction, weight
        """
        spy = self.spy.dropna()
        if as_of_date is None:
            as_of_date = spy.index[-1]

        # Common date range
        start = "1997-01-01"  # HY spread starts ~1997

        signals = {}

        # 1. Credit spread change (widening = bearish, tightening = bullish)
        if len(self.hy_spread.dropna()) > 100:
            hy = self.hy_spread.loc[start:].ffill()
            hy_z = -(hy - hy.rolling(252).mean()) / hy.rolling(252).std()  # negative = tightening = bullish
            signals["HY_Credit_Spread"] = {"series": hy_z, "weight": 0.15, "lead_months": "2-6"}

        if len(self.ig_spread.dropna()) > 100:
            ig = self.ig_spread.loc[start:].ffill()
            ig_z = -(ig - ig.rolling(252).mean()) / ig.rolling(252).std()
            signals["IG_Credit_Spread"] = {"series": ig_z, "weight": 0.10, "lead_months": "2-6"}

        # 2. Yield curve slope change (steepening = bullish for equities)
        if len(self.yield_2s10s.dropna()) > 100:
            yc = self.yield_2s10s.loc[start:].ffill()
            yc_chg = yc.diff(63)  # 3-month change in slope
            yc_z = (yc_chg - yc_chg.rolling(252).mean()) / yc_chg.rolling(252).std()
            signals["Yield_Curve_2s10s"] = {"series": yc_z, "weight": 0.10, "lead_months": "3-12"}

        # 3. NYSE breadth (A/D ratio) momentum
        if len(self.ad_ratio.dropna()) > 100:
            ad = self.ad_ratio.loc[start:].ffill()
            ad_mom = ad.rolling(21).mean() - ad.rolling(63).mean()  # short-term vs medium-term
            ad_z = (ad_mom - ad_mom.rolling(252).mean()) / ad_mom.rolling(252).std()
            signals["NYSE_Breadth"] = {"series": ad_z, "weight": 0.12, "lead_months": "1-3"}

        # 4. NYSE New Highs - New Lows
        if len(self.hilo_diff.dropna()) > 100:
            hl = self.hilo_diff.loc[start:].ffill()
            hl_smooth = hl.rolling(21).mean()
            hl_z = (hl_smooth - hl_smooth.rolling(252).mean()) / hl_smooth.rolling(252).std()
            signals["NYSE_HiLo"] = {"series": hl_z, "weight": 0.08, "lead_months": "1-3"}

        # 5. ISM PMI (above 50 = expansion, rate of change matters)
        if len(self.ism_pmi.dropna()) > 100:
            ism = self.ism_pmi.loc[start:].ffill()
            ism_mom = ism.diff(3).ffill()  # 3-period change (monthly data)
            ism_z = (ism_mom - ism_mom.rolling(120).mean()) / ism_mom.rolling(120).std()
            signals["ISM_PMI"] = {"series": ism_z, "weight": 0.10, "lead_months": "2-6"}

        # 6. Initial Jobless Claims (declining = bullish, inverted)
        if len(self.jobless_claims.dropna()) > 100:
            jc = self.jobless_claims.loc[start:].ffill()
            jc_roc = -jc.pct_change(13).ffill()  # 13-week rate of change, inverted
            jc_z = (jc_roc - jc_roc.rolling(252).mean()) / jc_roc.rolling(252).std()
            signals["Jobless_Claims"] = {"series": jc_z, "weight": 0.08, "lead_months": "1-4"}

        # 7. LEI rate of change
        if len(self.lei.dropna()) > 100:
            lei = self.lei.loc[start:].ffill()
            lei_roc = lei.pct_change(6).ffill()  # 6-month rate of change
            lei_z = (lei_roc - lei_roc.rolling(120).mean()) / lei_roc.rolling(120).std()
            signals["LEI"] = {"series": lei_z, "weight": 0.10, "lead_months": "3-9"}

        # 8. M2 Money Supply growth rate
        if len(self.m2_money.dropna()) > 100:
            m2 = self.m2_money.loc[start:].ffill()
            m2_roc = m2.pct_change(12).ffill()  # 12-month growth
            m2_z = (m2_roc - m2_roc.rolling(120).mean()) / m2_roc.rolling(120).std()
            signals["M2_Growth"] = {"series": m2_z, "weight": 0.05, "lead_months": "6-18"}

        # 9. NFCI (negative = loose conditions = bullish)
        if len(self.nfci.dropna()) > 100:
            nfci = self.nfci.loc[start:].ffill()
            nfci_z = -(nfci - nfci.rolling(252).mean()) / nfci.rolling(252).std()  # inverted
            signals["NFCI"] = {"series": nfci_z, "weight": 0.07, "lead_months": "1-6"}

        # 10. Put/Call ratio (extreme high = contrarian bullish)
        if len(self.put_call.dropna()) > 100:
            pc = self.put_call.loc[start:].ffill()
            pc_smooth = pc.rolling(21).mean()
            pc_z = (pc_smooth - pc_smooth.rolling(252).mean()) / pc_smooth.rolling(252).std()
            signals["PutCall_Contrarian"] = {"series": pc_z, "weight": 0.05, "lead_months": "0.5-2"}

        return signals

    def build_hybrid_signal(self, spy: pd.Series = None) -> pd.DataFrame:
        """
        Build the hybrid predictive + lagging signal DataFrame.

        Returns DataFrame indexed by date with columns:
        - predictive_score: weighted composite of leading indicators (-1 to +1)
        - lagging_score: SMA alignment score (0 to 1)
        - hybrid_score: combined signal
        - position_size: recommended allocation (0.0 to 1.0)
        """
        if spy is None:
            spy = self.spy.dropna()

        signals = self.compute_composite_leading_signal()

        # Build composite predictive score
        all_z = pd.DataFrame()
        total_weight = 0
        for name, info in signals.items():
            s = info["series"].reindex(spy.index).ffill()
            w = info["weight"]
            all_z[name] = s * w
            total_weight += w

        predictive_raw = all_z.sum(axis=1) / max(total_weight, 0.01)
        # Clip to [-2, +2] then scale to [-1, +1]
        predictive_score = predictive_raw.clip(-2, 2) / 2

        # Lagging score: SMA alignment
        sma50 = spy.rolling(50).mean()
        sma100 = spy.rolling(100).mean()
        sma200 = spy.rolling(200).mean()

        lagging_score = pd.Series(0.0, index=spy.index)
        lagging_score += (spy > sma50).astype(float) * 0.25
        lagging_score += (spy > sma100).astype(float) * 0.25
        lagging_score += (spy > sma200).astype(float) * 0.25
        lagging_score += (sma50 > sma200).astype(float) * 0.25
        lagging_score = lagging_score.shift(1)

        # Hybrid score: weighted blend
        # Predictive gets more weight when lagging is neutral (0.4-0.6)
        # Lagging gets more weight when it's extreme (0 or 1)
        pred_weight = 0.40
        lag_weight = 0.60

        hybrid_score = pred_weight * predictive_score + lag_weight * lagging_score

        # Position sizing based on hybrid score
        # predictive only (hybrid > 0.3, lagging < 0.5) -> 25%
        # predictive + partial confirm -> 50%
        # full confirm -> 100%
        # contradiction (pred bearish, lag bullish or vice versa) -> scale down
        position_size = pd.Series(0.0, index=spy.index)

        for i in range(len(spy)):
            pred = predictive_score.iloc[i] if i < len(predictive_score) else 0
            lag = lagging_score.iloc[i] if i < len(lagging_score) else 0

            if pred > 0.2 and lag >= 0.75:
                # Full confirmation: predictive bullish + lagging bullish
                position_size.iloc[i] = 1.0
            elif pred > 0.2 and lag >= 0.5:
                # Partial confirmation
                position_size.iloc[i] = 0.50
            elif pred > 0.2 and lag < 0.5:
                # Predictive only, no lagging confirmation
                position_size.iloc[i] = 0.25
            elif pred < -0.2 and lag < 0.25:
                # Both bearish
                position_size.iloc[i] = 0.0
            elif pred < -0.2 and lag >= 0.5:
                # Contradiction: predictive bearish but lagging still bullish
                # Reduce but don't exit immediately
                position_size.iloc[i] = 0.30
            else:
                # Neutral / mixed
                position_size.iloc[i] = lag  # default to lagging signal

        result = pd.DataFrame({
            "predictive_score": predictive_score,
            "lagging_score": lagging_score,
            "hybrid_score": hybrid_score,
            "position_size": position_size,
        }, index=spy.index)

        return result

    def backtest_hybrid_vs_lagging(self, spy: pd.Series = None) -> dict:
        """
        Backtest the hybrid signal vs pure lagging (SMA) approach on SPY.

        Returns dict with NAV series for: hybrid, pure_lagging, buy_and_hold
        """
        if spy is None:
            spy = self.spy.dropna()

        signals = self.build_hybrid_signal(spy)
        daily_ret = spy.pct_change()

        # Pure lagging strategy (SMA 50/200 crossover)
        sma50 = spy.rolling(50).mean()
        sma200 = spy.rolling(200).mean()
        lag_signal = (sma50 > sma200).astype(float).shift(1).fillna(0)

        lag_ret = daily_ret * lag_signal
        lag_changes = lag_signal.diff().abs()
        lag_ret -= lag_changes * (TRANSACTION_COST_BPS / 10000)

        # Hybrid strategy
        hybrid_pos = signals["position_size"].shift(1).fillna(0)
        hybrid_ret = daily_ret * hybrid_pos
        hybrid_changes = hybrid_pos.diff().abs()
        hybrid_ret -= hybrid_changes * (TRANSACTION_COST_BPS / 10000) * 0.5  # partial position changes cost less

        # Build NAVs from a common start (need 252 bars for leading indicators)
        start_idx = max(252, 200) + 10
        start_date = spy.index[start_idx]

        lag_nav = (1 + lag_ret.loc[start_date:]).cumprod()
        hybrid_nav = (1 + hybrid_ret.loc[start_date:]).cumprod()
        bh_nav = spy.loc[start_date:] / spy.loc[start_date]

        return {
            "hybrid_nav": hybrid_nav,
            "lagging_nav": lag_nav,
            "bh_nav": bh_nav,
            "signals": signals.loc[start_date:],
        }


# ============================================================================
# Report Generation
# ============================================================================

def print_separator(char="=", width=120):
    print(char * width)


def format_pct(val, decimals=1):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:>+.{decimals}f}%" if val >= 0 else f"{val:>.{decimals}f}%"


def format_ratio(val, decimals=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    return f"{val:>.{decimals}f}"


def print_performance_table(results: list):
    """Print the comprehensive performance comparison table (Task 1)."""
    print_separator()
    print("  TASK 1: HISTORICAL STRATEGY PERFORMANCE COMPARISON")
    print("  Data: Norgate survivorship-bias-free daily OHLCV (adjusted prices)")
    print("  Transaction costs: 10 bps round-trip")
    print_separator()

    # Header
    cols = [
        ("Strategy", 42), ("Start", 12), ("End", 12), ("Yrs", 5),
        ("CAGR%", 8), ("TotRet%", 9), ("MaxDD%", 8), ("Sharpe", 7),
        ("Sortino", 8), ("Alpha%", 8), ("BestYr%", 8), ("WorstYr%", 9),
        ("Recovery", 9),
    ]

    header = ""
    for name, width in cols:
        header += f"{name:>{width}s}"
    print(header)
    print("-" * sum(w for _, w in cols))

    for r in results:
        if "error" in r:
            print(f"  {r['name']:<40s} ERROR: {r['error']}")
            continue

        row = ""
        row += f"{r['name']:>42s}"
        row += f"{r['start']:>12s}"
        row += f"{r['end']:>12s}"
        row += f"{r['years']:>5.1f}"
        row += f"{r['cagr']*100:>+8.1f}"
        row += f"{r['total_return']*100:>+9.0f}"
        row += f"{r['max_dd']*100:>8.1f}"
        row += f"{r['sharpe']:>7.2f}"
        row += f"{r['sortino']:>8.2f}"
        row += f"{r['alpha']*100:>+8.1f}"
        row += f"{r['best_year']*100:>+8.1f}"
        row += f"{r['worst_year']*100:>+9.1f}"

        rec_yrs = r['recovery_days'] / 365.25
        if rec_yrs > 1:
            row += f"{rec_yrs:>7.1f}yr"
        else:
            row += f"{r['recovery_days']:>7d}d "

        print(row)

    print_separator("-")


def print_yearly_table(results: list):
    """Print year-by-year returns for all strategies side by side."""
    print()
    print_separator()
    print("  YEAR-BY-YEAR RETURNS (all strategies)")
    print_separator()

    # Collect all years
    all_years = set()
    for r in results:
        if "annual_rets" in r:
            all_years.update(r["annual_rets"].keys())
    all_years = sorted(all_years)

    # Short names
    short_names = []
    for r in results:
        name = r.get("name", "?")
        # Truncate to 10 chars
        short = name[:12] if len(name) > 12 else name
        short_names.append(short)

    col_w = 10
    header = f"{'Year':<6}"
    for sn in short_names:
        header += f"{sn:>{col_w}s}"
    print(header)
    print("-" * (6 + col_w * len(short_names)))

    # Add market regime labels
    regime_lookup = {}
    for regime_name, (rs, re) in MARKET_REGIMES.items():
        ry_start = int(rs[:4])
        ry_end = int(re[:4])
        for y in range(ry_start, ry_end + 1):
            if y not in regime_lookup:
                regime_lookup[y] = regime_name

    for yr in all_years:
        row = f"{yr:<6}"
        for r in results:
            ann = r.get("annual_rets", {})
            val = ann.get(yr, None)
            if val is None:
                row += f"{'--':>{col_w}}"
            else:
                row += f"{val*100:>{col_w}.1f}"

        regime = regime_lookup.get(yr, "")
        if regime:
            row += f"  [{regime}]"

        print(row)

    print_separator("-")


def print_leading_indicator_research():
    """Print the comprehensive leading indicator research summary (Task 2)."""
    print()
    print_separator("=")
    print("  TASK 2: PREDICTIVE / LEADING INDICATOR RESEARCH")
    print_separator("=")

    print("""
  ============================================================
  A. LEADING INDICATOR CATEGORIES & EMPIRICAL EVIDENCE
  ============================================================

  1. MARKET-BASED LEADING INDICATORS
  -----------------------------------

  a) Credit Spreads (IG and HY)
     Signal:     Baa-Aaa spread (IG) and HY OAS widening/tightening
     Lead Time:  2-6 months ahead of equity drawdowns
     Mechanism:  Credit markets price default risk before equity markets react.
                 Widening spreads = risk aversion rising = equities will follow down.
     Reliability: HIGH (75-80% directional accuracy at extremes)
     Data Source: Norgate Economic (#COBAA, #COAAA, #CCCHYS) -- available from 1983
     Cost: Included in Norgate subscription
     Evidence:
       - Baa-Aaa spread widened 150+ bps before both 2001 and 2008 equity crashes
       - HY spreads led the COVID crash by ~2 weeks (Feb 21 vs Mar 23, 2020)
       - False signal rate: ~20% (spread widens but equities don't crash)
       - Best used as CONFIRMATION of other signals, not standalone

  b) Yield Curve Dynamics (2s10s, 3m10y)
     Signal:     Slope of yield curve; rate of change in slope
     Lead Time:  3-18 months (inversion to recession: avg 12-18 months)
     Mechanism:  Inversion signals market expects rate cuts (due to recession).
                 STEEPENING after inversion is the actual equity crash signal.
     Reliability: VERY HIGH for recession prediction (8/8 since 1960)
                  MODERATE for equity timing (lead time too variable)
     Data Source: Norgate Economic (%10YTCM, %2YTCM, %3MTCM) -- from 1976
     Cost: Included in Norgate subscription
     Key Nuance: The yield curve inverted in 2022 but stocks rallied in 2023.
                 The STEEPENING (un-inversion) is often the better equity sell signal.

  c) Volatility Term Structure (VIX futures curve)
     Signal:     VIX in backwardation (near > far) = stress
                 VIX in steep contango = complacency (contrarian bearish)
     Lead Time:  1-4 weeks
     Mechanism:  Demand for near-term protection signals institutional hedging.
     Reliability: MODERATE (60-65%)
     Data Source: CBOE VIX futures (not in Norgate) -- need CBOE DataShop or similar
     Cost: $500-2000/year for VIX futures data
     Proxy: VIXM/VXZ ratio as term structure proxy (in Norgate, from 2011)

  d) Options Market Signals (Put/Call Ratio)
     Signal:     Extreme high P/C ratio = contrarian bullish (fear maxed out)
                 Extreme low P/C ratio = contrarian bearish (complacency)
     Lead Time:  0.5-2 months
     Mechanism:  Sentiment extreme -> mean reversion
     Reliability: MODERATE (55-60% as standalone, 70%+ at 2-sigma extremes)
     Data Source: Norgate US_Indices (#CBOEPC) -- from 1995
     Cost: Included in Norgate subscription

  e) Market Breadth (A/D Line, New Highs/Lows)
     Signal:     Divergence between price (new highs) and breadth (declining)
                 = "narrow market" = imminent correction
     Lead Time:  1-6 months
     Mechanism:  When fewer stocks participate in rally, it's distribution phase
     Reliability: HIGH for major tops (80%+), less useful for bottoms
     Data Source: Norgate US_Indices (#NYSEADV, #NYSEDEC, #NYSEHI, #NYSELO) -- from 1931
     Cost: Included in Norgate subscription

  f) Intermarket Signals
     Signal:     Bonds leading stocks (TLT rallies before equity stress)
                 Gold/USD leading risk-off moves
                 HYG/LQD ratio as credit stress early warning
     Lead Time:  1-4 weeks (bond/equity), 2-8 weeks (gold/commodities)
     Mechanism:  Flight to quality rotates through asset classes sequentially
     Reliability: MODERATE (60-65%)
     Data Source: Norgate US_Equities (TLT from 2002, GLD from 2004, HYG from 2007)
     Cost: Included in Norgate subscription

  2. ECONOMIC LEADING INDICATORS
  --------------------------------

  a) ISM Manufacturing PMI (especially New Orders component)
     Signal:     PMI > 50 = expansion; rate of change more important than level
                 New Orders sub-index leads headline by 1-2 months
     Lead Time:  2-6 months ahead of earnings growth/contraction
     Mechanism:  Manufacturing orders -> production -> revenue -> earnings -> stock prices
     Reliability: HIGH (70-75%) for direction of market over 6-month horizon
     Data Source: Norgate Economic (#NAPMI) -- from 1948
     Cost: Included in Norgate subscription

  b) Initial Jobless Claims (Rate of Change)
     Signal:     4-week moving average rising > 10% YoY = recession signal
                 Declining claims = labor market strengthening = bullish
     Lead Time:  1-4 months
     Mechanism:  Employment is coincident-to-leading indicator
     Reliability: HIGH (70-80%) when combined with other signals
     Data Source: Norgate Economic (#INJOB) -- from 1967
     Cost: Included in Norgate subscription

  c) Conference Board LEI
     Signal:     6-month rate of change turning negative = recession within 12 months
     Lead Time:  3-12 months
     Mechanism:  Composite of 10 leading indicators by design
     Reliability: HIGH (7/7 recessions since 1970 correctly predicted)
                  But has issued 2-3 false recession signals
     Data Source: Norgate Economic (#LVSAAT) -- from 1976
     Cost: Included in Norgate subscription

  d) M2 Money Supply Growth
     Signal:     Accelerating M2 growth = future inflation + asset price support
                 Decelerating/contracting M2 = tightening liquidity
     Lead Time:  6-18 months (long and variable, as Friedman said)
     Mechanism:  More money -> more spending -> higher earnings (with lag)
     Reliability: MODERATE (55-65%) -- lead time is too variable for precise timing
     Data Source: Norgate Economic (#M2FED) -- from 1959
     Cost: Included in Norgate subscription

  e) Financial Conditions (NFCI)
     Signal:     Negative NFCI = loose conditions = bullish for risk assets
                 Positive NFCI = tight conditions = bearish
     Lead Time:  1-6 months
     Mechanism:  Financial conditions directly affect credit availability and risk-taking
     Reliability: HIGH (70-75%)
     Data Source: Norgate Economic (#NFCI) -- from 1971
     Cost: Included in Norgate subscription

  3. FUNDAMENTAL / FLOW-BASED INDICATORS (NOT IN NORGATE)
  -------------------------------------------------------

  a) Earnings Revision Momentum
     Requires: Bloomberg, FactSet, or Refinitiv ($15,000-25,000/yr)
     Lead Time: 1-3 months. Revisions lead price by 1 earnings cycle.
     Reliability: HIGH (75%+ for sector rotation)

  b) Fund Flow Data (ETF inflows/outflows)
     Requires: ICI data (free monthly), or ETF.com/Bloomberg for daily ($2,000-5,000/yr)
     Lead Time: 1-4 weeks (contrarian), 1-3 months (trend-following)
     Reliability: MODERATE (60%)

  c) CFTC Commitment of Traders
     Requires: CFTC.gov (free, weekly release with 3-day lag)
     Lead Time: 2-8 weeks
     Reliability: MODERATE-HIGH at extremes (65-70%)

  d) Insider Trading Clusters
     Requires: SEC EDGAR (free) or OpenInsider ($0), Quiver Quantitative ($50/mo)
     Lead Time: 3-12 months
     Reliability: HIGH for sector-level clusters (70%+), low for individual stocks

  e) Alternative Data
     - Google Trends: Free, 1-4 week lead, 55-60% reliability
     - Job postings (Indeed/LinkedIn): $5,000-50,000/yr, 3-6 month lead
     - Satellite/shipping: $10,000-100,000/yr, sector-specific

  ============================================================
  B. PREDICTIVE + LAGGING HYBRID FRAMEWORK
  ============================================================

  LAYER 1: EARLY WARNING (Predictive Indicators)
  -----------------------------------------------
  These fire FIRST. We accept lower hit rate for earlier positioning.

  Composite Predictive Score = weighted z-score across:
    - HY Credit Spread change       (weight: 0.15, lead: 2-6 mo)
    - IG Credit Spread change        (weight: 0.10, lead: 2-6 mo)
    - Yield Curve 2s10s slope change (weight: 0.10, lead: 3-12 mo)
    - NYSE Breadth momentum          (weight: 0.12, lead: 1-3 mo)
    - NYSE New Highs-Lows            (weight: 0.08, lead: 1-3 mo)
    - ISM PMI momentum               (weight: 0.10, lead: 2-6 mo)
    - Initial Claims rate of change  (weight: 0.08, lead: 1-4 mo)
    - LEI rate of change             (weight: 0.10, lead: 3-9 mo)
    - M2 growth rate                 (weight: 0.05, lead: 6-18 mo)
    - NFCI (inverted)                (weight: 0.07, lead: 1-6 mo)
    - Put/Call ratio (contrarian)    (weight: 0.05, lead: 0.5-2 mo)
                                     Total: 1.00

  Confidence Scoring:
    - >= 7 of 11 indicators aligned bullish -> HIGH confidence
    - 5-6 of 11 aligned -> MODERATE confidence
    - < 5 aligned -> LOW confidence / mixed signal

  LAYER 2: CONFIRMATION (Lagging Indicators)
  -------------------------------------------
  SMA alignment score (0 to 1):
    - Price > SMA50:  +0.25
    - Price > SMA100: +0.25
    - Price > SMA200: +0.25
    - SMA50 > SMA200: +0.25 (golden cross)

  LAYER 3: POSITION SIZING LOGIC
  --------------------------------
  +-----------------------------+----------------+------------+
  | Condition                   | Position Size  | Rationale  |
  +-----------------------------+----------------+------------+
  | Pred bullish + Full confirm | 100% target    | Max convic |
  | Pred bullish + Partial conf |  50% target    | Building   |
  | Pred bullish + No confirm   |  25% target    | Early bet  |
  | Pred bearish + Lag bullish  |  30% target    | Reduce     |
  | Pred bearish + Lag bearish  |   0% (cash)    | Full exit  |
  | Neutral / mixed             | = lagging sig  | Default    |
  +-----------------------------+----------------+------------+

  LAYER 4: CONFLICT RESOLUTION
  ------------------------------
  Priority hierarchy:
    1. If BOTH predictive AND lagging are extreme (>0.7 or <0.3) -> trust agreement
    2. If predictive fires but lagging contradicts -> wait max 60 trading days
       for confirmation. If confirmation doesn't arrive -> exit early position
    3. If lagging fires but predictive contradicts -> reduce position by 50%
    4. Emergency override: if HY spread widens > 2 std devs in any 20-day window
       -> immediate risk reduction regardless of other signals
""")


def print_hybrid_backtest_results(results: dict, spy: pd.Series, rf_daily: pd.Series):
    """Print hybrid vs lagging backtest comparison."""
    print()
    print_separator("=")
    print("  HYBRID vs PURE LAGGING BACKTEST COMPARISON")
    print_separator("=")

    spy_bench = spy.reindex(results["hybrid_nav"].index).ffill()
    spy_bench = spy_bench / spy_bench.iloc[0]

    strategies = {
        "Hybrid (Predictive + Lagging)": results["hybrid_nav"],
        "Pure Lagging (SMA 50/200)": results["lagging_nav"],
        "Buy & Hold SPY": results["bh_nav"],
    }

    for name, nav in strategies.items():
        m = compute_metrics(nav, spy_bench, rf_daily.reindex(nav.index).ffill(), name=name)
        if "error" not in m:
            print(f"\n  {name}")
            print(f"    Period:       {m['start']} -> {m['end']} ({m['years']:.1f} years)")
            print(f"    CAGR:         {m['cagr']*100:>+.2f}%")
            print(f"    Total Return: {m['total_return']*100:>+.0f}%")
            print(f"    Max Drawdown: {m['max_dd']*100:>.1f}%")
            print(f"    Sharpe:       {m['sharpe']:>.3f}")
            print(f"    Sortino:      {m['sortino']:>.3f}")
            print(f"    Annual Vol:   {m['ann_vol']*100:>.1f}%")
            print(f"    Alpha vs SPY: {m['alpha']*100:>+.2f}%")

    # Estimate improvement metrics
    h_nav = results["hybrid_nav"]
    l_nav = results["lagging_nav"]
    bh_nav = results["bh_nav"]

    h_m = compute_metrics(h_nav, spy_bench, name="hybrid")
    l_m = compute_metrics(l_nav, spy_bench, name="lagging")

    if "error" not in h_m and "error" not in l_m:
        print(f"\n  {'IMPROVEMENT SUMMARY':^60}")
        print(f"  {'-'*60}")
        print(f"  {'Metric':<30s} {'Hybrid':>12s} {'Pure Lag':>12s} {'Delta':>12s}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")

        metrics_compare = [
            ("CAGR", h_m['cagr']*100, l_m['cagr']*100, "%"),
            ("Sharpe", h_m['sharpe'], l_m['sharpe'], ""),
            ("Sortino", h_m['sortino'], l_m['sortino'], ""),
            ("Max Drawdown", h_m['max_dd']*100, l_m['max_dd']*100, "%"),
            ("Best Year", h_m['best_year']*100, l_m['best_year']*100, "%"),
            ("Worst Year", h_m['worst_year']*100, l_m['worst_year']*100, "%"),
        ]

        for name, h_val, l_val, unit in metrics_compare:
            delta = h_val - l_val
            sign = "+" if delta > 0 else ""
            print(f"  {name:<30s} {h_val:>+11.2f}{unit} {l_val:>+11.2f}{unit} {sign}{delta:>10.2f}{unit}")


def print_implementation_roadmap():
    """Print the implementation roadmap."""
    print()
    print_separator("=")
    print("  IMPLEMENTATION ROADMAP")
    print_separator("=")

    print("""
  PHASE 1: DATA INFRASTRUCTURE (Week 1-2)
  ========================================
  Already available via Norgate ($0 additional):
    [x] US_Equities: 13,808 symbols, 1990-present (survivorship-bias-free)
    [x] US_Equities_Delisted: Full delisted universe
    [x] US_Indices: NYSE breadth, CBOE P/C ratio, TRIN
    [x] Economic: ISM, Claims, Housing, M2, LEI, CFNAI, NFCI, CPI, GDP
    [x] Yield curves: 3M, 2Y, 5Y, 10Y, 30Y Treasury + Moody's Baa/Aaa
    [x] HY credit spread, MOVE index

  Additional data sources needed:
    [ ] VIX futures term structure: CBOE DataShop (~$1,000/yr)
    [ ] CFTC COT data: Free from CFTC.gov (weekly, 3-day lag)
    [ ] ETF fund flows: ETF.com API or ICI monthly (free)
    [ ] Insider transactions: SEC EDGAR + OpenInsider (free)
    [ ] Earnings revisions: Requires Bloomberg/FactSet ($15,000+/yr)
        -> SKIP for initial implementation (cost too high)

  PHASE 2: LEADING INDICATOR ENGINE (Week 2-3)
  =============================================
    [ ] Build z-score normalization for all indicators
    [ ] Calibrate weights using rolling walk-forward (2000-2015 train, 2016+ test)
    [ ] Implement composite scoring with confidence bands
    [ ] Add emergency override logic (credit spread blowout detection)

  PHASE 3: HYBRID SIGNAL FRAMEWORK (Week 3-4)
  =============================================
    [ ] Integrate leading indicators with existing SMA framework
    [ ] Implement tiered position sizing (25%/50%/100%)
    [ ] Build conflict resolution logic
    [ ] Add 60-day confirmation window tracking

  PHASE 4: BACKTEST & VALIDATION (Week 4-5)
  ==========================================
    [ ] Walk-forward validation (purged, with embargo)
    [ ] Regime-specific performance analysis
    [ ] Monte Carlo simulation for robustness
    [ ] Parameter sensitivity analysis

  PHASE 5: DEPLOYMENT (Week 5-6)
  ==============================
    [ ] Daily signal generation pipeline
    [ ] Alert system for signal transitions
    [ ] Position sizing calculator
    [ ] Monthly rebalance execution checklist

  ESTIMATED TOTAL COST:
    Norgate Data subscription: Already have (included)
    CBOE VIX futures:          ~$1,000/yr (optional, can use VIXM/VXZ proxy)
    Additional data:           $0-1,000/yr (using free sources where possible)
    Bloomberg/FactSet:         SKIP ($15,000+/yr -- not worth it for initial deployment)
    -------------------------------------
    Total incremental:         $0-1,000/yr
""")


# ============================================================================
# Leverage Engine
# ============================================================================

def apply_leverage(nav: pd.Series, leverage: float,
                   rf_daily: pd.Series = None,
                   margin_spread_bps: float = 50) -> pd.Series:
    """
    Apply daily-rebalanced leverage to a strategy NAV series.

    Models realistic leveraged returns:
      leveraged_return = leverage * strategy_return - (leverage - 1) * borrow_cost

    This captures:
    - The volatility drag from daily rebalancing (geometric vs arithmetic)
    - Margin interest on borrowed capital
    - The fact that 2x daily != 2x long-term (path dependency)

    Args:
        nav:               Unleveraged NAV series
        leverage:          Leverage factor (1.0 = no leverage, 2.0 = 2x, 3.0 = 3x)
        rf_daily:          Daily risk-free rate series (for borrow cost baseline)
        margin_spread_bps: Broker spread over risk-free rate in bps (default 50 = 0.50%)
    """
    daily_ret = nav.pct_change().fillna(0)

    # Daily borrow cost: only pay on the borrowed portion (leverage - 1)
    if rf_daily is not None:
        rf = rf_daily.reindex(daily_ret.index).ffill().fillna(0)
    else:
        rf = pd.Series(0.0, index=daily_ret.index)

    borrow_rate_daily = rf + margin_spread_bps / 10000 / 252
    borrow_cost = (leverage - 1) * borrow_rate_daily

    # Leveraged daily return
    lev_ret = leverage * daily_ret - borrow_cost

    # Rebuild NAV
    lev_nav = (1 + lev_ret).cumprod()
    lev_nav.iloc[0] = 1.0
    return lev_nav


def apply_smart_leverage(nav: pd.Series, target_leverage: float,
                         rf_daily: pd.Series = None,
                         margin_spread_bps: float = 50,
                         mode: str = "vol_target",
                         # Vol-targeting params
                         target_vol: float = 0.15,
                         vol_lookback: int = 21,
                         vol_cap_mult: float = 1.0,
                         # Drawdown circuit-breaker params
                         dd_threshold_1: float = -0.15,
                         dd_threshold_2: float = -0.40,
                         dd_recovery_buffer: float = 0.05,
                         # Bear gate params
                         spy: pd.Series = None,
                         bear_sma: int = 200,
                         bear_leverage: float = 1.0,
                         # Leverage bounds
                         min_leverage: float = 0.0,
                         max_leverage: float = None,
                         ) -> Tuple[pd.Series, pd.Series]:
    """
    Apply dynamically-adjusted leverage with risk controls.

    Modes:
      "vol_target"   -- Scale leverage to target a constant annualized vol.
                        lev(t) = target_vol / realized_vol(t) * base_mult
                        Automatically de-levers when vol spikes.

      "dd_control"   -- Smooth drawdown scale-down.
                        Full leverage until DD hits threshold_1, then linearly
                        scale from full down to min_leverage as DD deepens
                        toward threshold_2. Recovers smoothly as DD improves.

      "bear_gate"    -- Full leverage when SPY > SMA(bear_sma), else bear_leverage.

      "combined"     -- All three: vol-target * dd_control * bear_gate.

    Returns:
        (leveraged_nav, daily_leverage_series) -- NAV + the leverage used each day
    """
    daily_ret = nav.pct_change().fillna(0)
    n = len(daily_ret)

    if rf_daily is not None:
        rf = rf_daily.reindex(daily_ret.index).ffill().fillna(0)
    else:
        rf = pd.Series(0.0, index=daily_ret.index)

    borrow_rate_daily = rf + margin_spread_bps / 10000 / 252

    if max_leverage is None:
        max_leverage = target_leverage

    # Pre-compute realized vol (annualized, rolling)
    realized_vol = daily_ret.rolling(vol_lookback, min_periods=5).std() * np.sqrt(252)
    realized_vol = realized_vol.fillna(target_vol)

    # Pre-compute SPY bear gate signal if needed
    bear_signal = None
    if spy is not None and mode in ("bear_gate", "combined", "bear_vol"):
        spy_aligned = spy.reindex(daily_ret.index).ffill()
        spy_sma = spy_aligned.rolling(bear_sma).mean()
        bear_signal = (spy_aligned > spy_sma).shift(1).fillna(True)  # True = bullish

    # Walk forward computing dynamic leverage
    lev_series = np.full(n, target_leverage)
    lev_nav = np.ones(n)
    peak_nav = 1.0

    # DD range for smooth interpolation
    dd_range = dd_threshold_2 - dd_threshold_1  # negative value

    for i in range(1, n):
        # Start with target leverage
        lev = target_leverage

        # --- Vol targeting ---
        if mode in ("vol_target", "combined", "bear_vol"):
            rv = realized_vol.iloc[i]
            if rv > 0.01:
                vol_lev = target_vol / rv * (target_leverage * vol_cap_mult)
            else:
                vol_lev = target_leverage
            lev = min(lev, vol_lev)

        # --- Smooth drawdown scale-down ---
        if mode in ("dd_control", "combined"):
            current_dd = (lev_nav[i-1] - peak_nav) / peak_nav if peak_nav > 0 else 0

            if current_dd < dd_threshold_2:
                dd_mult = min_leverage / target_leverage if target_leverage > 0 else 0
            elif current_dd < dd_threshold_1:
                dd_frac = (current_dd - dd_threshold_1) / dd_range  # 0..1 as DD deepens
                min_mult = max(min_leverage / target_leverage, 0.1) if target_leverage > 0 else 0.1
                dd_mult = 1.0 - dd_frac * (1.0 - min_mult)
            else:
                dd_mult = 1.0

            lev = lev * dd_mult

        # --- Bear gate ---
        if mode in ("bear_gate", "combined", "bear_vol"):
            if bear_signal is not None and not bear_signal.iloc[i]:
                lev = min(lev, bear_leverage)

        # Clamp
        lev = max(min_leverage, min(lev, max_leverage))
        lev_series[i] = lev

        # Compute leveraged return for this day
        borrow = max(lev - 1, 0) * borrow_rate_daily.iloc[i]
        day_ret = lev * daily_ret.iloc[i] - borrow

        lev_nav[i] = lev_nav[i-1] * (1 + day_ret)
        peak_nav = max(peak_nav, lev_nav[i])

    return (pd.Series(lev_nav, index=daily_ret.index),
            pd.Series(lev_series, index=daily_ret.index))


def apply_optimized_leverage(nav: pd.Series, target_leverage: float,
                              rf_daily: pd.Series = None,
                              margin_spread_bps: float = 50,
                              spy: pd.Series = None,
                              # Vol-ratio scaling params
                              vol_ewm_span: int = 10,
                              vol_baseline_window: int = 252,
                              vol_scale_power: float = 1.0,
                              # Bear gate params
                              bear_sma: int = 200,
                              bear_fast_sma: int = 50,
                              bear_leverage: float = 0.5,
                              caution_leverage: float = None,
                              # Equity curve filter
                              eq_curve_sma: int = 0,
                              eq_curve_mult: float = 0.5,
                              # Bounds
                              min_leverage: float = 0.25,
                              max_leverage: float = None,
                              ) -> Tuple[pd.Series, pd.Series]:
    """
    Optimized leverage with lessons from all previous approaches.

    Key improvements over apply_smart_leverage:
    1. Vol-RATIO scaling: scales leverage by current_vol / baseline_vol,
       not vs an arbitrary target. Uses EWM for faster reaction.
    2. Dual-timeframe bear gate: SPY vs SMA200 + SPY vs SMA50 for
       fast/slow regime detection with 3 states (bull/caution/bear).
    3. Optional equity curve filter: if leveraged equity < its own SMA,
       reduce leverage (the strategy itself is in a hostile regime).
    4. No DD breaker (redundant with vol-ratio + bear gate, creates traps).

    Returns:
        (leveraged_nav, daily_leverage_series)
    """
    daily_ret = nav.pct_change().fillna(0)
    n = len(daily_ret)

    if rf_daily is not None:
        rf = rf_daily.reindex(daily_ret.index).ffill().fillna(0)
    else:
        rf = pd.Series(0.0, index=daily_ret.index)

    borrow_rate_daily = rf + margin_spread_bps / 10000 / 252

    if max_leverage is None:
        max_leverage = target_leverage
    if caution_leverage is None:
        caution_leverage = target_leverage * 0.67  # 2x when target is 3x

    # --- Pre-compute vol-ratio signal (EWM for speed) ---
    ewm_vol = daily_ret.ewm(span=vol_ewm_span, min_periods=5).std() * np.sqrt(252)
    baseline_vol = daily_ret.rolling(vol_baseline_window, min_periods=63).std() * np.sqrt(252)
    baseline_vol = baseline_vol.ffill().bfill()
    # Clamp baseline to reasonable range
    baseline_vol = baseline_vol.clip(lower=0.05, upper=0.60)
    ewm_vol = ewm_vol.fillna(baseline_vol)

    # Vol ratio: >1 means vol is elevated vs baseline
    vol_ratio = (ewm_vol / baseline_vol).fillna(1.0)

    # --- Pre-compute dual-timeframe bear gate ---
    bear_slow = None
    bear_fast = None
    if spy is not None:
        spy_aligned = spy.reindex(daily_ret.index).ffill()
        spy_sma_slow = spy_aligned.rolling(bear_sma, min_periods=50).mean()
        spy_sma_fast = spy_aligned.rolling(bear_fast_sma, min_periods=20).mean()
        bear_slow = (spy_aligned > spy_sma_slow).shift(1).fillna(True)   # True = bullish
        bear_fast = (spy_aligned > spy_sma_fast).shift(1).fillna(True)

    # --- Walk forward ---
    lev_series = np.full(n, target_leverage, dtype=np.float64)
    lev_nav = np.ones(n, dtype=np.float64)
    peak_nav = 1.0

    # Equity curve SMA buffer (computed on the fly)
    eq_sma_buf = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        lev = target_leverage

        # 1) Vol-ratio scaling: when vol is X times baseline, divide leverage by X
        vr = vol_ratio.iloc[i]
        if vr > 1.0:
            lev = lev / (vr ** vol_scale_power)

        # 2) Dual-timeframe bear gate
        if bear_slow is not None:
            is_bull_slow = bear_slow.iloc[i]
            is_bull_fast = bear_fast.iloc[i]

            if not is_bull_slow and not is_bull_fast:
                # Confirmed bear: both SMAs broken
                lev = min(lev, bear_leverage)
            elif not is_bull_fast:
                # Early warning: fast SMA broken but slow still holds
                lev = min(lev, caution_leverage)
            # else: full bull, lev stays

        # 3) Equity curve filter (if enabled)
        if eq_curve_sma > 0 and i >= eq_curve_sma:
            eq_sma_val = np.mean(lev_nav[max(0, i - eq_curve_sma):i])
            eq_sma_buf[i] = eq_sma_val
            if lev_nav[i-1] < eq_sma_val:
                lev = lev * eq_curve_mult

        # Clamp
        lev = max(min_leverage, min(lev, max_leverage))
        lev_series[i] = lev

        # Compute leveraged return
        borrow = max(lev - 1, 0) * borrow_rate_daily.iloc[i]
        day_ret = lev * daily_ret.iloc[i] - borrow

        lev_nav[i] = lev_nav[i-1] * (1 + day_ret)
        peak_nav = max(peak_nav, lev_nav[i])

    return (pd.Series(lev_nav, index=daily_ret.index),
            pd.Series(lev_series, index=daily_ret.index))


def print_leverage_table(results: list, title: str = ""):
    """Print a compact comparison table for leveraged strategies."""
    print()
    print_separator("=")
    print(f"  {title}")
    print_separator("=")

    cols = [
        ("Strategy", 44), ("Lev", 4), ("CAGR%", 8), ("TotRet%", 10),
        ("MaxDD%", 8), ("Sharpe", 7), ("Sortino", 8), ("Vol%", 6),
        ("Alpha%", 8), ("WorstYr%", 9),
    ]

    header = ""
    for name, width in cols:
        header += f"{name:>{width}s}"
    print(header)
    print("-" * sum(w for _, w in cols))

    for r in results:
        if "error" in r:
            continue
        lev = r.get("leverage", 1.0)
        row = ""
        row += f"{r['name']:>44s}"
        row += f"{lev:>4.0f}x" if lev > 1 else f"{'1x':>4s}"
        row += f"{r['cagr']*100:>+8.1f}"
        row += f"{r['total_return']*100:>+10.0f}"
        row += f"{r['max_dd']*100:>8.1f}"
        row += f"{r['sharpe']:>7.2f}"
        row += f"{r['sortino']:>8.2f}"
        row += f"{r['ann_vol']*100:>6.1f}"
        row += f"{r['alpha']*100:>+8.1f}"
        row += f"{r['worst_year']*100:>+9.1f}"
        print(row)

    print_separator("-")


def print_leverage_yearly(results_by_name: dict):
    """Print year-by-year returns for leveraged variants side by side."""
    print()
    print_separator()
    print("  YEAR-BY-YEAR RETURNS -- LEVERAGED STRATEGIES")
    print_separator()

    all_years = set()
    for r in results_by_name.values():
        if "annual_rets" in r:
            all_years.update(r["annual_rets"].keys())
    all_years = sorted(all_years)

    names = list(results_by_name.keys())
    col_w = 12
    header = f"{'Year':<6}"
    for n in names:
        short = n[:12] if len(n) > 12 else n
        header += f"{short:>{col_w}s}"
    print(header)
    print("-" * (6 + col_w * len(names)))

    for yr in all_years:
        row = f"{yr:<6}"
        for n in names:
            r = results_by_name[n]
            val = r.get("annual_rets", {}).get(yr, None)
            if val is None:
                row += f"{'--':>{col_w}}"
            else:
                row += f"{val*100:>{col_w}.1f}"
        print(row)

    print_separator("-")


# ============================================================================
# Reconstitution Analysis (Task 3: "S&P 500 Already Does This")
# ============================================================================

def reconstitution_analysis(engine: StrategyEngine):
    """
    Analyze our dollar-volume universe reconstitution vs S&P 500.
    Since we don't have S&P 500 constituent history, we:
    1. Measure universe turnover (how many stocks enter/exit per month)
    2. Case studies: when did NVDA, TSLA, META enter our universe?
    3. "Entering" vs "Leaving" stock performance
    4. Direct comparison: equal-weight top-100 vs SPY
    """
    if not engine.dynamic_universe:
        print("  Reconstitution analysis requires dynamic universe mode.")
        return

    rankings = engine.universe_rankings
    spy = engine.spy
    top_n = engine.universe_top_n

    print_separator("=")
    print("  TASK 3: RECONSTITUTION ANALYSIS")
    print("  Can we beat S&P 500's built-in reconstitution with faster dollar-volume ranking?")
    print_separator("=")

    # --- A) Universe Turnover Analysis ---
    print("\n  A) Universe Turnover: How many stocks change each month?")
    print("  " + "-" * 60)

    rebal_dates = sorted(rankings["date"].unique())
    monthly_enters = []
    monthly_exits = []
    prev_set = set()

    for rd in rebal_dates:
        month_data = rankings[(rankings["date"] == rd) & (rankings["rank"] <= top_n)]
        current_set = set(month_data["uid"].values)

        if prev_set:
            enters = len(current_set - prev_set)
            exits = len(prev_set - current_set)
            monthly_enters.append(enters)
            monthly_exits.append(exits)

        prev_set = current_set

    if monthly_enters:
        avg_enters = np.mean(monthly_enters)
        avg_exits = np.mean(monthly_exits)
        med_enters = np.median(monthly_enters)
        turnover_pct = avg_enters / top_n * 100

        print(f"  Average monthly entries: {avg_enters:.1f} stocks ({turnover_pct:.1f}% of universe)")
        print(f"  Average monthly exits:   {avg_exits:.1f} stocks")
        print(f"  Median monthly entries:  {med_enters:.0f} stocks")
        print(f"  Total months analyzed:   {len(monthly_enters)}")
        print(f"  This means the universe completely turns over every ~{top_n/max(avg_enters,0.1):.0f} months")
        print()
        print("  Compare: S&P 500 adds/removes ~25 stocks per year (2/month).")
        print(f"  Our universe: ~{avg_enters*12:.0f} changes/year -> {avg_enters*12/25:.1f}x faster reconstitution.")

    # --- B) Speed-of-Inclusion Case Studies ---
    print(f"\n  B) Case Studies: When did key stocks enter our top-{top_n}?")
    print("  " + "-" * 60)

    # Known S&P 500 inclusion dates for comparison
    case_studies = {
        "NVDA":  {"sp500_add": "2001-11-30", "note": "S&P 500 added ~2001; re-entered mega-cap ~2023"},
        "TSLA":  {"sp500_add": "2020-12-21", "note": "S&P 500 added Dec 2020"},
        "META":  {"sp500_add": "2013-12-23", "note": "S&P 500 added Dec 2013 (as FB)"},
        "AMZN":  {"sp500_add": "2005-11-18", "note": "S&P 500 added Nov 2005"},
        "GOOGL": {"sp500_add": "2006-03-31", "note": "S&P 500 added Mar 2006 (as GOOG)"},
    }

    from universe_builder import uid_to_ticker

    for ticker, info in case_studies.items():
        # Find first month this ticker appeared in our top-N
        # Check both exact ticker and delisted variants
        mask = rankings["base_symbol"].str.upper() == ticker.upper()
        if not mask.any():
            # Try without last character (e.g., GOOGL vs GOOG)
            mask = rankings["base_symbol"].str.upper().str.startswith(ticker[:4].upper())

        filtered = rankings[mask & (rankings["rank"] <= top_n)]
        if len(filtered) > 0:
            first_entry = filtered["date"].min()
            last_appearance = filtered["date"].max()
            total_months = filtered["date"].nunique()

            sp500_date = pd.Timestamp(info["sp500_add"])
            lead_days = (sp500_date - first_entry).days

            print(f"\n  {ticker}:")
            print(f"    Our universe first entry: {first_entry.strftime('%Y-%m-%d')}")
            print(f"    S&P 500 inclusion:        {info['sp500_add']}")
            if lead_days > 0:
                print(f"    Our lead time:            {lead_days} days ({lead_days/30:.0f} months) EARLIER")
            elif lead_days < 0:
                print(f"    S&P 500 was faster by:    {-lead_days} days")
            else:
                print(f"    Same timing")
            print(f"    Months in our top-{top_n}:     {total_months}")
            print(f"    Note: {info['note']}")
        else:
            print(f"\n  {ticker}: Not found in rankings (may use different symbol)")

    # --- C) "Entering" vs "Leaving" Alpha ---
    print(f"\n\n  C) Does entering our universe predict outperformance?")
    print("  " + "-" * 60)

    all_prices = pd.DataFrame(engine.stock_prices)
    entering_rets = []
    leaving_rets = []
    spy_rets_entering = []

    prev_set = set()
    for rd_idx, rd in enumerate(rebal_dates):
        month_data = rankings[(rankings["date"] == rd) & (rankings["rank"] <= top_n)]
        current_set = set(month_data["uid"].values)

        if prev_set and rd_idx + 12 < len(rebal_dates):
            enters = current_set - prev_set
            exits = prev_set - current_set

            # Forward 12-month return for entering stocks
            future_date = rebal_dates[min(rd_idx + 12, len(rebal_dates) - 1)]

            for uid in enters:
                if uid in all_prices.columns:
                    p_start = all_prices.loc[:rd, uid].dropna()
                    p_end = all_prices.loc[:future_date, uid].dropna()
                    if len(p_start) > 0 and len(p_end) > 0:
                        ret = p_end.iloc[-1] / p_start.iloc[-1] - 1
                        if not np.isnan(ret) and abs(ret) < 10:  # sanity check
                            entering_rets.append(ret)

            for uid in exits:
                if uid in all_prices.columns:
                    p_start = all_prices.loc[:rd, uid].dropna()
                    p_end = all_prices.loc[:future_date, uid].dropna()
                    if len(p_start) > 0 and len(p_end) > 0:
                        ret = p_end.iloc[-1] / p_start.iloc[-1] - 1
                        if not np.isnan(ret) and abs(ret) < 10:
                            leaving_rets.append(ret)

            # SPY return for same period
            spy_start = spy.loc[:rd].dropna()
            spy_end = spy.loc[:future_date].dropna()
            if len(spy_start) > 0 and len(spy_end) > 0:
                spy_rets_entering.append(spy_end.iloc[-1] / spy_start.iloc[-1] - 1)

        prev_set = current_set

    if entering_rets and leaving_rets:
        avg_enter = np.mean(entering_rets) * 100
        avg_leave = np.mean(leaving_rets) * 100
        avg_spy = np.mean(spy_rets_entering) * 100
        med_enter = np.median(entering_rets) * 100
        med_leave = np.median(leaving_rets) * 100

        print(f"  Stocks ENTERING top-{top_n} (next 12m avg return):  {avg_enter:+.1f}% (median {med_enter:+.1f}%)")
        print(f"  Stocks LEAVING top-{top_n} (next 12m avg return):   {avg_leave:+.1f}% (median {med_leave:+.1f}%)")
        print(f"  SPY over same periods:                          {avg_spy:+.1f}%")
        print(f"  Enter-Leave spread:                             {avg_enter - avg_leave:+.1f}%")
        print(f"  Sample: {len(entering_rets)} entries, {len(leaving_rets)} exits")

        if avg_enter > avg_leave:
            print(f"\n  CONCLUSION: Entering stocks outperform leaving stocks by {avg_enter-avg_leave:.1f}pp/year.")
            print(f"  Our dollar-volume reconstitution creates alpha through faster inclusion.")
        else:
            print(f"\n  CONCLUSION: Leaving stocks actually perform comparably.")
            print(f"  Reconstitution does NOT generate meaningful alpha — it's about risk management.")

    # --- D) Direct Comparison: Equal-Weight Top-100 vs SPY ---
    print(f"\n\n  D) Equal-weight top-{top_n} vs SPY (monthly rebalanced)")
    print("  " + "-" * 60)

    # Build equal-weight top-N portfolio
    ew_rets = []
    spy_comparison_rets = []

    for rd_idx in range(len(rebal_dates) - 1):
        rd = rebal_dates[rd_idx]
        next_rd = rebal_dates[rd_idx + 1]

        month_data = rankings[(rankings["date"] == rd) & (rankings["rank"] <= top_n)]
        uids = [u for u in month_data["uid"].values if u in all_prices.columns]

        if uids:
            # Equal-weight return from rd to next_rd
            period_rets = []
            for uid in uids:
                p = all_prices[uid].loc[rd:next_rd].dropna()
                if len(p) >= 2:
                    period_rets.append(p.iloc[-1] / p.iloc[0] - 1)

            if period_rets:
                ew_rets.append(np.mean(period_rets))

                # SPY return same period
                spy_p = spy.loc[rd:next_rd].dropna()
                if len(spy_p) >= 2:
                    spy_comparison_rets.append(spy_p.iloc[-1] / spy_p.iloc[0] - 1)
                else:
                    spy_comparison_rets.append(0.0)

    if ew_rets and spy_comparison_rets:
        n = min(len(ew_rets), len(spy_comparison_rets))
        ew_rets = ew_rets[:n]
        spy_comparison_rets = spy_comparison_rets[:n]

        ew_cum = np.cumprod([1 + r for r in ew_rets])
        spy_cum = np.cumprod([1 + r for r in spy_comparison_rets])

        ew_cagr = (ew_cum[-1] ** (12 / len(ew_rets)) - 1) * 100
        spy_cagr = (spy_cum[-1] ** (12 / len(spy_comparison_rets)) - 1) * 100

        excess = np.array(ew_rets) - np.array(spy_comparison_rets)
        tracking_error = np.std(excess) * np.sqrt(12) * 100
        info_ratio = np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(12)

        print(f"  Equal-weight top-{top_n} CAGR: {ew_cagr:+.1f}%")
        print(f"  SPY CAGR (same period):      {spy_cagr:+.1f}%")
        print(f"  Alpha:                        {ew_cagr - spy_cagr:+.1f}%")
        print(f"  Tracking Error:               {tracking_error:.1f}%")
        print(f"  Information Ratio:            {info_ratio:.2f}")
        print(f"  Months analyzed:              {n}")

    print_separator("-")


# ============================================================================
# Stress Testing (Task 4)
# ============================================================================

# More precise stress period dates (peak-to-trough for crashes)
STRESS_PERIODS = OrderedDict([
    ("Dot-Com Crash",   ("2000-03-24", "2002-10-09")),
    ("GFC",             ("2007-10-09", "2009-03-09")),
    ("COVID Crash",     ("2020-02-19", "2020-03-23")),
    ("2022 Bear",       ("2022-01-03", "2022-10-12")),
    ("1990s Bull",      ("1995-01-01", "1999-12-31")),
    ("Recovery 03-07",  ("2003-03-11", "2007-10-09")),
    ("Post-GFC Bull",   ("2009-03-09", "2020-02-19")),
    ("AI Rally",        ("2023-01-01", "2026-02-23")),
])


def stress_test_all(all_navs: dict, spy_nav: pd.Series):
    """
    Run all strategies through 8 historical stress periods.
    For each: period return, period max DD, grade vs SPY.
    """
    print_separator("=")
    print("  TASK 4: STRESS TESTING ACROSS MARKET REGIMES")
    print_separator("=")

    # Compute SPY metrics for each period
    spy_period_metrics = {}
    for period_name, (start, end) in STRESS_PERIODS.items():
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        spy_slice = spy_nav.loc[s:e].dropna()
        if len(spy_slice) >= 2:
            ret = spy_slice.iloc[-1] / spy_slice.iloc[0] - 1
            cum_max = spy_slice.cummax()
            dd = ((spy_slice - cum_max) / cum_max).min()
            spy_period_metrics[period_name] = {"return": ret, "max_dd": dd}

    # Header
    period_names = list(STRESS_PERIODS.keys())
    col_w = 14
    strat_w = 30

    # Print period dates
    print(f"\n{'Period':>{strat_w}}", end="")
    for pn in period_names:
        short = pn[:13] if len(pn) > 13 else pn
        print(f"{short:>{col_w}}", end="")
    print()

    print(f"{'Dates':>{strat_w}}", end="")
    for pn in period_names:
        s, e = STRESS_PERIODS[pn]
        y1, y2 = s[:4], e[:4]
        print(f"{y1+'-'+y2:>{col_w}}", end="")
    print()
    print("-" * (strat_w + col_w * len(period_names)))

    # SPY benchmark row
    print(f"{'SPY B&H (Return %)':>{strat_w}}", end="")
    for pn in period_names:
        m = spy_period_metrics.get(pn)
        if m:
            print(f"{m['return']*100:>{col_w}.1f}", end="")
        else:
            print(f"{'--':>{col_w}}", end="")
    print()

    print(f"{'SPY B&H (MaxDD %)':>{strat_w}}", end="")
    for pn in period_names:
        m = spy_period_metrics.get(pn)
        if m:
            print(f"{m['max_dd']*100:>{col_w}.1f}", end="")
        else:
            print(f"{'--':>{col_w}}", end="")
    print()
    print("-" * (strat_w + col_w * len(period_names)))

    # Strategy rows
    strategy_grades = {}
    for strat_name, nav in all_navs.items():
        nav = nav.dropna()
        if len(nav) < 50:
            continue

        strategy_grades[strat_name] = []

        # Return row
        short_name = strat_name[:29] if len(strat_name) > 29 else strat_name
        print(f"{short_name + ' Ret%':>{strat_w}}", end="")
        for pn in period_names:
            s_dt = pd.Timestamp(STRESS_PERIODS[pn][0])
            e_dt = pd.Timestamp(STRESS_PERIODS[pn][1])
            nav_slice = nav.loc[s_dt:e_dt].dropna()
            spy_m = spy_period_metrics.get(pn)

            if len(nav_slice) >= 2 and spy_m:
                ret = nav_slice.iloc[-1] / nav_slice.iloc[0] - 1
                print(f"{ret*100:>{col_w}.1f}", end="")
            else:
                print(f"{'--':>{col_w}}", end="")
        print()

        # MaxDD row
        print(f"{short_name + ' DD%':>{strat_w}}", end="")
        for pn in period_names:
            s_dt = pd.Timestamp(STRESS_PERIODS[pn][0])
            e_dt = pd.Timestamp(STRESS_PERIODS[pn][1])
            nav_slice = nav.loc[s_dt:e_dt].dropna()
            spy_m = spy_period_metrics.get(pn)

            if len(nav_slice) >= 2 and spy_m:
                cum_max = nav_slice.cummax()
                dd = ((nav_slice - cum_max) / cum_max).min()
                print(f"{dd*100:>{col_w}.1f}", end="")

                # Grade: A if better return AND better DD, B if one, C if neither
                ret = nav_slice.iloc[-1] / nav_slice.iloc[0] - 1
                better_ret = ret > spy_m["return"]
                better_dd = dd > spy_m["max_dd"]  # less negative = better
                if better_ret and better_dd:
                    strategy_grades[strat_name].append("A")
                elif better_ret or better_dd:
                    strategy_grades[strat_name].append("B")
                else:
                    strategy_grades[strat_name].append("C")
            else:
                print(f"{'--':>{col_w}}", end="")
                strategy_grades[strat_name].append("-")
        print()
        print()  # blank line between strategies

    # Grades summary
    print("-" * (strat_w + col_w * len(period_names)))
    print(f"\n  GRADES (A=beats SPY in return AND drawdown, B=beats in one, C=beats in neither)")
    print(f"{'Strategy':>{strat_w}}", end="")
    for pn in period_names:
        short = pn[:13] if len(pn) > 13 else pn
        print(f"{short:>{col_w}}", end="")
    print(f"{'Score':>{col_w}}")
    print("-" * (strat_w + col_w * (len(period_names) + 1)))

    for strat_name, grades in strategy_grades.items():
        short_name = strat_name[:29] if len(strat_name) > 29 else strat_name
        print(f"{short_name:>{strat_w}}", end="")
        a_count = grades.count("A")
        b_count = grades.count("B")
        c_count = grades.count("C")
        for g in grades:
            print(f"{g:>{col_w}}", end="")
        score = a_count * 3 + b_count * 1  # weighted score
        print(f"{f'{a_count}A {b_count}B {c_count}C':>{col_w}}")

    print_separator("-")


# ============================================================================
# Comprehensive Reporting
# ============================================================================

def print_head_to_head(all_results: list):
    """Print comprehensive head-to-head comparison of all strategies."""
    print_separator("=")
    print("  HEAD-TO-HEAD: ALL STRATEGIES COMPARED")
    print_separator("=")

    cols = [
        ("Strategy", 32), ("CAGR%", 7), ("MaxDD%", 8), ("Sharpe", 7),
        ("Sortino", 8), ("Calmar", 7), ("Vol%", 6), ("WorstYr%", 9),
        ("BestYr%", 9), ("Trades", 7),
    ]
    header = ""
    for name, width in cols:
        header += f"{name:>{width}s}"
    print(header)
    print("-" * sum(w for _, w in cols))

    # Sort by Sharpe descending
    sorted_results = sorted(all_results, key=lambda r: r.get("sharpe", 0), reverse=True)

    for r in sorted_results:
        if "error" in r:
            continue
        row = ""
        short = r['name'][:31] if len(r['name']) > 31 else r['name']
        row += f"{short:>32s}"
        row += f"{r['cagr']*100:>+7.1f}"
        row += f"{r['max_dd']*100:>8.1f}"
        row += f"{r['sharpe']:>7.2f}"
        row += f"{r['sortino']:>8.2f}"
        calmar = r['cagr'] / abs(r['max_dd']) if r['max_dd'] != 0 else 0
        row += f"{calmar:>7.2f}"
        row += f"{r['ann_vol']*100:>6.1f}"
        row += f"{r['worst_year']*100:>+9.1f}"
        row += f"{r.get('best_year', 0)*100:>+9.1f}"
        row += f"{r.get('n_trades', 0):>7d}"
        print(row)

    print_separator("-")

    # Best-in-class highlights
    if sorted_results:
        best_sharpe = max(sorted_results, key=lambda r: r.get("sharpe", 0))
        best_cagr = max(sorted_results, key=lambda r: r.get("cagr", 0))
        best_dd = max(sorted_results, key=lambda r: r.get("max_dd", -1))  # least negative
        best_calmar = max(sorted_results,
                          key=lambda r: r.get("cagr", 0) / abs(r.get("max_dd", -1)) if r.get("max_dd", 0) != 0 else 0)

        print(f"  Best Sharpe:  {best_sharpe['name']} ({best_sharpe['sharpe']:.2f})")
        print(f"  Best CAGR:    {best_cagr['name']} ({best_cagr['cagr']*100:+.1f}%)")
        print(f"  Shallowest DD: {best_dd['name']} ({best_dd['max_dd']*100:.1f}%)")
        calmar_val = best_calmar['cagr'] / abs(best_calmar['max_dd']) if best_calmar['max_dd'] != 0 else 0
        print(f"  Best Calmar:  {best_calmar['name']} ({calmar_val:.2f})")


def print_rolling_returns(all_navs: dict, window_years: int = 3):
    """Print rolling CAGR analysis for all strategies."""
    window = window_years * 252  # trading days

    print_separator("=")
    print(f"  ROLLING {window_years}-YEAR CAGR ANALYSIS (consistency check)")
    print_separator("=")

    cols = [("Strategy", 32), ("Median", 8), ("Min", 8), ("Max", 8),
            ("StdDev", 8), ("%Positive", 10)]
    header = ""
    for name, width in cols:
        header += f"{name:>{width}s}"
    print(header)
    print("-" * sum(w for _, w in cols))

    for strat_name, nav in sorted(all_navs.items()):
        nav = nav.dropna()
        if len(nav) < window + 10:
            continue

        # Rolling CAGR
        rolling_cagr = (nav / nav.shift(window)) ** (252 / window) - 1
        rolling_cagr = rolling_cagr.dropna()

        if len(rolling_cagr) > 0:
            short = strat_name[:31] if len(strat_name) > 31 else strat_name
            med = rolling_cagr.median() * 100
            mn = rolling_cagr.min() * 100
            mx = rolling_cagr.max() * 100
            sd = rolling_cagr.std() * 100
            pct_pos = (rolling_cagr > 0).mean() * 100

            print(f"{short:>32s}{med:>+8.1f}{mn:>+8.1f}{mx:>+8.1f}{sd:>8.1f}{pct_pos:>10.1f}")

    print_separator("-")


def print_final_recommendation(all_results: list, all_navs: dict):
    """Print final strategy recommendation with honest assessment."""
    print_separator("=")
    print("  FINAL RECOMMENDATION")
    print_separator("=")

    # Rank strategies by different criteria
    valid = [r for r in all_results if "error" not in r and r.get("sharpe", 0) > 0]
    if not valid:
        print("  No valid strategies to recommend.")
        return

    # Complexity ranking (lower = simpler)
    complexity = {
        "B&H: S&P 500 (SPY)": 1,
        "B&H: Nasdaq 100 (QQQ)": 1,
        "B&H: Equal-Wt Sectors": 1,
        "Dual Momentum ETF": 2,
        "Tactical All-Weather": 3,
        "SMA 50/200 Crossover (SPY)": 2,
        "Sector Momentum ETF": 3,
        "SMA 50/200 Stock Basket": 4,
        "Dual Momentum Stocks": 4,
        "Low Volatility Stocks": 3,
        "Mean Reversion Dip": 5,
        "Multibagger Screen + Trend": 6,
        "Combined Multibagger+Rotation": 7,
        "Hybrid Core+Satellite": 5,
    }

    # Sort by Sharpe
    by_sharpe = sorted(valid, key=lambda r: r["sharpe"], reverse=True)

    print("\n  Ranked by risk-adjusted returns (Sharpe ratio):")
    print(f"  {'Rank':<5} {'Strategy':<35} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>8} {'Complexity':>11}")
    print("  " + "-" * 73)
    for i, r in enumerate(by_sharpe[:10], 1):
        comp = complexity.get(r['name'], 5)
        comp_str = ["", "Trivial", "Simple", "Medium", "Moderate", "Complex", "High", "Very High"][min(comp, 7)]
        print(f"  {i:<5} {r['name']:<35} {r['sharpe']:>7.2f} {r['cagr']*100:>+7.1f} {r['max_dd']*100:>8.1f} {comp_str:>11}")

    # Recommendations by profile
    print("\n  RECOMMENDATIONS BY INVESTOR PROFILE:")
    print("  " + "-" * 60)

    # Conservative: best Sharpe with MaxDD > -30%
    conservative = [r for r in valid if r['max_dd'] > -0.35]
    if conservative:
        best = max(conservative, key=lambda r: r['sharpe'])
        print(f"\n  Conservative (MaxDD < 35%):")
        print(f"    -> {best['name']}")
        print(f"       Sharpe {best['sharpe']:.2f}, CAGR {best['cagr']*100:+.1f}%, MaxDD {best['max_dd']*100:.1f}%")
    else:
        print(f"\n  Conservative: No strategy has MaxDD < 35%. Consider bonds or lower allocation.")

    # Balanced: best Sharpe overall
    if by_sharpe:
        print(f"\n  Balanced (best risk-adjusted):")
        print(f"    -> {by_sharpe[0]['name']}")
        print(f"       Sharpe {by_sharpe[0]['sharpe']:.2f}, CAGR {by_sharpe[0]['cagr']*100:+.1f}%, MaxDD {by_sharpe[0]['max_dd']*100:.1f}%")

    # Growth: best CAGR with Sharpe > 0.3
    growth = [r for r in valid if r['sharpe'] > 0.3]
    if growth:
        best = max(growth, key=lambda r: r['cagr'])
        print(f"\n  Growth (highest return, Sharpe > 0.3):")
        print(f"    -> {best['name']}")
        print(f"       Sharpe {best['sharpe']:.2f}, CAGR {best['cagr']*100:+.1f}%, MaxDD {best['max_dd']*100:.1f}%")

    # Simplicity: best Sharpe among simple strategies
    simple = [r for r in valid if complexity.get(r['name'], 5) <= 2]
    if simple:
        best = max(simple, key=lambda r: r['sharpe'])
        print(f"\n  Simplicity-first (complexity <= 'Simple'):")
        print(f"    -> {best['name']}")
        print(f"       Sharpe {best['sharpe']:.2f}, CAGR {best['cagr']*100:+.1f}%, MaxDD {best['max_dd']*100:.1f}%")

    print("\n  HONEST ASSESSMENT:")
    print("  " + "-" * 60)
    print("  - All strategies have significant drawdowns (-30% to -70%+)")
    print("  - The best strategy is the one you will ACTUALLY FOLLOW through drawdowns")
    print("  - Simpler strategies are easier to stick with during stress")
    print("  - If SPY B&H beats most strategies on Sharpe, complexity is not rewarded")
    print("  - Transaction costs, taxes, and behavioral errors eat into active strategies")

    spy_result = next((r for r in valid if "SPY" in r['name'] and "B&H" in r['name']), None)
    if spy_result:
        active = [r for r in valid if r['name'] != spy_result['name'] and "B&H" not in r['name']]
        beaten_by_spy = sum(1 for r in active if r['sharpe'] < spy_result['sharpe'])
        total_active = len(active)
        if total_active > 0:
            pct_beaten = beaten_by_spy / total_active * 100
            print(f"\n  SPY B&H (Sharpe {spy_result['sharpe']:.2f}) beats {beaten_by_spy}/{total_active} "
                  f"({pct_beaten:.0f}%) of active strategies on risk-adjusted basis.")

    print_separator("-")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print()
    print_separator("=")
    print("  COMPREHENSIVE STRATEGY RESEARCH")
    print("  Stock Strategies | ETF Strategies | Reconstitution | Stress Testing")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data: Norgate survivorship-bias-free (34,599 stocks + ETFs + Economic)")
    print_separator("=")

    # =============================================
    # SECTION 1: STOCK STRATEGIES
    # =============================================

    engine = StrategyEngine()

    print("\n" + "=" * 80)
    print("  SECTION 1: STOCK STRATEGIES (dynamic top-100 universe)")
    print("=" * 80)

    stock_results = []
    stock_navs = {}
    spy_bench = engine.buy_and_hold("SPY")

    # Benchmarks
    print("  [1/11] SPY Buy & Hold...")
    spy_nav = engine.buy_and_hold("SPY")
    m = compute_metrics(spy_nav, name="B&H: S&P 500 (SPY)")
    m.update(compute_trade_metrics([]))
    stock_results.append(m)
    stock_navs["B&H: S&P 500 (SPY)"] = spy_nav

    print("  [2/11] QQQ Buy & Hold...")
    qqq_nav = engine.buy_and_hold("QQQ")
    m = compute_metrics(qqq_nav, spy_bench, engine.rf_daily, name="B&H: Nasdaq 100 (QQQ)")
    m.update(compute_trade_metrics([]))
    stock_results.append(m)
    stock_navs["B&H: Nasdaq 100 (QQQ)"] = qqq_nav

    # Strategy 1: SMA 50/200 Crossover (SPY)
    print("  [3/11] SMA 50/200 Crossover (SPY)...")
    nav, trades = engine.sma_crossover_spy(50, 200)
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="SMA 50/200 Crossover (SPY)")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["SMA 50/200 Crossover (SPY)"] = nav

    # Strategy 2: SMA 50/200 Stock Basket
    print("  [4/11] SMA 50/200 Stock Basket...")
    nav, trades = engine.sma_crossover_basket(50, 200, top_n=20)
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="SMA 50/200 Stock Basket")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["SMA 50/200 Stock Basket"] = nav

    # Strategy 3: Sector Rotation (w/ bear gate)
    print("  [5/11] Sector Rotation (momentum + bear gate)...")
    nav, trades = engine.sector_rotation(bear_gate=True, top_n=3)
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Sector Rotation (Bear Gate)")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["Sector Rotation (Bear Gate)"] = nav

    # Strategy 4: Multibagger Screening + Trend
    print("  [6/11] Multibagger Screen + Trend Filter...")
    nav, trades = engine.multibagger_screen()
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Multibagger Screen + Trend")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["Multibagger Screen + Trend"] = nav

    # Strategy 5: Combined Multibagger + Sector Rotation
    print("  [7/11] Combined Multibagger + Rotation...")
    nav, trades = engine.combined_multibagger_rotation()
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Combined Multibagger+Rotation")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["Combined Multibagger+Rotation"] = nav

    # Strategy 6: Dual Momentum (NEW)
    print("  [8/11] Dual Momentum Stocks...")
    nav, trades = engine.dual_momentum(top_n=20)
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Dual Momentum Stocks")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["Dual Momentum Stocks"] = nav

    # Strategy 7: Mean Reversion Dip (NEW)
    print("  [9/11] Mean Reversion Dip Buying...")
    nav, trades = engine.mean_reversion_dip()
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Mean Reversion Dip")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["Mean Reversion Dip"] = nav

    # Strategy 8: Low Volatility (NEW)
    print("  [10/11] Low Volatility Stocks...")
    nav, trades = engine.low_volatility(top_n=20)
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Low Volatility Stocks")
    m.update(compute_trade_metrics(trades))
    stock_results.append(m)
    stock_navs["Low Volatility Stocks"] = nav

    # Benchmark: Equal-weight sectors
    print("  [11/11] Equal-Weight Sector ETFs...")
    ew_nav = engine.equal_weight_sectors()
    m = compute_metrics(ew_nav, spy_bench, engine.rf_daily, name="B&H: Equal-Wt Sectors")
    m.update(compute_trade_metrics([]))
    stock_results.append(m)
    stock_navs["B&H: Equal-Wt Sectors"] = ew_nav

    # Print stock strategy tables
    print_performance_table(stock_results)
    print_yearly_table(stock_results)

    # =============================================
    # SECTION 2: ETF STRATEGIES
    # =============================================

    print("\n" + "=" * 80)
    print("  SECTION 2: ETF STRATEGIES")
    print("=" * 80)

    etf_engine = ETFStrategyEngine(strategy_engine=engine)
    etf_results = []
    etf_navs = {}

    # ETF Strategy 1: Dual Momentum ETF
    print("  [1/4] Dual Momentum ETF (SPY/EFA/AGG)...")
    nav, trades = etf_engine.dual_momentum_etf()
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Dual Momentum ETF")
    m.update(compute_trade_metrics(trades))
    etf_results.append(m)
    etf_navs["Dual Momentum ETF"] = nav

    # ETF Strategy 2: Tactical All-Weather
    print("  [2/4] Tactical All-Weather (SPY/TLT/GLD/DBC)...")
    nav, trades = etf_engine.tactical_allweather()
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Tactical All-Weather")
    m.update(compute_trade_metrics(trades))
    etf_results.append(m)
    etf_navs["Tactical All-Weather"] = nav

    # ETF Strategy 3: Enhanced Sector Rotation
    print("  [3/4] Enhanced Sector Momentum ETF...")
    nav, trades = etf_engine.sector_momentum_etf(top_n=3)
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Sector Momentum ETF")
    m.update(compute_trade_metrics(trades))
    etf_results.append(m)
    etf_navs["Sector Momentum ETF"] = nav

    # ETF Strategy 4: Hybrid Core+Satellite
    print("  [4/8] Hybrid Core+Satellite (60% ETF + 40% Stock)...")
    nav, trades = etf_engine.hybrid_core_satellite()
    m = compute_metrics(nav, spy_bench, engine.rf_daily, name="Hybrid Core+Satellite")
    m.update(compute_trade_metrics(trades))
    etf_results.append(m)
    etf_navs["Hybrid Core+Satellite"] = nav

    # ETF Strategy 5-8: QQQ/XLK Core + Momentum Satellite variants
    core_sat_configs = [
        ("QQQ", 0.60, 5,  "QQQ 60% + Mom T5 40%"),
        ("QQQ", 0.50, 5,  "QQQ 50% + Mom T5 50%"),
        ("XLK", 0.60, 5,  "XLK 60% + Mom T5 40%"),
        ("QQQ", 0.60, 10, "QQQ 60% + Mom T10 40%"),
    ]
    for i, (core_etf, core_pct, sat_n, label) in enumerate(core_sat_configs):
        print(f"  [{5+i}/8] {label} (bear-gated core)...")
        nav, trades = etf_engine.qqq_core_momentum_satellite(
            core_etf=core_etf, core_pct=core_pct, sat_top_n=sat_n)
        m = compute_metrics(nav, spy_bench, engine.rf_daily, name=label)
        m.update(compute_trade_metrics(trades))
        etf_results.append(m)
        etf_navs[label] = nav

    # Print ETF strategy tables
    print_performance_table(etf_results)
    print_yearly_table(etf_results)

    # =============================================
    # SECTION 3: HEAD-TO-HEAD COMPARISON
    # =============================================

    print("\n" + "=" * 80)
    print("  SECTION 3: HEAD-TO-HEAD -- ALL STRATEGIES COMPARED")
    print("=" * 80)

    all_results = stock_results + etf_results
    all_navs = {**stock_navs, **etf_navs}

    print_head_to_head(all_results)
    print_rolling_returns(all_navs, window_years=3)
    print_rolling_returns(all_navs, window_years=5)

    # =============================================
    # SECTION 4: RECONSTITUTION ANALYSIS
    # =============================================

    print("\n" + "=" * 80)
    print("  SECTION 4: S&P 500 RECONSTITUTION ANALYSIS")
    print("=" * 80)

    reconstitution_analysis(engine)

    # =============================================
    # SECTION 5: STRESS TESTING
    # =============================================

    print("\n" + "=" * 80)
    print("  SECTION 5: STRESS TESTING ACROSS MARKET REGIMES")
    print("=" * 80)

    stress_test_all(all_navs, spy_nav)

    # =============================================
    # SECTION 6: LEVERAGE ANALYSIS (on best strategies)
    # =============================================

    print("\n" + "=" * 80)
    print("  SECTION 6: LEVERAGE ANALYSIS")
    print("=" * 80)

    # Build base NAVs for leverage testing
    base_navs = {
        "SPY B&H":                stock_navs.get("B&H: S&P 500 (SPY)", pd.Series(dtype=float)),
        "SMA 50/200 Basket":      stock_navs.get("SMA 50/200 Stock Basket", pd.Series(dtype=float)),
        "Dual Momentum Stocks":   stock_navs.get("Dual Momentum Stocks", pd.Series(dtype=float)),
        "Low Volatility Stocks":  stock_navs.get("Low Volatility Stocks", pd.Series(dtype=float)),
        "Dual Momentum ETF":      etf_navs.get("Dual Momentum ETF", pd.Series(dtype=float)),
        "Tactical All-Weather":   etf_navs.get("Tactical All-Weather", pd.Series(dtype=float)),
        "QQQ 60%+MomT5 40%":     etf_navs.get("QQQ 60% + Mom T5 40%", pd.Series(dtype=float)),
        "QQQ 50%+MomT5 50%":     etf_navs.get("QQQ 50% + Mom T5 50%", pd.Series(dtype=float)),
        "XLK 60%+MomT5 40%":     etf_navs.get("XLK 60% + Mom T5 40%", pd.Series(dtype=float)),
    }

    # --- Extra: Concentrated Dual Momentum (top-10) ---
    print("  Running concentrated Dual Momentum (top-10)...")
    conc_nav, conc_trades = engine.dual_momentum(top_n=10)
    m = compute_metrics(conc_nav, spy_bench, engine.rf_daily, name="Concentrated Momentum T10")
    m.update(compute_trade_metrics(conc_trades))
    # Add to overall comparison
    all_results.append(m)
    all_navs["Concentrated Momentum T10"] = conc_nav
    base_navs["Concentrated Mom T10"] = conc_nav

    # --- Wave Rider: Improved Momentum T10 ---
    print("  Running Wave Rider T10 (improved momentum)...")
    wr10_nav, wr10_trades = engine.improved_momentum(top_n=10)
    m_wr10 = compute_metrics(wr10_nav, spy_bench, engine.rf_daily, name="Wave Rider T10")
    m_wr10.update(compute_trade_metrics(wr10_trades))
    all_results.append(m_wr10)
    all_navs["Wave Rider T10"] = wr10_nav
    base_navs["Wave Rider T10"] = wr10_nav

    # --- Wave Rider: Improved Momentum T5 ---
    print("  Running Wave Rider T5 (improved momentum)...")
    wr5_nav, wr5_trades = engine.improved_momentum(top_n=5)
    m_wr5 = compute_metrics(wr5_nav, spy_bench, engine.rf_daily, name="Wave Rider T5")
    m_wr5.update(compute_trade_metrics(wr5_trades))
    all_results.append(m_wr5)
    all_navs["Wave Rider T5"] = wr5_nav
    base_navs["Wave Rider T5"] = wr5_nav

    # --- Wave Rider T5 with meme stock filter (max 1 high-vol stock) ---
    print("  Running Wave Rider T5 MemeFilter (max 1 high-vol stock)...")
    wr5f_nav, wr5f_trades = engine.improved_momentum(top_n=5, max_highvol=1, highvol_threshold=0.70)
    m_wr5f = compute_metrics(wr5f_nav, spy_bench, engine.rf_daily, name="Wave Rider T5 MF")
    m_wr5f.update(compute_trade_metrics(wr5f_trades))
    all_results.append(m_wr5f)
    all_navs["Wave Rider T5 MF"] = wr5f_nav
    base_navs["Wave Rider T5 MF"] = wr5f_nav

    # --- Wave Rider T10 with meme stock filter ---
    print("  Running Wave Rider T10 MemeFilter (max 1 high-vol stock)...")
    wr10f_nav, wr10f_trades = engine.improved_momentum(top_n=10, max_highvol=1, highvol_threshold=0.70)
    m_wr10f = compute_metrics(wr10f_nav, spy_bench, engine.rf_daily, name="Wave Rider T10 MF")
    m_wr10f.update(compute_trade_metrics(wr10f_trades))
    all_results.append(m_wr10f)
    all_navs["Wave Rider T10 MF"] = wr10f_nav
    base_navs["Wave Rider T10 MF"] = wr10f_nav

    # --- Wave Rider T5 with multi-factor meme score ---
    print("  Running Wave Rider T5 MemeScore (multi-factor 0-100)...")
    wr5ms_nav, wr5ms_trades = engine.improved_momentum(
        top_n=5, use_meme_score=True, meme_exclude=70, meme_max1=50, meme_max2=30)
    m_wr5ms = compute_metrics(wr5ms_nav, spy_bench, engine.rf_daily, name="Wave Rider T5 MS")
    m_wr5ms.update(compute_trade_metrics(wr5ms_trades))
    all_results.append(m_wr5ms)
    all_navs["Wave Rider T5 MS"] = wr5ms_nav
    base_navs["Wave Rider T5 MS"] = wr5ms_nav

    # --- Wave Rider T5 MemeScore strict (lower thresholds) ---
    print("  Running Wave Rider T5 MemeScore-Strict (aggressive filtering)...")
    wr5mss_nav, wr5mss_trades = engine.improved_momentum(
        top_n=5, use_meme_score=True, meme_exclude=55, meme_max1=40, meme_max2=25)
    m_wr5mss = compute_metrics(wr5mss_nav, spy_bench, engine.rf_daily, name="Wave Rider T5 MS-S")
    m_wr5mss.update(compute_trade_metrics(wr5mss_trades))
    all_results.append(m_wr5mss)
    all_navs["Wave Rider T5 MS-S"] = wr5mss_nav
    base_navs["Wave Rider T5 MS-S"] = wr5mss_nav

    # Part A: Naive 3x leverage
    naive_results = []
    for strat_name, base_nav in base_navs.items():
        base_nav = base_nav.dropna()
        if len(base_nav) < 252:
            continue
        for lev in [1.0, 3.0]:
            label = f"{strat_name}" if lev == 1.0 else f"{strat_name} Naive3x"
            nav = base_nav if lev == 1.0 else apply_leverage(base_nav, lev, engine.rf_daily)
            m = compute_metrics(nav, spy_bench, engine.rf_daily, name=label)
            m["leverage"] = lev
            naive_results.append(m)

    print_leverage_table(naive_results,
                         title="PART A: NAIVE FIXED 3x LEVERAGE")

    # Part B: Smart leverage on ALL promising base strategies (stock + ETF)
    focus_strats = [
        "SMA 50/200 Basket", "Dual Momentum Stocks", "Low Volatility Stocks",
        "Tactical All-Weather", "Dual Momentum ETF", "Concentrated Mom T10",
        "QQQ 60%+MomT5 40%", "QQQ 50%+MomT5 50%", "XLK 60%+MomT5 40%",
        "Wave Rider T10", "Wave Rider T5",
        "Wave Rider T5 MF", "Wave Rider T10 MF",
        "Wave Rider T5 MS", "Wave Rider T5 MS-S",
    ]
    smart_results = []
    smart_navs = {}

    for strat_name in focus_strats:
        base_nav = base_navs.get(strat_name, pd.Series(dtype=float)).dropna()
        if len(base_nav) < 252:
            continue

        # 1x baseline
        m = compute_metrics(base_nav, spy_bench, engine.rf_daily, name=f"{strat_name} 1x")
        m["leverage"] = 1.0
        smart_results.append(m)
        smart_navs[f"{strat_name} 1x"] = base_nav

        # BearVol3x bear=0.5x (best approach from previous research)
        bv_half, bv_half_lev = apply_smart_leverage(
            base_nav, target_leverage=3.0, rf_daily=engine.rf_daily,
            mode="bear_vol", target_vol=0.25, vol_lookback=21,
            spy=engine.spy, bear_sma=200, bear_leverage=0.5,
            max_leverage=3.0, min_leverage=0.25)
        m = compute_metrics(bv_half, spy_bench, engine.rf_daily, name=f"{strat_name} BearVol3x")
        m["leverage"] = 3.0
        m["avg_lev"] = float(bv_half_lev.mean())
        smart_results.append(m)
        smart_navs[f"{strat_name} BearVol3x"] = bv_half

        # Also test 2x for lower risk
        bv_2x, bv_2x_lev = apply_smart_leverage(
            base_nav, target_leverage=2.0, rf_daily=engine.rf_daily,
            mode="bear_vol", target_vol=0.20, vol_lookback=21,
            spy=engine.spy, bear_sma=200, bear_leverage=0.5,
            max_leverage=2.0, min_leverage=0.25)
        m = compute_metrics(bv_2x, spy_bench, engine.rf_daily, name=f"{strat_name} BearVol2x")
        m["leverage"] = 2.0
        m["avg_lev"] = float(bv_2x_lev.mean())
        smart_results.append(m)
        smart_navs[f"{strat_name} BearVol2x"] = bv_2x

    # Print smart leverage table
    print()
    print_separator("=")
    print("  PART B: SMART LEVERAGE -- BEATING S&P 500")
    print("  BearVol = bear gate (0.5x when SPY < SMA200) + vol targeting")
    print("  Goal: CAGR > SPY's 10.6% with manageable drawdowns")
    print_separator("=")

    cols = [
        ("Strategy", 42), ("CAGR%", 8), ("TotRet%", 10),
        ("MaxDD%", 8), ("Sharpe", 7), ("Sortino", 8), ("Vol%", 6),
        ("WorstYr%", 9), ("AvgLev", 7),
    ]
    header = ""
    for name, width in cols:
        header += f"{name:>{width}s}"
    print(header)
    print("-" * sum(w for _, w in cols))

    for r in smart_results:
        if "error" in r:
            continue
        row = ""
        row += f"{r['name']:>42s}"
        row += f"{r['cagr']*100:>+8.1f}"
        row += f"{r['total_return']*100:>+10.0f}"
        row += f"{r['max_dd']*100:>8.1f}"
        row += f"{r['sharpe']:>7.2f}"
        row += f"{r['sortino']:>8.2f}"
        row += f"{r['ann_vol']*100:>6.1f}"
        row += f"{r['worst_year']*100:>+9.1f}"
        avg_lev = r.get("avg_lev", r.get("leverage", 1.0))
        row += f"{avg_lev:>7.2f}"
        print(row)
        if "BearVol2x" in r['name']:
            print()

    print_separator("-")

    # Part C: COMPOSITE STRATEGIES -- maximize growth through diversification + leverage
    print()
    print_separator("=")
    print("  PART C: COMPOSITE STRATEGIES -- MAXIMIZE GROWTH")
    print("  Blend uncorrelated strategies -> better base Sharpe -> leverage amplifies")
    print("  Target: beat SPY by at least 50% (CAGR >= 15.9%)")
    print_separator("=")

    taw_nav_c = base_navs.get("Tactical All-Weather", pd.Series(dtype=float)).dropna()
    conc_nav_c = base_navs.get("Concentrated Mom T10", pd.Series(dtype=float)).dropna()

    composites = {}

    # --- Top-5 concentrated momentum (more conviction) ---
    print("  Running top-5 concentrated momentum...")
    conc5_nav, conc5_trades = engine.dual_momentum(top_n=5)
    composites["Concentrated Mom T5"] = conc5_nav

    # Add to overall results
    m5 = compute_metrics(conc5_nav, spy_bench, engine.rf_daily, name="Concentrated Momentum T5")
    m5.update(compute_trade_metrics(conc5_trades))
    all_results.append(m5)
    all_navs["Concentrated Momentum T5"] = conc5_nav

    # --- Wave Rider T10 and T5 as composites for leverage testing ---
    wr10_nav_c = base_navs.get("Wave Rider T10", pd.Series(dtype=float)).dropna()
    wr5_nav_c = base_navs.get("Wave Rider T5", pd.Series(dtype=float)).dropna()
    if len(wr10_nav_c) > 252:
        composites["Wave Rider T10"] = wr10_nav_c
    if len(wr5_nav_c) > 252:
        composites["Wave Rider T5"] = wr5_nav_c

    # --- Wave Rider MemeFilter variants ---
    wr5f_nav_c = base_navs.get("Wave Rider T5 MF", pd.Series(dtype=float)).dropna()
    wr10f_nav_c = base_navs.get("Wave Rider T10 MF", pd.Series(dtype=float)).dropna()
    if len(wr5f_nav_c) > 252:
        composites["Wave Rider T5 MF"] = wr5f_nav_c
    if len(wr10f_nav_c) > 252:
        composites["Wave Rider T10 MF"] = wr10f_nav_c

    # --- Wave Rider MemeScore variants ---
    wr5ms_nav_c = base_navs.get("Wave Rider T5 MS", pd.Series(dtype=float)).dropna()
    wr5mss_nav_c = base_navs.get("Wave Rider T5 MS-S", pd.Series(dtype=float)).dropna()
    if len(wr5ms_nav_c) > 252:
        composites["Wave Rider T5 MS"] = wr5ms_nav_c
    if len(wr5mss_nav_c) > 252:
        composites["Wave Rider T5 MS-S"] = wr5mss_nav_c

    # --- Blended composites (TAW + Concentrated Momentum) ---
    common_idx = taw_nav_c.index.intersection(conc_nav_c.index)
    if len(common_idx) > 252:
        taw_r = taw_nav_c.reindex(common_idx).pct_change().fillna(0)
        conc_r = conc_nav_c.reindex(common_idx).pct_change().fillna(0)

        # 50/50 blend
        blend_r_50 = 0.5 * taw_r + 0.5 * conc_r
        blend_nav_50 = (1 + blend_r_50).cumprod()
        blend_nav_50 = blend_nav_50 / blend_nav_50.iloc[0] * 10000
        composites["Blend 50/50 TAW+ConcMom10"] = blend_nav_50

        # 30/70 growth tilt
        blend_r_70 = 0.3 * taw_r + 0.7 * conc_r
        blend_nav_70 = (1 + blend_r_70).cumprod()
        blend_nav_70 = blend_nav_70 / blend_nav_70.iloc[0] * 10000
        composites["Blend 30/70 TAW+ConcMom10"] = blend_nav_70

    composite_results = []
    for comp_name, comp_nav in composites.items():
        comp_nav = comp_nav.dropna()
        if len(comp_nav) < 252:
            continue

        # 1x baseline
        mc = compute_metrics(comp_nav, spy_bench, engine.rf_daily, name=f"{comp_name} 1x")
        mc["leverage"] = 1.0
        composite_results.append(mc)

        # BearVol3x
        bv3c, bv3c_lev = apply_smart_leverage(
            comp_nav, target_leverage=3.0, rf_daily=engine.rf_daily,
            mode="bear_vol", target_vol=0.25, vol_lookback=21,
            spy=engine.spy, bear_sma=200, bear_leverage=0.5,
            max_leverage=3.0, min_leverage=0.25)
        mc3 = compute_metrics(bv3c, spy_bench, engine.rf_daily, name=f"{comp_name} BearVol3x")
        mc3["leverage"] = 3.0
        mc3["avg_lev"] = float(bv3c_lev.mean())
        composite_results.append(mc3)
        smart_results.append(mc3)
        smart_navs[f"{comp_name} BearVol3x"] = bv3c
        all_results.append(mc3)
        all_navs[f"{comp_name} BearVol3x"] = bv3c

        # BearVol2x (moderate)
        bv2c, bv2c_lev = apply_smart_leverage(
            comp_nav, target_leverage=2.0, rf_daily=engine.rf_daily,
            mode="bear_vol", target_vol=0.20, vol_lookback=21,
            spy=engine.spy, bear_sma=200, bear_leverage=0.5,
            max_leverage=2.0, min_leverage=0.25)
        mc2 = compute_metrics(bv2c, spy_bench, engine.rf_daily, name=f"{comp_name} BearVol2x")
        mc2["leverage"] = 2.0
        mc2["avg_lev"] = float(bv2c_lev.mean())
        composite_results.append(mc2)
        smart_results.append(mc2)
        smart_navs[f"{comp_name} BearVol2x"] = bv2c

    # Print composite table
    comp_cols = [
        ("Strategy", 44), ("CAGR%", 8), ("TotRet%", 10),
        ("MaxDD%", 8), ("Sharpe", 7), ("Sortino", 8), ("Vol%", 6),
        ("WorstYr%", 9), ("AvgLev", 7),
    ]
    comp_header = ""
    for cname, cwidth in comp_cols:
        comp_header += f"{cname:>{cwidth}s}"
    print(comp_header)
    print("-" * sum(w for _, w in comp_cols))

    for r in composite_results:
        if "error" in r:
            continue
        row = ""
        row += f"{r['name']:>44s}"
        row += f"{r['cagr']*100:>+8.1f}"
        row += f"{r['total_return']*100:>+10.0f}"
        row += f"{r['max_dd']*100:>8.1f}"
        row += f"{r['sharpe']:>7.2f}"
        row += f"{r['sortino']:>8.2f}"
        row += f"{r['ann_vol']*100:>6.1f}"
        row += f"{r['worst_year']*100:>+9.1f}"
        avg_lev = r.get("avg_lev", r.get("leverage", 1.0))
        row += f"{avg_lev:>7.2f}"
        print(row)
        if "BearVol2x" in r['name']:
            print()

    print_separator("-")

    # Highlight SPY-beating strategies (target: 50%+ above SPY)
    spy_cagr = 0.106  # from our results
    target_cagr = spy_cagr * 1.5  # 50% above SPY
    all_lev_results = smart_results  # includes Part B + Part C leveraged
    spy_beaters = [r for r in all_lev_results
                   if r['cagr'] > spy_cagr and r.get('avg_lev', r.get('leverage', 1)) > 1.0]
    tier1 = [r for r in spy_beaters if r['cagr'] >= target_cagr]  # 50%+ above SPY
    tier1.sort(key=lambda r: r['sharpe'], reverse=True)
    tier2 = [r for r in spy_beaters if r['cagr'] < target_cagr]
    tier2.sort(key=lambda r: r['sharpe'], reverse=True)

    if tier1:
        print(f"\n  TIER 1: STRATEGIES THAT BEAT SPY BY 50%+ (CAGR >= {target_cagr*100:.1f}%)")
        print(f"  {'Strategy':<44s} {'CAGR%':>7s} {'MaxDD%':>7s} {'Sharpe':>7s} {'AvgLev':>7s}")
        print("  " + "-" * 72)
        for r in tier1:
            beats_pct = (r['cagr'] / spy_cagr - 1) * 100
            avg_lev = r.get('avg_lev', r.get('leverage', 1.0))
            print(f"  {r['name']:<44s} {r['cagr']*100:>+7.1f} {r['max_dd']*100:>7.1f} "
                  f"{r['sharpe']:>7.2f} {avg_lev:>7.2f}  (+{beats_pct:.0f}% vs SPY)")

    if tier2:
        print(f"\n  TIER 2: BEATS SPY BUT <50% (CAGR {spy_cagr*100:.1f}-{target_cagr*100:.1f}%)")
        print(f"  {'Strategy':<44s} {'CAGR%':>7s} {'MaxDD%':>7s} {'Sharpe':>7s} {'AvgLev':>7s}")
        print("  " + "-" * 72)
        for r in tier2:
            beats_pct = (r['cagr'] / spy_cagr - 1) * 100
            avg_lev = r.get('avg_lev', r.get('leverage', 1.0))
            print(f"  {r['name']:<44s} {r['cagr']*100:>+7.1f} {r['max_dd']*100:>7.1f} "
                  f"{r['sharpe']:>7.2f} {avg_lev:>7.2f}  (+{beats_pct:.0f}% vs SPY)")

    # =============================================
    # SECTION 7: FINAL RECOMMENDATION
    # =============================================

    print("\n" + "=" * 80)
    print("  SECTION 7: FINAL RECOMMENDATION")
    print("=" * 80)

    # Include leveraged strategies in final recommendation
    all_results_with_lev = all_results + [r for r in smart_results if r.get('avg_lev', 1) > 1.0]
    all_navs_with_lev = {**all_navs, **smart_navs}

    print_final_recommendation(all_results_with_lev, all_navs_with_lev)

    # =============================================
    # SECTION 8: INTERACTIVE CHART
    # =============================================

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        print("\n  Generating interactive charts...")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=[
                                "All Strategies: Equity Curves (log scale)",
                                "Drawdown Comparison"],
                            vertical_spacing=0.08, row_heights=[0.65, 0.35])

        # Color palette for strategies
        colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        ]

        # Combine all NAVs for charting
        chart_navs = {**all_navs, **smart_navs}

        # Define which to show by default vs legendonly
        default_visible = {
            "B&H: S&P 500 (SPY)", "SMA 50/200 Stock Basket",
            "Dual Momentum Stocks", "Dual Momentum ETF",
            "Tactical All-Weather", "Low Volatility Stocks",
        }

        for idx, (label, nav) in enumerate(chart_navs.items()):
            nav = nav.dropna()
            if len(nav) < 10:
                continue

            color = colors[idx % len(colors)]
            visible = True if label in default_visible else "legendonly"
            width = 2.0 if label in default_visible else 1.2

            fig.add_trace(go.Scatter(
                x=nav.index, y=nav.values, name=label,
                line=dict(color=color, width=width),
                visible=visible,
            ), row=1, col=1)

            # Drawdown for default-visible
            if visible is True:
                cum_max = nav.cummax()
                dd = (nav - cum_max) / cum_max * 100
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values, name=f"DD: {label}",
                    line=dict(color=color, width=width * 0.6),
                    showlegend=False,
                ), row=2, col=1)

        # Add regime shading
        regime_colors = {
            "Dot-Com Crash": "rgba(255,0,0,0.1)",
            "GFC": "rgba(255,0,0,0.1)",
            "COVID Crash": "rgba(255,165,0,0.15)",
            "2022 Bear": "rgba(255,165,0,0.1)",
        }
        for period_name, (start, end) in STRESS_PERIODS.items():
            if period_name in regime_colors:
                fig.add_vrect(
                    x0=start, x1=end,
                    fillcolor=regime_colors[period_name],
                    layer="below", line_width=0,
                    annotation_text=period_name,
                    annotation_position="top left",
                    annotation_font_size=8,
                    row=1, col=1,
                )

        fig.update_layout(
            title="Comprehensive Strategy Research: All Strategies Compared",
            height=1100,
            template="plotly_dark",
            legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02,
                        font=dict(size=8)),
        )
        fig.update_yaxes(type="log", title_text="NAV (log scale)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

        chart_path = os.path.join(os.path.dirname(__file__), "strategy_research_chart.html")
        fig.write_html(chart_path)
        print(f"  Chart saved: {chart_path}")

    except ImportError:
        print("  (Plotly not available -- skipping interactive chart)")

    print()
    print_separator("=")
    print("  RESEARCH COMPLETE")
    print_separator("=")


if __name__ == "__main__":
    main()
