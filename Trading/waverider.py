"""
Wave Rider T5 MS BearVol2x — Production Strategy Module.

Cross-sectional momentum rotation: rank top-100 US equities by multi-timeframe
Carhart momentum (risk-adjusted, SMA200 filtered), select top-5 with hysteresis,
apply 6-factor meme score graduated filtering, overlay BearVol2x leverage
(vol-target 20%, max 2x, bear gate 0.5x when SPY < SMA200).

Verified performance (1991-2026): CAGR +33.0%, Sharpe 1.01, MaxDD -59.6%.

This module is the single source of truth. All other scripts import from here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths — Norgate data + cached universe files
# ---------------------------------------------------------------------------
NORGATE_EQ = r"D:\Experiments\norgate_data\US_Equities"
NORGATE_ECON = r"D:\Experiments\norgate_data\Economic"
CACHE_DIR = r"D:\Experiments\Trading\data_cache"

RISK_FREE_TICKER = "%3MTCM"  # 3-month T-bill (annualized %)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class WaveRiderConfig:
    """All tunable parameters for WR T5 MS BearVol2x."""

    # --- Portfolio construction ---
    top_n: int = 5
    exit_band_mult: float = 2.5
    transaction_cost_bps: int = 10
    rebalance_freq: int = 21  # trading days

    # --- Momentum signal ---
    weight_12m: float = 0.40
    weight_6m: float = 0.35
    weight_3m: float = 0.25
    vol_adj_floor: float = 0.05

    # --- 6-factor meme score thresholds ---
    meme_exclude: int = 70
    meme_max1: int = 50
    meme_max2: int = 30

    # --- BearVol2x leverage ---
    target_vol: float = 0.20
    target_leverage: float = 2.0
    bear_leverage: float = 0.5
    min_leverage: float = 0.25
    max_leverage: float = 2.0
    bear_sma: int = 200
    vol_lookback: int = 21
    margin_spread_bps: int = 50

    # --- Universe ---
    universe_top_n: int = 100

    # --- Deduplication (same company, multiple share classes) ---
    dedup_map: Dict[str, str] = field(default_factory=lambda: {
        "GOOG": "GOOGL",   # Alphabet Class C -> Class A (keep GOOGL)
    })

    @property
    def exit_band(self) -> int:
        return int(self.top_n * self.exit_band_mult)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Full backtest output."""
    nav_leveraged: pd.Series
    nav_unlevered: pd.Series
    leverage_series: pd.Series
    holdings_log: Dict[pd.Timestamp, List[str]]
    filtered_log: Dict[pd.Timestamp, List[str]]
    trades_log: Dict[pd.Timestamp, int]
    rebalance_dates: List[pd.Timestamp]
    dates: pd.DatetimeIndex
    spy_price: pd.Series
    composite: pd.DataFrame
    meme_scores: pd.DataFrame


@dataclass
class PortfolioSignal:
    """Current portfolio recommendation for live trading."""
    date: pd.Timestamp
    holdings: List[str]          # UIDs
    holdings_clean: List[str]    # base symbols
    weights: Dict[str, float]
    meme_scores: Dict[str, float]
    leverage: float
    bear_regime: bool
    realized_vol: float
    nav_lev: float
    nav_unlev: float
    buys: List[str]
    sells: List[str]
    candidates: List[Dict]       # top-20 ranked candidates


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached universe prices and rankings."""
    prices = pd.read_parquet(os.path.join(CACHE_DIR, "universe_prices_top150.parquet"))
    rankings = pd.read_parquet(os.path.join(CACHE_DIR, "universe_rankings_top150.parquet"))
    return prices, rankings


def load_spy() -> pd.Series:
    """Load SPY price series from Norgate."""
    path = os.path.join(NORGATE_EQ, "SPY.parquet")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].sort_index()


def load_risk_free(dates: pd.DatetimeIndex) -> pd.Series:
    """Load 3-month T-bill daily rate, aligned to dates."""
    path = os.path.join(NORGATE_ECON, f"{RISK_FREE_TICKER}.parquet")
    if not os.path.exists(path):
        return pd.Series(0.0, index=dates)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    col = "Close" if "Close" in df.columns else df.columns[0]
    rf_annual = df[col].sort_index()
    return (rf_annual / 100 / 252).reindex(dates).ffill().fillna(0)


def clean_uid(uid: str) -> str:
    """Convert UID like 'DELL-200308' to 'DELL*', or leave 'AAPL' as is."""
    m = re.match(r"^(.+)-(\d{6})$", uid)
    return m.group(1) + "*" if m else uid


# ---------------------------------------------------------------------------
# Core Strategy
# ---------------------------------------------------------------------------
class WaveRiderStrategy:
    """
    Wave Rider T5 MS BearVol2x.

    Multi-timeframe Carhart momentum, risk-adjusted by volatility, SMA200
    trend filter, hysteresis band, 6-factor meme score graduated filtering,
    BearVol2x leverage overlay.
    """

    def __init__(self, config: Optional[WaveRiderConfig] = None):
        self.cfg = config or WaveRiderConfig()

    # ------------------------------------------------------------------
    # Signal computation (vectorized)
    # ------------------------------------------------------------------
    def compute_signals(
        self, prices: pd.DataFrame, membership: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute momentum composite score for all stocks on all dates.

        Returns:
            (composite, daily_rets, vol_63d)
            composite: risk-adjusted blended momentum, NaN where ineligible
        """
        daily_rets = prices.pct_change()
        c = self.cfg

        mom_12m = prices.pct_change(252).shift(1)
        mom_6m = prices.pct_change(126).shift(1)
        mom_3m = prices.pct_change(63).shift(1)
        mom_1m = prices.pct_change(21).shift(1)

        carhart_12 = mom_12m - mom_1m
        carhart_6 = mom_6m - mom_1m
        carhart_3 = mom_3m - mom_1m
        blended = c.weight_12m * carhart_12 + c.weight_6m * carhart_6 + c.weight_3m * carhart_3

        vol_63d = daily_rets.rolling(63).std().shift(1) * np.sqrt(252)
        risk_adj = blended / (vol_63d + c.vol_adj_floor)

        sma200 = prices.rolling(200).mean()
        trend_ok = (prices > sma200).shift(1).fillna(False)

        composite = risk_adj.copy()
        composite[~trend_ok] = np.nan
        composite[membership == 0] = np.nan

        return composite, daily_rets, vol_63d

    # ------------------------------------------------------------------
    # Meme score (vectorized)
    # ------------------------------------------------------------------
    def compute_meme_scores(
        self,
        prices: pd.DataFrame,
        daily_rets: pd.DataFrame,
        vol_63d: pd.DataFrame,
        rankings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        6-factor meme score (0-115 raw points).

        Factors:
            1. Volatility level       (0-25)
            2. Parabolic 3m move      (0-25)
            3. Price-SMA200 stretch   (0-20)
            4. Momentum concentration (0-15)
            5. Volatility acceleration(0-15)
            6. Universe tenure        (0-15)
        """
        idx, cols = prices.index, prices.columns
        c = self.cfg

        # Factor 1: Volatility level
        f1 = pd.DataFrame(0.0, index=idx, columns=cols)
        f1[vol_63d > 0.40] = 8
        f1[vol_63d > 0.60] = 15
        f1[vol_63d > 0.80] = 20
        f1[vol_63d > 1.00] = 25

        # Factor 2: Parabolic 3m move
        ret_3m_abs = prices.pct_change(63).shift(1).abs()
        f2 = pd.DataFrame(0.0, index=idx, columns=cols)
        f2[ret_3m_abs > 0.50] = 8
        f2[ret_3m_abs > 1.00] = 18
        f2[ret_3m_abs > 2.00] = 25

        # Factor 3: Price-SMA200 stretch
        sma200 = prices.rolling(200).mean()
        stretch = (prices / sma200).shift(1)
        f3 = pd.DataFrame(0.0, index=idx, columns=cols)
        f3[stretch > 1.3] = 8
        f3[stretch > 2.0] = 15
        f3[stretch > 3.0] = 20

        # Factor 4: Momentum concentration
        mom_1m = prices.pct_change(21).shift(1)
        mom_12m = prices.pct_change(252).shift(1)
        both_pos = (mom_1m > 0) & (mom_12m > 0.01)
        conc = pd.DataFrame(0.0, index=idx, columns=cols)
        conc[both_pos] = (mom_1m[both_pos] / mom_12m[both_pos]).clip(0, 2)
        f4 = pd.DataFrame(0.0, index=idx, columns=cols)
        f4[conc > 0.3] = 5
        f4[conc > 0.5] = 10
        f4[conc > 0.8] = 15

        # Factor 5: Volatility acceleration
        vol_21d = daily_rets.rolling(21).std().shift(1) * np.sqrt(252)
        vol_126d = daily_rets.rolling(126).std().shift(1) * np.sqrt(252)
        vol_accel = vol_21d / (vol_126d + 0.01)
        f5 = pd.DataFrame(0.0, index=idx, columns=cols)
        f5[vol_accel > 1.0] = 5
        f5[vol_accel > 1.5] = 10
        f5[vol_accel > 2.5] = 15

        # Factor 6: Universe tenure (ETF/index membership proxy)
        f6 = self._compute_tenure_scores(prices, rankings)

        return f1 + f2 + f3 + f4 + f5 + f6

    def _compute_tenure_scores(
        self, prices: pd.DataFrame, rankings: pd.DataFrame
    ) -> pd.DataFrame:
        """Factor 6: penalize stocks newly entering top-100."""
        c = self.cfg
        f6 = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        top100 = rankings[rankings["rank"] <= c.universe_top_n]
        rank_dates = sorted(rankings["date"].unique())

        tenure: Dict[str, int] = {}
        tenure_snapshots: Dict = {}
        for rdate in rank_dates:
            current_top = set(top100[top100["date"] == rdate]["uid"].values)
            new_tenure = {uid: tenure.get(uid, 0) + 1 for uid in current_top}
            tenure = new_tenure
            tenure_snapshots[rdate] = dict(tenure)

        col_set = set(prices.columns)
        for i, rdate in enumerate(rank_dates):
            snap = tenure_snapshots[rdate]
            if i + 1 < len(rank_dates):
                end = rank_dates[i + 1]
                mask = (prices.index >= rdate) & (prices.index < end)
            else:
                mask = prices.index >= rdate
            for uid, months in snap.items():
                if uid in col_set:
                    if months <= 2:
                        f6.loc[mask, uid] = 15
                    elif months <= 5:
                        f6.loc[mask, uid] = 8
        return f6

    # ------------------------------------------------------------------
    # Membership mask
    # ------------------------------------------------------------------
    def build_membership(
        self, prices: pd.DataFrame, rankings: pd.DataFrame
    ) -> pd.DataFrame:
        """Point-in-time top-N universe membership mask (1/0)."""
        c = self.cfg
        mask = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        rank_dates = sorted(rankings["date"].unique())
        col_set = set(prices.columns)

        for i, rdate in enumerate(rank_dates):
            top = set(
                rankings[
                    (rankings["date"] == rdate) & (rankings["rank"] <= c.universe_top_n)
                ]["uid"].values
            ) & col_set
            if i + 1 < len(rank_dates):
                period = (mask.index >= rdate) & (mask.index < rank_dates[i + 1])
            else:
                period = mask.index >= rdate
            for uid in top:
                mask.loc[period, uid] = 1
        return mask

    # ------------------------------------------------------------------
    # Portfolio selection (single rebalance date)
    # ------------------------------------------------------------------
    def select_portfolio(
        self,
        date: pd.Timestamp,
        composite: pd.DataFrame,
        meme_scores: pd.DataFrame,
        current_holdings: set,
    ) -> Tuple[set, List[str]]:
        """
        Select top-N stocks with hysteresis + graduated meme filtering.

        Returns:
            (portfolio_set, filtered_out_list)
        """
        c = self.cfg
        scores = composite.loc[date].dropna().sort_values(ascending=False)
        if len(scores) == 0:
            return set(), []

        top_fresh = set(scores.head(c.top_n).index)
        top_band = set(scores.head(c.exit_band).index)

        retained = current_holdings & top_band
        remaining = c.top_n - len(retained)
        new_entries = [s for s in scores.index if s in top_fresh and s not in retained]
        portfolio = retained | set(new_entries[: max(remaining, 0)])

        if len(portfolio) > c.top_n:
            scored = sorted(
                [(s, scores.get(s, -999)) for s in portfolio], key=lambda x: -x[1]
            )
            portfolio = set(s for s, _ in scored[: c.top_n])

        # --- Graduated meme filtering ---
        ms_today = meme_scores.loc[date]

        def _ms(s: str) -> float:
            v = ms_today.get(s, 0)
            return v if pd.notna(v) else 0

        before = set(portfolio)

        # Tier 1: exclude pure meme
        portfolio -= {s for s in portfolio if _ms(s) > c.meme_exclude}

        # Tier 2: max 1 meme-adjacent
        meme_adj = [s for s in portfolio if _ms(s) > c.meme_max1]
        if len(meme_adj) > 1:
            ma_sorted = sorted(meme_adj, key=lambda s: scores.get(s, -999), reverse=True)
            portfolio -= set(ma_sorted[1:])

        # Tier 3: max 2 gray-zone
        gray = [s for s in portfolio if _ms(s) > c.meme_max2]
        if len(gray) > 2:
            gz_sorted = sorted(gray, key=lambda s: scores.get(s, -999), reverse=True)
            portfolio -= set(gz_sorted[2:])

        # --- Deduplication (same company, multiple share classes) ---
        if c.dedup_map:
            # Build canonical symbol for each UID in portfolio
            def _canonical(uid: str) -> str:
                base = clean_uid(uid).rstrip("*")
                return c.dedup_map.get(base, base)

            from collections import defaultdict
            groups = defaultdict(list)
            for s in portfolio:
                groups[_canonical(s)].append(s)
            for canon, members in groups.items():
                if len(members) > 1:
                    # Keep highest-scored, remove rest
                    members.sort(key=lambda s: scores.get(s, -999), reverse=True)
                    portfolio -= set(members[1:])

        filtered_out = list(before - portfolio)

        # Refill with clean stocks (skip dedup variants already held)
        held_canonicals = set()
        if c.dedup_map:
            held_canonicals = {_canonical(s) for s in portfolio}
        for s in scores.index:
            if len(portfolio) >= c.top_n:
                break
            if s in portfolio:
                continue
            if _ms(s) > c.meme_max2:
                continue
            if c.dedup_map:
                if _canonical(s) in held_canonicals:
                    continue
            portfolio.add(s)
            if c.dedup_map:
                held_canonicals.add(_canonical(s))

        return portfolio, filtered_out

    # ------------------------------------------------------------------
    # Full backtest
    # ------------------------------------------------------------------
    def backtest(
        self,
        prices: pd.DataFrame,
        spy_price: pd.Series,
        rankings: pd.DataFrame,
        rf_daily: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run full historical backtest from first available date."""
        c = self.cfg

        membership = self.build_membership(prices, rankings)
        composite, daily_rets, vol_63d = self.compute_signals(prices, membership)
        meme_scores = self.compute_meme_scores(prices, daily_rets, vol_63d, rankings)

        warmup = prices.index[max(252, 200) + 5]
        dates = prices.loc[warmup:].index

        # --- Unlevered backtest ---
        nav_val = 1.0
        nav_vals: List[float] = []
        current_holdings: set = set()
        current_weights = pd.Series(0.0, index=prices.columns)

        rebal_dates: List[pd.Timestamp] = []
        holdings_log: Dict[pd.Timestamp, List[str]] = {}
        filtered_log: Dict[pd.Timestamp, List[str]] = {}
        trades_log: Dict[pd.Timestamp, int] = {}

        for i, date in enumerate(dates):
            if i == 0:
                nav_vals.append(1.0)
                continue

            if i % c.rebalance_freq == 0:
                portfolio, filtered_out = self.select_portfolio(
                    date, composite, meme_scores, current_holdings
                )
                buys = portfolio - current_holdings
                sells = current_holdings - portfolio

                current_holdings = portfolio
                new_weights = pd.Series(0.0, index=prices.columns)
                if len(portfolio) > 0:
                    w = 1.0 / len(portfolio)
                    for sym in portfolio:
                        new_weights[sym] = w
                current_weights = new_weights

                rebal_dates.append(date)
                holdings_log[date] = sorted(portfolio)
                filtered_log[date] = filtered_out
                trades_log[date] = len(buys) + len(sells)

            day_ret = (current_weights * daily_rets.loc[date]).sum()
            if i % c.rebalance_freq == 0:
                day_ret -= c.transaction_cost_bps / 10000
            nav_val *= 1 + day_ret
            nav_vals.append(nav_val)

        nav_unlevered = pd.Series(nav_vals, index=dates)

        # --- BearVol2x overlay ---
        nav_leveraged, lev_series = self._apply_bearvol2x(
            nav_unlevered, spy_price, dates, rf_daily
        )

        return BacktestResult(
            nav_leveraged=nav_leveraged,
            nav_unlevered=nav_unlevered,
            leverage_series=lev_series,
            holdings_log=holdings_log,
            filtered_log=filtered_log,
            trades_log=trades_log,
            rebalance_dates=rebal_dates,
            dates=dates,
            spy_price=spy_price,
            composite=composite,
            meme_scores=meme_scores,
        )

    # ------------------------------------------------------------------
    # BearVol2x leverage overlay
    # ------------------------------------------------------------------
    def _apply_bearvol2x(
        self,
        nav_unlevered: pd.Series,
        spy_price: pd.Series,
        dates: pd.DatetimeIndex,
        rf_daily: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Apply vol-targeting + bear gate leverage overlay."""
        c = self.cfg

        spy_aligned = spy_price.reindex(dates).ffill()
        spy_sma = spy_aligned.rolling(c.bear_sma).mean()
        bear_signal = (spy_aligned > spy_sma).shift(1).fillna(True)

        unlev_rets = nav_unlevered.pct_change().fillna(0)
        realized_vol = unlev_rets.rolling(c.vol_lookback, min_periods=5).std() * np.sqrt(252)
        realized_vol = realized_vol.fillna(c.target_vol)

        if rf_daily is None:
            rf_daily = load_risk_free(dates)
        else:
            rf_daily = rf_daily.reindex(dates).ffill().fillna(0)

        borrow_rate = rf_daily + c.margin_spread_bps / 10000 / 252

        n = len(dates)
        nav_lev = np.ones(n, dtype=np.float64)
        lev_arr = np.full(n, c.target_leverage, dtype=np.float64)

        for i in range(1, n):
            rv = realized_vol.iloc[i]
            vol_lev = (c.target_vol / rv * c.target_leverage) if rv > 0.01 else c.target_leverage
            lev = min(c.target_leverage, vol_lev)

            if not bear_signal.iloc[i]:
                lev = min(lev, c.bear_leverage)

            lev = max(c.min_leverage, min(lev, c.max_leverage))
            lev_arr[i] = lev

            borrow = max(lev - 1, 0) * borrow_rate.iloc[i]
            day_ret = lev * unlev_rets.iloc[i] - borrow
            nav_lev[i] = nav_lev[i - 1] * (1 + day_ret)

        return pd.Series(nav_lev, index=dates), pd.Series(lev_arr, index=dates)

    # ------------------------------------------------------------------
    # Current portfolio signal (for live trading)
    # ------------------------------------------------------------------
    def current_portfolio(
        self,
        prices: pd.DataFrame,
        spy_price: pd.Series,
        rankings: pd.DataFrame,
        rf_daily: Optional[pd.Series] = None,
    ) -> PortfolioSignal:
        """
        Run backtest through today and return current state.

        Must replay history because hysteresis depends on prior holdings.
        """
        result = self.backtest(prices, spy_price, rankings, rf_daily)
        c = self.cfg

        if len(result.rebalance_dates) == 0:
            raise ValueError("No rebalance dates in backtest — check data range")

        last_rebal = result.rebalance_dates[-1]
        holdings = result.holdings_log[last_rebal]
        holdings_clean = [clean_uid(uid) for uid in holdings]

        # Meme scores for current holdings
        ms_today = result.meme_scores.loc[last_rebal]
        ms_dict = {}
        for uid in holdings:
            v = ms_today.get(uid, 0)
            ms_dict[clean_uid(uid)] = float(v) if pd.notna(v) else 0.0

        # Current leverage
        lev_today = float(result.leverage_series.iloc[-1])

        # Bear regime
        spy_aligned = spy_price.reindex(result.dates).ffill()
        spy_sma = spy_aligned.rolling(c.bear_sma).mean()
        bear_regime = bool(spy_aligned.iloc[-1] < spy_sma.iloc[-1])

        # Realized vol
        unlev_rets = result.nav_unlevered.pct_change().fillna(0)
        rv = unlev_rets.iloc[-c.vol_lookback:].std() * np.sqrt(252)

        # Buys/sells vs previous rebalance
        if len(result.rebalance_dates) >= 2:
            prev_rebal = result.rebalance_dates[-2]
            prev_holdings = set(result.holdings_log[prev_rebal])
        else:
            prev_holdings = set()
        current_set = set(holdings)
        buys = [clean_uid(s) for s in sorted(current_set - prev_holdings)]
        sells = [clean_uid(s) for s in sorted(prev_holdings - current_set)]

        # Top-20 candidates
        scores = result.composite.loc[last_rebal].dropna().sort_values(ascending=False)
        candidates = []
        for uid in scores.head(20).index:
            sc = float(scores[uid])
            ms = float(ms_today.get(uid, 0)) if pd.notna(ms_today.get(uid, 0)) else 0.0
            if ms <= c.meme_max2:
                classification = "Clean Growth"
            elif ms <= c.meme_max1:
                classification = "Gray Zone"
            elif ms <= c.meme_exclude:
                classification = "Meme-Adjacent"
            else:
                classification = "Pure Meme"
            candidates.append({
                "uid": uid,
                "symbol": clean_uid(uid),
                "composite": sc,
                "meme_score": ms,
                "classification": classification,
                "in_portfolio": uid in current_set,
            })

        return PortfolioSignal(
            date=last_rebal,
            holdings=holdings,
            holdings_clean=holdings_clean,
            weights={clean_uid(uid): 1.0 / len(holdings) for uid in holdings} if holdings else {},
            meme_scores=ms_dict,
            leverage=lev_today,
            bear_regime=bear_regime,
            realized_vol=float(rv),
            nav_lev=float(result.nav_leveraged.iloc[-1]),
            nav_unlev=float(result.nav_unlevered.iloc[-1]),
            buys=buys,
            sells=sells,
            candidates=candidates,
        )


# ---------------------------------------------------------------------------
# Metrics helpers (for reporting)
# ---------------------------------------------------------------------------
def compute_nav_metrics(nav: pd.Series, label: str = "") -> Dict:
    """Compute CAGR, Sharpe, Sortino, MaxDD from NAV series."""
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    if n_years <= 0 or nav.iloc[0] <= 0:
        return {}
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / n_years) - 1
    daily_r = nav.pct_change().dropna()
    sharpe = (daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
    down = daily_r[daily_r < 0].std() * np.sqrt(252)
    sortino = (daily_r.mean() * 252 / down) if down > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    return {
        "label": label,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "total_return": total_ret,
        "n_years": n_years,
    }
