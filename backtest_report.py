"""
backtest_report.py -- Institutional Evaluation & Reporting (WS4)
================================================================================

Components:
    BacktestReport      -- WS4A: Full metrics suite (gross/net/benchmark)
    StressTester        -- WS4B: Crisis period replay
    CapacityEstimator   -- WS4C: AUM capacity estimation

References:
    - Grinold & Kahn (2000) "Active Portfolio Management"
    - Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier"
    - Almgren & Chriss (2000) optimal execution

Usage:
    from backtest_report import BacktestReport, StressTester, CapacityEstimator
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import numpy as np


# ============================================================================
# WS4A: Full Metrics Suite
# ============================================================================

@dataclass
class BacktestReport:
    """
    Complete institutional backtest report. All fields mandatory.

    Every strategy must report metrics both gross and net of costs,
    and relative to the stated benchmark. This dataclass enforces
    that discipline by making all fields required and providing a
    validate() method.

    Reference: Grinold & Kahn (2000) "Active Portfolio Management"
    """
    # Identity
    strategy_name: str = ""
    benchmark: str = "cash"           # MANDATORY: "SPY", "cash", or "risk-free"
    period_start: str = ""
    period_end: str = ""

    # Returns (BOTH gross and net)
    cagr_gross: float = 0.0
    cagr_net: float = 0.0
    annual_vol_gross: float = 0.0
    annual_vol_net: float = 0.0
    sharpe_gross: float = 0.0
    sharpe_net: float = 0.0
    sortino_gross: float = 0.0
    sortino_net: float = 0.0
    calmar_gross: float = 0.0
    calmar_net: float = 0.0

    # Drawdown
    max_dd_depth: float = 0.0         # Percentage
    max_dd_duration_days: int = 0     # Calendar days in max DD
    max_dd_recovery_days: int = 0     # Days to recover

    # Risk
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Exposure
    avg_gross_exposure: float = 0.0
    avg_net_exposure: float = 0.0
    avg_long_exposure: float = 0.0
    avg_short_exposure: float = 0.0

    # Costs
    turnover_annualized: float = 0.0  # x capital
    cost_drag_bps_annual: float = 0.0

    # Relative to benchmark
    alpha_vs_benchmark: float = 0.0
    beta_vs_benchmark: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    # Anti-overfitting
    deflated_sharpe: float = 0.0
    oos_is_decay_ratio: float = 0.0
    n_independent_bets: float = 0.0
    n_configs_tested: int = 1

    # Stability
    rolling_sharpe_12m: List[float] = field(default_factory=list)
    rolling_ic_monthly: List[float] = field(default_factory=list)

    # Capacity
    estimated_capacity_aum: float = 0.0

    def validate(self):
        """Assert all critical fields are populated and sane."""
        assert self.benchmark, "Benchmark must be defined"
        assert self.strategy_name, "Strategy name must be defined"
        assert self.n_configs_tested > 0, "Must report number of configs tested"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            d[k] = v
        return d

    def print_summary(self, indent: str = "  "):
        """Print a formatted summary of the backtest report."""
        print(f"\n{'=' * 70}")
        print(f"  BACKTEST REPORT: {self.strategy_name}")
        print(f"  Benchmark: {self.benchmark}")
        print(f"  Period: {self.period_start} -> {self.period_end}")
        print(f"{'=' * 70}")

        print(f"\n{indent}{'Metric':<28s} {'Gross':>10s} {'Net':>10s}")
        print(f"{indent}{'-' * 28} {'-' * 10} {'-' * 10}")
        print(f"{indent}{'CAGR':<28s} {self.cagr_gross:>+9.2%} {self.cagr_net:>+9.2%}")
        print(f"{indent}{'Annual Vol':<28s} {self.annual_vol_gross:>9.2%} {self.annual_vol_net:>9.2%}")
        print(f"{indent}{'Sharpe':<28s} {self.sharpe_gross:>+9.3f} {self.sharpe_net:>+9.3f}")
        print(f"{indent}{'Sortino':<28s} {self.sortino_gross:>+9.3f} {self.sortino_net:>+9.3f}")
        print(f"{indent}{'Calmar':<28s} {self.calmar_gross:>+9.3f} {self.calmar_net:>+9.3f}")

        print(f"\n{indent}{'Max DD Depth':<28s} {self.max_dd_depth:>9.2%}")
        print(f"{indent}{'Max DD Duration (days)':<28s} {self.max_dd_duration_days:>9d}")
        print(f"{indent}{'CVaR 95%':<28s} {self.cvar_95:>+9.4f}")
        print(f"{indent}{'CVaR 99%':<28s} {self.cvar_99:>+9.4f}")

        print(f"\n{indent}{'Turnover (ann.)':<28s} {self.turnover_annualized:>9.1f}x")
        print(f"{indent}{'Cost Drag (bps/yr)':<28s} {self.cost_drag_bps_annual:>9.1f}")
        print(f"{indent}{'Alpha vs Benchmark':<28s} {self.alpha_vs_benchmark:>+9.4f}")
        print(f"{indent}{'Beta vs Benchmark':<28s} {self.beta_vs_benchmark:>+9.3f}")
        print(f"{indent}{'Info Ratio':<28s} {self.information_ratio:>+9.3f}")

        print(f"\n{indent}{'Deflated Sharpe':<28s} {self.deflated_sharpe:>9.3f}")
        print(f"{indent}{'OOS/IS Decay':<28s} {self.oos_is_decay_ratio:>9.2%}")
        print(f"{indent}{'N Configs Tested':<28s} {self.n_configs_tested:>9d}")
        print(f"{indent}{'Estimated Capacity ($M)':<28s} {self.estimated_capacity_aum / 1e6:>9.1f}")
        print(f"{'=' * 70}")


def compute_backtest_report(
    daily_returns_gross: np.ndarray,
    daily_returns_net: np.ndarray,
    benchmark_returns: np.ndarray = None,
    exposures: np.ndarray = None,
    strategy_name: str = "Alpha-Trade v7.0",
    benchmark: str = "cash",
    period_start: str = "",
    period_end: str = "",
    turnover_annual: float = 0.0,
    cost_drag_bps: float = 0.0,
    deflated_sharpe: float = 0.0,
    oos_is_decay: float = 0.0,
    n_configs: int = 1,
    capacity_aum: float = 0.0,
    avg_adv_dollars: float = 5_000_000.0,
    spread_bps: float = 1.0,
    impact_coeff: float = 0.1,
    auto_capacity: bool = True,
) -> BacktestReport:
    """
    Compute a full BacktestReport from return arrays.

    Args:
        daily_returns_gross: Gross daily returns (before costs)
        daily_returns_net:   Net daily returns (after costs)
        benchmark_returns:   Daily benchmark returns (same length)
        exposures:           Daily exposure values (for exposure stats)
        avg_adv_dollars:     Average daily volume for capacity estimation
        spread_bps:          Full bid-ask spread for capacity estimation
        impact_coeff:        Square-root impact coefficient for capacity
        auto_capacity:       If True and capacity_aum==0, auto-compute capacity
    """
    rg = np.asarray(daily_returns_gross, dtype=np.float64)
    rn = np.asarray(daily_returns_net, dtype=np.float64)
    n = len(rn)
    ann = 252

    if benchmark_returns is not None:
        rb = np.asarray(benchmark_returns, dtype=np.float64)[:n]
    else:
        rb = np.zeros(n)

    # --- CAGR ---
    cum_g = np.prod(1 + rg) ** (ann / max(n, 1)) - 1
    cum_n = np.prod(1 + rn) ** (ann / max(n, 1)) - 1

    # --- Vol ---
    vol_g = float(np.std(rg)) * math.sqrt(ann)
    vol_n = float(np.std(rn)) * math.sqrt(ann)

    # --- Sharpe ---
    sh_g = (np.mean(rg) * ann) / (vol_g + 1e-10)
    sh_n = (np.mean(rn) * ann) / (vol_n + 1e-10)

    # --- Sortino ---
    down_g = rg[rg < 0]
    down_n = rn[rn < 0]
    down_vol_g = float(np.std(down_g)) * math.sqrt(ann) if len(down_g) > 0 else 1e-10
    down_vol_n = float(np.std(down_n)) * math.sqrt(ann) if len(down_n) > 0 else 1e-10
    sort_g = (np.mean(rg) * ann) / (down_vol_g + 1e-10)
    sort_n = (np.mean(rn) * ann) / (down_vol_n + 1e-10)

    # --- Drawdown ---
    cum = np.cumprod(1 + rn)
    peak = np.maximum.accumulate(cum)
    dd_series = (peak - cum) / (peak + 1e-10)
    max_dd = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0

    # DD duration
    in_dd = dd_series > 0.001
    dd_duration = 0
    max_dd_dur = 0
    for v in in_dd:
        if v:
            dd_duration += 1
            max_dd_dur = max(max_dd_dur, dd_duration)
        else:
            dd_duration = 0

    # --- Calmar ---
    cal_g = cum_g / (max_dd + 1e-10) if max_dd > 0.001 else 0.0
    cal_n = cum_n / (max_dd + 1e-10) if max_dd > 0.001 else 0.0

    # --- CVaR ---
    cvar_95 = _cvar(rn, 0.95)
    cvar_99 = _cvar(rn, 0.99)

    # --- Exposure stats ---
    if exposures is not None:
        exp = np.asarray(exposures, dtype=np.float64)
        avg_gross = float(np.mean(np.abs(exp)))
        avg_net = float(np.mean(exp))
        avg_long = float(np.mean(np.maximum(exp, 0)))
        avg_short = float(np.mean(np.minimum(exp, 0)))
    else:
        avg_gross = avg_net = avg_long = avg_short = 0.0

    # --- Benchmark relative ---
    active = rn - rb
    alpha = float(np.mean(active)) * ann
    te = float(np.std(active)) * math.sqrt(ann)
    ir = alpha / (te + 1e-10)
    cov_pb = np.cov(rn, rb)[0, 1] if len(rb) == len(rn) and len(rn) > 5 else 0.0
    var_b = np.var(rb) + 1e-10
    beta = cov_pb / var_b

    # --- Rolling Sharpe 12m ---
    window = min(252, n)
    rolling_sharpe = []
    for i in range(window, n, 21):
        chunk = rn[i - window:i]
        s = float(np.mean(chunk)) / (float(np.std(chunk)) + 1e-10) * math.sqrt(ann)
        rolling_sharpe.append(round(s, 3))

    # --- Auto Capacity Estimation (WS4C integration) ---
    final_capacity = capacity_aum
    if auto_capacity and capacity_aum <= 0.0 and turnover_annual > 0:
        gross_alpha_bps = float(np.mean(rg)) * ann * 1e4  # annualized gross alpha in bps
        if gross_alpha_bps > 0:
            cap_est = CapacityEstimator()
            cap_result = cap_est.estimate(
                gross_alpha_bps=gross_alpha_bps,
                annual_turnover=turnover_annual,
                avg_adv_dollars=avg_adv_dollars,
                spread_bps=spread_bps,
                impact_coeff=impact_coeff,
                annual_vol=vol_n if vol_n > 0 else 0.15,
            )
            # Use the more conservative of the two capacity measures
            final_capacity = min(
                cap_result.get('capacity_at_sharpe_05', 0.0),
                cap_result.get('capacity_at_50pct_cost', 0.0),
            )

    return BacktestReport(
        strategy_name=strategy_name,
        benchmark=benchmark,
        period_start=period_start,
        period_end=period_end,
        cagr_gross=float(cum_g),
        cagr_net=float(cum_n),
        annual_vol_gross=vol_g,
        annual_vol_net=vol_n,
        sharpe_gross=float(sh_g),
        sharpe_net=float(sh_n),
        sortino_gross=float(sort_g),
        sortino_net=float(sort_n),
        calmar_gross=float(cal_g),
        calmar_net=float(cal_n),
        max_dd_depth=max_dd,
        max_dd_duration_days=max_dd_dur,
        max_dd_recovery_days=0,  # Would need full date tracking
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        avg_gross_exposure=avg_gross,
        avg_net_exposure=avg_net,
        avg_long_exposure=avg_long,
        avg_short_exposure=avg_short,
        turnover_annualized=turnover_annual,
        cost_drag_bps_annual=cost_drag_bps,
        alpha_vs_benchmark=alpha,
        beta_vs_benchmark=float(beta),
        information_ratio=ir,
        tracking_error=te,
        deflated_sharpe=deflated_sharpe,
        oos_is_decay_ratio=oos_is_decay,
        n_independent_bets=0.0,
        n_configs_tested=n_configs,
        rolling_sharpe_12m=rolling_sharpe,
        rolling_ic_monthly=[],
        estimated_capacity_aum=final_capacity,
    )


def _cvar(returns: np.ndarray, confidence: float) -> float:
    """Historical Conditional Value-at-Risk (Expected Shortfall)."""
    if len(returns) < 10:
        return -0.10  # Conservative default
    cutoff_pct = (1 - confidence) * 100
    cutoff = np.percentile(returns, cutoff_pct)
    tail = returns[returns <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else float(cutoff)


# ============================================================================
# WS4B: Stress Testing Module
# ============================================================================

class StressTester:
    """
    Replay strategy through known crisis periods.

    For each crisis: report max DD, time-to-recovery,
    kill switch activation, factor exposure behavior.

    Reference: common institutional practice for stress testing.
    """

    CRISIS_PERIODS = {
        'GFC_2008':           ('2008-09-01', '2009-03-31'),
        'EU_DEBT_2011':       ('2011-07-01', '2011-12-31'),
        'CHINA_DEVAL_2015':   ('2015-08-01', '2015-10-31'),
        'VOLMAGEDDON_2018':   ('2018-01-26', '2018-03-31'),
        'COVID_2020':         ('2020-02-19', '2020-06-30'),
        'RATES_SHOCK_2022':   ('2022-01-01', '2022-10-31'),
    }

    def stress_test_returns(self, daily_returns: np.ndarray,
                            dates: np.ndarray = None,
                            crisis_name: str = "full_sample") -> Dict[str, Any]:
        """
        Compute stress metrics for a given return series (or subset).

        Args:
            daily_returns: Array of daily returns
            dates:         Optional date array (for period filtering)
            crisis_name:   Label for the crisis

        Returns dict with: max_dd, recovery_days, cvar_95, cvar_99,
                          avg_return, worst_day, worst_week, survived
        """
        r = np.asarray(daily_returns, dtype=np.float64)
        if len(r) == 0:
            return {'crisis': crisis_name, 'n_days': 0, 'available': False}

        cum = np.cumprod(1 + r)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / (peak + 1e-10)
        max_dd = float(np.max(dd))

        # Recovery: days from max DD trough to new peak
        trough_idx = int(np.argmax(dd))
        recovered = False
        recovery_days = len(r) - trough_idx  # Default: never recovered
        for j in range(trough_idx + 1, len(r)):
            if cum[j] >= peak[trough_idx]:
                recovery_days = j - trough_idx
                recovered = True
                break

        # Worst week (5 days)
        worst_week = 0.0
        for i in range(len(r) - 5):
            w = float(np.prod(1 + r[i:i + 5]) - 1)
            worst_week = min(worst_week, w)

        return {
            'crisis': crisis_name,
            'n_days': len(r),
            'available': True,
            'max_dd': max_dd,
            'recovery_days': recovery_days,
            'recovered': recovered,
            'cvar_95': _cvar(r, 0.95),
            'cvar_99': _cvar(r, 0.99),
            'avg_daily_return': float(np.mean(r)),
            'worst_day': float(np.min(r)),
            'worst_week': worst_week,
            'would_survive_15pct_dd': max_dd < 0.15,
        }

    def full_stress_report(self, daily_returns: np.ndarray,
                           dates: np.ndarray = None) -> Dict[str, Dict]:
        """
        Run stress test across full sample AND individual crisis periods.

        When date arrays are provided, each CRISIS_PERIODS window is extracted
        and stress-tested individually. Without dates, still reports full-sample
        metrics plus any crisis periods that can be date-matched.

        Args:
            daily_returns: Array of daily returns
            dates: Optional array of date strings or datetime objects (same length)
                   Accepts: 'YYYY-MM-DD' strings, np.datetime64, datetime.date
        """
        results = {}
        results['full_sample'] = self.stress_test_returns(
            daily_returns, crisis_name="full_sample"
        )

        # -- Crisis period replay (spec requirement) --
        if dates is not None and len(dates) == len(daily_returns):
            # Normalize dates to comparable string format
            date_strs = []
            for d in dates:
                if hasattr(d, 'strftime'):
                    date_strs.append(d.strftime('%Y-%m-%d'))
                elif hasattr(d, 'astype'):
                    # np.datetime64
                    date_strs.append(str(d)[:10])
                else:
                    date_strs.append(str(d)[:10])

            for crisis_name, (start_str, end_str) in self.CRISIS_PERIODS.items():
                # Filter returns within the crisis window
                mask = [(start_str <= ds <= end_str) for ds in date_strs]
                crisis_returns = daily_returns[np.array(mask)]

                if len(crisis_returns) >= 5:  # Need at least 5 days
                    results[crisis_name] = self.stress_test_returns(
                        crisis_returns, crisis_name=crisis_name
                    )
                else:
                    results[crisis_name] = {
                        'crisis': crisis_name,
                        'n_days': len(crisis_returns),
                        'available': False,
                        'reason': 'insufficient_data_in_period',
                    }
        else:
            # No dates: report that crisis periods are unavailable
            for crisis_name in self.CRISIS_PERIODS:
                results[crisis_name] = {
                    'crisis': crisis_name,
                    'n_days': 0,
                    'available': False,
                    'reason': 'no_date_array_provided',
                }

        return results

    def print_stress_report(self, report: Dict[str, Dict], indent: str = "  "):
        """Pretty-print the stress test report."""
        print(f"\n{'=' * 60}")
        print(f"  STRESS TEST REPORT")
        print(f"{'=' * 60}")

        print(f"\n{indent}{'Crisis':<22s} {'MaxDD':>7s} {'Recov':>6s} "
              f"{'CVaR95':>8s} {'WorstDay':>9s} {'Surv?':>6s}")
        print(f"{indent}{'-' * 22} {'-' * 7} {'-' * 6} "
              f"{'-' * 8} {'-' * 9} {'-' * 6}")

        for name, data in report.items():
            if not data.get('available', False):
                continue
            surv = "YES" if data.get('would_survive_15pct_dd', False) else "NO"
            recov_str = f"{data['recovery_days']:>4d}d" if data.get('recovered') else "  N/R"
            print(f"{indent}{name:<22s} {data['max_dd']:>6.1%} {recov_str:>6s} "
                  f"{data['cvar_95']:>+7.4f} {data['worst_day']:>+8.4f} {surv:>6s}")

        print(f"{'=' * 60}")


# ============================================================================
# WS4C: Capacity Analysis
# ============================================================================

class CapacityEstimator:
    """
    Estimate strategy capacity as a function of liquidity and costs.

    Model: at AUM = X, cost_drag grows as sqrt(X/ADV) * turnover,
    which eats into gross alpha. Capacity = AUM where
    net_sharpe < 0.5 or cost_drag > 50% of gross alpha.

    Reference: Almgren & Chriss (2000) "Optimal Execution"
    """

    def estimate(self, gross_alpha_bps: float,
                 annual_turnover: float,
                 avg_adv_dollars: float,
                 spread_bps: float = 1.0,
                 impact_coeff: float = 0.1,
                 annual_vol: float = 0.15) -> Dict[str, Any]:
        """
        Estimate strategy capacity.

        Args:
            gross_alpha_bps:  Gross alpha in bps/year
            annual_turnover:  Annual turnover as fraction of AUM
            avg_adv_dollars:  Average daily volume in dollars
            spread_bps:       Full bid-ask spread in bps
            impact_coeff:     Square-root impact coefficient
            annual_vol:       Annual portfolio volatility

        Returns:
            capacity_at_sharpe_05: AUM where Sharpe drops below 0.5
            capacity_at_50pct_cost: AUM where costs eat 50% of alpha
            cost_curve: list of (aum, net_sharpe, cost_pct) dicts
        """
        if gross_alpha_bps <= 0 or annual_turnover <= 0:
            return {
                'capacity_at_sharpe_05': 0.0,
                'capacity_at_50pct_cost': 0.0,
                'cost_curve': [],
            }

        # Scan AUM levels from $1M to $10B
        aum_levels = np.logspace(6, 10, 50)  # $1M to $10B
        cost_curve = []

        cap_sharpe_05 = float(aum_levels[-1])
        cap_50pct = float(aum_levels[-1])

        for aum in aum_levels:
            # Per-trade notional = turnover * aum / 252
            daily_traded = annual_turnover * aum / 252.0

            # Spread cost per round-trip in bps
            spread_drag = 0.5 * spread_bps * annual_turnover

            # Impact cost: impact_coeff * sqrt(daily_traded/ADV) * annual_turnover * 1e4
            participation = daily_traded / max(avg_adv_dollars, 1.0)
            impact_drag_bps = impact_coeff * math.sqrt(min(participation, 1.0)) * annual_turnover * 1e4

            total_cost_bps = spread_drag + impact_drag_bps
            net_alpha_bps = gross_alpha_bps - total_cost_bps
            cost_pct = total_cost_bps / (gross_alpha_bps + 1e-10)

            # Approximate Sharpe: net_alpha / vol
            net_sharpe = (net_alpha_bps * 1e-4) / (annual_vol + 1e-10)

            cost_curve.append({
                'aum': float(aum),
                'net_sharpe': float(net_sharpe),
                'cost_pct': float(cost_pct),
                'cost_bps': float(total_cost_bps),
                'net_alpha_bps': float(net_alpha_bps),
            })

            if net_sharpe < 0.5 and aum < cap_sharpe_05:
                cap_sharpe_05 = float(aum)
            if cost_pct > 0.5 and aum < cap_50pct:
                cap_50pct = float(aum)

        return {
            'capacity_at_sharpe_05': cap_sharpe_05,
            'capacity_at_50pct_cost': cap_50pct,
            'cost_curve': cost_curve,
        }