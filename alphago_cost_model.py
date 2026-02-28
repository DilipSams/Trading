"""
alphago_cost_model.py -- Canonical transaction cost model (L3 + L4 shared)

This is the SINGLE source of truth for cost estimation across the entire
pipeline.  Both L3 (PortfolioConstructor, cost-adjusted mu) and the
simulator / live fill engine must import from here.

Model:
    total_cost = half_spread_cost + impact_cost + fees_cost

Where:
    half_spread_cost = 0.5 * spread_bps * 1e-4 * trade_notional
    impact_cost      = impact_coeff * sqrt(trade_notional / adv_dollars) * trade_notional
    fees_cost        = fees_pct * trade_notional

All inputs are in dollar terms.  The caller converts exposure fractions
to notional before calling.

References:
    - Almgren & Chriss (2000) optimal execution framework
    - Kissell & Glantz (2003) practical market impact
"""

from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised cost breakdown for a single trade."""
    half_spread_cost: float
    impact_cost: float
    fees_cost: float
    total_cost: float          # sum of the above (always >= 0)
    trade_notional: float      # echo back for logging
    adv_dollars: float         # echo back for logging
    participation_rate: float  # trade_notional / adv_dollars


def estimate_cost_dollars(
    *,
    mid: float,
    trade_notional: float,
    adv_dollars: float,
    spread_bps: float = 1.0,
    impact_coeff: float = 0.1,
    fees_pct: float = 0.0,
) -> CostBreakdown:
    """
    Estimate total transaction cost in dollars.

    Parameters
    --------
    mid : float
        Current mid price (used only for logging context; cost is
        computed in notional space).
    trade_notional : float
        Absolute dollar value of the trade (|delta_exposure| * portfolio_value).
    adv_dollars : float
        Average daily volume in dollars (rolling mean of volume * price).
    spread_bps : float
        Full bid-ask spread in basis points.
    impact_coeff : float
        Square-root impact coefficient.  Typical values: 0.05-0.20.
    fees_pct : float
        Proportional fee (commission) as a fraction (e.g. 0.0001 = 1 bps).

    Returns
    -----
    CostBreakdown
        Itemised cost (all values >= 0).
    """
    trade_notional = abs(float(trade_notional))
    adv_dollars = max(float(adv_dollars), 1.0)  # floor to avoid div-by-zero

    # Half spread
    half_spread_cost = 0.5 * float(spread_bps) * 1e-4 * trade_notional

    # Square-root market impact: impact_coeff * sqrt(participation) * notional
    participation = trade_notional / adv_dollars
    impact_cost = float(impact_coeff) * math.sqrt(min(participation, 1.0)) * trade_notional

    # Proportional fees / commission
    fees_cost = float(fees_pct) * trade_notional

    total = half_spread_cost + impact_cost + fees_cost

    return CostBreakdown(
        half_spread_cost=half_spread_cost,
        impact_cost=impact_cost,
        fees_cost=fees_cost,
        total_cost=total,
        trade_notional=trade_notional,
        adv_dollars=adv_dollars,
        participation_rate=participation,
    )


def fill_price_from_cost(
    *,
    mid: float,
    side: str,
    trade_notional: float,
    adv_dollars: float,
    spread_bps: float = 1.0,
    impact_coeff: float = 0.1,
    fees_pct: float = 0.0,
) -> tuple[float, CostBreakdown]:
    """
    Compute a fill price that embeds spread + impact (fees kept separate).

    Parameters
    --------
    mid : float
        Mid price of the asset.
    side : str
        "buy" or "sell".
    trade_notional : float
        Absolute dollar notional of the trade.
    adv_dollars : float
        Average daily volume in dollars.
    spread_bps, impact_coeff, fees_pct :
        Same as estimate_cost_dollars.

    Returns
    -----
    (fill_price, CostBreakdown)
        fill_price includes spread + impact adverse move.
        Fees are NOT embedded in the fill price (keep them as a separate
        cash debit to avoid double-counting with commission).
    """
    cb = estimate_cost_dollars(
        mid=mid,
        trade_notional=trade_notional,
        adv_dollars=adv_dollars,
        spread_bps=spread_bps,
        impact_coeff=impact_coeff,
        fees_pct=fees_pct,
    )

    # Adverse price move = (half_spread + impact) / trade_notional * mid
    # i.e. move mid by the cost-per-dollar fraction
    if trade_notional > 0:
        adverse_bps = (cb.half_spread_cost + cb.impact_cost) / trade_notional
    else:
        adverse_bps = 0.0

    sign = 1.0 if side == "buy" else -1.0
    fill = mid * (1.0 + sign * adverse_bps)

    return fill, cb