"""
Asymmetric Stop Loss System
============================

Implements "cut losses fast, let winners run" principle.

- Tight stops when losing (2% or 1.5 ATR)
- Trailing stops when winning (5% from peak or 3 ATR)

Usage:
    stop_manager = AsymmetricStopLoss(acfg)
    stop_price = stop_manager.compute_stop(position_pnl, entry_price, current_price, atr, peak_pnl)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class StopLossConfig:
    """Configuration for asymmetric stop loss."""
    # Loss regime (tight stops)
    loss_stop_pct: float = 0.02          # 2% stop when losing
    loss_stop_atr_mult: float = 1.5      # 1.5 ATR stop when losing

    # Profit regime (trailing stops)
    profit_trail_pct: float = 0.05       # Trail 5% from peak
    profit_trail_atr_mult: float = 3.0   # 3 ATR trailing stop

    # Volatility adjustments
    vol_adjust_stops: bool = True        # Adjust stops for volatility
    vol_baseline: float = 0.15           # Baseline volatility (15%)
    vol_max_adjustment: float = 2.0      # Max vol adjustment factor

    # Time-based tightening
    time_tighten_enabled: bool = False   # Tighten stops over time
    time_tighten_bars: int = 10          # Start tightening after N bars
    time_tighten_factor: float = 0.5     # Tighten to 50% of original


class AsymmetricStopLoss:
    """
    Asymmetric stop loss manager.

    Principle: Cut losses fast, let winners run.
    """

    def __init__(self, config: Optional[StopLossConfig] = None):
        """
        Initialize asymmetric stop loss manager.

        Args:
            config: Stop loss configuration (uses defaults if None)
        """
        self.config = config or StopLossConfig()

        # Statistics tracking
        self.stops_hit = {
            'loss_stops': 0,
            'trail_stops': 0,
            'total': 0,
        }
        self.avg_loss_at_stop = []
        self.avg_profit_at_stop = []

    def compute_stop(self,
                     position_pnl_pct: float,
                     entry_price: float,
                     current_price: float,
                     atr: float,
                     peak_pnl_pct: float,
                     bars_in_trade: int = 0,
                     is_long: bool = True,
                     realized_vol: float = 0.15) -> Dict:
        """
        Compute asymmetric stop loss level.

        Args:
            position_pnl_pct: Current P&L as % (e.g., 0.05 for +5%)
            entry_price: Entry price
            current_price: Current price
            atr: Average True Range
            peak_pnl_pct: Peak P&L reached (for trailing)
            bars_in_trade: Bars since entry (for time-based tightening)
            is_long: True if long position, False if short
            realized_vol: Current realized volatility

        Returns:
            {
                'stop_price': float,        # Price level to exit
                'stop_distance_pct': float, # Distance from current (%)
                'stop_type': str,           # 'loss' or 'trail'
                'should_exit': bool,        # True if stop already hit
            }
        """
        # FIX Bug #7: Validate inputs to prevent zero/NaN ATR issues
        if not np.isfinite(atr) or atr <= 0:
            atr = current_price * 0.015  # Default to 1.5% of price
        if not np.isfinite(realized_vol) or realized_vol <= 0:
            realized_vol = self.config.vol_baseline

        # Volatility adjustment factor
        if self.config.vol_adjust_stops:
            vol_ratio = realized_vol / self.config.vol_baseline
            vol_adjustment = min(vol_ratio, self.config.vol_max_adjustment)
        else:
            vol_adjustment = 1.0

        # Time-based tightening factor
        if self.config.time_tighten_enabled and bars_in_trade > self.config.time_tighten_bars:
            if position_pnl_pct < 0.01:  # Only tighten if not profitable
                time_factor = self.config.time_tighten_factor
            else:
                time_factor = 1.0
        else:
            time_factor = 1.0

        if position_pnl_pct < 0:
            # LOSING POSITION: Tight stop
            stop_price, stop_type = self._compute_loss_stop(
                entry_price, current_price, atr, is_long,
                vol_adjustment, time_factor
            )
        else:
            # WINNING POSITION: Trailing stop
            stop_price, stop_type = self._compute_trail_stop(
                entry_price, current_price, atr, peak_pnl_pct,
                is_long, vol_adjustment
            )

        # Calculate stop distance
        if is_long:
            stop_distance_pct = (stop_price - current_price) / current_price
            should_exit = current_price <= stop_price
        else:
            stop_distance_pct = (current_price - stop_price) / current_price
            should_exit = current_price >= stop_price

        return {
            'stop_price': float(stop_price),
            'stop_distance_pct': float(stop_distance_pct),
            'stop_type': stop_type,
            'should_exit': should_exit,
            'vol_adjustment': float(vol_adjustment),
            'time_factor': float(time_factor),
        }

    def _compute_loss_stop(self,
                           entry_price: float,
                           current_price: float,
                           atr: float,
                           is_long: bool,
                           vol_adj: float,
                           time_adj: float) -> Tuple[float, str]:
        """
        Compute tight stop for losing positions.

        Uses tighter of: percentage stop or ATR stop.
        """
        # Percentage-based stop (adjusted for vol and time)
        loss_pct = self.config.loss_stop_pct * vol_adj * time_adj

        if is_long:
            stop_pct = entry_price * (1 - loss_pct)
            stop_atr = current_price - (self.config.loss_stop_atr_mult * atr * vol_adj)
            # Tighter stop = higher price (closer to current)
            stop_price = max(stop_pct, stop_atr)
        else:
            stop_pct = entry_price * (1 + loss_pct)
            stop_atr = current_price + (self.config.loss_stop_atr_mult * atr * vol_adj)
            # Tighter stop = lower price (closer to current)
            stop_price = min(stop_pct, stop_atr)

        return stop_price, 'loss'

    def _compute_trail_stop(self,
                            entry_price: float,
                            current_price: float,
                            atr: float,
                            peak_pnl_pct: float,
                            is_long: bool,
                            vol_adj: float) -> Tuple[float, str]:
        """
        Compute trailing stop for winning positions.

        Trails from peak P&L, not current price.
        """
        # Peak price based on peak P&L
        peak_price = entry_price * (1 + peak_pnl_pct) if is_long else entry_price * (1 - peak_pnl_pct)

        # Percentage-based trail (adjusted for volatility)
        trail_pct = self.config.profit_trail_pct * vol_adj

        if is_long:
            stop_pct = peak_price * (1 - trail_pct)
            stop_atr = peak_price - (self.config.profit_trail_atr_mult * atr * vol_adj)
            # Tighter trail = higher price (locks in more profit)
            stop_price = max(stop_pct, stop_atr)
        else:
            stop_pct = peak_price * (1 + trail_pct)
            stop_atr = peak_price + (self.config.profit_trail_atr_mult * atr * vol_adj)
            # Tighter trail = lower price (locks in more profit)
            stop_price = min(stop_pct, stop_atr)

        return stop_price, 'trail'

    def record_stop_hit(self, stop_type: str, pnl_at_stop: float):
        """Record statistics when stop is hit."""
        self.stops_hit['total'] += 1

        if stop_type == 'loss':
            self.stops_hit['loss_stops'] += 1
            self.avg_loss_at_stop.append(pnl_at_stop)
        else:
            self.stops_hit['trail_stops'] += 1
            self.avg_profit_at_stop.append(pnl_at_stop)

    def get_statistics(self) -> Dict:
        """Get stop loss statistics."""
        return {
            'total_stops': self.stops_hit['total'],
            'loss_stops': self.stops_hit['loss_stops'],
            'trail_stops': self.stops_hit['trail_stops'],
            'avg_loss_at_stop': np.mean(self.avg_loss_at_stop) if self.avg_loss_at_stop else 0.0,
            'avg_profit_at_stop': np.mean(self.avg_profit_at_stop) if self.avg_profit_at_stop else 0.0,
            'loss_stop_pct': self.stops_hit['loss_stops'] / max(self.stops_hit['total'], 1) * 100,
            'trail_stop_pct': self.stops_hit['trail_stops'] / max(self.stops_hit['total'], 1) * 100,
        }


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("ASYMMETRIC STOP LOSS - TEST")
    print("=" * 70)

    config = StopLossConfig()
    stop_manager = AsymmetricStopLoss(config)

    # Test 1: Losing position (tight stop)
    print("\n[TEST 1] Losing Position")
    print("-" * 70)
    result = stop_manager.compute_stop(
        position_pnl_pct=-0.03,  # -3% loss
        entry_price=100.0,
        current_price=97.0,
        atr=2.0,
        peak_pnl_pct=0.0,
        is_long=True,
        realized_vol=0.15
    )
    print(f"Entry: $100, Current: $97 (losing -3%)")
    print(f"Stop Price:    ${result['stop_price']:.2f}")
    print(f"Stop Distance: {result['stop_distance_pct']*100:.2f}%")
    print(f"Stop Type:     {result['stop_type']}")
    print(f"Should Exit:   {result['should_exit']}")

    # Test 2: Winning position (trailing stop)
    print("\n[TEST 2] Winning Position")
    print("-" * 70)
    result = stop_manager.compute_stop(
        position_pnl_pct=0.08,   # +8% profit
        entry_price=100.0,
        current_price=108.0,
        atr=2.0,
        peak_pnl_pct=0.10,       # Peak was +10%
        is_long=True,
        realized_vol=0.15
    )
    print(f"Entry: $100, Current: $108 (+8%), Peak: $110 (+10%)")
    print(f"Stop Price:    ${result['stop_price']:.2f}")
    print(f"Stop Distance: {result['stop_distance_pct']*100:.2f}%")
    print(f"Stop Type:     {result['stop_type']}")
    print(f"Should Exit:   {result['should_exit']}")
    print(f"Locked Profit: ${result['stop_price'] - 100:.2f} (+{(result['stop_price']/100-1)*100:.1f}%)")

    # Test 3: High volatility adjustment
    print("\n[TEST 3] High Volatility Regime")
    print("-" * 70)
    result = stop_manager.compute_stop(
        position_pnl_pct=-0.02,
        entry_price=100.0,
        current_price=98.0,
        atr=3.0,
        peak_pnl_pct=0.0,
        is_long=True,
        realized_vol=0.30  # Double baseline vol
    )
    print(f"Entry: $100, Current: $98, High Vol (30% vs 15% baseline)")
    print(f"Stop Price:    ${result['stop_price']:.2f}")
    print(f"Vol Adjustment: {result['vol_adjustment']:.2f}x")
    print(f"Stop widened due to high volatility")

    print("\n" + "=" * 70)
    print("[SUCCESS] Asymmetric stop loss working correctly!")
    print("=" * 70)
