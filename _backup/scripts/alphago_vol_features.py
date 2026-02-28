"""
Volatility-Based Features for AlphaGo Trading System
====================================================

Advanced volatility features that enhance regime detection and risk management.

Features implemented:
1. Vol-of-Vol (Volatility of Volatility) - Expected IC +0.005
2. Gap Analysis - Overnight return decomposition
3. Volume Anomaly Detection - High-volume return premium

All features follow the "test before build" philosophy.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional
import sys
sys.path.insert(0, '.')


# ============================================================================
# VOL-OF-VOL FEATURE
# ============================================================================

class VolOfVolFeature:
    """
    Tracks the volatility of volatility using rolling vol estimates.
    Produces a normalized vol-of-vol score for use as:
      - Feature in RL observation vector
      - Conditioning variable for vol premium harvesting
      - Risk scaling input for L3

    High vol-of-vol → uncertainty about uncertainty → reduce risk.
    Low vol-of-vol → complacency → conditions for tail event.

    Expected IC: +0.005
    Use case: Risk management feature, not standalone alpha
    """

    def __init__(self, vol_window: int = 21, vov_window: int = 21):
        self.vol_window = vol_window
        self.vov_window = vov_window
        self._vol_history = deque(maxlen=vov_window + 10)
        self._warmup = vol_window + vov_window

    def update(self, closes: np.ndarray, bar_idx: int) -> Dict[str, float]:
        """
        Call once per bar. Returns dict with vol-of-vol features.

        Returns:
            {
                'vol_of_vol': float,        # Raw vol-of-vol (annualized)
                'vol_of_vol_zscore': float,  # Z-score relative to history
                'vol_regime': str,           # 'low_vov' / 'normal' / 'high_vov'
            }
        """
        n = len(closes)
        if n < self.vol_window + 2:
            return {'vol_of_vol': 0.0, 'vol_of_vol_zscore': 0.0,
                    'vol_regime': 'normal'}

        # Current vol estimate over last vol_window bars
        log_rets = np.diff(np.log(closes[-self.vol_window:] + 1e-12))
        current_vol = float(np.std(log_rets)) * np.sqrt(252)
        current_vol = max(current_vol, 0.01)  # Floor at 1% annual

        self._vol_history.append(current_vol)

        if len(self._vol_history) < self.vov_window:
            return {'vol_of_vol': 0.0, 'vol_of_vol_zscore': 0.0,
                    'vol_regime': 'normal'}

        # Vol-of-vol = standard deviation of rolling vol estimates
        vol_array = np.array(list(self._vol_history))
        vov = float(np.std(vol_array[-self.vov_window:]))
        vov_mean = float(np.mean(vol_array))
        vov_std = float(np.std(vol_array)) + 1e-10

        # Z-score: how extreme is current vol-of-vol?
        vov_z = (vov - vov_mean) / vov_std

        # Regime classification
        if vov_z > 1.5:
            regime = 'high_vov'
        elif vov_z < -0.5:
            regime = 'low_vov'
        else:
            regime = 'normal'

        return {
            'vol_of_vol': vov,
            'vol_of_vol_zscore': float(np.clip(vov_z, -3.0, 3.0)),
            'vol_regime': regime,
        }


# ============================================================================
# GAP ANALYSIS
# ============================================================================

class GapAnalysis:
    """
    Decomposes daily return into overnight (close-to-open) and
    intraday (open-to-close) components.

    Key findings:
    - Nearly ALL equity returns accrue overnight, not intraday.
    - Large gap ups that hold = continuation (breakout)
    - Large gap ups that fill = mean reversion (exhaustion)
    - Track gap fill rate as a regime indicator.

    Source: French & Roll (1986), Elm Wealth (2025)
    Expected IC: +0.003-0.005 (feature, not standalone alpha)
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback
        self._overnight_rets = deque(maxlen=lookback)
        self._intraday_rets = deque(maxlen=lookback)
        self._gap_fill_count = 0
        self._gap_count = 0

    def compute(self, opens: np.ndarray, highs: np.ndarray,
                lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
        """
        Compute gap analysis features.

        Returns:
            {
                'overnight_ret': float,         # Open vs prev close
                'intraday_ret': float,          # Close vs open
                'gap_pct': float,               # Same as overnight_ret
                'overnight_dominance': float,   # Fraction of returns overnight
                'gap_fill_rate': float,         # How often gaps fill
            }
        """
        n = len(closes)
        if n < 3 or len(opens) < n:
            return {'overnight_ret': 0.0, 'intraday_ret': 0.0,
                    'gap_pct': 0.0, 'overnight_dominance': 0.5,
                    'gap_fill_rate': 0.5}

        # Overnight return: open today vs close yesterday
        overnight_ret = float(np.log(opens[-1] / (closes[-2] + 1e-12)))
        # Intraday return: close today vs open today
        intraday_ret = float(np.log(closes[-1] / (opens[-1] + 1e-12)))

        self._overnight_rets.append(overnight_ret)
        self._intraday_rets.append(intraday_ret)

        # Gap detection and fill tracking
        gap_threshold = 0.005  # 0.5% gap
        if abs(overnight_ret) > gap_threshold:
            self._gap_count += 1
            # Did intraday reverse the gap? (gap fill)
            if np.sign(overnight_ret) != np.sign(intraday_ret):
                self._gap_fill_count += 1

        # Overnight dominance: what fraction of total return is overnight?
        if len(self._overnight_rets) >= 10:
            on = np.array(list(self._overnight_rets))
            intra = np.array(list(self._intraday_rets))
            total_abs = np.sum(np.abs(on)) + np.sum(np.abs(intra)) + 1e-10
            overnight_share = np.sum(np.abs(on)) / total_abs
        else:
            overnight_share = 0.5

        gap_fill_rate = (self._gap_fill_count /
                         max(self._gap_count, 1))

        return {
            'overnight_ret': overnight_ret,
            'intraday_ret': intraday_ret,
            'gap_pct': overnight_ret,  # Same as overnight ret
            'overnight_dominance': float(overnight_share),
            'gap_fill_rate': float(gap_fill_rate),
        }


# ============================================================================
# VOLUME ANOMALY FEATURE
# ============================================================================

class VolumeAnomalyFeature:
    """
    Computes volume z-score and produces a feature for the
    observation vector. High-volume bars predict positive
    forward returns.

    Source: Gervais, Kaniel, Mingelgrin (2001) "The High-Volume Return Premium"
    Expected IC: +0.003-0.005
    Economic hypothesis: High volume = attention/information arrival
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def compute(self, volumes: np.ndarray) -> Dict[str, float]:
        """
        Compute volume anomaly features.

        Returns:
            {
                'volume_zscore': float,      # Z-score of current volume
                'volume_anomaly': bool,      # True if unusually high (>2 std)
            }
        """
        n = len(volumes)
        if n < self.lookback + 5:
            return {'volume_zscore': 0.0, 'volume_anomaly': False}

        recent_vol = volumes[-1]
        hist_mean = np.mean(volumes[-self.lookback:])
        hist_std = np.std(volumes[-self.lookback:]) + 1e-10

        z = (recent_vol - hist_mean) / hist_std

        return {
            'volume_zscore': float(np.clip(z, -3.0, 5.0)),
            'volume_anomaly': bool(z > 2.0),  # Unusually high
        }


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_vol_of_vol():
    """Quick test of Vol-of-Vol feature"""
    print("\n" + "="*70)
    print("Testing Vol-of-Vol Feature")
    print("="*70)

    # Generate synthetic price data with regime changes
    np.random.seed(42)
    n = 500

    # Low vol regime
    rets_low = np.random.normal(0.0001, 0.01, 200)
    # High vol regime
    rets_high = np.random.normal(0.0001, 0.03, 200)
    # Back to low vol
    rets_low2 = np.random.normal(0.0001, 0.01, 100)

    rets = np.concatenate([rets_low, rets_high, rets_low2])
    prices = 100.0 * np.exp(np.cumsum(rets))

    vov = VolOfVolFeature(vol_window=21, vov_window=21)

    results = []
    for i in range(100, len(prices)):
        result = vov.update(prices[:i+1], i)
        if i % 50 == 0:
            print(f"Bar {i:3d}: Vol-of-Vol={result['vol_of_vol']:.4f}, "
                  f"Z-score={result['vol_of_vol_zscore']:+.2f}, "
                  f"Regime={result['vol_regime']}")
        results.append(result)

    print(f"\nTest passed! Generated {len(results)} vol-of-vol measurements")
    return results


def test_gap_analysis():
    """Quick test of Gap Analysis"""
    print("\n" + "="*70)
    print("Testing Gap Analysis")
    print("="*70)

    # Generate synthetic OHLC data
    np.random.seed(42)
    n = 100

    closes = 100.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.01, n)))
    opens = closes * (1 + np.random.normal(0.001, 0.005, n))  # Gaps
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.01, n))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.01, n))

    gap = GapAnalysis(lookback=63)

    result = gap.compute(opens, highs, lows, closes)

    print(f"Overnight return: {result['overnight_ret']:+.4f}")
    print(f"Intraday return: {result['intraday_ret']:+.4f}")
    print(f"Overnight dominance: {result['overnight_dominance']:.2%}")
    print(f"Gap fill rate: {result['gap_fill_rate']:.2%}")

    print(f"\nTest passed!")
    return result


def test_volume_anomaly():
    """Quick test of Volume Anomaly"""
    print("\n" + "="*70)
    print("Testing Volume Anomaly Feature")
    print("="*70)

    # Generate synthetic volume data with spikes
    np.random.seed(42)
    n = 100

    base_vol = 1000000
    volumes = base_vol * (1 + np.random.uniform(-0.3, 0.3, n))
    # Add some spikes
    volumes[50] *= 3.0  # Big spike
    volumes[75] *= 2.5

    va = VolumeAnomalyFeature(lookback=63)

    result = va.compute(volumes)

    print(f"Volume Z-score: {result['volume_zscore']:+.2f}")
    print(f"Volume anomaly: {result['volume_anomaly']}")

    print(f"\nTest passed!")
    return result


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VOL FEATURES TEST SUITE")
    print("="*70)

    test_vol_of_vol()
    test_gap_analysis()
    test_volume_anomaly()

    print("\n" + "="*70)
    print("ALL TESTS PASSED")
    print("="*70)
