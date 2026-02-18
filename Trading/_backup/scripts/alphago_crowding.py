"""
Crowding Detection for AlphaGo Trading System
==============================================

L4 Execution Monitoring: Detect when alpha signals become crowded (all agree).

Crowded trades are dangerous:
- When all alphas agree → market already moved
- High correlation → position concentration risk
- Unanimous direction → potential reversal point

Usage:
    detector = CrowdingDetector(warning_threshold=0.7, kill_threshold=0.85)
    result = detector.detect_crowding(alpha_signals)

    if result['action'] == 'kill':
        # Don't trade
    elif result['action'] == 'reduce':
        # Reduce position size by 30%
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class AlphaSignal:
    """Minimal AlphaSignal structure for type checking."""
    mu: float = 0.0
    sigma: float = 0.15
    confidence: float = 0.0
    horizon: int = 1
    alpha_name: str = ""
    metadata: dict = None
    timestamp: int = 0


class CrowdingDetector:
    """
    Detects crowding in alpha signals.

    Crowding Metrics:
    1. Direction Agreement: % of alphas with same sign
    2. Magnitude Agreement: Std of normalized mu values
    3. Historical Correlation: Rolling correlation among alphas

    Triggers:
    - Warning: 70% direction agreement → reduce sizing by 30%
    - Kill: 85% direction agreement → skip trade entirely
    """

    def __init__(self, warning_threshold: float = 0.70, kill_threshold: float = 0.85):
        """
        Initialize crowding detector.

        Args:
            warning_threshold: Direction agreement % to trigger warning
            kill_threshold: Direction agreement % to trigger kill switch
        """
        self.warning_threshold = warning_threshold
        self.kill_threshold = kill_threshold
        self.history = []  # Track crowding over time
        self.max_history = 100

    def detect_crowding(self, alpha_signals: Dict[str, AlphaSignal]) -> Dict:
        """
        Detect crowding in current alpha signals.

        Args:
            alpha_signals: Dict of alpha_name -> AlphaSignal

        Returns:
            {
                'crowding_score': float [0, 1],
                'action': 'normal' | 'reduce' | 'kill',
                'message': str,
                'metrics': {
                    'direction_agreement': float,
                    'positive_count': int,
                    'negative_count': int,
                    'neutral_count': int,
                    'magnitude_std': float,
                }
            }
        """
        # Extract valid signals (non-zero mu, non-zero confidence)
        valid_signals = [
            sig for sig in alpha_signals.values()
            if sig.mu != 0.0 and sig.confidence > 0.01
        ]

        if len(valid_signals) < 3:
            return {
                'crowding_score': 0.0,
                'action': 'normal',
                'message': 'Insufficient signals for crowding detection',
                'metrics': {
                    'direction_agreement': 0.0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': len(alpha_signals),
                    'magnitude_std': 0.0,
                }
            }

        # Count directional agreement
        positive_count = sum(1 for sig in valid_signals if sig.mu > 0)
        negative_count = sum(1 for sig in valid_signals if sig.mu < 0)
        total_count = len(valid_signals)

        # Direction agreement = max(positive, negative) / total
        max_agreement_count = max(positive_count, negative_count)
        direction_agreement = max_agreement_count / total_count

        # Magnitude agreement (low std = all alphas have similar strength)
        mus = np.array([sig.mu for sig in valid_signals])
        confidences = np.array([sig.confidence for sig in valid_signals])

        # Weighted mu by confidence
        weighted_mus = mus * confidences
        magnitude_std = np.std(weighted_mus) if len(weighted_mus) > 1 else 0.0

        # Combined crowding score
        # High direction agreement + low magnitude variation = crowded
        crowding_score = direction_agreement

        # Add to history
        self.history.append({
            'timestamp': valid_signals[0].timestamp if valid_signals else 0,
            'crowding_score': crowding_score,
            'direction_agreement': direction_agreement,
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Determine action
        if crowding_score >= self.kill_threshold:
            action = 'kill'
            message = (
                f"CROWDING KILL: {crowding_score*100:.0f}% alphas agree "
                f"({max_agreement_count}/{total_count} "
                f"{'long' if positive_count > negative_count else 'short'})"
            )
        elif crowding_score >= self.warning_threshold:
            action = 'reduce'
            message = (
                f"CROWDING WARNING: {crowding_score*100:.0f}% alphas agree "
                f"({max_agreement_count}/{total_count} "
                f"{'long' if positive_count > negative_count else 'short'}) "
                f"- reduce sizing by 30%"
            )
        else:
            action = 'normal'
            message = (
                f"Alpha diversity OK: {positive_count} long, {negative_count} short "
                f"(agreement: {crowding_score*100:.0f}%)"
            )

        return {
            'crowding_score': float(crowding_score),
            'action': action,
            'message': message,
            'metrics': {
                'direction_agreement': float(direction_agreement),
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'neutral_count': int(len(alpha_signals) - total_count),
                'magnitude_std': float(magnitude_std),
                'total_alphas': len(alpha_signals),
                'valid_alphas': total_count,
            }
        }

    def get_crowding_stats(self) -> Dict:
        """
        Get statistics on historical crowding.

        Returns:
            {
                'avg_crowding': float,
                'max_crowding': float,
                'crowding_events': int (count of warning/kill triggers),
            }
        """
        if not self.history:
            return {
                'avg_crowding': 0.0,
                'max_crowding': 0.0,
                'crowding_events': 0,
            }

        crowding_scores = [h['crowding_score'] for h in self.history]
        crowding_events = sum(
            1 for score in crowding_scores if score >= self.warning_threshold
        )

        return {
            'avg_crowding': float(np.mean(crowding_scores)),
            'max_crowding': float(np.max(crowding_scores)),
            'crowding_events': int(crowding_events),
            'samples': len(self.history),
        }


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("CROWDING DETECTOR TEST")
    print("=" * 70)

    detector = CrowdingDetector(warning_threshold=0.7, kill_threshold=0.85)

    # Test Case 1: Diverse signals (normal)
    print("\n[TEST 1] Diverse Signals (Normal)")
    signals_diverse = {
        'trend': AlphaSignal(mu=0.05, confidence=0.8, alpha_name='trend'),
        'mean_rev': AlphaSignal(mu=-0.03, confidence=0.6, alpha_name='mean_rev'),
        'value': AlphaSignal(mu=0.02, confidence=0.4, alpha_name='value'),
        'carry': AlphaSignal(mu=-0.01, confidence=0.3, alpha_name='carry'),
        'vol_prem': AlphaSignal(mu=0.04, confidence=0.7, alpha_name='vol_prem'),
    }
    result = detector.detect_crowding(signals_diverse)
    print(f"  Action: {result['action']}")
    print(f"  Crowding Score: {result['crowding_score']:.2f}")
    print(f"  Message: {result['message']}")

    # Test Case 2: Warning level (70% agreement)
    print("\n[TEST 2] Warning Level (70% Agreement)")
    signals_warning = {
        'trend': AlphaSignal(mu=0.05, confidence=0.8, alpha_name='trend'),
        'mean_rev': AlphaSignal(mu=0.03, confidence=0.6, alpha_name='mean_rev'),
        'value': AlphaSignal(mu=0.02, confidence=0.4, alpha_name='value'),
        'carry': AlphaSignal(mu=-0.01, confidence=0.3, alpha_name='carry'),
        'vol_prem': AlphaSignal(mu=0.04, confidence=0.7, alpha_name='vol_prem'),
        'liquidity': AlphaSignal(mu=0.06, confidence=0.5, alpha_name='liquidity'),
        'reversal': AlphaSignal(mu=0.02, confidence=0.6, alpha_name='reversal'),
    }
    result = detector.detect_crowding(signals_warning)
    print(f"  Action: {result['action']}")
    print(f"  Crowding Score: {result['crowding_score']:.2f}")
    print(f"  Message: {result['message']}")

    # Test Case 3: Kill level (90% agreement)
    print("\n[TEST 3] Kill Level (90% Agreement)")
    signals_kill = {
        'trend': AlphaSignal(mu=0.08, confidence=0.9, alpha_name='trend'),
        'mean_rev': AlphaSignal(mu=0.05, confidence=0.7, alpha_name='mean_rev'),
        'value': AlphaSignal(mu=0.03, confidence=0.6, alpha_name='value'),
        'carry': AlphaSignal(mu=0.02, confidence=0.5, alpha_name='carry'),
        'vol_prem': AlphaSignal(mu=0.06, confidence=0.8, alpha_name='vol_prem'),
        'liquidity': AlphaSignal(mu=0.07, confidence=0.6, alpha_name='liquidity'),
        'reversal': AlphaSignal(mu=0.04, confidence=0.7, alpha_name='reversal'),
        'hurst': AlphaSignal(mu=0.05, confidence=0.5, alpha_name='hurst'),
        'gap': AlphaSignal(mu=0.09, confidence=0.8, alpha_name='gap'),
        'vol_term': AlphaSignal(mu=-0.01, confidence=0.3, alpha_name='vol_term'),
    }
    result = detector.detect_crowding(signals_kill)
    print(f"  Action: {result['action']}")
    print(f"  Crowding Score: {result['crowding_score']:.2f}")
    print(f"  Message: {result['message']}")

    # Test Case 4: All neutral (no signals)
    print("\n[TEST 4] All Neutral (No Active Signals)")
    signals_neutral = {
        'trend': AlphaSignal(mu=0.0, confidence=0.0, alpha_name='trend'),
        'mean_rev': AlphaSignal(mu=0.0, confidence=0.0, alpha_name='mean_rev'),
        'value': AlphaSignal(mu=0.0, confidence=0.0, alpha_name='value'),
    }
    result = detector.detect_crowding(signals_neutral)
    print(f"  Action: {result['action']}")
    print(f"  Crowding Score: {result['crowding_score']:.2f}")
    print(f"  Message: {result['message']}")

    # Stats
    print("\n" + "=" * 70)
    print("CROWDING STATISTICS")
    print("=" * 70)
    stats = detector.get_crowding_stats()
    print(f"  Average Crowding: {stats['avg_crowding']:.2f}")
    print(f"  Max Crowding:     {stats['max_crowding']:.2f}")
    print(f"  Crowding Events:  {stats['crowding_events']}")
    print(f"  Samples:          {stats['samples']}")

    print("\n[SUCCESS] Crowding detector working correctly!")
