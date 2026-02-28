"""
Modified alpha architecture for testing signal inversions.

Changes from original alphago_architecture.py:
1. SeasonalityAlpha: Inverts signal (mu *= -1.0) to test if anti-predictive becomes predictive
2. VolatilityPremiumAlpha: Inverts signal (buy vol instead of sell) to test contrarian strategy

Usage:
    # Temporarily replace alphago_architecture.py with this file
    # Or modify import in alphago_layering.py to use this module

Tests hypothesis:
    If seasonality IC = -0.014 to -0.046, does inverting produce IC = +0.014 to +0.046?
    If vol_premium IC = -0.062, does inverting produce IC = +0.062?
"""

# This is a patch file - copy the modified methods into alphago_architecture.py
# or import selectively

SEASONALITY_INVERSION_PATCH = """
# In SeasonalityAlpha._generate_from_calendar(), line 1012:
# ORIGINAL:
#     mu = float(np.average(signals, weights=weights)) * 252  # Annualize

# INVERTED (for testing):
mu_raw = float(np.average(signals, weights=weights)) * 252
mu = -1.0 * mu_raw  # INVERSION: Flip calendar signal since IC was negative

# Add to metadata for tracking:
metadata={
    'dow': dow,
    'month': month,
    'quarter': quarter,
    'is_tom': is_tom,
    'n_features': len(signals),
    'mu_raw': mu_raw,  # Original signal before inversion
    'inverted': True,
}
"""

VOLPREMIUM_INVERSION_PATCH = """
# In VolatilityPremiumAlpha.generate(), lines 1108-1113:
# ORIGINAL:
#     if vol_ratio > 1.2:
#         mu = -0.02  # Vol spike -> expect calm -> slight short bias
#     elif vol_ratio < 0.8:
#         mu = 0.02   # Vol crush -> expect expansion -> slight long bias
#     else:
#         mu = 0.0

# INVERTED (for testing - buy vol instead of sell):
if vol_ratio > 1.2:
    mu = +0.02  # INVERTED: Vol spike -> BUY vol (expect continued turbulence)
elif vol_ratio < 0.8:
    mu = -0.02  # INVERTED: Vol crush -> SELL vol (expect continued calm)
else:
    mu = 0.0

# Add tracking:
metadata={
    'short_vol': short_vol,
    'long_vol': long_vol,
    'vol_ratio': vol_ratio,
    'vol_of_vol': vov,
    'inverted': True,  # Mark as inverted for analysis
}
"""

VALUE_TIMING_NOTE = """
# Value timing investigation requires portfolio-level changes, not alpha-level.
# The alpha horizon is already set to 63 bars (3 months) in config.
# To test if actual 15-bar holding improves Sharpe:

Option 1: Modify L3 PortfolioConstructor to honor alpha horizons
    - Track entry timestamp for each position
    - Only rebalance position after horizon bars have elapsed
    - Requires adding position aging logic

Option 2: Create a separate backtester that:
    - Takes value alpha signals
    - Holds for exactly 15 bars
    - Measures realized return over that period
    - Computes Sharpe of non-overlapping 15-bar trades

Option 3: Use the multi-horizon profiling IC at 15-bar as proxy
    - Already done: IC(value, 15-bar) = +0.055 (t=1.9)
    - Compare to IC(value, 1-bar) = +0.018 (t=2.4)
    - 3x higher IC at 15-bar suggests longer holding is better
"""

def apply_seasonality_inversion():
    """
    Modify alphago_architecture.py to invert seasonality signal.

    Returns the patch as a string that can be applied.
    """
    return SEASONALITY_INVERSION_PATCH

def apply_volpremium_inversion():
    """
    Modify alphago_architecture.py to invert vol premium signal.

    Returns the patch as a string that can be applied.
    """
    return VOLPREMIUM_INVERSION_PATCH

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("SEASONALITY INVERSION PATCH:")
    print("="*80)
    print(SEASONALITY_INVERSION_PATCH)

    print("\n" + "="*80)
    print("VOL PREMIUM INVERSION PATCH:")
    print("="*80)
    print(VOLPREMIUM_INVERSION_PATCH)

    print("\n" + "="*80)
    print("VALUE TIMING NOTE:")
    print("="*80)
    print(VALUE_TIMING_NOTE)
