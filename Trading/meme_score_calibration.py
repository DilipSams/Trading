#!/usr/bin/env python3
"""
Meme Score Calibration — Empirical Threshold Optimization.

Research-only script: analyzes historical data (1990-2026) to empirically
calibrate the 6-factor meme score thresholds used in waverider.py.

Approach:
  1. Scan all stocks in universe_builder history (~1,287 unique symbols)
  2. Identify parabolic rally candidates (3m return > 100%)
  3. Classify by subsequent behavior:
     - MEME BLOWUP: >60% drawdown within 12 months of peak
     - SUSTAINED GROWTH: stays within 30% of peak for 2+ years
  4. Compute meme scores at peak dates using WaveRider's 6-factor model
  5. Analyze factor distributions, ROC curves, optimal thresholds
  6. Compare vs current hardcoded thresholds

Output: comprehensive report with factor importance, threshold recommendations,
        regime stability analysis.

Usage:
    python meme_score_calibration.py
    python meme_score_calibration.py --top-n 200 --min-rally 0.80
"""

import os
import sys
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure sibling imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

NORGATE_ROOT = r"D:\Experiments\norgate_data"

# Current WaveRider thresholds (from waverider.py WaveRiderConfig)
CURRENT_THRESHOLDS = {
    "meme_exclude": 70,   # Score > 70 → excluded from portfolio
    "meme_max1": 50,      # Score > 50 → max position 1 slot
    "meme_max2": 30,      # Score > 30 → max position 2 slots
}

# Factor point allocations (current)
FACTOR_RANGES = {
    "f1_volatility":  {"max": 25, "thresholds": [0.40, 0.60, 0.80, 1.00], "points": [8, 15, 20, 25]},
    "f2_parabolic":   {"max": 25, "thresholds": [0.50, 1.00, 2.00], "points": [8, 18, 25]},
    "f3_sma_stretch": {"max": 20, "thresholds": [1.30, 2.00, 3.00], "points": [8, 15, 20]},
    "f4_mom_conc":    {"max": 15, "thresholds": [0.30, 0.50, 0.80], "points": [5, 10, 15]},
    "f5_vol_accel":   {"max": 15, "thresholds": [1.00, 1.50, 2.50], "points": [5, 10, 15]},
    "f6_tenure":      {"max": 15, "note": "≤2 months=15, ≤5 months=8"},
}

# Market regimes for regime-stratified validation
REGIMES = [
    ("1990s Bull",     "1991-01-01", "2000-03-24"),
    ("Dot-Com Bust",   "2000-03-24", "2002-10-09"),
    ("Mid-2000s Bull", "2002-10-09", "2007-10-09"),
    ("GFC",            "2007-10-09", "2009-03-09"),
    ("QE Bull",        "2009-03-09", "2020-02-19"),
    ("Post-COVID",     "2020-02-19", "2026-12-31"),
]

# Known meme/blowup stocks for validation
KNOWN_MEME = {"GME", "AMC", "BBBY", "KOSS", "BB", "NOK", "CLOV", "WISH", "SPCE"}
KNOWN_BLOWUPS_HISTORICAL = {
    "ENRNQ", "WCOM", "TYCO", "PMCSQ", "JDSU", "CSCO",  # dot-com era
    "LEHMQ", "BSCQ", "WAMUQ",  # GFC era
}


# ===========================================================================
# STEP 1: Build labeled dataset
# ===========================================================================

def load_universe_data(top_n=150):
    """Load PIT universe rankings and catalog for all historical symbols."""
    from universe_builder import build_universe_cache, _parse_catalog

    rankings, prices = build_universe_cache(top_n=top_n)
    catalog = _parse_catalog(NORGATE_ROOT)

    print(f"  Universe: {rankings['uid'].nunique()} unique symbols, "
          f"{len(rankings)} rank records")
    print(f"  Price matrix: {prices.shape}")
    return rankings, prices, catalog


def _load_symbol_ohlcv(uid, catalog):
    """Load full OHLCV for a single symbol from Norgate."""
    uid_to_path = dict(zip(catalog["uid"], catalog["filepath"]))
    fpath = uid_to_path.get(uid)
    if fpath is None or not os.path.exists(fpath):
        return None
    try:
        df = pd.read_parquet(fpath)
        df.index = pd.to_datetime(df.index)
        # Standardize columns
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl == "open": col_map[c] = "Open"
            elif cl == "high": col_map[c] = "High"
            elif cl == "low": col_map[c] = "Low"
            elif cl == "close" and "unadj" not in cl: col_map[c] = "Close"
            elif cl in ("volume", "vol"): col_map[c] = "Volume"
        df = df.rename(columns=col_map)
        if "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def find_parabolic_rallies(prices, rankings, min_rally=1.0, lookback_months=3):
    """
    Identify stocks with parabolic rallies (3-month return > min_rally).

    Returns:
        list of dicts: [{'uid': str, 'peak_date': Timestamp, 'rally_pct': float,
                         'subsequent_max_dd': float, 'label': str}, ...]
    """
    print(f"\n  Scanning for parabolic rallies (3m return > {min_rally:.0%})...")
    t0 = time.time()

    lookback_bars = lookback_months * 21  # ~63 bars
    results = []
    n_symbols = len(prices.columns)

    for i, uid in enumerate(prices.columns):
        if i % 200 == 0:
            print(f"    Scanning: {i}/{n_symbols} symbols ({i/n_symbols*100:.0f}%)...",
                  end="\r")

        col = prices[uid].dropna()
        if len(col) < lookback_bars + 252:  # Need enough history
            continue

        # Find 3-month returns
        ret_3m = col.pct_change(lookback_bars)

        # Find all dates where 3m return exceeds threshold
        rally_dates = ret_3m[ret_3m > min_rally].index
        if len(rally_dates) == 0:
            continue

        # Group into rally episodes (within 63 bars = same rally)
        episodes = []
        prev_dt = None
        for dt in rally_dates:
            if prev_dt is None or (dt - prev_dt).days > 90:
                episodes.append([dt])
            else:
                episodes[-1].append(dt)
            prev_dt = dt

        # For each episode, find the peak and classify
        for episode in episodes:
            # Peak = highest close during the episode window
            ep_start = episode[0]
            ep_end = episode[-1] + pd.Timedelta(days=30)  # extend slightly
            ep_slice = col.loc[ep_start:ep_end]
            if len(ep_slice) == 0:
                continue
            peak_date = ep_slice.idxmax()
            peak_price = ep_slice.max()
            rally_pct = float(ret_3m.get(episode[-1], 0))

            # Compute subsequent max drawdown (12 months after peak)
            post_peak_end = peak_date + pd.Timedelta(days=365)
            post_peak = col.loc[peak_date:post_peak_end]
            if len(post_peak) < 20:
                continue
            min_after_peak = post_peak.min()
            max_dd = (min_after_peak - peak_price) / peak_price

            # Classify
            if max_dd < -0.60:
                label = "meme_blowup"
            elif max_dd > -0.30 and len(col.loc[peak_date:peak_date + pd.Timedelta(days=730)]) > 252:
                # Stayed within 30% of peak for at least 1 year of data
                post_2y = col.loc[peak_date:peak_date + pd.Timedelta(days=730)]
                if len(post_2y) > 252 and post_2y.min() / peak_price > 0.70:
                    label = "sustained_growth"
                else:
                    label = "ambiguous"
            else:
                label = "ambiguous"

            results.append({
                "uid": uid,
                "peak_date": peak_date,
                "rally_pct": rally_pct,
                "subsequent_max_dd": max_dd,
                "label": label,
            })

    elapsed = time.time() - t0
    print(f"    Scanning: {n_symbols}/{n_symbols} symbols (100%) - "
          f"{elapsed:.1f}s                    ")

    n_meme = sum(1 for r in results if r["label"] == "meme_blowup")
    n_growth = sum(1 for r in results if r["label"] == "sustained_growth")
    n_ambig = sum(1 for r in results if r["label"] == "ambiguous")
    print(f"  Found {len(results)} parabolic rally episodes:")
    print(f"    Meme blowups:     {n_meme}")
    print(f"    Sustained growth: {n_growth}")
    print(f"    Ambiguous:        {n_ambig}")

    return results


# ===========================================================================
# STEP 2: Compute meme scores at peak dates
# ===========================================================================

def compute_factor_scores_at_peaks(episodes, prices, rankings, catalog):
    """
    For each labeled episode, compute the 6 individual factor scores at peak_date.

    Returns DataFrame with columns: uid, peak_date, label, f1-f6, composite,
                                    rally_pct, subsequent_max_dd
    """
    print(f"\n  Computing 6-factor meme scores at {len(episodes)} peak dates...")
    t0 = time.time()

    # Pre-compute shared data
    daily_rets = prices.pct_change()
    vol_63d = daily_rets.rolling(63).std() * np.sqrt(252)

    # Factor computations (vectorized, same logic as waverider.py)
    # Factor 1: Volatility level
    f1_all = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    f1_all[vol_63d > 0.40] = 8
    f1_all[vol_63d > 0.60] = 15
    f1_all[vol_63d > 0.80] = 20
    f1_all[vol_63d > 1.00] = 25

    # Factor 2: Parabolic 3m move
    ret_3m_abs = prices.pct_change(63).shift(1).abs()
    f2_all = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    f2_all[ret_3m_abs > 0.50] = 8
    f2_all[ret_3m_abs > 1.00] = 18
    f2_all[ret_3m_abs > 2.00] = 25

    # Factor 3: Price-SMA200 stretch
    sma200 = prices.rolling(200).mean()
    stretch = (prices / sma200).shift(1)
    f3_all = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    f3_all[stretch > 1.3] = 8
    f3_all[stretch > 2.0] = 15
    f3_all[stretch > 3.0] = 20

    # Factor 4: Momentum concentration
    mom_1m = prices.pct_change(21).shift(1)
    mom_12m = prices.pct_change(252).shift(1)
    both_pos = (mom_1m > 0) & (mom_12m > 0.01)
    conc = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    conc[both_pos] = (mom_1m[both_pos] / mom_12m[both_pos]).clip(0, 2)
    f4_all = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    f4_all[conc > 0.3] = 5
    f4_all[conc > 0.5] = 10
    f4_all[conc > 0.8] = 15

    # Factor 5: Volatility acceleration
    vol_21d = daily_rets.rolling(21).std().shift(1) * np.sqrt(252)
    vol_126d = daily_rets.rolling(126).std().shift(1) * np.sqrt(252)
    vol_accel = vol_21d / (vol_126d + 0.01)
    f5_all = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    f5_all[vol_accel > 1.0] = 5
    f5_all[vol_accel > 1.5] = 10
    f5_all[vol_accel > 2.5] = 15

    # Factor 6: Universe tenure (simplified: use rankings directly)
    # Build tenure from rankings
    top100 = rankings[rankings["rank"] <= 100]
    rank_dates = sorted(rankings["date"].unique())
    tenure = {}
    tenure_snapshots = {}
    for rdate in rank_dates:
        current_top = set(top100[top100["date"] == rdate]["uid"].values)
        new_tenure = {uid: tenure.get(uid, 0) + 1 for uid in current_top}
        tenure = new_tenure
        tenure_snapshots[rdate] = dict(tenure)

    f6_all = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
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
                    f6_all.loc[mask, uid] = 15
                elif months <= 5:
                    f6_all.loc[mask, uid] = 8

    print(f"  Factor matrices computed in {time.time()-t0:.1f}s")

    # Extract factor values at each peak date
    records = []
    for ep in episodes:
        uid = ep["uid"]
        dt = ep["peak_date"]
        if uid not in prices.columns:
            continue
        if dt not in prices.index:
            # Find nearest date
            idx = prices.index.searchsorted(dt)
            if idx >= len(prices.index):
                continue
            dt = prices.index[idx]

        try:
            f1 = float(f1_all.loc[dt, uid]) if dt in f1_all.index else 0
            f2 = float(f2_all.loc[dt, uid]) if dt in f2_all.index else 0
            f3 = float(f3_all.loc[dt, uid]) if dt in f3_all.index else 0
            f4 = float(f4_all.loc[dt, uid]) if dt in f4_all.index else 0
            f5 = float(f5_all.loc[dt, uid]) if dt in f5_all.index else 0
            f6 = float(f6_all.loc[dt, uid]) if dt in f6_all.index else 0
            composite = f1 + f2 + f3 + f4 + f5 + f6

            records.append({
                "uid": uid,
                "peak_date": dt,
                "label": ep["label"],
                "rally_pct": ep["rally_pct"],
                "subsequent_max_dd": ep["subsequent_max_dd"],
                "f1_volatility": f1,
                "f2_parabolic": f2,
                "f3_sma_stretch": f3,
                "f4_mom_conc": f4,
                "f5_vol_accel": f5,
                "f6_tenure": f6,
                "composite": composite,
            })
        except (KeyError, IndexError):
            continue

    df = pd.DataFrame(records)
    print(f"  Extracted scores for {len(df)} episodes ({time.time()-t0:.1f}s total)")
    return df


# ===========================================================================
# STEP 3: Distribution analysis
# ===========================================================================

def analyze_factor_distributions(df):
    """Per-factor histogram statistics and KS-test for meme vs growth separation."""
    from scipy import stats

    print("\n" + "=" * 80)
    print("FACTOR DISTRIBUTION ANALYSIS")
    print("=" * 80)

    meme = df[df["label"] == "meme_blowup"]
    growth = df[df["label"] == "sustained_growth"]

    if len(meme) == 0 or len(growth) == 0:
        print("  Insufficient labeled data for distribution analysis.")
        return {}

    factors = ["f1_volatility", "f2_parabolic", "f3_sma_stretch",
               "f4_mom_conc", "f5_vol_accel", "f6_tenure", "composite"]
    factor_labels = {
        "f1_volatility": "F1: Volatility Level (0-25)",
        "f2_parabolic": "F2: Parabolic Move (0-25)",
        "f3_sma_stretch": "F3: SMA200 Stretch (0-20)",
        "f4_mom_conc": "F4: Momentum Conc. (0-15)",
        "f5_vol_accel": "F5: Vol Acceleration (0-15)",
        "f6_tenure": "F6: Tenure Score (0-15)",
        "composite": "COMPOSITE (0-115)",
    }

    results = {}
    print(f"\n  {'Factor':<35s} {'Meme Mean':>10s} {'Growth Mean':>12s} "
          f"{'KS Stat':>8s} {'p-value':>10s} {'Discriminative':>14s}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*8} {'-'*10} {'-'*14}")

    for f in factors:
        m_vals = meme[f].dropna().values
        g_vals = growth[f].dropna().values
        m_mean = np.mean(m_vals) if len(m_vals) > 0 else 0
        g_mean = np.mean(g_vals) if len(g_vals) > 0 else 0

        # KS test
        if len(m_vals) >= 5 and len(g_vals) >= 5:
            ks_stat, ks_p = stats.ks_2samp(m_vals, g_vals)
        else:
            ks_stat, ks_p = 0, 1.0

        disc = "STRONG" if ks_p < 0.01 else "MODERATE" if ks_p < 0.05 else "WEAK"

        print(f"  {factor_labels.get(f, f):<35s} {m_mean:>10.1f} {g_mean:>12.1f} "
              f"{ks_stat:>8.3f} {ks_p:>10.4f} {disc:>14s}")

        results[f] = {
            "meme_mean": m_mean, "growth_mean": g_mean,
            "ks_stat": ks_stat, "ks_p": ks_p, "disc": disc,
        }

    # Factor importance ranking by KS statistic
    print(f"\n  FACTOR IMPORTANCE (by KS statistic — higher = more discriminative):")
    ranked = sorted(results.items(), key=lambda x: x[1]["ks_stat"], reverse=True)
    for i, (f, r) in enumerate(ranked, 1):
        print(f"    {i}. {factor_labels.get(f, f):<35s}  KS={r['ks_stat']:.3f}  "
              f"p={r['ks_p']:.4f}  [{r['disc']}]")

    return results


# ===========================================================================
# STEP 4: ROC / threshold optimization
# ===========================================================================

def compute_roc_and_thresholds(df):
    """Compute ROC curves and optimal thresholds for each factor."""
    print("\n" + "=" * 80)
    print("ROC ANALYSIS & THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Binary labels: meme_blowup=1, sustained_growth=0
    labeled = df[df["label"].isin(["meme_blowup", "sustained_growth"])].copy()
    labeled["is_meme"] = (labeled["label"] == "meme_blowup").astype(int)

    if len(labeled) < 20:
        print("  Insufficient labeled data for ROC analysis.")
        return {}

    factors = ["f1_volatility", "f2_parabolic", "f3_sma_stretch",
               "f4_mom_conc", "f5_vol_accel", "f6_tenure", "composite"]

    results = {}
    print(f"\n  {'Factor':<35s} {'AUC':>6s} {'Optimal Thresh':>15s} "
          f"{'Youden J':>9s} {'TPR':>6s} {'FPR':>6s}")
    print(f"  {'-'*35} {'-'*6} {'-'*15} {'-'*9} {'-'*6} {'-'*6}")

    for f in factors:
        y_true = labeled["is_meme"].values
        y_score = labeled[f].values

        # Manual ROC computation (no sklearn dependency)
        thresholds = np.unique(y_score)
        tpr_list, fpr_list = [], []
        for thresh in thresholds:
            pred_pos = y_score >= thresh
            tp = np.sum(pred_pos & (y_true == 1))
            fp = np.sum(pred_pos & (y_true == 0))
            fn = np.sum(~pred_pos & (y_true == 1))
            tn = np.sum(~pred_pos & (y_true == 0))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        tpr_arr = np.array(tpr_list)
        fpr_arr = np.array(fpr_list)

        # AUC (trapezoidal)
        # Sort by FPR for proper AUC computation
        sort_idx = np.argsort(fpr_arr)
        auc = np.trapz(tpr_arr[sort_idx], fpr_arr[sort_idx])

        # Youden's J = TPR - FPR → optimal threshold
        j_scores = tpr_arr - fpr_arr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        best_tpr = tpr_arr[best_idx]
        best_fpr = fpr_arr[best_idx]
        best_j = j_scores[best_idx]

        print(f"  {f:<35s} {auc:>6.3f} {best_thresh:>15.0f} "
              f"{best_j:>9.3f} {best_tpr:>5.0%} {best_fpr:>5.0%}")

        results[f] = {
            "auc": auc, "optimal_threshold": best_thresh,
            "youden_j": best_j, "tpr": best_tpr, "fpr": best_fpr,
            "thresholds": thresholds, "tpr_arr": tpr_arr, "fpr_arr": fpr_arr,
        }

    return results


# ===========================================================================
# STEP 5: Regime-stratified validation
# ===========================================================================

def regime_analysis(df, roc_results):
    """Test if optimal thresholds are stable across market regimes."""
    print("\n" + "=" * 80)
    print("REGIME-STRATIFIED VALIDATION")
    print("=" * 80)

    labeled = df[df["label"].isin(["meme_blowup", "sustained_growth"])].copy()
    labeled["is_meme"] = (labeled["label"] == "meme_blowup").astype(int)

    factors = ["composite", "f1_volatility", "f2_parabolic"]

    for f in factors:
        print(f"\n  --- {f} ---")
        print(f"  {'Regime':<20s} {'N':>5s} {'N_meme':>7s} {'Opt Thresh':>11s} "
              f"{'Youden J':>9s} {'TPR':>6s} {'FPR':>6s}")
        print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*11} {'-'*9} {'-'*6} {'-'*6}")

        regime_thresholds = []
        for rname, rstart, rend in REGIMES:
            mask = ((labeled["peak_date"] >= rstart) &
                    (labeled["peak_date"] <= rend))
            subset = labeled[mask]
            n_meme = sum(subset["is_meme"])
            n_total = len(subset)

            if n_total < 5 or n_meme < 2:
                print(f"  {rname:<20s} {n_total:>5d} {n_meme:>7d} {'(insufficient)':>11s}")
                continue

            y_true = subset["is_meme"].values
            y_score = subset[f].values
            thresholds = np.unique(y_score)

            best_j, best_thresh, best_tpr, best_fpr = -1, 0, 0, 0
            for thresh in thresholds:
                pred_pos = y_score >= thresh
                tp = np.sum(pred_pos & (y_true == 1))
                fp = np.sum(pred_pos & (y_true == 0))
                fn = np.sum(~pred_pos & (y_true == 1))
                tn = np.sum(~pred_pos & (y_true == 0))
                tpr = tp / max(tp + fn, 1)
                fpr = fp / max(fp + tn, 1)
                j = tpr - fpr
                if j > best_j:
                    best_j, best_thresh = j, thresh
                    best_tpr, best_fpr = tpr, fpr

            print(f"  {rname:<20s} {n_total:>5d} {n_meme:>7d} {best_thresh:>11.0f} "
                  f"{best_j:>9.3f} {best_tpr:>5.0%} {best_fpr:>5.0%}")
            regime_thresholds.append(best_thresh)

        if regime_thresholds:
            std = np.std(regime_thresholds)
            mean = np.mean(regime_thresholds)
            print(f"  {'STABILITY':.<20s} mean={mean:.0f}  std={std:.1f}  "
                  f"CV={std/max(mean,1):.2f}")


# ===========================================================================
# STEP 6: Band calibration
# ===========================================================================

def calibrate_bands(df):
    """Test current meme score bands against empirical data."""
    print("\n" + "=" * 80)
    print("BAND CALIBRATION (Current vs Empirical)")
    print("=" * 80)

    labeled = df[df["label"].isin(["meme_blowup", "sustained_growth"])].copy()
    labeled["is_meme"] = (labeled["label"] == "meme_blowup").astype(int)

    if len(labeled) < 10:
        print("  Insufficient data.")
        return

    # Current bands
    bands = [
        ("Safe (0-30)",     0,  30),
        ("Watch (31-50)",  31,  50),
        ("Caution (51-70)", 51, 70),
        ("Exclude (>70)",  71, 200),
    ]

    print(f"\n  {'Band':<22s} {'Total':>6s} {'Meme':>6s} {'Growth':>7s} "
          f"{'Precision':>10s} {'Recall':>7s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*7} {'-'*10} {'-'*7}")

    total_meme = labeled["is_meme"].sum()
    for band_name, lo, hi in bands:
        mask = (labeled["composite"] >= lo) & (labeled["composite"] <= hi)
        band_data = labeled[mask]
        n_total = len(band_data)
        n_meme = band_data["is_meme"].sum()
        n_growth = n_total - n_meme
        precision = n_meme / max(n_total, 1)
        recall = n_meme / max(total_meme, 1)

        print(f"  {band_name:<22s} {n_total:>6d} {n_meme:>6d} {n_growth:>7d} "
              f"{precision:>10.1%} {recall:>7.1%}")

    # Key questions
    print(f"\n  KEY DIAGNOSTIC QUESTIONS:")
    # What fraction of true meme blowups scored >70?
    meme_above_70 = labeled[(labeled["is_meme"] == 1) & (labeled["composite"] > 70)]
    total_meme_count = labeled["is_meme"].sum()
    print(f"    Q: What % of true meme blowups scored >70 (exclude threshold)?")
    print(f"    A: {len(meme_above_70)}/{total_meme_count} = "
          f"{len(meme_above_70)/max(total_meme_count,1):.1%} "
          f"(recall at exclude threshold)")

    # What fraction of >70 scores were actual blowups?
    above_70 = labeled[labeled["composite"] > 70]
    above_70_meme = above_70["is_meme"].sum()
    print(f"    Q: What % of stocks scoring >70 were actual blowups?")
    print(f"    A: {above_70_meme}/{len(above_70)} = "
          f"{above_70_meme/max(len(above_70),1):.1%} "
          f"(precision at exclude threshold)")

    # What about >50?
    above_50 = labeled[labeled["composite"] > 50]
    above_50_meme = above_50["is_meme"].sum()
    meme_above_50 = labeled[(labeled["is_meme"] == 1) & (labeled["composite"] > 50)]
    print(f"    Q: At threshold >50: precision={above_50_meme/max(len(above_50),1):.1%}, "
          f"recall={len(meme_above_50)/max(total_meme_count,1):.1%}")


# ===========================================================================
# STEP 7: Output report
# ===========================================================================

def print_final_report(df, dist_results, roc_results):
    """Print comprehensive final report."""
    print("\n" + "=" * 80)
    print("FINAL REPORT: MEME SCORE THRESHOLD CALIBRATION")
    print("=" * 80)

    labeled = df[df["label"].isin(["meme_blowup", "sustained_growth"])]

    print(f"""
HYPOTHESIS: 6-factor meme score with empirically optimized thresholds
            can distinguish parabolic meme blowups from sustained growth stocks.

SIGNAL:     Composite score from OHLCV-only factors (volatility, parabolic move,
            SMA stretch, momentum concentration, vol acceleration, tenure).

ECONOMIC RATIONALE:
  Speculative stocks exhibit measurable OHLCV patterns: extreme volatility,
  parabolic price moves, detachment from moving averages, and concentrated
  momentum. These patterns precede blowups in a statistically significant
  fraction of cases.

DATASET:
  Total parabolic rally episodes: {len(df)}
  Labeled meme blowups:          {sum(df['label']=='meme_blowup')}
  Labeled sustained growth:      {sum(df['label']=='sustained_growth')}
  Ambiguous (excluded):          {sum(df['label']=='ambiguous')}
  Period: {df['peak_date'].min().strftime('%Y-%m-%d')} to {df['peak_date'].max().strftime('%Y-%m-%d')}
""")

    # Current vs optimized thresholds
    if roc_results and "composite" in roc_results:
        comp = roc_results["composite"]
        current_auc = comp["auc"]
        current_thresh = CURRENT_THRESHOLDS["meme_exclude"]
        optimal_thresh = comp["optimal_threshold"]

        print(f"CURRENT vs OPTIMIZED (composite score):")
        print(f"  Current exclude threshold: >{current_thresh}  "
              f"(AUC={current_auc:.3f})")
        print(f"  Optimal threshold (Youden): >{optimal_thresh:.0f}  "
              f"(TPR={comp['tpr']:.0%}, FPR={comp['fpr']:.0%})")

        if abs(optimal_thresh - current_thresh) <= 10:
            verdict = "KEEP current thresholds (within 10 points of optimal)"
        elif optimal_thresh < current_thresh:
            verdict = f"CONSIDER LOWERING to {optimal_thresh:.0f} (more conservative)"
        else:
            verdict = f"CONSIDER RAISING to {optimal_thresh:.0f} (less aggressive)"
        print(f"\n  VERDICT: {verdict}")

    # Per-factor comparison
    if roc_results:
        print(f"\n  PER-FACTOR OPTIMAL THRESHOLDS:")
        for f in ["f1_volatility", "f2_parabolic", "f3_sma_stretch",
                   "f4_mom_conc", "f5_vol_accel", "f6_tenure"]:
            if f in roc_results:
                r = roc_results[f]
                print(f"    {f:<20s}: optimal={r['optimal_threshold']:.0f}  "
                      f"AUC={r['auc']:.3f}  J={r['youden_j']:.3f}")

    # Known meme stock validation
    print(f"\n  KNOWN MEME STOCK VALIDATION:")
    from universe_builder import uid_to_ticker
    for uid in df["uid"].unique():
        ticker = uid_to_ticker(uid)
        if ticker in KNOWN_MEME or ticker in KNOWN_BLOWUPS_HISTORICAL:
            ep = df[df["uid"] == uid]
            if len(ep) > 0:
                row = ep.iloc[0]
                print(f"    {ticker:<8s}: composite={row['composite']:.0f}  "
                      f"label={row['label']:<18s}  "
                      f"rally={row['rally_pct']:+.0%}  "
                      f"dd={row['subsequent_max_dd']:.0%}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Meme Score Calibration — Empirical Threshold Optimization")
    parser.add_argument("--top-n", type=int, default=150,
                        help="PIT universe size (default: 150)")
    parser.add_argument("--min-rally", type=float, default=1.0,
                        help="Minimum 3-month return to qualify as parabolic rally "
                             "(default: 1.0 = 100%%)")
    parser.add_argument("--save-csv", type=str, default=None,
                        help="Save labeled dataset to CSV file")
    args = parser.parse_args()

    print("=" * 80)
    print("MEME SCORE CALIBRATION — Empirical Threshold Optimization")
    print("=" * 80)
    print(f"  Config: top_n={args.top_n}, min_rally={args.min_rally:.0%}")
    print(f"  Current thresholds: exclude>{CURRENT_THRESHOLDS['meme_exclude']}, "
          f"max1>{CURRENT_THRESHOLDS['meme_max1']}, "
          f"max2>{CURRENT_THRESHOLDS['meme_max2']}")

    # Step 1: Load data
    print("\n" + "=" * 80)
    print("STEP 1: LOAD UNIVERSE DATA")
    print("=" * 80)
    rankings, prices, catalog = load_universe_data(top_n=args.top_n)

    # Step 2: Find parabolic rallies and classify
    print("\n" + "=" * 80)
    print("STEP 2: IDENTIFY & CLASSIFY PARABOLIC RALLIES")
    print("=" * 80)
    episodes = find_parabolic_rallies(
        prices, rankings, min_rally=args.min_rally)

    if len(episodes) < 10:
        print("ERROR: Too few parabolic rally episodes found. "
              "Try lowering --min-rally.")
        return

    # Step 3: Compute factor scores at peaks
    print("\n" + "=" * 80)
    print("STEP 3: COMPUTE FACTOR SCORES AT PEAK DATES")
    print("=" * 80)
    df = compute_factor_scores_at_peaks(episodes, prices, rankings, catalog)

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"  Saved labeled dataset to: {args.save_csv}")

    # Step 4: Distribution analysis
    dist_results = analyze_factor_distributions(df)

    # Step 5: ROC / threshold optimization
    roc_results = compute_roc_and_thresholds(df)

    # Step 6: Regime-stratified validation
    regime_analysis(df, roc_results)

    # Step 7: Band calibration
    calibrate_bands(df)

    # Step 8: Final report
    print_final_report(df, dist_results, roc_results)

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print("  This is a RESEARCH script. Do NOT modify waverider.py based on these")
    print("  results without review. Discuss findings before making threshold changes.")


if __name__ == "__main__":
    main()
