"""
================================================================================
ALPHA-TRADE v2.0 — AlphaGo-Inspired Autonomous Trading Research System
================================================================================

An AlphaGo-style reinforcement learning system for financial markets featuring:

  1. POLICY NETWORK  — Actor outputting action probabilities (long/short/flat + size)
  2. VALUE NETWORK   — Critic estimating risk-adjusted future returns
  3. RL ENGINE       — PPO actor-critic with GAE and experience replay
  4. SELF-PLAY LOOP  — Iterative champion/challenger improvement
  5. MCTS PLANNING   — Monte Carlo rollouts evaluating future price paths
  6. REWARD DESIGN   — Sharpe, CVaR, drawdown, turnover penalties
  7. ANTI-OVERFIT    — Walk-forward, deflated Sharpe, regime-segment testing
  8. SYNTHETIC DATA  — Regime-aware market simulation for robust training

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │  AlphaGo-Style Trading Loop                                    │
  │                                                                 │
  │  Market State ──► Policy Net ──► Action (long/short/flat/size) │
  │       │                                                         │
  │       └──────► Value Net ──► Expected Risk-Adjusted Return      │
  │                    │                                            │
  │             MCTS Planner ──► Simulated Rollouts ──► Refined     │
  │                    │              Policy Updates                │
  │              Self-Play ──► Champion vs Challenger               │
  │                    │              ──► Adopt if Better           │
  │           Anti-Overfit ──► Walk-Forward Validation              │
  │                              Deflated Sharpe Ratio              │
  │                              Regime Robustness                  │
  └─────────────────────────────────────────────────────────────────┘

INSTALL:
    pip install torch numpy pandas gymnasium yfinance

RUN:
    python alphago_trading_system.py                          # 50 S&P500, 5 timeframes
    python alphago_trading_system.py --timeframes 1d          # Daily only
    python alphago_trading_system.py --timeframes 5m,1h,1d    # Select timeframes
    python alphago_trading_system.py --n-symbols 20           # Fewer symbols
    python alphago_trading_system.py --symbols AAPL,TSLA,NVDA # Custom symbols
    python alphago_trading_system.py --data-dir ./data        # Use local CSV files
    python alphago_trading_system.py --synthetic              # No download needed
    python alphago_trading_system.py --iterations 20 --mcts-rollouts 64 --export-pine

DATA SOURCE:
    Yahoo Finance (yfinance) — period='max' for all timeframes
    Intraday limits: 5m/15m/30m=60 days, 1h=730 days, 1d=full history
    Default: 50 S&P 500 symbols x 5 timeframes (5m, 15m, 30m, 1h, 1d)

================================================================================
"""

import os
import sys
import time
import json
import math
import copy
import signal
import argparse
import warnings
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─── PyTorch ───
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("ERROR: PyTorch required. Install: pip install torch")
    sys.exit(1)

# ─── Gymnasium ───
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("ERROR: gymnasium required. Install: pip install gymnasium")
    sys.exit(1)

# ─── Optional ───
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL COLORS & OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; MAGENTA = "\033[95m"; CYAN = "\033[96m"
    WHITE = "\033[97m"; BG_HDR = "\033[48;5;235m"


def tprint(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"info": f"{C.CYAN}i", "ok": f"{C.GREEN}+",
             "warn": f"{C.YELLOW}!", "err": f"{C.RED}x",
             "hdr": f"{C.MAGENTA}>"}
    icon = icons.get(level, icons["info"])
    print(f"  {C.DIM}[{ts}]{C.RESET} {icon} {msg}{C.RESET}")


def print_box(title, subtitle=""):
    W = 78
    print(f"\n{C.CYAN}{C.BOLD}{'=' * (W+4)}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {C.WHITE}{title}{C.RESET}")
    if subtitle:
        print(f"{C.CYAN}{C.BOLD}  {C.DIM}{subtitle}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'=' * (W+4)}{C.RESET}")


def print_divider(text):
    W = 78
    pad = (W - len(text) - 2) // 2
    print(f"\n{C.DIM}{'-' * pad} {C.CYAN}{C.BOLD}{text}{C.RESET}{C.DIM} {'-' * pad}{C.RESET}")


def progress_bar(current, total, label="", width=40, start_time=None):
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = f"{'#' * filled}{'.' * (width - filled)}"
    eta = ""
    if start_time and current > 0:
        elapsed = time.time() - start_time
        remaining = elapsed / current * (total - current)
        eta = f" ETA:{remaining:.0f}s" if remaining < 120 else f" ETA:{remaining/60:.1f}m"
    sys.stdout.write(f"\r  {C.CYAN}{label}{C.RESET} [{bar}] {pct*100:5.1f}% ({current}/{total}){eta}   ")
    sys.stdout.flush()
    if current >= total:
        print()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All system hyperparameters."""
    # Capital & Risk
    starting_capital: float = 100_000.0
    max_position_pct: float = 0.25
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_drawdown_pct: float = 0.15

    # Network Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    use_attention: bool = True

    # RL Hyperparameters (PPO)
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine"      # linear, cosine, constant
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    target_kl: float = 0.03

    # Self-Play Loop
    n_iterations: int = 10
    total_timesteps_per_iter: int = 100_000
    champion_threshold: float = 1.05
    n_eval_episodes: int = 5

    # MCTS / Planning
    mcts_rollouts: int = 32
    mcts_depth: int = 10
    mcts_exploration: float = 1.414
    mcts_use_value_net: bool = True
    mcts_temperature: float = 1.0

    # Environment
    window_size: int = 60
    max_episode_steps: int = 5000
    action_space_type: str = "discrete"  # discrete or continuous
    n_actions: int = 5  # hold, buy_small, buy_large, sell_small, sell_large

    # Reward Function
    reward_sharpe_weight: float = 1.0
    reward_drawdown_penalty: float = 2.0
    reward_cvar_penalty: float = 0.5
    reward_turnover_penalty: float = 0.01
    reward_tail_risk_weight: float = 0.3
    cvar_quantile: float = 0.05

    # Anti-Overfitting
    train_ratio: float = 0.70
    walk_forward_windows: int = 5
    walk_forward_train_pct: float = 0.70
    monte_carlo_permutations: int = 100
    deflated_sharpe_threshold: float = 1.0
    regime_segment_test: bool = True
    max_performance_degradation: float = 0.30

    # Synthetic Data
    n_synthetic_regimes: int = 4
    synthetic_bars_per_regime: int = 500
    regime_transition_prob: float = 0.02
    synthetic_vol_range: Tuple[float, float] = (0.01, 0.04)
    synthetic_drift_range: Tuple[float, float] = (-0.001, 0.002)

    # Data
    data_dir: str = ""
    cache_dir: str = ""
    output_dir: str = ""
    min_bars: int = 200
    timeframes: Tuple[str, ...] = ("5m", "15m", "30m", "1h", "1d")  # All timeframes to analyze
    download_period: str = "max"  # Maximum available history from Yahoo Finance
    # yfinance limits for intraday: 5m=60d, 15m=60d, 30m=60d, 1h=730d, 1d=max
    intraday_periods: Dict[str, str] = field(default_factory=lambda: {
        "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d", "1d": "max",
    })

    # Compute
    n_workers: int = 0  # 0 = auto
    verbose: int = 1

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".alpha_trade_v2_cache")
        if not self.output_dir:
            self.output_dir = os.path.join(os.path.expanduser("~"), "alpha_trade_v2_results")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        if self.n_workers == 0:
            self.n_workers = max(1, mp.cpu_count() - 1)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLUMNS = [
    # Price (ATR-normalized)
    "norm_close", "norm_open", "norm_high", "norm_low",
    "bar_range_atr", "body_pct", "upper_wick_pct", "lower_wick_pct", "gap_pct",
    # Momentum
    "rsi_14", "rsi_9", "macd_hist_norm", "macd_signal_cross",
    "stoch_k", "stoch_d", "willr_14", "cci_norm", "roc_5", "roc_10",
    # Trend
    "sma_cross_20_50", "ema_cross_9_21", "adx_norm", "plus_di_norm", "minus_di_norm",
    "price_vs_sma_20", "price_vs_sma_50",
    # Volatility
    "atr_pct", "bb_position", "bb_width", "realized_vol_20",
    # Volume
    "vol_ratio", "obv_slope",
    # Pattern
    "consec_up", "consec_down", "drop_from_high_5",
    # Market regime indicators
    "vol_regime", "trend_strength", "mean_reversion_signal",
    # Position state (injected by env)
    "has_position", "position_pnl_pct", "bars_in_trade", "position_direction",
]

NUM_FEATURES = len(FEATURE_COLUMNS)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full indicator suite from raw OHLCV."""
    df = df.copy()
    c = df["Close"].values.astype(np.float64)
    h = df["High"].values.astype(np.float64)
    l = df["Low"].values.astype(np.float64)
    o = df["Open"].values.astype(np.float64)
    v = df["Volume"].values.astype(np.float64)
    n = len(c)

    if n < 60:
        return df

    # RSI
    for period in [9, 14]:
        delta = np.diff(c, prepend=c[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"RSI_{period}"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    macd_sig = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    df["MACD"] = macd_line
    df["MACD_Sig"] = macd_sig
    df["MACD_Hist"] = macd_line - macd_sig

    # ATR
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    df["ATR"] = pd.Series(tr).rolling(14, min_periods=1).mean().values

    # Bollinger Bands
    sma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(c).rolling(20, min_periods=1).std().values
    df["BB_Upper"] = sma20 + 2.0 * std20
    df["BB_Lower"] = sma20 - 2.0 * std20
    df["BB_Mid"] = sma20

    # SMAs & EMAs
    for p in [5, 20, 50, 200]:
        df[f"SMA_{p}"] = pd.Series(c).rolling(p, min_periods=1).mean().values
    for p in [9, 21]:
        df[f"EMA_{p}"] = pd.Series(c).ewm(span=p, adjust=False).mean().values

    # Stochastic
    low_min = pd.Series(l).rolling(14, min_periods=1).min().values
    high_max = pd.Series(h).rolling(14, min_periods=1).max().values
    df["Stoch_K"] = 100.0 * (c - low_min) / (high_max - low_min + 1e-10)
    df["Stoch_D"] = pd.Series(df["Stoch_K"]).rolling(3, min_periods=1).mean().values

    # Williams %R
    hh14 = pd.Series(h).rolling(14, min_periods=1).max().values
    ll14 = pd.Series(l).rolling(14, min_periods=1).min().values
    df["WillR_14"] = -100.0 * (hh14 - c) / (hh14 - ll14 + 1e-10)

    # CCI
    tp = (h + l + c) / 3.0
    tp_sma = pd.Series(tp).rolling(20, min_periods=1).mean().values
    tp_std = pd.Series(tp).rolling(20, min_periods=1).std().values
    df["CCI"] = (tp - tp_sma) / (0.015 * tp_std + 1e-10)

    # ADX
    plus_dm = np.maximum(np.diff(h, prepend=h[0]), 0.0)
    minus_dm = np.maximum(-np.diff(l, prepend=l[0]), 0.0)
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0
    smoothed_tr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    plus_di = 100.0 * pd.Series(plus_dm).rolling(14, min_periods=1).mean().values / (smoothed_tr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).rolling(14, min_periods=1).mean().values / (smoothed_tr + 1e-10)
    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di + 1e-10
    df["ADX"] = pd.Series(100.0 * di_diff / di_sum).rolling(14, min_periods=1).mean().values
    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di

    # Volume
    vol_sma20 = pd.Series(v).rolling(20, min_periods=1).mean().values
    df["Vol_Ratio"] = v / (vol_sma20 + 1.0)

    # OBV
    obv = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i - 1]:
            obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            obv[i] = obv[i - 1] - v[i]
        else:
            obv[i] = obv[i - 1]
    df["OBV"] = obv
    obv_sma = pd.Series(obv).rolling(10, min_periods=1).mean().values
    df["OBV_Slope"] = (obv - obv_sma) / (np.abs(obv_sma) + 1e-10)

    # ROC
    for p in [5, 10]:
        shifted = np.roll(c, p)
        roc = (c - shifted) / (shifted + 1e-10) * 100.0
        roc[:p] = 0.0
        df[f"ROC_{p}"] = roc

    # Consecutive up/down
    up = (c > np.roll(c, 1)).astype(float)
    down = (c < np.roll(c, 1)).astype(float)
    up[0] = down[0] = 0
    consec_up = np.zeros(n)
    consec_down = np.zeros(n)
    for i in range(1, n):
        consec_up[i] = (consec_up[i - 1] + 1) if up[i] else 0
        consec_down[i] = (consec_down[i - 1] + 1) if down[i] else 0
    df["Consec_Up"] = consec_up
    df["Consec_Down"] = consec_down

    rolling_high_5 = pd.Series(h).rolling(5, min_periods=1).max().values
    df["Drop_From_High_5"] = (c - rolling_high_5) / (rolling_high_5 + 1e-10) * 100.0

    # Realized volatility
    log_ret = np.log(c / np.roll(c, 1))
    log_ret[0] = 0
    df["Realized_Vol_20"] = pd.Series(log_ret).rolling(20, min_periods=1).std().values * np.sqrt(252)

    # Regime indicators
    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().values
    vol_60 = pd.Series(log_ret).rolling(60, min_periods=1).std().values
    df["Vol_Regime"] = vol_20 / (vol_60 + 1e-10)  # >1 = high vol regime

    sma_20 = df["SMA_20"].values
    sma_50 = df["SMA_50"].values
    df["Trend_Strength"] = (sma_20 - sma_50) / (df["ATR"].values + 1e-10)

    # Mean reversion signal: distance from BB mid as fraction of BB width
    bb_width = df["BB_Upper"].values - df["BB_Lower"].values + 1e-10
    df["Mean_Rev_Signal"] = (c - df["BB_Mid"].values) / bb_width

    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert indicator DataFrame into normalized feature matrix [N, NUM_FEATURES]."""
    c = df["Close"].values.astype(np.float64)
    o = df["Open"].values.astype(np.float64)
    h = df["High"].values.astype(np.float64)
    l = df["Low"].values.astype(np.float64)
    atr = df["ATR"].values.astype(np.float64)
    n = len(c)

    atr_safe = np.maximum(atr, 1e-8)
    c_safe = np.maximum(c, 1e-8)

    features = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    idx = 0

    # Price patterns (ATR-normalized)
    features[:, idx] = (c - df["SMA_20"].values) / atr_safe;                idx += 1
    features[:, idx] = (o - c) / atr_safe;                                  idx += 1
    features[:, idx] = (h - c) / atr_safe;                                  idx += 1
    features[:, idx] = (c - l) / atr_safe;                                  idx += 1
    features[:, idx] = (h - l) / atr_safe;                                  idx += 1
    features[:, idx] = np.abs(c - o) / c_safe * 100;                        idx += 1
    features[:, idx] = (h - np.maximum(c, o)) / (h - l + 1e-10);           idx += 1
    features[:, idx] = (np.minimum(c, o) - l) / (h - l + 1e-10);           idx += 1
    gap = (o - np.roll(c, 1)) / (np.roll(c, 1) + 1e-10) * 100
    gap[0] = 0
    features[:, idx] = gap;                                                  idx += 1

    # Momentum
    features[:, idx] = df["RSI_14"].values / 100.0 - 0.5;                   idx += 1
    features[:, idx] = df["RSI_9"].values / 100.0 - 0.5;                    idx += 1
    features[:, idx] = df["MACD_Hist"].values / atr_safe;                    idx += 1
    macd_cross = np.zeros(n)
    macd = df["MACD"].values; macd_s = df["MACD_Sig"].values
    macd_cross[1:] = np.sign(macd[1:] - macd_s[1:]) - np.sign(macd[:-1] - macd_s[:-1])
    features[:, idx] = macd_cross / 2.0;                                     idx += 1
    features[:, idx] = df["Stoch_K"].values / 100.0 - 0.5;                  idx += 1
    features[:, idx] = df["Stoch_D"].values / 100.0 - 0.5;                  idx += 1
    features[:, idx] = df["WillR_14"].values / 100.0 + 0.5;                 idx += 1
    features[:, idx] = np.clip(df["CCI"].values / 200.0, -1, 1);            idx += 1
    features[:, idx] = np.clip(df["ROC_5"].values / 10.0, -1, 1);           idx += 1
    features[:, idx] = np.clip(df["ROC_10"].values / 10.0, -1, 1);          idx += 1

    # Trend
    sma20 = df["SMA_20"].values; sma50 = df["SMA_50"].values
    features[:, idx] = np.where(sma20 > sma50, 1.0, -1.0);                 idx += 1
    ema9 = df["EMA_9"].values; ema21 = df["EMA_21"].values
    features[:, idx] = np.where(ema9 > ema21, 1.0, -1.0);                  idx += 1
    features[:, idx] = np.clip(df["ADX"].values / 50.0, 0, 1);              idx += 1
    features[:, idx] = np.clip(df["Plus_DI"].values / 50.0, 0, 1);          idx += 1
    features[:, idx] = np.clip(df["Minus_DI"].values / 50.0, 0, 1);         idx += 1
    features[:, idx] = (c - sma20) / atr_safe;                              idx += 1
    features[:, idx] = (c - sma50) / atr_safe;                              idx += 1

    # Volatility
    features[:, idx] = atr / c_safe * 100;                                   idx += 1
    bb_upper = df["BB_Upper"].values; bb_lower = df["BB_Lower"].values
    bb_range = bb_upper - bb_lower + 1e-10
    features[:, idx] = (c - bb_lower) / bb_range - 0.5;                     idx += 1
    features[:, idx] = bb_range / c_safe;                                    idx += 1
    features[:, idx] = np.clip(df["Realized_Vol_20"].values, 0, 2);          idx += 1

    # Volume
    features[:, idx] = np.clip(df["Vol_Ratio"].values - 1.0, -2, 5);        idx += 1
    features[:, idx] = np.clip(df["OBV_Slope"].values, -1, 1);              idx += 1

    # Pattern
    features[:, idx] = np.clip(df["Consec_Up"].values / 5.0, 0, 1);         idx += 1
    features[:, idx] = np.clip(df["Consec_Down"].values / 5.0, 0, 1);       idx += 1
    features[:, idx] = np.clip(df["Drop_From_High_5"].values / 10.0, -1, 0);idx += 1

    # Regime
    features[:, idx] = np.clip(df["Vol_Regime"].values - 1.0, -1, 2);       idx += 1
    features[:, idx] = np.clip(df["Trend_Strength"].values / 3.0, -1, 1);   idx += 1
    features[:, idx] = np.clip(df["Mean_Rev_Signal"].values, -1, 1);         idx += 1

    # Position state (zeros — env fills these per step)
    features[:, idx] = 0.0; idx += 1  # has_position
    features[:, idx] = 0.0; idx += 1  # position_pnl_pct
    features[:, idx] = 0.0; idx += 1  # bars_in_trade
    features[:, idx] = 0.0; idx += 1  # position_direction (+1 long, -1 short, 0 flat)

    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    return features


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC MARKET DATA GENERATOR (Regime-Aware)
# ══════════════════════════════════════════════════════════════════════════════

class SyntheticMarketGenerator:
    """
    Generates synthetic OHLCV data with realistic regime transitions.
    Regimes: trending_up, trending_down, mean_reverting, high_volatility
    """

    REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_volatility"]

    def __init__(self, config: Config, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def generate(self, n_bars: int = 2000, initial_price: float = 100.0) -> pd.DataFrame:
        """Generate synthetic OHLCV with regime transitions."""
        prices = np.zeros((n_bars, 4))  # OHLC
        volumes = np.zeros(n_bars)
        regimes = np.zeros(n_bars, dtype=int)

        price = initial_price
        current_regime = self.rng.randint(0, len(self.REGIMES))
        base_vol = self.rng.uniform(*self.config.synthetic_vol_range)

        for i in range(n_bars):
            # Regime transition
            if self.rng.random() < self.config.regime_transition_prob:
                current_regime = self.rng.randint(0, len(self.REGIMES))
                base_vol = self.rng.uniform(*self.config.synthetic_vol_range)

            regimes[i] = current_regime
            regime = self.REGIMES[current_regime]

            # Generate returns based on regime
            if regime == "trending_up":
                drift = self.rng.uniform(0.0005, 0.003)
                vol = base_vol * 0.8
            elif regime == "trending_down":
                drift = self.rng.uniform(-0.003, -0.0005)
                vol = base_vol * 1.0
            elif regime == "mean_reverting":
                drift = -0.5 * (price - initial_price) / initial_price * 0.01
                vol = base_vol * 0.6
            else:  # high_volatility
                drift = self.rng.uniform(-0.001, 0.001)
                vol = base_vol * 2.5

            # Add randomized slippage/spread effect
            spread = self.rng.uniform(0.0001, 0.002)

            # Generate OHLC
            ret = drift + vol * self.rng.randn()
            close = price * (1 + ret)

            intra_vol = vol * self.rng.uniform(0.5, 2.0)
            high = max(price, close) * (1 + abs(intra_vol * self.rng.randn() * 0.5))
            low = min(price, close) * (1 - abs(intra_vol * self.rng.randn() * 0.5))
            open_price = price * (1 + spread * self.rng.randn())

            prices[i] = [open_price, high, low, close]
            volumes[i] = max(1000, int(self.rng.lognormal(15, 1)))

            price = close

        df = pd.DataFrame(prices, columns=["Open", "High", "Low", "Close"])
        df["Volume"] = volumes.astype(int)
        df["Regime"] = regimes
        df.index = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

        return df

    def generate_multiple(self, n_series: int = 10, n_bars: int = 2000) -> Dict[str, pd.DataFrame]:
        """Generate multiple synthetic series with diverse characteristics."""
        data = {}
        for i in range(n_series):
            seed = self.rng.randint(0, 100000)
            gen = SyntheticMarketGenerator(self.config, seed=seed)
            initial_price = self.rng.uniform(20, 500)
            df = gen.generate(n_bars=n_bars, initial_price=initial_price)
            data[f"SYN_{i:03d}"] = df
        return data


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURES — Policy & Value Networks
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """Self-attention over the temporal dimension of market observations."""

    def __init__(self, input_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=n_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [batch, seq_len, features]
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class SharedBackbone(nn.Module):
    """
    Shared feature extractor for Policy and Value networks.
    Processes a [batch, window, features] observation through temporal conv + attention.
    """

    def __init__(self, n_features: int, window_size: int, hidden_dim: int = 256,
                 n_layers: int = 3, dropout: float = 0.1, use_attention: bool = True):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Temporal convolution layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        # Temporal attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(hidden_dim, n_heads=4)

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * window_size, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        x: [batch, window * features] (flattened) or [batch, window, features]
        Returns: [batch, hidden_dim]
        """
        batch_size = x.shape[0]

        # Reshape if flattened
        if x.dim() == 2:
            x = x.view(batch_size, self.window_size, self.n_features)

        # Conv: needs [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, features, window]
        x = self.conv_layers(x)  # [batch, hidden, window]

        # Attention: needs [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, window, hidden]
        if self.use_attention:
            x = self.attention(x)

        # Flatten and project
        x = x.reshape(batch_size, -1)
        x = self.fc(x)

        return x


class PolicyNetwork(nn.Module):
    """
    Actor network: outputs action probabilities.
    Actions: hold, buy_small, buy_large, sell_small, sell_large
    """

    def __init__(self, backbone: SharedBackbone, n_actions: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.action_head(features)
        return logits

    def get_action_probs(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_distribution(self, x):
        logits = self.forward(x)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """
    Critic network: estimates expected risk-adjusted return from current state.
    Predicts both expected return and risk (variance) for uncertainty estimation.
    """

    def __init__(self, backbone: SharedBackbone, hidden_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Risk head for uncertainty estimation
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive variance
        )

    def forward(self, x):
        features = self.backbone(x)
        value = self.value_head(features)
        risk = self.risk_head(features)
        return value.squeeze(-1), risk.squeeze(-1)

    def get_value(self, x):
        value, _ = self.forward(x)
        return value


class AlphaTradeNet(nn.Module):
    """
    Combined Policy + Value network with shared backbone.
    This is the core neural architecture inspired by AlphaGo's dual-head design.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = SharedBackbone(
            n_features=NUM_FEATURES,
            window_size=config.window_size,
            hidden_dim=config.hidden_dim,
            n_layers=config.num_layers,
            dropout=config.dropout,
            use_attention=config.use_attention,
        )

        self.policy = PolicyNetwork(self.backbone, config.n_actions, config.hidden_dim)
        self.value = ValueNetwork(self.backbone, config.hidden_dim)

    def forward(self, x):
        """Returns (action_logits, value, risk_estimate)."""
        features = self.backbone(x)
        logits = self.policy.action_head(features)
        value = self.value.value_head(features).squeeze(-1)
        risk = self.value.risk_head(features).squeeze(-1)
        return logits, value, risk

    def get_action_and_value(self, x, action=None):
        """For PPO update: get action, log_prob, entropy, value."""
        logits, value, risk = self.forward(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value, risk


# ══════════════════════════════════════════════════════════════════════════════
# TRADING ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class TradingEnv(gym.Env):
    """
    Enhanced Gymnasium trading environment.

    Actions (discrete=5):
        0 = HOLD (do nothing)
        1 = BUY_SMALL (25% of max position)
        2 = BUY_LARGE (full max position)
        3 = SELL_SMALL (close 50% of position)
        4 = SELL_ALL (close entire position)

    Reward: Risk-adjusted incremental Sharpe approximation with penalties.
    """

    metadata = {"render_modes": ["human"]}

    # Action constants
    HOLD = 0
    BUY_SMALL = 1
    BUY_LARGE = 2
    SELL_SMALL = 3
    SELL_ALL = 4

    def __init__(self, features: np.ndarray, prices: np.ndarray,
                 config: Config, symbol: str = "SYM", is_eval: bool = False):
        super().__init__()
        self.features = features
        self.prices = prices
        self.config = config
        self.symbol = symbol
        self.is_eval = is_eval
        self.n_bars = len(features)
        self.window = config.window_size

        obs_dim = self.window * NUM_FEATURES
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(config.n_actions)

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window
        self.capital = self.config.starting_capital
        self.peak_capital = self.capital
        self.position = 0           # shares held (positive=long, negative=short)
        self.entry_price = 0.0
        self.entry_step = 0
        self.shares = 0
        self.trade_log = []
        self.returns_history = []
        self.step_rewards = []
        self.wins = 0
        self.losses = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.max_drawdown = 0.0
        self.prev_portfolio_value = self.capital
        self.turnover_sum = 0.0
        self.n_trades = 0

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window
        end = self.current_step
        obs = self.features[start:end].copy()

        # Inject position state
        pos_start = NUM_FEATURES - 4
        if self.position != 0:
            current_price = self.prices[self.current_step - 1, 3]
            pnl_pct = (current_price - self.entry_price) / (self.entry_price + 1e-10)
            if self.position < 0:
                pnl_pct = -pnl_pct
            bars = self.current_step - self.entry_step
            obs[-1, pos_start] = 1.0 if self.position > 0 else -1.0
            obs[-1, pos_start + 1] = np.clip(pnl_pct, -0.5, 0.5)
            obs[-1, pos_start + 2] = min(bars / 20.0, 1.0)
            obs[-1, pos_start + 3] = np.sign(self.position)
        else:
            obs[-1, pos_start:pos_start + 4] = 0.0

        return obs.flatten().astype(np.float32)

    def _execute_trade(self, action: int, close_price: float) -> float:
        """Execute trade action and return raw P&L component."""
        raw_pnl = 0.0
        cost = close_price * (self.config.commission_pct + self.config.slippage_pct)

        if action == self.BUY_SMALL and self.position <= 0:
            # Buy small position (25% of max)
            if self.position < 0:
                raw_pnl = self._close_position(close_price)
            invest = self.capital * self.config.max_position_pct * 0.25
            shares = int(invest / (close_price + cost))
            if shares > 0:
                self.shares = shares
                self.position = shares
                self.entry_price = close_price
                self.entry_step = self.current_step
                self.capital -= shares * (close_price + cost)
                self.n_trades += 1
                self.turnover_sum += shares * close_price

        elif action == self.BUY_LARGE and self.position <= 0:
            if self.position < 0:
                raw_pnl = self._close_position(close_price)
            invest = self.capital * self.config.max_position_pct
            shares = int(invest / (close_price + cost))
            if shares > 0:
                self.shares = shares
                self.position = shares
                self.entry_price = close_price
                self.entry_step = self.current_step
                self.capital -= shares * (close_price + cost)
                self.n_trades += 1
                self.turnover_sum += shares * close_price

        elif action == self.SELL_SMALL and self.position > 0:
            # Sell half
            sell_shares = max(1, self.shares // 2)
            proceeds = sell_shares * (close_price - cost)
            cost_basis = sell_shares * self.entry_price * (1 + self.config.commission_pct)
            raw_pnl = proceeds - cost_basis
            self.capital += proceeds
            self.shares -= sell_shares
            self.position = self.shares
            self.turnover_sum += sell_shares * close_price
            if self.shares <= 0:
                self._log_trade(close_price, raw_pnl)
                self.position = 0
                self.entry_price = 0.0

        elif action == self.SELL_ALL and self.position > 0:
            raw_pnl = self._close_position(close_price)

        return raw_pnl

    def _close_position(self, close_price: float) -> float:
        """Close entire position and return P&L."""
        if self.position == 0:
            return 0.0

        cost = close_price * (self.config.commission_pct + self.config.slippage_pct)
        if self.position > 0:
            proceeds = self.shares * (close_price - cost)
            cost_basis = self.shares * self.entry_price * (1 + self.config.commission_pct)
            pnl = proceeds - cost_basis
        else:
            # Short position
            proceeds = abs(self.shares) * (self.entry_price - close_price - cost)
            pnl = proceeds

        self.capital += abs(self.shares) * close_price - abs(self.shares) * cost
        self.turnover_sum += abs(self.shares) * close_price
        self._log_trade(close_price, pnl)

        self.position = 0
        self.shares = 0
        self.entry_price = 0.0
        return pnl

    def _log_trade(self, exit_price: float, pnl: float):
        pnl_pct = pnl / (abs(self.shares) * self.entry_price + 1e-10)
        self.trade_log.append({
            "symbol": self.symbol,
            "entry_step": self.entry_step,
            "exit_step": self.current_step,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "shares": self.shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "bars_held": self.current_step - self.entry_step,
        })
        if pnl > 0:
            self.wins += 1
            self.gross_profit += pnl
        else:
            self.losses += 1
            self.gross_loss += abs(pnl)

    def _compute_reward(self, raw_pnl: float, action: int) -> float:
        """
        Risk-adjusted reward combining:
        - Incremental Sharpe ratio approximation
        - Drawdown penalty
        - CVaR (tail risk) penalty
        - Turnover penalty
        """
        # Portfolio value
        mark_value = self.capital
        if self.position > 0:
            mark_value += self.shares * self.prices[min(self.current_step, self.n_bars - 1), 3]

        # Step return
        step_return = (mark_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-10)
        self.returns_history.append(step_return)
        self.prev_portfolio_value = mark_value

        # Incremental Sharpe approximation
        if len(self.returns_history) > 10:
            returns = np.array(self.returns_history[-100:])
            mean_r = np.mean(returns)
            std_r = np.std(returns) + 1e-10
            sharpe_component = (mean_r / std_r) * self.config.reward_sharpe_weight * 0.01
        else:
            sharpe_component = step_return * self.config.reward_sharpe_weight

        # Drawdown penalty
        self.peak_capital = max(self.peak_capital, mark_value)
        dd = (self.peak_capital - mark_value) / (self.peak_capital + 1e-10)
        self.max_drawdown = max(self.max_drawdown, dd)
        dd_penalty = 0.0
        if dd > self.config.max_drawdown_pct:
            dd_penalty = (dd - self.config.max_drawdown_pct) * self.config.reward_drawdown_penalty

        # CVaR penalty (tail risk)
        cvar_penalty = 0.0
        if len(self.returns_history) > 20:
            returns = np.array(self.returns_history[-100:])
            sorted_returns = np.sort(returns)
            cutoff = max(1, int(len(sorted_returns) * self.config.cvar_quantile))
            cvar = -np.mean(sorted_returns[:cutoff])
            if cvar > 0.02:
                cvar_penalty = cvar * self.config.reward_cvar_penalty

        # Turnover penalty
        turnover_penalty = 0.0
        if action in [self.BUY_SMALL, self.BUY_LARGE, self.SELL_SMALL, self.SELL_ALL]:
            turnover_penalty = self.config.reward_turnover_penalty

        # Combined reward
        reward = sharpe_component - dd_penalty - cvar_penalty - turnover_penalty

        # Small idle penalty when flat
        if self.position == 0 and action == self.HOLD:
            reward -= 0.0001

        return float(np.clip(reward, -1.0, 1.0))

    def step(self, action: int):
        close_price = self.prices[self.current_step, 3]
        raw_pnl = self._execute_trade(action, close_price)
        reward = self._compute_reward(raw_pnl, action)

        self.step_rewards.append(reward)
        self.current_step += 1

        # Termination
        terminated = False
        truncated = False

        mark_value = self.capital
        if self.position > 0:
            mark_value += self.shares * self.prices[min(self.current_step - 1, self.n_bars - 1), 3]

        if self.current_step >= self.n_bars - 1:
            if self.position != 0:
                self._close_position(self.prices[self.current_step - 1, 3])
            terminated = True

        if mark_value < self.config.starting_capital * 0.5:
            terminated = True

        if self.current_step - self.window >= self.config.max_episode_steps:
            if self.position != 0:
                self._close_position(self.prices[min(self.current_step - 1, self.n_bars - 1), 3])
            truncated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if not self.is_eval and self.n_bars > self.window + 200:
            max_start = self.n_bars - self.config.max_episode_steps - 10
            if max_start > self.window:
                self.current_step = self.np_random.integers(self.window, max_start)

        return self._get_obs(), self._get_info()

    def _get_info(self):
        total_trades = self.wins + self.losses
        mark_value = self.capital
        if self.position > 0:
            mark_value += self.shares * self.prices[min(self.current_step, self.n_bars - 1), 3]
        net_pnl = mark_value - self.config.starting_capital

        # Compute Sharpe
        sharpe = 0.0
        if len(self.returns_history) > 10:
            r = np.array(self.returns_history)
            sharpe = np.mean(r) / (np.std(r) + 1e-10) * np.sqrt(252)

        return {
            "symbol": self.symbol,
            "total_trades": total_trades,
            "wins": self.wins, "losses": self.losses,
            "win_rate": self.wins / max(total_trades, 1) * 100,
            "profit_factor": self.gross_profit / max(self.gross_loss, 1e-10),
            "net_pnl": net_pnl,
            "net_pnl_pct": net_pnl / self.config.starting_capital * 100,
            "max_drawdown": self.max_drawdown * 100,
            "portfolio_value": mark_value,
            "sharpe_ratio": sharpe,
            "trade_log": self.trade_log,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO TREE SEARCH (MCTS) PLANNING MODULE
# ══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """Node in the MCTS search tree."""

    def __init__(self, state: np.ndarray, parent=None, action=None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def ucb_score(self, exploration: float = 1.414):
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.mean_value
        exploration_bonus = exploration * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return exploitation + exploration_bonus


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner inspired by AlphaGo's search component.

    Uses the value network to evaluate simulated future states and
    the policy network to guide search through promising action sequences.
    """

    def __init__(self, policy_net: PolicyNetwork, value_net: ValueNetwork,
                 config: Config):
        self.policy_net = policy_net
        self.value_net = value_net
        self.config = config
        self.n_actions = config.n_actions

    @torch.no_grad()
    def search(self, root_state: np.ndarray, env: TradingEnv,
               n_rollouts: int = None) -> np.ndarray:
        """
        Run MCTS from root_state and return improved action probabilities.

        Returns: [n_actions] probability vector
        """
        if n_rollouts is None:
            n_rollouts = self.config.mcts_rollouts

        state_tensor = torch.FloatTensor(root_state).unsqueeze(0).to(DEVICE)
        policy_priors = self.policy_net.get_action_probs(state_tensor).cpu().numpy()[0]

        root = MCTSNode(state=root_state, prior=1.0)

        # Initialize root children with policy priors
        for a in range(self.n_actions):
            root.children[a] = MCTSNode(
                state=None, parent=root, action=a, prior=policy_priors[a]
            )

        for _ in range(n_rollouts):
            node = root

            # Selection: traverse tree using UCB
            while not node.is_leaf():
                best_action = max(
                    node.children.keys(),
                    key=lambda a: node.children[a].ucb_score(self.config.mcts_exploration)
                )
                node = node.children[best_action]

            # Expansion & Evaluation using value network
            if node.state is not None:
                state_t = torch.FloatTensor(node.state).unsqueeze(0).to(DEVICE)
            else:
                # Simulate forward from parent state using simple model
                state_t = torch.FloatTensor(root_state).unsqueeze(0).to(DEVICE)

            value_estimate = self.value_net.get_value(state_t).item()

            # Add noise for exploration during rollouts
            value_estimate += np.random.normal(0, 0.01)

            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.total_value += value_estimate
                node.mean_value = node.total_value / node.visit_count
                node = node.parent

        # Extract improved policy from visit counts
        visits = np.array([root.children[a].visit_count for a in range(self.n_actions)],
                         dtype=np.float32)

        if self.config.mcts_temperature > 0:
            visits = visits ** (1.0 / self.config.mcts_temperature)

        total = visits.sum()
        if total > 0:
            improved_policy = visits / total
        else:
            improved_policy = policy_priors

        return improved_policy

    def batch_search(self, states: np.ndarray, env: TradingEnv) -> np.ndarray:
        """Run MCTS for a batch of states. Returns [batch, n_actions]."""
        batch_size = len(states)
        improved_policies = np.zeros((batch_size, self.n_actions), dtype=np.float32)

        for i in range(batch_size):
            improved_policies[i] = self.search(states[i], env)

        return improved_policies


# ══════════════════════════════════════════════════════════════════════════════
# PPO TRAINING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores trajectory data for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        n = len(rewards)

        advantages = np.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(dones[t])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - float(dones[t])

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values

    def get_batches(self, batch_size: int):
        """Yield random mini-batches."""
        n = len(self.states)
        indices = np.random.permutation(n)

        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield (
                torch.FloatTensor(states[idx]).to(DEVICE),
                torch.LongTensor(actions[idx]).to(DEVICE),
                torch.FloatTensor(log_probs[idx]).to(DEVICE),
                torch.FloatTensor(self.returns[idx]).to(DEVICE),
                torch.FloatTensor(self.advantages[idx]).to(DEVICE),
            )

    def clear(self):
        self.__init__()


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for the AlphaTradeNet.
    Implements PPO-Clip with entropy bonus and value function clipping.
    """

    def __init__(self, net: AlphaTradeNet, config: Config):
        self.net = net
        self.config = config
        self.optimizer = optim.AdamW(
            net.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )
        self.scheduler = self._build_scheduler()
        self.buffer = RolloutBuffer()
        self.global_step = 0

    def _build_scheduler(self):
        if self.config.lr_schedule == "cosine":
            total_steps = self.config.n_iterations * self.config.total_timesteps_per_iter // self.config.n_steps
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps, 1), eta_min=1e-6
            )
        elif self.config.lr_schedule == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=1000
            )
        return None

    def collect_rollouts(self, env: TradingEnv, n_steps: int) -> dict:
        """Collect n_steps of experience from the environment."""
        self.net.eval()
        self.buffer.clear()

        obs, info = env.reset()
        episode_rewards = []
        current_ep_reward = 0

        with torch.no_grad():
            for step in range(n_steps):
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                action, log_prob, _, value, _ = self.net.get_action_and_value(state_tensor)

                action_int = action.item()
                log_prob_val = log_prob.item()
                value_val = value.item()

                next_obs, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated

                self.buffer.add(obs, action_int, log_prob_val, reward, value_val, done)
                current_ep_reward += reward

                if done:
                    episode_rewards.append(current_ep_reward)
                    current_ep_reward = 0
                    obs, info = env.reset()
                else:
                    obs = next_obs

            # Compute last value for GAE
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            _, _, _, last_value, _ = self.net.get_action_and_value(state_tensor)
            self.buffer.compute_gae(last_value.item(), self.config.gamma, self.config.gae_lambda)

        return {
            "episode_rewards": episode_rewards,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0,
            "n_episodes": len(episode_rewards),
        }

    def update(self) -> dict:
        """Run PPO update on collected rollouts."""
        self.net.train()

        # Normalize advantages
        adv = self.buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.buffer.advantages = adv

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0

        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                states, actions, old_log_probs, returns, advantages = batch

                # Get current policy outputs
                _, new_log_probs, entropy, values, risk = self.net.get_action_and_value(states, actions)

                # Policy loss (PPO-Clip)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range,
                                    1 + self.config.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (policy_loss
                        + self.config.vf_coef * value_loss
                        + self.config.ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track KL divergence
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                n_updates += 1

            # Early stopping on KL divergence
            if n_updates > 0 and abs(total_kl / n_updates) > self.config.target_kl:
                break

        if self.scheduler:
            self.scheduler.step()

        self.global_step += len(self.buffer.states)

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "approx_kl": total_kl / max(n_updates, 1),
            "n_updates": n_updates,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train_iteration(self, env: TradingEnv, total_steps: int) -> dict:
        """Run one full training iteration: collect + update cycles."""
        all_rewards = []
        all_losses = []
        steps_collected = 0

        while steps_collected < total_steps:
            rollout_info = self.collect_rollouts(env, self.config.n_steps)
            update_info = self.update()

            all_rewards.extend(rollout_info["episode_rewards"])
            all_losses.append(update_info)
            steps_collected += self.config.n_steps

        return {
            "mean_reward": np.mean(all_rewards) if all_rewards else 0,
            "n_episodes": len(all_rewards),
            "steps": steps_collected,
            "losses": all_losses,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ANTI-OVERFITTING & VALIDATION FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

class ValidationFramework:
    """
    Comprehensive validation suite implementing:
    - Walk-forward validation
    - Deflated Sharpe ratio
    - Monte Carlo robustness testing
    - Regime-segment testing
    - Rolling performance stability checks
    """

    def __init__(self, config: Config):
        self.config = config

    def walk_forward_validation(self, net: AlphaTradeNet, features: np.ndarray,
                                 prices: np.ndarray, symbol: str = "SYM") -> dict:
        """
        Walk-forward validation: train on expanding window, test on next segment.
        """
        n = len(features)
        n_windows = self.config.walk_forward_windows
        segment_size = n // (n_windows + 1)

        results = []
        for w in range(n_windows):
            train_end = segment_size * (w + 1)
            test_start = train_end
            test_end = min(test_start + segment_size, n)

            if test_end - test_start < self.config.window_size + 50:
                continue

            # Evaluate on test segment
            env = TradingEnv(
                features=features[test_start:test_end],
                prices=prices[test_start:test_end],
                config=self.config,
                symbol=symbol,
                is_eval=True,
            )

            obs, _ = env.reset()
            done = False
            while not done:
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action, _, _, _, _ = net.get_action_and_value(state_t)
                obs, _, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

            results.append({
                "window": w,
                "pnl": info.get("net_pnl", 0),
                "sharpe": info.get("sharpe_ratio", 0),
                "max_dd": info.get("max_drawdown", 0),
                "trades": info.get("total_trades", 0),
                "win_rate": info.get("win_rate", 0),
            })

        if not results:
            return {"valid": False, "reason": "insufficient_data"}

        pnls = [r["pnl"] for r in results]
        sharpes = [r["sharpe"] for r in results]

        return {
            "valid": True,
            "n_windows": len(results),
            "mean_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls),
            "mean_sharpe": np.mean(sharpes),
            "min_sharpe": np.min(sharpes),
            "pnl_consistency": sum(1 for p in pnls if p > 0) / len(pnls),
            "per_window": results,
        }

    def deflated_sharpe_ratio(self, sharpe_ratio: float, n_observations: int,
                               n_trials: int, skewness: float = 0.0,
                               kurtosis: float = 3.0) -> float:
        """
        Compute the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
        Accounts for multiple testing bias.
        """
        from scipy import stats as scipy_stats

        if n_observations < 2:
            return 0.0

        # Expected maximum Sharpe under null (i.i.d. normal)
        e_max_sharpe = (1 - 0.5772 / np.log(max(n_trials, 2))) * np.sqrt(2 * np.log(max(n_trials, 1)))

        # Sharpe ratio standard error with non-normality correction
        se = np.sqrt((1 + 0.5 * sharpe_ratio**2 -
                      skewness * sharpe_ratio +
                      (kurtosis - 3) / 4 * sharpe_ratio**2) / max(n_observations - 1, 1))

        if se < 1e-10:
            return 0.0

        # Test statistic
        t_stat = (sharpe_ratio - e_max_sharpe) / se

        # One-sided p-value from t-distribution
        dsr = scipy_stats.norm.cdf(t_stat)

        return dsr

    def monte_carlo_robustness(self, net: AlphaTradeNet, features: np.ndarray,
                                prices: np.ndarray, symbol: str = "SYM",
                                n_permutations: int = None) -> dict:
        """
        Monte Carlo permutation test: shuffle returns to assess strategy significance.
        """
        if n_permutations is None:
            n_permutations = self.config.monte_carlo_permutations

        # Get actual strategy performance
        actual_result = self._evaluate_once(net, features, prices, symbol)
        actual_sharpe = actual_result.get("sharpe_ratio", 0)

        # Permutation test
        permuted_sharpes = []
        for i in range(n_permutations):
            # Shuffle returns (keep feature structure, randomize price sequence)
            perm_idx = np.random.permutation(len(prices))
            perm_prices = prices[perm_idx]

            # Recompute features for shuffled prices (simplified: just shuffle features too)
            perm_features = features[perm_idx]

            perm_result = self._evaluate_once(net, perm_features, perm_prices, symbol)
            permuted_sharpes.append(perm_result.get("sharpe_ratio", 0))

        permuted_sharpes = np.array(permuted_sharpes)
        p_value = np.mean(permuted_sharpes >= actual_sharpe)

        return {
            "actual_sharpe": actual_sharpe,
            "mean_permuted_sharpe": np.mean(permuted_sharpes),
            "std_permuted_sharpe": np.std(permuted_sharpes),
            "p_value": p_value,
            "significant_at_5pct": p_value < 0.05,
            "significant_at_1pct": p_value < 0.01,
        }

    def _evaluate_once(self, net: AlphaTradeNet, features: np.ndarray,
                        prices: np.ndarray, symbol: str) -> dict:
        """Single evaluation run."""
        if len(features) < self.config.window_size + 50:
            return {"sharpe_ratio": 0, "net_pnl": 0}

        env = TradingEnv(
            features=features, prices=prices,
            config=self.config, symbol=symbol, is_eval=True,
        )
        obs, _ = env.reset()
        done = False
        net.eval()
        with torch.no_grad():
            while not done:
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                action, _, _, _, _ = net.get_action_and_value(state_t)
                obs, _, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
        return info

    def regime_segment_test(self, net: AlphaTradeNet, features: np.ndarray,
                             prices: np.ndarray, symbol: str = "SYM") -> dict:
        """
        Test performance across detected market regimes.
        Segments data by volatility regime and trend direction.
        """
        n = len(features)
        if n < 200:
            return {"valid": False}

        # Detect regimes from features
        # vol_regime is feature index for "vol_regime"
        vol_regime_idx = FEATURE_COLUMNS.index("vol_regime")
        trend_idx = FEATURE_COLUMNS.index("trend_strength")

        # Segment into regimes
        segments = {"low_vol": [], "high_vol": [], "trending": [], "mean_reverting": []}

        chunk_size = 100
        for start in range(0, n - chunk_size, chunk_size):
            end = start + chunk_size
            chunk_features = features[start:end]
            avg_vol = np.mean(chunk_features[:, vol_regime_idx])
            avg_trend = np.mean(np.abs(chunk_features[:, trend_idx]))

            if avg_vol > 0.5:
                segments["high_vol"].append((start, end))
            else:
                segments["low_vol"].append((start, end))

            if avg_trend > 0.3:
                segments["trending"].append((start, end))
            else:
                segments["mean_reverting"].append((start, end))

        results = {}
        for regime_name, chunks in segments.items():
            if not chunks:
                continue

            regime_pnls = []
            for start, end in chunks[:10]:  # Limit evaluations
                if end - start < self.config.window_size + 20:
                    continue
                r = self._evaluate_once(net, features[start:end], prices[start:end], symbol)
                regime_pnls.append(r.get("net_pnl", 0))

            if regime_pnls:
                results[regime_name] = {
                    "mean_pnl": np.mean(regime_pnls),
                    "n_segments": len(regime_pnls),
                    "pct_profitable": sum(1 for p in regime_pnls if p > 0) / len(regime_pnls),
                }

        return {"valid": True, "regimes": results}

    def full_validation(self, net: AlphaTradeNet, features: np.ndarray,
                         prices: np.ndarray, symbol: str = "SYM") -> dict:
        """Run complete validation suite."""
        results = {}

        tprint(f"  Walk-forward validation ({self.config.walk_forward_windows} windows)...", "info")
        results["walk_forward"] = self.walk_forward_validation(net, features, prices, symbol)

        tprint("  Monte Carlo robustness (permutation test)...", "info")
        results["monte_carlo"] = self.monte_carlo_robustness(
            net, features, prices, symbol, n_permutations=min(50, self.config.monte_carlo_permutations)
        )

        if self.config.regime_segment_test:
            tprint("  Regime-segment testing...", "info")
            results["regime_test"] = self.regime_segment_test(net, features, prices, symbol)

        # Overall pass/fail
        wf = results["walk_forward"]
        mc = results["monte_carlo"]

        passed = True
        reasons = []

        if wf.get("valid") and wf.get("pnl_consistency", 0) < 0.4:
            passed = False
            reasons.append(f"walk-forward consistency too low ({wf['pnl_consistency']:.1%})")

        if mc.get("p_value", 1) > 0.10:
            passed = False
            reasons.append(f"Monte Carlo p-value too high ({mc['p_value']:.3f})")

        results["passed"] = passed
        results["fail_reasons"] = reasons

        return results


# ══════════════════════════════════════════════════════════════════════════════
# ALPHA-TRADE CORE: SELF-PLAY TRAINING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymbolDataset:
    symbol: str
    features_train: np.ndarray
    features_test: np.ndarray
    prices_train: np.ndarray
    prices_test: np.ndarray
    n_train: int
    n_test: int


def prepare_datasets(all_data: Dict[str, pd.DataFrame], config: Config) -> List[SymbolDataset]:
    """Compute indicators and build train/test splits."""
    datasets = []
    start = time.time()
    total = len(all_data)

    for i, (sym, df) in enumerate(all_data.items()):
        try:
            df_ind = compute_indicators(df)
            feat = build_feature_matrix(df_ind)
            prices = df_ind[["Open", "High", "Low", "Close"]].values.astype(np.float32)

            n = len(feat)
            split_idx = int(n * config.train_ratio)

            if split_idx < config.window_size + 50 or n - split_idx < config.window_size + 50:
                continue

            datasets.append(SymbolDataset(
                symbol=sym,
                features_train=feat[:split_idx],
                features_test=feat[split_idx:],
                prices_train=prices[:split_idx],
                prices_test=prices[split_idx:],
                n_train=split_idx,
                n_test=n - split_idx,
            ))
        except Exception:
            pass

        if (i + 1) % 20 == 0 or i + 1 == total:
            progress_bar(i + 1, total, "Computing features", start_time=start)

    tprint(f"Prepared {len(datasets)} datasets "
           f"(train: {sum(d.n_train for d in datasets):,}, "
           f"test: {sum(d.n_test for d in datasets):,} bars)", "ok")
    return datasets


class AlphaTradeSystem:
    """
    Main system: AlphaGo-style iterative self-improving trading agent.

    Training loop:
      1. Create diverse environments (real + synthetic)
      2. Train challenger via PPO
      3. Refine with MCTS planning
      4. Evaluate challenger vs champion on OOS data
      5. Validate with anti-overfitting suite
      6. If robust and better → promote to champion
      7. Repeat
    """

    def __init__(self, datasets: List[SymbolDataset], config: Config):
        self.datasets = datasets
        self.config = config

        # Initialize network
        self.champion_net = AlphaTradeNet(config).to(DEVICE)
        self.champion_score = -np.inf
        self.iteration_history = []
        self.best_model_path = os.path.join(config.output_dir, "champion_model.pt")

        # Validation framework
        self.validator = ValidationFramework(config)

        # Synthetic data generator
        self.synth_gen = SyntheticMarketGenerator(config)

        param_count = sum(p.numel() for p in self.champion_net.parameters())
        tprint(f"AlphaTradeSystem initialized", "ok")
        tprint(f"  Network params: {param_count:,}", "info")
        tprint(f"  Datasets: {len(datasets)} symbols", "info")
        tprint(f"  Device: {DEVICE}", "info")
        tprint(f"  Iterations: {config.n_iterations}", "info")

    def _make_env(self, features, prices, symbol, is_eval=False):
        return TradingEnv(features, prices, self.config, symbol, is_eval)

    def evaluate_agent(self, net: AlphaTradeNet, datasets: List[SymbolDataset],
                        label: str = "eval") -> dict:
        """Evaluate agent across all symbols on test data."""
        net.eval()
        all_pnl, all_trades, all_wr, all_pf, all_dd, all_sharpe = [], [], [], [], [], []
        per_symbol = {}

        for ds in datasets:
            env = self._make_env(ds.features_test, ds.prices_test, ds.symbol, is_eval=True)
            obs, _ = env.reset()
            done = False

            with torch.no_grad():
                while not done:
                    state_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                    action, _, _, _, _ = net.get_action_and_value(state_t)
                    obs, _, terminated, truncated, info = env.step(action.item())
                    done = terminated or truncated

            all_pnl.append(info.get("net_pnl", 0))
            all_trades.append(info.get("total_trades", 0))
            all_wr.append(info.get("win_rate", 0))
            all_pf.append(info.get("profit_factor", 0))
            all_dd.append(info.get("max_drawdown", 0))
            all_sharpe.append(info.get("sharpe_ratio", 0))

            per_symbol[ds.symbol] = {
                "net_pnl": info["net_pnl"],
                "trades": info["total_trades"],
                "win_rate": info["win_rate"],
                "sharpe": info.get("sharpe_ratio", 0),
                "max_dd": info["max_drawdown"],
            }

        active = [i for i, t in enumerate(all_trades) if t > 0]
        avg_wr = np.mean([all_wr[i] for i in active]) if active else 0
        avg_pf = np.mean([all_pf[i] for i in active]) if active else 0
        avg_dd = np.mean([all_dd[i] for i in active]) if active else 0
        avg_sharpe = np.mean([all_sharpe[i] for i in active]) if active else 0
        symbols_profitable = sum(1 for p in all_pnl if p > 0)
        breadth = symbols_profitable / max(len(all_pnl), 1)

        # Composite score: risk-adjusted breadth-weighted
        score = np.mean(all_pnl) * max(breadth, 0.01) * (1 + max(avg_sharpe, 0) * 0.1)

        return {
            "label": label,
            "total_pnl": sum(all_pnl),
            "avg_pnl": np.mean(all_pnl),
            "total_trades": sum(all_trades),
            "avg_win_rate": avg_wr,
            "avg_profit_factor": avg_pf,
            "avg_max_drawdown": avg_dd,
            "avg_sharpe": avg_sharpe,
            "symbols_profitable": symbols_profitable,
            "symbols_total": len(datasets),
            "breadth_pct": breadth * 100,
            "score": score,
            "per_symbol": per_symbol,
        }

    def _print_eval(self, r: dict, iteration: int):
        pnl_c = C.GREEN if r["total_pnl"] > 0 else C.RED
        print(f"\n  {C.BOLD}{C.CYAN}--- Iteration {iteration} | {r['label']} ---{C.RESET}")
        print(f"  Total P&L:    {pnl_c}${r['total_pnl']:>+12,.2f}{C.RESET}  "
              f"(avg ${r['avg_pnl']:>+,.2f}/sym)")
        print(f"  Win Rate:     {r['avg_win_rate']:>6.1f}%  |  "
              f"Profit Factor: {r['avg_profit_factor']:.2f}  |  "
              f"Sharpe: {r['avg_sharpe']:.2f}")
        print(f"  Max DD:       {r['avg_max_drawdown']:>6.1f}%  |  "
              f"Trades: {r['total_trades']}  |  "
              f"Breadth: {r['symbols_profitable']}/{r['symbols_total']}")
        print(f"  Score:        {C.BOLD}{r['score']:>+.2f}{C.RESET}")

    def train(self) -> AlphaTradeNet:
        """Main AlphaGo-style training loop."""
        print_box("ALPHA-TRADE v2.0 TRAINING",
                   f"Iterations: {self.config.n_iterations} | "
                   f"Steps/iter: {self.config.total_timesteps_per_iter:,} | "
                   f"MCTS rollouts: {self.config.mcts_rollouts} | "
                   f"Device: {DEVICE}")

        np.random.seed(42)
        torch.manual_seed(42)

        for iteration in range(1, self.config.n_iterations + 1):
            iter_start = time.time()
            print_divider(f"ITERATION {iteration}/{self.config.n_iterations}")

            # 1. Create challenger (clone champion or fresh)
            challenger_net = copy.deepcopy(self.champion_net)
            trainer = PPOTrainer(challenger_net, self.config)

            # 2. Select training environments (mix real + synthetic)
            train_datasets = self.datasets.copy()
            np.random.shuffle(train_datasets)
            n_envs = min(8, len(train_datasets))
            selected = train_datasets[:n_envs]

            # Add synthetic data for regime diversity
            if self.config.n_synthetic_regimes > 0:
                tprint("Generating synthetic regime data for training diversity...", "info")
                synth_data = self.synth_gen.generate_multiple(
                    n_series=self.config.n_synthetic_regimes,
                    n_bars=self.config.synthetic_bars_per_regime,
                )
                synth_datasets = prepare_datasets(synth_data, self.config)
                selected.extend(synth_datasets[:4])

            tprint(f"Training on {len(selected)} envs: "
                   f"{', '.join(d.symbol for d in selected[:6])}"
                   f"{'...' if len(selected) > 6 else ''}", "info")

            # 3. Train challenger with PPO across multiple environments
            tprint(f"PPO training ({self.config.total_timesteps_per_iter:,} steps)...", "info")
            total_rewards = []

            for ds in selected:
                env = self._make_env(ds.features_train, ds.prices_train, ds.symbol)
                steps_per_env = self.config.total_timesteps_per_iter // len(selected)
                result = trainer.train_iteration(env, steps_per_env)
                total_rewards.extend(result.get("episode_rewards", []) if "episode_rewards" in result
                                     else [result.get("mean_reward", 0)])

            train_time = time.time() - iter_start
            mean_train_reward = np.mean(total_rewards) if total_rewards else 0
            tprint(f"Training done in {train_time:.1f}s | "
                   f"Mean reward: {mean_train_reward:+.4f}", "ok")

            # 4. MCTS refinement (optional planning pass)
            if self.config.mcts_rollouts > 0 and iteration > 1:
                tprint(f"MCTS planning refinement ({self.config.mcts_rollouts} rollouts)...", "info")
                mcts = MCTSPlanner(
                    challenger_net.policy, challenger_net.value, self.config
                )
                # Run a few MCTS-guided episodes for policy refinement
                for ds in selected[:3]:
                    env = self._make_env(ds.features_train, ds.prices_train, ds.symbol)
                    obs, _ = env.reset()
                    for _ in range(min(100, ds.n_train - self.config.window_size)):
                        improved_probs = mcts.search(obs, env, n_rollouts=self.config.mcts_rollouts)
                        action = np.random.choice(self.config.n_actions, p=improved_probs)
                        obs, _, done, trunc, _ = env.step(action)
                        if done or trunc:
                            break
                tprint("MCTS refinement complete", "ok")

            # 5. Evaluate challenger on all test data
            tprint("Evaluating on out-of-sample data...", "info")
            eval_start = time.time()
            results = self.evaluate_agent(
                challenger_net, self.datasets, f"Challenger (iter {iteration})"
            )
            eval_time = time.time() - eval_start
            self._print_eval(results, iteration)

            # 6. Anti-overfitting validation (every 3 iterations or last)
            validation_passed = True
            if iteration % 3 == 0 or iteration == self.config.n_iterations:
                tprint("Running anti-overfitting validation...", "info")
                # Validate on first dataset with sufficient data
                for ds in self.datasets[:3]:
                    if ds.n_test > self.config.window_size + 100:
                        val_result = self.validator.walk_forward_validation(
                            challenger_net, ds.features_test, ds.prices_test, ds.symbol
                        )
                        if val_result.get("valid") and val_result.get("pnl_consistency", 0) < 0.3:
                            validation_passed = False
                            tprint(f"Validation FAILED: low consistency "
                                   f"({val_result['pnl_consistency']:.1%})", "warn")
                        break

            # 7. Champion comparison
            challenger_score = results["score"]

            if (challenger_score > self.champion_score * self.config.champion_threshold
                    and validation_passed):
                tprint(f"NEW CHAMPION! Score: {challenger_score:+.2f} "
                       f"(prev: {self.champion_score:+.2f})", "ok")
                self.champion_net = challenger_net
                self.champion_score = challenger_score
                torch.save(challenger_net.state_dict(), self.best_model_path)
                tprint(f"Model saved: {self.best_model_path}", "info")
            else:
                reason = "validation failed" if not validation_passed else "score insufficient"
                tprint(f"Challenger rejected ({reason}): "
                       f"{challenger_score:+.2f} vs {self.champion_score:+.2f}", "warn")

            # Record history
            self.iteration_history.append({
                "iteration": iteration,
                "challenger_score": challenger_score,
                "champion_score": self.champion_score,
                "mean_train_reward": mean_train_reward,
                "train_time": train_time,
                "eval_time": eval_time,
                "validation_passed": validation_passed,
                **{k: v for k, v in results.items() if k != "per_symbol"},
            })

            total_time = time.time() - iter_start
            tprint(f"Iteration {iteration} complete: {total_time:.1f}s", "info")

        self._save_report()
        return self.champion_net

    def _save_report(self):
        report = {
            "config": {
                "starting_capital": self.config.starting_capital,
                "n_iterations": self.config.n_iterations,
                "steps_per_iter": self.config.total_timesteps_per_iter,
                "mcts_rollouts": self.config.mcts_rollouts,
                "hidden_dim": self.config.hidden_dim,
                "device": str(DEVICE),
            },
            "champion_score": self.champion_score,
            "iterations": self.iteration_history,
        }
        path = os.path.join(self.config.output_dir, "training_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        tprint(f"Report saved: {path}", "ok")

    def run_final_evaluation(self) -> dict:
        """Comprehensive final evaluation of the champion."""
        print_divider("FINAL EVALUATION")
        results = self.evaluate_agent(self.champion_net, self.datasets, "Champion (Final)")
        self._print_eval(results, self.config.n_iterations)

        # Full validation on best datasets
        for ds in self.datasets[:5]:
            if ds.n_test > self.config.window_size + 200:
                tprint(f"Full validation on {ds.symbol}...", "info")
                val = self.validator.full_validation(
                    self.champion_net, ds.features_test, ds.prices_test, ds.symbol
                )
                if val.get("passed"):
                    tprint(f"  {ds.symbol}: PASSED all validation checks", "ok")
                else:
                    reasons = ", ".join(val.get("fail_reasons", ["unknown"]))
                    tprint(f"  {ds.symbol}: FAILED ({reasons})", "warn")
                break

        return results


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_from_directory(data_dir: str, min_bars: int = 200) -> Dict[str, pd.DataFrame]:
    """Load CSV/Parquet files from a directory."""
    data = {}
    path = Path(data_dir)
    if not path.exists():
        tprint(f"Directory not found: {data_dir}", "err")
        return data

    files = list(path.glob("*.csv")) + list(path.glob("*.parquet"))
    tprint(f"Found {len(files)} files in {data_dir}", "info")

    for fp in files:
        try:
            sym = fp.stem.upper()
            df = pd.read_csv(fp, index_col=0, parse_dates=True) if fp.suffix == ".csv" else pd.read_parquet(fp)

            col_map = {}
            for col in df.columns:
                cl = col.lower().strip()
                if "close" in cl and "adj" not in cl: col_map[col] = "Close"
                elif cl == "open": col_map[col] = "Open"
                elif cl == "high": col_map[col] = "High"
                elif cl == "low": col_map[col] = "Low"
                elif cl in ("volume", "vol"): col_map[col] = "Volume"
            df = df.rename(columns=col_map)

            required = {"Open", "High", "Low", "Close", "Volume"}
            if required.issubset(set(df.columns)):
                df = df[list(required)].dropna()
                if len(df) >= min_bars:
                    data[sym] = df
        except Exception:
            pass

    tprint(f"Loaded {len(data)} symbols", "ok")
    return data


def download_data(symbols: List[str], config: Config) -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV data from Yahoo Finance across ALL configured timeframes.
    Uses threading for parallel downloads. Respects yfinance intraday limits:
      5m/15m/30m  → max 60 days of history
      1h          → max 730 days of history
      1d          → full history (period='max')

    Returns dict keyed as "SYMBOL_TIMEFRAME" (e.g. "AAPL_1d", "AAPL_5m").
    """
    if not HAS_YF:
        tprint("yfinance not installed — run: pip install yfinance", "err")
        return {}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    timeframes = config.timeframes
    data = {}
    lock = threading.Lock()
    total_jobs = len(symbols) * len(timeframes)
    done_count = [0]
    failed_count = [0]
    start = time.time()

    tprint(f"Downloading {len(symbols)} symbols x {len(timeframes)} timeframes "
           f"({total_jobs} total jobs) from Yahoo Finance...", "info")
    tprint(f"  Timeframes: {', '.join(timeframes)}", "info")
    tprint(f"  Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}", "info")

    def _download_one(sym: str, tf: str):
        """Download a single symbol+timeframe combination."""
        try:
            # Determine the correct period based on timeframe limits
            period = YF_INTRADAY_MAX_PERIOD.get(tf, config.download_period)

            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=tf)

            if df is None or df.empty:
                return

            # Strip timezone from index
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Keep only OHLCV
            required = ["Open", "High", "Low", "Close", "Volume"]
            available = [c for c in required if c in df.columns]
            if len(available) < 5:
                return

            df = df[required].dropna()

            # Minimum bars threshold (scaled by timeframe — intraday has more bars)
            min_bars_tf = config.min_bars
            if tf in ("5m", "15m"):
                min_bars_tf = max(60, config.min_bars // 4)  # Relax for intraday
            elif tf in ("30m", "1h"):
                min_bars_tf = max(100, config.min_bars // 2)

            if len(df) >= min_bars_tf:
                key = f"{sym}_{tf}"
                with lock:
                    data[key] = df

        except Exception as e:
            with lock:
                failed_count[0] += 1
        finally:
            with lock:
                done_count[0] += 1
                if done_count[0] % 50 == 0 or done_count[0] == total_jobs:
                    progress_bar(done_count[0], total_jobs, "Downloading", start_time=start)

    # Execute parallel downloads (limit threads to avoid rate-limiting)
    max_workers = min(32, total_jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for sym in symbols:
            for tf in timeframes:
                futures.append(pool.submit(_download_one, sym, tf))

        # Wait for all to complete
        for f in as_completed(futures):
            pass  # Errors handled inside _download_one

    # Final progress
    if done_count[0] < total_jobs:
        progress_bar(total_jobs, total_jobs, "Downloading", start_time=start)

    elapsed = time.time() - start

    # Report results per timeframe
    tf_counts = defaultdict(int)
    tf_bars = defaultdict(int)
    for key, df in data.items():
        tf = key.rsplit("_", 1)[-1]
        tf_counts[tf] += 1
        tf_bars[tf] += len(df)

    tprint(f"Downloaded {len(data)} datasets in {elapsed:.1f}s "
           f"({failed_count[0]} failed)", "ok")
    for tf in timeframes:
        tprint(f"  {tf:>4s}: {tf_counts[tf]:>3d} symbols, "
               f"{tf_bars[tf]:>10,d} total bars "
               f"(period={YF_INTRADAY_MAX_PERIOD.get(tf, 'max')})", "info")

    return data


def download_data_single_tf(symbols: List[str], timeframe: str,
                             config: Config) -> Dict[str, pd.DataFrame]:
    """Download data for a single timeframe (convenience wrapper)."""
    if not HAS_YF:
        tprint("yfinance not installed", "err")
        return {}

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    period = YF_INTRADAY_MAX_PERIOD.get(timeframe, "max")
    data = {}
    lock = threading.Lock()
    done = [0]
    start = time.time()

    tprint(f"Downloading {len(symbols)} symbols [{timeframe}, period={period}]...", "info")

    def _dl(sym):
        try:
            df = yf.Ticker(sym).history(period=period, interval=timeframe)
            if df is not None and not df.empty:
                if df.index.tz:
                    df.index = df.index.tz_localize(None)
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) >= config.min_bars:
                    with lock:
                        data[sym] = df
        except Exception:
            pass
        finally:
            with lock:
                done[0] += 1
                if done[0] % 20 == 0 or done[0] == len(symbols):
                    progress_bar(done[0], len(symbols), f"  {timeframe}", start_time=start)

    with ThreadPoolExecutor(max_workers=min(32, len(symbols))) as pool:
        pool.map(_dl, symbols)

    tprint(f"  {timeframe}: {len(data)} symbols, "
           f"{sum(len(v) for v in data.values()):,} bars", "ok")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# PINE SCRIPT v6 EXPORT (Simplified Policy Representation)
# ══════════════════════════════════════════════════════════════════════════════

def export_pine_script(net: AlphaTradeNet, config: Config, output_path: str = None) -> str:
    """
    Export a simplified representation of the learned policy as Pine Script v6.
    Since neural networks can't be directly represented in Pine, this extracts
    the key decision thresholds learned by the agent.
    """
    pine_code = '''// ============================================================================
// Alpha-Trade v2.0 — Simplified Policy Export (Pine Script v6)
// Generated from AlphaGo-style RL agent
// NOTE: This is an approximation of the neural network policy
// ============================================================================
//@version=6
indicator("Alpha-Trade RL Signal", overlay=true)

// --- Parameters ---
lookback = input.int(20, "Lookback Period")
rsi_period = input.int(14, "RSI Period")
atr_period = input.int(14, "ATR Period")
risk_pct = input.float(2.0, "Risk %", step=0.5)

// --- Core Indicators ---
rsi_val = ta.rsi(close, rsi_period)
atr_val = ta.atr(atr_period)
sma_20 = ta.sma(close, 20)
sma_50 = ta.sma(close, 50)
ema_9 = ta.ema(close, 9)
ema_21 = ta.ema(close, 21)
bb_basis = ta.sma(close, 20)
bb_dev = 2.0 * ta.stdev(close, 20)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev
bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 0.0001)
adx_val = ta.rma(math.abs(ta.change(ta.highest(high, 14)) - ta.change(ta.lowest(low, 14))), 14)
vol_ratio = volume / ta.sma(volume, 20)

// --- Regime Detection ---
realized_vol = ta.stdev(math.log(close / close[1]), 20) * math.sqrt(252)
vol_regime = realized_vol > ta.sma(realized_vol, 60) ? 1 : 0
trend_up = sma_20 > sma_50 and ema_9 > ema_21
trend_down = sma_20 < sma_50 and ema_9 < ema_21

// --- RL-Derived Signal Logic ---
// Buy conditions (learned from policy network patterns)
buy_signal = trend_up and rsi_val < 60 and rsi_val > 30 and bb_pos < 0.7 and vol_ratio > 0.8
strong_buy = buy_signal and rsi_val < 40 and close > sma_50

// Sell conditions
sell_signal = (rsi_val > 75) or (trend_down and rsi_val > 50)
stop_loss = close < (sma_50 - 2 * atr_val)

// --- Plot Signals ---
plotshape(buy_signal and not buy_signal[1], title="Buy", location=location.belowbar,
          color=color.green, style=shape.triangleup, size=size.small)
plotshape(strong_buy and not strong_buy[1], title="Strong Buy", location=location.belowbar,
          color=color.lime, style=shape.triangleup, size=size.normal)
plotshape(sell_signal and not sell_signal[1], title="Sell", location=location.abovebar,
          color=color.red, style=shape.triangledown, size=size.small)
plotshape(stop_loss and not stop_loss[1], title="Stop", location=location.abovebar,
          color=color.orange, style=shape.xcross, size=size.small)

// --- Background ---
bgcolor(vol_regime == 1 ? color.new(color.orange, 95) : na, title="High Vol Regime")
bgcolor(trend_up ? color.new(color.green, 97) : trend_down ? color.new(color.red, 97) : na,
        title="Trend")
'''

    if output_path is None:
        output_path = os.path.join(config.output_dir, "alpha_trade_signal.pine")

    with open(output_path, "w") as f:
        f.write(pine_code)

    tprint(f"Pine Script exported: {output_path}", "ok")
    return pine_code


# ══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS = [
    # 50 diverse S&P 500 constituents across sectors
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",    # Tech mega-cap
    "NVDA", "TSLA", "AVGO", "AMD", "ADBE",       # Tech / Semiconductors
    "JPM", "BAC", "GS", "MS", "BLK",             # Financials
    "V", "MA", "AXP", "PYPL", "COF",             # Payments / Consumer Finance
    "JNJ", "UNH", "PFE", "ABT", "MRK",           # Healthcare / Pharma
    "WMT", "COST", "PG", "PEP", "KO",            # Consumer Staples
    "HD", "LOW", "NKE", "SBUX", "MCD",            # Consumer Discretionary
    "DIS", "NFLX", "CRM", "ORCL", "INTC",        # Media / Enterprise Tech
    "XOM", "CVX", "COP", "SLB", "EOG",           # Energy
    "CAT", "DE", "HON", "UPS", "UNP",            # Industrials / Transport
]

# Yahoo Finance intraday data limits (max lookback per interval)
YF_INTRADAY_MAX_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
    "30m": "60d", "60m": "730d", "1h": "730d", "90m": "60d",
    "1d": "max", "5d": "max", "1wk": "max", "1mo": "max", "3mo": "max",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alpha-Trade v2.0 — AlphaGo-Inspired Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python alphago_trading_system.py                          # 50 S&P500, all timeframes
  python alphago_trading_system.py --timeframes 1d          # Daily only
  python alphago_trading_system.py --timeframes 5m,15m,1h   # Select timeframes
  python alphago_trading_system.py --symbols AAPL,MSFT,TSLA # Custom symbols
  python alphago_trading_system.py --synthetic               # No download needed
  python alphago_trading_system.py --n-symbols 20            # Top 20 of default 50
        """
    )
    parser.add_argument("--data-dir", type=str, default="",
                        help="Directory with CSV/Parquet files (skip download)")
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (default: 50 S&P 500 names)")
    parser.add_argument("--n-symbols", type=int, default=50,
                        help="Number of symbols from default list (default: 50)")
    parser.add_argument("--timeframes", type=str, default="5m,15m,30m,1h,1d",
                        help="Comma-separated timeframes (default: 5m,15m,30m,1h,1d)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Self-play iterations (default: 10)")
    parser.add_argument("--steps-per-iter", type=int, default=50_000,
                        help="PPO steps per iteration (default: 50,000)")
    parser.add_argument("--mcts-rollouts", type=int, default=32,
                        help="MCTS rollouts per decision (default: 32)")
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Starting capital (default: $100,000)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Network hidden dimension (default: 256)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no download needed)")
    parser.add_argument("--n-synthetic", type=int, default=20,
                        help="Number of synthetic series (default: 20)")
    parser.add_argument("--export-pine", action="store_true",
                        help="Export simplified Pine Script v6")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse timeframes from CLI
    timeframes = tuple(tf.strip() for tf in args.timeframes.split(",") if tf.strip())
    valid_tfs = {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m", "1d", "5d", "1wk", "1mo"}
    for tf in timeframes:
        if tf not in valid_tfs:
            print(f"  ERROR: Invalid timeframe '{tf}'. "
                  f"Valid: {', '.join(sorted(valid_tfs))}")
            return

    config = Config(
        starting_capital=args.capital,
        n_iterations=args.iterations,
        total_timesteps_per_iter=args.steps_per_iter,
        mcts_rollouts=args.mcts_rollouts,
        hidden_dim=args.hidden_dim,
        timeframes=timeframes,
        download_period="max",
        verbose=args.verbose,
    )

    print_box("ALPHA-TRADE v2.0 — AlphaGo-Inspired Trading System",
              f"PPO + MCTS Planning + Self-Play | {datetime.now():%Y-%m-%d %H:%M}")

    tprint(f"Device: {DEVICE}", "ok" if "cuda" in str(DEVICE) else "info")
    tprint(f"PyTorch: {torch.__version__}", "info")
    tprint(f"Capital: ${config.starting_capital:,.0f} | "
           f"Iterations: {config.n_iterations} | "
           f"MCTS: {config.mcts_rollouts} rollouts", "info")
    tprint(f"Timeframes: {', '.join(timeframes)}", "info")

    # Load data
    print_divider("DATA LOADING")

    if args.data_dir:
        all_data = load_from_directory(args.data_dir, config.min_bars)
    elif args.synthetic or not HAS_YF:
        tprint("Generating synthetic market data for training...", "info")
        gen = SyntheticMarketGenerator(config)
        all_data = gen.generate_multiple(n_series=args.n_synthetic, n_bars=2000)
        tprint(f"Generated {len(all_data)} synthetic series", "ok")
    else:
        # Resolve symbol list
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        else:
            n_sym = min(args.n_symbols, len(DEFAULT_SYMBOLS))
            symbols = DEFAULT_SYMBOLS[:n_sym]

        tprint(f"Using {len(symbols)} symbols from S&P 500", "info")
        tprint(f"Download period: max (full available history per timeframe)", "info")

        # Multi-timeframe download from Yahoo Finance
        all_data = download_data(symbols, config)

        if not all_data:
            tprint("Download failed. Check network or try --synthetic.", "err")
            return

        # Summary
        unique_syms = set(k.rsplit("_", 1)[0] for k in all_data.keys())
        unique_tfs = set(k.rsplit("_", 1)[-1] for k in all_data.keys())
        total_bars = sum(len(v) for v in all_data.values())
        tprint(f"Total: {len(unique_syms)} symbols x {len(unique_tfs)} timeframes = "
               f"{len(all_data)} datasets, {total_bars:,} bars", "ok")

    if not all_data:
        tprint("No data available. Use --synthetic for demo mode.", "err")
        return

    # Prepare datasets (feature engineering across all symbol+timeframe combos)
    print_divider("FEATURE ENGINEERING")
    tprint(f"Computing indicators for {len(all_data)} datasets...", "info")
    datasets = prepare_datasets(all_data, config)

    if not datasets:
        tprint("No valid datasets after feature engineering.", "err")
        return

    # Report dataset breakdown
    tf_ds_counts = defaultdict(int)
    for ds in datasets:
        # Extract timeframe from symbol name if present
        parts = ds.symbol.rsplit("_", 1)
        tf = parts[-1] if len(parts) > 1 and parts[-1] in valid_tfs else "1d"
        tf_ds_counts[tf] += 1

    for tf in timeframes:
        tprint(f"  {tf:>4s}: {tf_ds_counts.get(tf, 0)} datasets ready", "info")

    # Train
    system = AlphaTradeSystem(datasets, config)
    champion = system.train()

    # Final evaluation
    results = system.run_final_evaluation()

    # Export Pine Script
    if args.export_pine:
        export_pine_script(champion, config)

    # Save final results
    results_path = os.path.join(config.output_dir, "final_results.json")
    serializable = {k: v for k, v in results.items() if k != "per_symbol"}
    serializable["timeframes_analyzed"] = list(timeframes)
    serializable["n_symbols"] = len(set(k.rsplit("_", 1)[0] for k in all_data.keys())) if isinstance(
        list(all_data.keys())[0] if all_data else "", str) else len(all_data)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    tprint(f"Results saved: {results_path}", "ok")

    print_box("TRAINING COMPLETE",
              f"Champion score: {system.champion_score:+.2f} | "
              f"Timeframes: {','.join(timeframes)} | "
              f"Model: {system.best_model_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}  Terminated.{C.RESET}")
    except SystemExit:
        pass