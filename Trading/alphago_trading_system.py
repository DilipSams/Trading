"""
================================================================================
ALPHA-TRADE v3.0 - GPU-Accelerated AlphaGo-Inspired Trading System
================================================================================

Full GPU-parallel reinforcement learning system for financial markets.

v3.0 IMPROVEMENTS (from expert review):
  1. Target-position action space: {-1.0, -0.5, 0, +0.5, +1.0} (long+short+flat)
  2. Consistent fill-price transaction cost model (no double-counting)
  3. Stationary log-return reward + proportional turnover + quadratic drawdown
  4. Risk head trained via auxiliary loss (predicts forward realized vol)
  5. MCTS distilled into policy via KL divergence (search--learning)
  6. Champion gating: additive margin (fixed negative score bug)
  7. Soft reward clipping via tanh (preserves tail info)

GPU ACCELERATION FEATURES:
  - Mixed-Precision Training (FP16/BF16 via torch.amp)
  - torch.compile() JIT fusion (PyTorch 2.0+)
  - Vectorized Multi-Environment Rollouts (N envs stepped in parallel)
  - GPU-Resident Rollout Buffers with Pinned Memory Transfers
  - Batched MCTS with forward simulation + path reward backup
  - cuDNN Benchmark Mode, Multi-GPU DataParallel, Fused AdamW

INSTALL:
    pip install torch numpy pandas gymnasium yfinance

RUN:
    python alphago_trading_system.py                     # 50 S&P500 x 5 timeframes
    python alphago_trading_system.py --backbone transformer  # Transformer backbone
    python alphago_trading_system.py --n-envs 32         # More parallel envs
    python alphago_trading_system.py --no-amp             # Disable mixed precision
    python alphago_trading_system.py --synthetic          # No download needed
================================================================================
"""

import os, sys, time, json, math, copy, argparse, warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd

# Shared cost model (WI-2: single source of truth for cost math)
# FIX Â§4.1: Use correct export names from alphago_cost_model.py
try:
    from alphago_cost_model import fill_price_from_cost, estimate_cost_dollars, CostBreakdown
    HAS_COST_MODEL = True
except ImportError:
    HAS_COST_MODEL = False

# Asymmetric stop loss system
try:
    from alphago_stop_loss import AsymmetricStopLoss, StopLossConfig
    HAS_ASYMMETRIC_STOPS = True
except ImportError:
    HAS_ASYMMETRIC_STOPS = False
    print("[WARNING] alphago_stop_loss.py not found - asymmetric stops disabled")

# FIX Â§4.6: Targeted warning suppression only â€” keep RuntimeWarning visible
# to catch NaN propagation, overflow, and data integrity issues.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*urllib3.*")
# RuntimeWarning intentionally NOT suppressed â€” silent NaN propagation is dangerous

# === PyTorch + CUDA ===
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    HAS_TORCH = True
except ImportError:
    print("ERROR: PyTorch required. pip install torch"); sys.exit(1)

# === GPU Detection & Performance Config ===
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
N_GPUS = torch.cuda.device_count() if HAS_CUDA else 0

if HAS_CUDA:
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True          # Auto-tune conv kernels
    # FIX Â§4.2: Default to deterministic=True for reproducibility (Absolute Rule #6).
    # benchmark is disabled when deterministic is True (set in Config.__post_init__).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 on Ampere+
    torch.backends.cudnn.allow_tf32 = True
elif HAS_MPS:
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# FIX 4: Use non-deprecated torch.amp API (PyTorch 2.4+)
HAS_AMP = HAS_CUDA and hasattr(torch.amp, 'autocast')
HAS_COMPILE = hasattr(torch, "compile")
# Triton is required for torch.compile but unavailable on Windows
HAS_TRITON = False
if HAS_COMPILE:
    try:
        import triton  # noqa: F401
        HAS_TRITON = True
    except ImportError:
        pass
    if sys.platform == "win32":
        HAS_TRITON = False  # Triton doesn't support Windows
if not HAS_TRITON:
    HAS_COMPILE = False

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("ERROR: gymnasium required. pip install gymnasium"); sys.exit(1)

try:
    import yfinance as yf; HAS_YF = True
except ImportError:
    HAS_YF = False


# ============================================================================
# TERMINAL OUTPUT
# ============================================================================
class C:
    RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
    RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"
    BLUE="\033[94m"; MAGENTA="\033[95m"; CYAN="\033[96m"; WHITE="\033[97m"

def tprint(msg, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    icons = {"info":f"{C.CYAN}i","ok":f"{C.GREEN}+","warn":f"{C.YELLOW}!",
             "err":f"{C.RED}x","gpu":f"{C.MAGENTA}G"}
    print(f"  {C.DIM}[{ts}]{C.RESET} {icons.get(level,icons['info'])} {msg}{C.RESET}")

def print_box(title, sub=""):
    print(f"\n{C.CYAN}{C.BOLD}{'='*82}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}  {C.WHITE}{title}{C.RESET}")
    if sub: print(f"{C.CYAN}{C.BOLD}  {C.DIM}{sub}{C.RESET}")
    print(f"{C.CYAN}{C.BOLD}{'='*82}{C.RESET}")

def print_divider(text):
    pad = (76 - len(text)) // 2
    print(f"\n{C.DIM}{'-'*pad} {C.CYAN}{C.BOLD}{text}{C.RESET}{C.DIM} {'-'*pad}{C.RESET}")

def progress_bar(cur, tot, label="", width=40, start_time=None):
    pct = cur / max(tot, 1); f = int(width * pct)
    eta = ""
    if start_time and cur > 0:
        r = (time.time()-start_time)/cur*(tot-cur)
        eta = f" ETA:{r:.0f}s" if r < 120 else f" ETA:{r/60:.1f}m"
    sys.stdout.write(f"\r  {C.CYAN}{label}{C.RESET} [{'#'*f}{'.'*(width-f)}] {pct*100:5.1f}% ({cur}/{tot}){eta}   ")
    sys.stdout.flush()
    if cur >= tot: print()

def print_gpu_info():
    if not HAS_CUDA:
        tprint(f"Device: CPU ({mp.cpu_count()} cores)" + (" | MPS available" if HAS_MPS else ""), "info")
        return
    for i in range(N_GPUS):
        p = torch.cuda.get_device_properties(i)
        tprint(f"GPU {i}: {p.name} | {p.total_memory/1024**3:.1f}GB | SM{p.major}.{p.minor} | {p.multi_processor_count} SMs", "gpu")
    tprint(f"cuDNN benchmark: ON | TF32: ON | AMP: {'available' if HAS_AMP else 'N/A'}", "gpu")


# ============================================================================
# Terminal chart utilities  (zero dependencies — pure Unicode + ANSI)
# ============================================================================
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values, width=20):
    """Render numeric series as compact Unicode sparkline string.

    Returns a string of width characters using block elements ▁▂▃▄▅▆▇█.
    If len(values) > width, downsamples by averaging buckets.
    """
    if values is None or len(values) == 0:
        return ""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return ""

    # Downsample to width by averaging buckets
    if len(arr) > width:
        bucket = len(arr) / width
        arr = np.array([np.mean(arr[int(i * bucket):int((i + 1) * bucket)])
                        for i in range(width)])

    vmin, vmax = np.min(arr), np.max(arr)
    rng = vmax - vmin
    if rng == 0:
        return _SPARK_CHARS[4] * len(arr)

    # Map to 0..7
    indices = np.clip(((arr - vmin) / rng * 7).astype(int), 0, 7)
    return "".join(_SPARK_CHARS[i] for i in indices)


def hbar_chart(items, width=40, title="", indent=4):
    """Print horizontal bar chart with green/red bars.

    Args:
        items: list of (label, value) tuples
        width: max bar width in characters
        title: chart title
        indent: left indent spaces
    """
    if not items:
        return
    pad = " " * indent
    max_label = max(len(label) for label, _ in items)
    max_abs = max(abs(v) for _, v in items) or 1

    lines = []
    if title:
        lines.append(f"\n{pad}{C.BOLD}{title}{C.RESET}")
        lines.append(f"{pad}{C.DIM}{'─' * (max_label + width + 20)}{C.RESET}")

    for label, value in items:
        bar_len = int(abs(value) / max_abs * width)
        bar_len = max(bar_len, 1) if value != 0 else 0
        color = C.GREEN if value >= 0 else C.RED
        bar = "█" * bar_len
        sign = "+" if value >= 0 else ""
        lines.append(f"{pad}{label:<{max_label}s}  {color}{bar}{C.RESET}  {sign}${value:,.2f}")

    lines.append(f"{pad}{C.DIM}{'─' * (max_label + width + 20)}{C.RESET}")
    print("\n".join(lines))


def _chart_common(values, width, fmt, dates=None):
    """Shared setup for line charts: downsample, compute range, format labels."""
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    ds_dates = None
    if len(arr) > width:
        bucket = len(arr) / width
        if dates is not None and len(dates) == len(np.array(values, dtype=float)[np.isfinite(np.array(values, dtype=float))]):
            ds_dates = [dates[int(i * bucket)] for i in range(width)]
        arr = np.array([np.mean(arr[int(i * bucket):int((i + 1) * bucket)])
                        for i in range(width)])
    elif dates is not None and len(dates) == len(arr):
        ds_dates = list(dates)
    return arr, ds_dates


def _year_axis(ds_dates, n, label_width, indent=4):
    """Build an X-axis string with year labels at year-boundary positions."""
    if ds_dates is None or len(ds_dates) < 2:
        return None
    # Extract year from each date string (YYYY-MM-DD)
    years = [d[:4] for d in ds_dates]
    # Find positions where year changes
    markers = []  # [(column, year_str), ...]
    markers.append((0, years[0]))
    for i in range(1, len(years)):
        if years[i] != years[i - 1]:
            markers.append((i, years[i]))
    # Build label line — skip labels that would overlap or extend past chart edge
    axis = [" "] * n
    last_end = -2  # position after last placed label
    for col, yr in markers:
        if col < last_end + 1:      # need at least 1 space gap
            continue
        if col + len(yr) > n:       # would be truncated at chart edge
            continue
        for j, ch in enumerate(yr):
            axis[col + j] = ch
        last_end = col + len(yr)
    pad = " " * indent
    return f"{pad}{' ' * label_width}  {''.join(axis)}"


def _fmt_y(v, fmt):
    if fmt == "$":
        if abs(v) >= 1_000_000: return f"${v/1e6:.1f}M"
        if abs(v) >= 1_000:     return f"${v/1e3:.0f}k"
        return f"${v:.0f}"
    elif fmt == "%":
        return f"{v:.1f}%"
    else:
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"


def _compute_rows(arr, vmin, vmax, rng, height):
    """Map array values to row indices (row 0 = top = vmax)."""
    rows = []
    for v in arr:
        r = int((vmax - v) / rng * (height - 1) + 0.5)
        rows.append(max(0, min(height - 1, r)))
    return rows


def _draw_line_on_canvas(canvas, rows, n):
    """Draw a connected line onto a canvas using box-drawing characters.

    For each transition from column c to c+1:
      - flat: ─
      - descending (row increases): ╮ at top, │ fill, ╰ at bottom
      - ascending  (row decreases): ╯ at bottom, │ fill, ╭ at top
    All transition characters are placed in column c+1.
    """
    # First column: start marker
    canvas[rows[0]][0] = "─"

    for c in range(1, n):
        pr, cr = rows[c - 1], rows[c]
        if pr == cr:
            canvas[cr][c] = "─"
        elif pr < cr:                       # descending (value drops)
            canvas[pr][c] = "╮"
            for r in range(pr + 1, cr):
                canvas[r][c] = "│"
            canvas[cr][c] = "╰"
        else:                               # ascending (value rises)
            canvas[cr][c] = "╭"
            for r in range(cr + 1, pr):
                canvas[r][c] = "│"
            canvas[pr][c] = "╯"


_LINE_CHARS = set("╮╰╭╯│─")


def line_chart(values, width=60, height=10, title="", indent=4, fmt="$", dates=None):
    """Print ASCII line chart in the terminal using box-drawing characters."""
    arr, ds_dates = _chart_common(values, width, fmt, dates=dates)
    if len(arr) < 2:
        return

    n = len(arr)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    rng = vmax - vmin
    if rng == 0:
        rng = abs(vmax) * 0.1 or 1.0
        vmin -= rng / 2; vmax += rng / 2; rng = vmax - vmin

    pad = " " * indent
    y_labels = [_fmt_y(vmin + rng * i / (height - 1), fmt) for i in range(height)]
    label_width = max(len(lbl) for lbl in y_labels)

    rows = _compute_rows(arr, vmin, vmax, rng, height)
    canvas = [[" "] * n for _ in range(height)]
    _draw_line_on_canvas(canvas, rows, n)

    lines = []
    if title:
        lines.append(f"\n{pad}{C.BOLD}{title}{C.RESET}")
        lines.append(f"{pad}{' ' * label_width} ┌{'─' * n}┐")

    for row_idx in range(height):
        y_idx = height - 1 - row_idx
        label = y_labels[y_idx].rjust(label_width)
        colored = ""
        for ch in canvas[row_idx]:
            if ch in _LINE_CHARS:
                colored += f"{C.GREEN}{ch}{C.RESET}"
            else:
                colored += ch
        lines.append(f"{pad}{C.DIM}{label}{C.RESET} ┤{colored}│")

    lines.append(f"{pad}{' ' * label_width} └{'─' * n}┘")
    year_line = _year_axis(ds_dates, n, label_width, indent)
    if year_line:
        lines.append(year_line)
    print("\n".join(lines))


def dual_line_chart(values1, values2, width=60, height=12, title="",
                    label1="Strategy", label2="SPY", indent=4, fmt="$",
                    dates=None):
    """Print ASCII line chart with two overlaid series using box-drawing chars."""
    arr1, ds_dates = _chart_common(values1, width, fmt, dates=dates)
    arr2, _ = _chart_common(values2, width, fmt)
    if len(arr1) < 2:
        return

    # Align lengths (use shorter)
    n = min(len(arr1), len(arr2)) if len(arr2) > 1 else len(arr1)
    arr1 = arr1[:n]
    arr2 = arr2[:n] if len(arr2) > 1 else None
    if ds_dates is not None:
        ds_dates = ds_dates[:n]

    # Global min/max across both series
    vmin, vmax = float(np.min(arr1)), float(np.max(arr1))
    if arr2 is not None:
        vmin = min(vmin, float(np.min(arr2)))
        vmax = max(vmax, float(np.max(arr2)))

    rng = vmax - vmin
    if rng == 0:
        rng = abs(vmax) * 0.1 or 1.0
        vmin -= rng / 2; vmax += rng / 2; rng = vmax - vmin

    pad = " " * indent
    y_labels = [_fmt_y(vmin + rng * i / (height - 1), fmt) for i in range(height)]
    label_width = max(len(lbl) for lbl in y_labels)

    # Draw each series on its own canvas, then merge
    canvas1 = [[" "] * n for _ in range(height)]
    rows1 = _compute_rows(arr1, vmin, vmax, rng, height)
    _draw_line_on_canvas(canvas1, rows1, n)

    canvas2 = None
    if arr2 is not None:
        canvas2 = [[" "] * n for _ in range(height)]
        rows2 = _compute_rows(arr2, vmin, vmax, rng, height)
        _draw_line_on_canvas(canvas2, rows2, n)

    # Print
    lines = []
    if title:
        lines.append(f"\n{pad}{C.BOLD}{title}{C.RESET}")
        legend = (f"{pad}  {C.GREEN}──{C.RESET} {label1}    "
                  f"{C.CYAN}──{C.RESET} {label2}")
        lines.append(legend)
        lines.append(f"{pad}{' ' * label_width} ┌{'─' * n}┐")

    for row_idx in range(height):
        y_idx = height - 1 - row_idx
        label = y_labels[y_idx].rjust(label_width)
        row_str = ""
        for c in range(n):
            c1 = canvas1[row_idx][c]
            c2 = canvas2[row_idx][c] if canvas2 else " "
            has1 = c1 in _LINE_CHARS
            has2 = c2 in _LINE_CHARS
            if has1 and has2:
                row_str += f"{C.YELLOW}{c1}{C.RESET}"    # overlap
            elif has1:
                row_str += f"{C.GREEN}{c1}{C.RESET}"     # series 1
            elif has2:
                row_str += f"{C.CYAN}{c2}{C.RESET}"      # series 2
            else:
                row_str += " "
        lines.append(f"{pad}{C.DIM}{label}{C.RESET} ┤{row_str}│")

    lines.append(f"{pad}{' ' * label_width} └{'─' * n}┘")
    year_line = _year_axis(ds_dates, n, label_width, indent)
    if year_line:
        lines.append(year_line)
    print("\n".join(lines))


def multi_line_chart(series_list, width=70, height=14, title="", indent=4,
                     fmt="%", dates=None):
    """Print ASCII line chart with N overlaid series.

    series_list : list of (label, values_array) in display-priority order.
                  The first series is drawn last (highest priority on overlap).
    Colors assigned in order: GREEN, CYAN, YELLOW, MAGENTA, BLUE, WHITE, RED.
    """
    if not series_list:
        return

    COLORS = [C.GREEN, C.CYAN, C.YELLOW, C.MAGENTA, C.BLUE, C.WHITE, C.RED]

    # Downsample / align every series to the same chart width
    processed = []
    ref_dates = None
    for label, values in series_list:
        arr, ds = _chart_common(values, width, fmt, dates=dates)
        if len(arr) < 2:
            continue
        processed.append((label, arr))
        if ref_dates is None:
            ref_dates = ds
    if not processed:
        return

    n = min(len(arr) for _, arr in processed)
    processed = [(lbl, arr[:n]) for lbl, arr in processed]
    if ref_dates is not None:
        ref_dates = ref_dates[:n]

    # Global y-axis range across ALL series
    vmin = min(float(np.min(arr)) for _, arr in processed)
    vmax = max(float(np.max(arr)) for _, arr in processed)
    rng = vmax - vmin
    if rng == 0:
        rng = abs(vmax) * 0.1 or 1.0
        vmin -= rng / 2; vmax += rng / 2; rng = vmax - vmin

    pad = " " * indent
    y_labels = [_fmt_y(vmin + rng * i / (height - 1), fmt) for i in range(height)]
    label_width = max(len(lbl) for lbl in y_labels)

    # Build one canvas per series
    canvases = []
    for _, arr in processed:
        canvas = [[" "] * n for _ in range(height)]
        rows = _compute_rows(arr, vmin, vmax, rng, height)
        _draw_line_on_canvas(canvas, rows, n)
        canvases.append(canvas)

    lines = []
    if title:
        lines.append(f"\n{pad}{C.BOLD}{title}{C.RESET}")
        legend_parts = [
            f"{COLORS[i % len(COLORS)]}──{C.RESET} {lbl}"
            for i, (lbl, _) in enumerate(processed)
        ]
        lines.append(f"{pad}  " + "    ".join(legend_parts))
        lines.append(f"{pad}{' ' * label_width} ┌{'─' * n}┐")

    for row_idx in range(height):
        y_idx = height - 1 - row_idx
        label = y_labels[y_idx].rjust(label_width)
        row_str = ""
        for c in range(n):
            cell = " "
            # Iterate in reverse priority so first series wins on overlap
            for i in range(len(canvases) - 1, -1, -1):
                if canvases[i][row_idx][c] in _LINE_CHARS:
                    col = COLORS[i % len(COLORS)]
                    cell = f"{col}{canvases[i][row_idx][c]}{C.RESET}"
            row_str += cell
        lines.append(f"{pad}{C.DIM}{label}{C.RESET} ┤{row_str}│")

    lines.append(f"{pad}{' ' * label_width} └{'─' * n}┘")
    year_line = _year_axis(ref_dates, n, label_width, indent)
    if year_line:
        lines.append(year_line)
    print("\n".join(lines))


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    starting_capital: float = 10_000.0
    max_position_pct: float = 0.95        # Max fraction of capital in position
    max_leverage: float = 1.0             # Max leverage (1.0 = no leverage)
    commission_pct: float = 0.001         # Commission as fraction of notional
    slippage_pct: float = 0.0005          # Slippage as fraction of price
    max_drawdown_pct: float = 0.15
    # Trailing Stops
    use_trailing_stops: bool = True
    trailing_stop_lookback: int = 5
    trailing_stop_initial_distance: float = 0.0
    # Asymmetric Stop Loss
    use_asymmetric_stops: bool = True
    loss_stop_pct: float = 0.015              # 1.5% stop when losing (grid search optimal)
    loss_stop_atr_mult: float = 1.5           # 1.5 ATR stop when losing
    profit_trail_pct: float = 0.05            # Trail 5% from peak (grid search optimal)
    profit_trail_atr_mult: float = 3.0        # 3 ATR trailing stop
    vol_adjust_stops: bool = True             # Adjust stops for volatility
    vol_baseline: float = 0.15                # Baseline volatility (15%)
    vol_max_adjustment: float = 2.0           # Max vol adjustment factor
    time_tighten_enabled: bool = False        # Tighten stops over time
    time_tighten_bars: int = 10               # Start tightening after N bars
    time_tighten_factor: float = 0.5          # Tighten to 50% of original
    # Network
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    use_attention: bool = True
    backbone_type: str = "conv_attn"  # "conv_attn" or "transformer"
    # Derived (set in __post_init__)
    obs_dim: int = 0
    # PPO
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine_warm_restarts"  # "cosine" or "cosine_warm_restarts"
    lr_restart_period: int = 0                 # T_0 for warm restarts (0 = auto: per-iteration)
    lr_restart_mult: int = 2                   # T_mult: period doubles each restart (SGDR)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.03
    ent_floor: float = 0.3                     # Entropy floor — boost ent_coef if entropy drops below
    ent_boost_factor: float = 3.0              # Multiply ent_coef by this when below floor
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    batch_size: int = 4096              # GPU-optimized: 8x larger (was 512)
    n_epochs: int = 10
    target_kl: float = 0.03
    plateau_patience: int = 2                  # Reset LR if champion score stalls for N iterations
    # Risk Head (auxiliary loss -- predicts forward realized volatility)
    risk_coef: float = 0.1              # Weight of risk auxiliary loss in PPO
    risk_horizon: int = 20              # N bars forward for realized vol target
    # MCTS Distillation
    mcts_kl_coef: float = 0.5           # Weight of KL(-_net || -_mcts) in PPO loss
    # GPU
    use_amp: bool = True
    use_compile: bool = True
    use_multi_gpu: bool = True
    n_envs: int = 64                    # GPU-optimized: 4x more parallel (was 16)
    pin_memory: bool = True
    no_subproc: bool = True              # SubprocVecEnv disabled (use TorchVecEnv for GPU parallelism)
    # Self-Play (increased budget: 8 iterations × 100k steps = 800k total)
    n_iterations: int = 8
    total_timesteps_per_iter: int = 100_000
    champion_margin: float = 0.1        # Additive margin (was multiplicative -- broken on negatives)
    # MCTS — Deep Search (AlphaGo-style 15-20 step lookahead)
    mcts_rollouts: int = 256            # UP from 32: 8x more sims → tree depth ~10-12
    mcts_exploration: float = 1.0       # DOWN from 1.414: deeper principal variation
    mcts_temperature: float = 1.0
    mcts_batch_size: int = 64           # DOWN from 128: fewer roots, more rollouts per root
    # Environment -- target-position action space
    # Actions represent target exposure: {-1.0, -0.5, 0, +0.5, +1.0}
    # Enables: long, short, flat, continuous sizing -- all native
    window_size: int = 60
    max_episode_steps: int = 5000
    n_actions: int = 5
    action_targets: Tuple[float,...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    # Reward (stationary base + auxiliary penalties)
    reward_scale: float = 100.0           # Scale log-returns to useful range
    reward_drawdown_penalty: float = 2.0  # Continuous quadratic DD penalty
    reward_turnover_cost: float = 0.05    # Proportional to dollar turnover (was 0.5 — caused zero trades)
    reward_holding_bonus: float = 0.0     # Removed: was rewarding FLAT, contributing to zero trades
    cvar_quantile: float = 0.05
    reward_cvar_penalty: float = 0.5
    reward_soft_clip_scale: float = 3.0   # tanh(reward/scale) soft clip
    use_dsr: bool = True
    dsr_eta: float = 0.001
    # Validation
    train_ratio: float = 0.70
    walk_forward_windows: int = 5
    wf_consistency_min: float = 0.4        # Min walk-forward PnL consistency
    monte_carlo_permutations: int = 100
    regime_segment_test: bool = True
    deflated_sharpe_threshold: float = 0.05  # p-value for deflated Sharpe gate
    perm_p_threshold: float = 0.10           # p-value for permutation/sign-flip gate (separate!)
    val_k: int = 3                         # Number of symbols to validate on
    val_pass_ratio: float = 0.5            # Fraction that must pass validation
    # Microstructure robustness gates
    micro_sharpe_std_max: float = 1.5      # Max Sharpe std across friction seeds
    micro_sharpe_min: float = -0.5         # Min Sharpe across any friction seed
    micro_n_seeds: int = 5                 # Number of friction seeds
    micro_n_symbols: int = 2              # Symbols to test microstructure on
    # MCTS distillation
    mcts_distill_temp: float = 1.5         # Temperature to soften MCTS targets (>1 = softer)
    # MCTS exploration noise (AlphaGo-style Dirichlet)
    mcts_dirichlet_alpha: float = 0.3      # Dirichlet concentration parameter
    mcts_dirichlet_eps: float = 0.25       # Blend: (1-eps)*prior + eps*noise
    # MCTS planning realism
    mcts_stochastic: bool = True           # Use stochastic rollouts (not realized tape)
    mcts_bootstrap_block: int = 5          # Block size for regime-conditioned bootstrap
    mcts_sim_horizon: int = 60             # UP from 20: support 60-bar (3-month) lookahead
    # Virtual Loss (AlphaGo parallel exploration trick)
    mcts_virtual_loss: float = 3.0        # Magnitude of virtual loss (discourages revisiting)
    mcts_use_virtual_loss: bool = True    # Enable virtual loss for deeper tree exploration
    # Progressive Widening (reduce effective branching 5→3 for deeper search)
    mcts_progressive_widening: bool = True  # Expand only top-K children initially
    mcts_pw_alpha: float = 0.5           # Widen when N^alpha > |children|
    mcts_pw_max_children: int = 3        # Hard cap: start with top-3 by prior probability
    # Continuation Rollouts (add +5 depth per rollout cheaply, no GPU)
    mcts_continuation_steps: int = 5     # Extra greedy steps after leaf expansion
    # Iterative Deepening (scale rollouts with training iteration)
    mcts_schedule_enabled: bool = True   # Scale rollouts as value network improves
    mcts_schedule_base: int = 64         # Rollouts at iteration 2 (shallow, network still weak)
    mcts_schedule_max: int = 512         # Rollouts at final iteration (full depth)
    # Wave batching (GPU saturation)
    mcts_wave_size: int = 0              # Rollouts per GPU batch wave (0 = auto: target batch ~256)
    # --- Institutional splitting / leakage control ---
    val_ratio: float = 0.10                # Fraction for validation (time-based)
    holdout_ratio: float = 0.10            # Fraction for holdout (never seen in training)
    embargo_bars: int = 100                # Gap between splits (with purge_gap=21, total separation=121 bars covers Trend/Hurst 126-bar lookback)
    split_by_symbol: bool = True           # Create disjoint symbol pools
    holdout_seed: int = 123                # Seed for reproducible symbol pool assignment
    # --- Execution realism ---
    spread_bps: float = 1.0               # Bid/ask spread in basis points
    impact_coef: float = 0.15             # Market impact strength
    adv_window: int = 20                   # Rolling volume window for ADV
    use_impact_model: bool = True          # FIX Â§4.5: Always use canonical sqrt-impact model
    trade_at_next_open: bool = False       # Execute at next bar's open (validation realism)
    borrow_bps_daily: float = 30.0         # Short borrow cost (bps/day)
    bars_per_year: int = 252               # Bars per year (adjust by timeframe)
    # --- Constrained RL (Lagrangian) ---
    use_lagrangian: bool = True            # Enable constraint multipliers
    target_dd: float = 0.15               # Max drawdown target
    target_turnover_frac: float = 5.0     # Annual turnover fraction target (was 0.50 — too restrictive)
    target_cvar: float = 0.02             # CVaR constraint target
    lagrangian_lr: float = 0.01           # Multiplier learning rate
    # --- Validation methodology ---
    validate_delay: bool = True            # Test with next-open execution in validation
    placebo_feature_shuffle: bool = True   # Run placebo test (shuffled features)
    placebo_fail_pnl_threshold: float = 0.0  # Placebo should not profit
    # --- Capital sizing mode ---
    use_realized_capital_sizing: bool = False  # Size on starting_capital + realized PnL only (not MTM)
    # --- Engineering ---
    deterministic_eval: bool = True        # Use argmax (not sampling) in evaluation
    # --- VWAP/TWAP order slicing (#5) ---
    use_order_slicing: bool = True         # Slice large orders into child fills
    n_slices: int = 5                      # Number of child orders per rebalance
    slice_mode: str = "TWAP"              # "TWAP" or "VWAP"
    # --- Kill switches & safety (#7) ---
    max_episode_loss_pct: float = 0.05       # Max loss per episode window (fraction of capital)
    max_turnover_spike: float = 5.0        # Max step turnover / avg step turnover ratio
    drift_z_threshold: float = 6.0         # Feature distribution drift z-score for kill
    use_kill_switches: bool = True         # Enable safety monitors
    # --- Regime detection (#4) ---
    regime_vol_lookback: int = 20          # Volatility lookback for regime
    regime_trend_lookback: int = 50        # Trend lookback for regime
    n_regimes: int = 4                     # Number of regime states
    # --- Cash yield & turnover annualization (#8) ---
    cash_yield_bps_annual: float = 400.0   # Risk-free rate on idle cash (bps/year)
    bars_per_year_map: Dict[str, int] = field(default_factory=lambda: {
        "1m":252*390,"2m":252*195,"5m":252*78,"15m":252*26,
        "30m":252*13,"60m":252*7,"1h":252*7,"90m":252*5,
        "1d":252,"5d":52,"1wk":52,"1mo":12})
    # --- Action masking (#8) ---
    use_action_masking: bool = True        # Restrict actions when near ruin/max DD
    dd_mask_threshold: float = 0.20        # DD level to start masking risky actions (was 0.12 — too aggressive)
    # Synthetic
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
    timeframes: Tuple[str, ...] = ("5m", "15m", "30m", "1h", "1d")
    download_period: str = "max"
    intraday_periods: Dict[str, str] = field(default_factory=lambda: {
        "5m":"60d","15m":"60d","30m":"60d","1h":"730d","1d":"max"})
    n_workers: int = 0
    verbose: int = 1

    def __post_init__(self):
        if not self.cache_dir: self.cache_dir = os.path.join(os.path.expanduser("~"), ".alpha_trade_v2_cache")
        if not self.output_dir: self.output_dir = os.path.join(os.path.expanduser("~"), "alpha_trade_v2_results")
        os.makedirs(self.cache_dir, exist_ok=True); os.makedirs(self.output_dir, exist_ok=True)
        if self.n_workers == 0: self.n_workers = max(1, mp.cpu_count() - 1)
        if not HAS_CUDA: self.use_amp = False; self.use_multi_gpu = False; self.pin_memory = False
        if not HAS_COMPILE: self.use_compile = False
        if self.use_multi_gpu and N_GPUS > 1: self.batch_size *= N_GPUS
        self.obs_dim = int(self.window_size * NUM_FEATURES)
        # FIX Â§4.2: Apply cuDNN deterministic settings based on config
        if HAS_CUDA:
            if self.deterministic_eval:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
FEATURE_COLUMNS = [
    "norm_close","norm_open","norm_high","norm_low","bar_range_atr","body_pct",
    "upper_wick_pct","lower_wick_pct","gap_pct",
    "rsi_14","rsi_9","macd_hist_norm","macd_signal_cross","stoch_k","stoch_d",
    "willr_14","cci_norm","roc_5","roc_10",
    "sma_cross_20_50","ema_cross_9_21","adx_norm","plus_di_norm","minus_di_norm",
    "price_vs_sma_20","price_vs_sma_50",
    "atr_pct","range_ratio","bb_position","bb_width","realized_vol_20",  # Tier 1: Added range_ratio
    "vol_ratio","obv_slope","consec_up","consec_down","drop_from_high_5",
    "vol_regime","trend_strength","mean_reversion_signal",
    "hurst_exponent","vol_of_vol","atr_regime_z",  # Tier 1 & 3: Regime and vol features
    "dd_duration_norm","dd_vol_signal","lyapunov",  # NEW: DD+Vol and Lyapunov features
    "regime_0","regime_1","regime_2","regime_3",
    "has_position","position_pnl_pct","bars_in_trade","position_direction",
]
NUM_FEATURES = len(FEATURE_COLUMNS)

def compute_indicators(df):
    df = df.copy()
    # Ensure Volume exists (synthetic data or stripped CSVs may lack it)
    if "Volume" not in df.columns:
        df["Volume"] = 1_000_000
    c,h,l,o,v = [df[x].values.astype(np.float64) for x in ["Close","High","Low","Open","Volume"]]
    n = len(c)
    if n < 60: return df
    for p in [9, 14]:
        d = np.diff(c, prepend=c[0]); g = np.where(d>0,d,0.); lo = np.where(d<0,-d,0.)
        ag = pd.Series(g).rolling(p,min_periods=1).mean().values
        al = pd.Series(lo).rolling(p,min_periods=1).mean().values
        df[f"RSI_{p}"] = 100. - 100./(1.+ag/(al+1e-10))
    e12 = pd.Series(c).ewm(span=12,adjust=False).mean().values
    e26 = pd.Series(c).ewm(span=26,adjust=False).mean().values
    ml = e12 - e26; ms = pd.Series(ml).ewm(span=9,adjust=False).mean().values
    df["MACD"]=ml; df["MACD_Sig"]=ms; df["MACD_Hist"]=ml-ms
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1)))); tr[0]=h[0]-l[0]
    df["ATR"] = pd.Series(tr).rolling(14,min_periods=1).mean().values
    s20 = pd.Series(c).rolling(20,min_periods=1).mean().values
    st20 = pd.Series(c).rolling(20,min_periods=1).std().values
    df["BB_Upper"]=s20+2*st20; df["BB_Lower"]=s20-2*st20; df["BB_Mid"]=s20
    for p in [5,20,50,200]: df[f"SMA_{p}"] = pd.Series(c).rolling(p,min_periods=1).mean().values
    for p in [9,21]: df[f"EMA_{p}"] = pd.Series(c).ewm(span=p,adjust=False).mean().values
    lm = pd.Series(l).rolling(14,min_periods=1).min().values
    hm = pd.Series(h).rolling(14,min_periods=1).max().values
    df["Stoch_K"]=100*(c-lm)/(hm-lm+1e-10)
    df["Stoch_D"]=pd.Series(df["Stoch_K"]).rolling(3,min_periods=1).mean().values
    hh=pd.Series(h).rolling(14,min_periods=1).max().values
    ll=pd.Series(l).rolling(14,min_periods=1).min().values
    df["WillR_14"]=-100*(hh-c)/(hh-ll+1e-10)
    tp=(h+l+c)/3; tps=pd.Series(tp).rolling(20,min_periods=1).mean().values
    tpst=pd.Series(tp).rolling(20,min_periods=1).std().values
    df["CCI"]=(tp-tps)/(0.015*tpst+1e-10)
    pdm=np.maximum(np.diff(h,prepend=h[0]),0.); mdm=np.maximum(-np.diff(l,prepend=l[0]),0.)
    pdm[pdm<mdm]=0.; mdm[mdm<pdm]=0.
    str_=pd.Series(tr).rolling(14,min_periods=1).mean().values
    pdi=100*pd.Series(pdm).rolling(14,min_periods=1).mean().values/(str_+1e-10)
    mdi=100*pd.Series(mdm).rolling(14,min_periods=1).mean().values/(str_+1e-10)
    df["ADX"]=pd.Series(100*np.abs(pdi-mdi)/(pdi+mdi+1e-10)).rolling(14,min_periods=1).mean().values
    df["Plus_DI"]=pdi; df["Minus_DI"]=mdi
    vs=pd.Series(v).rolling(20,min_periods=1).mean().values; df["Vol_Ratio"]=v/(vs+1.)
    _dir = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(_dir * v)
    df["OBV"]=obv; os_=pd.Series(obv).rolling(10,min_periods=1).mean().values
    df["OBV_Slope"]=(obv-os_)/(np.abs(os_)+1e-10)
    for p in [5,10]:
        sh=np.roll(c,p); r_=(c-sh)/(sh+1e-10)*100; r_[:p]=0; df[f"ROC_{p}"]=r_
    up=(c>np.roll(c,1)).astype(float); dn=(c<np.roll(c,1)).astype(float); up[0]=dn[0]=0
    cu=np.zeros(n); cd=np.zeros(n)
    for i in range(1,n): cu[i]=(cu[i-1]+1) if up[i] else 0; cd[i]=(cd[i-1]+1) if dn[i] else 0
    df["Consec_Up"]=cu; df["Consec_Down"]=cd
    rh5=pd.Series(h).rolling(5,min_periods=1).max().values
    df["Drop_From_High_5"]=(c-rh5)/(rh5+1e-10)*100
    lr_=np.log(c/np.roll(c,1)); lr_[0]=0
    df["Realized_Vol_20"]=pd.Series(lr_).rolling(20,min_periods=1).std().values*np.sqrt(252)
    v20=pd.Series(lr_).rolling(20,min_periods=1).std().values
    v60=pd.Series(lr_).rolling(60,min_periods=1).std().values
    df["Vol_Regime"]=v20/(v60+1e-10)
    df["Trend_Strength"]=(df["SMA_20"].values-df["SMA_50"].values)/(df["ATR"].values+1e-10)
    bw=df["BB_Upper"].values-df["BB_Lower"].values+1e-10
    df["Mean_Rev_Signal"]=(c-df["BB_Mid"].values)/bw

    # Tier 1: Hurst exponent for regime detection (vectorized R/S analysis)
    # Compute rolling Hurst on 126-bar windows (6 months)
    hurst_window = 126
    hurst_vals = np.full(n, 0.5)  # Default to 0.5 (random walk)
    if n >= hurst_window + 1:
        from numpy.lib.stride_tricks import sliding_window_view as _swv_h
        # Windows of prices: shape (n_windows, hurst_window+1)
        # window_prices[w] = c[w : w + hurst_window + 1]
        # Result goes into hurst_vals[w + hurst_window]
        _price_windows = _swv_h(c, hurst_window + 1)  # (n - hurst_window, 127)
        _n_hw = len(_price_windows)

        lags = [10, 20, 40, 63]
        log_lags = np.log(np.array(lags, dtype=np.float64))

        # For each lag, compute R/S for ALL windows at once
        log_rs_all = np.zeros((_n_hw, len(lags)), dtype=np.float64)
        for li, lag in enumerate(lags):
            # sub = window_prices[-lag:] for each window → _price_windows[:, -lag:]
            subs = _price_windows[:, -lag:]  # (n_hw, lag)
            means = subs.mean(axis=1, keepdims=True)  # (n_hw, 1)
            cumdevs = np.cumsum(subs - means, axis=1)  # (n_hw, lag)
            R = cumdevs.max(axis=1) - cumdevs.min(axis=1)  # (n_hw,)
            S = subs.std(axis=1) + 1e-10  # (n_hw,)
            log_rs_all[:, li] = np.log(R / S + 1e-10)

        # Vectorized polyfit: slope = cov(log_lags, log_rs) / var(log_lags)
        # log_lags is the same for all windows, so var(log_lags) is a constant
        _ll_mean = log_lags.mean()
        _ll_var = np.sum((log_lags - _ll_mean) ** 2)
        _lr_mean = log_rs_all.mean(axis=1)  # (n_hw,)
        # cov = sum((log_lags[i] - mean) * (log_rs[w,i] - mean_w)) for each w
        _cov = np.sum((log_lags[None, :] - _ll_mean) * (log_rs_all - _lr_mean[:, None]),
                       axis=1)  # (n_hw,)
        _slopes = _cov / (_ll_var + 1e-15)
        hurst_vals[hurst_window:hurst_window + _n_hw] = np.clip(_slopes, 0.0, 1.0)
    df["Hurst_Exponent"] = hurst_vals

    # Tier 3: Vol-of-vol (volatility of volatility) — vectorized rolling std
    # Old code: vov[idx] = np.std(vals[idx-20:idx]) — window ends BEFORE idx
    # So we shift(1) to align: rolling at idx-1 = std of [idx-20, idx-1], placed at idx
    df["Vol_of_Vol"] = pd.Series(df["Realized_Vol_20"].values).rolling(
        20, min_periods=20).std(ddof=0).shift(1).fillna(0).values

    # Tier 3: ATR regime (z-score of ATR relative to 60-bar history) — vectorized
    # Old code: mean/std of atr_vals[idx-60:idx] (excludes current), z = (current - mean) / std
    atr_vals = df["ATR"].values
    _atr_s = pd.Series(atr_vals)
    _atr_rm = _atr_s.rolling(60, min_periods=60).mean().shift(1)
    _atr_rs = _atr_s.rolling(60, min_periods=60).std(ddof=0).shift(1) + 1e-10
    df["ATR_Regime_Z"] = ((_atr_s - _atr_rm) / _atr_rs).fillna(0).values

    # NEW: DD Duration (bars since price peak)
    dd_duration = np.zeros(n)
    peak_idx = 0
    for i in range(n):
        if c[i] >= c[peak_idx]:
            peak_idx = i
            dd_duration[i] = 0
        else:
            dd_duration[i] = i - peak_idx
    df["DD_Duration"] = dd_duration
    df["DD_Duration_Norm"] = np.clip(dd_duration / 60, 0, 2)  # Normalize: 60 bars = 1.0

    # NEW: DD+Vol Signal (combine DD duration with vol regime)
    vol_regime_vals = df["Vol_Regime"].values
    dd_vol_signal = df["DD_Duration_Norm"].values * vol_regime_vals
    df["DD_Vol_Signal"] = dd_vol_signal

    # NEW: Lyapunov Exponent (chaos detection) — VECTORIZED
    df["Lyapunov"] = _precompute_lyapunov_vectorized(c)

    return df


def _precompute_lyapunov_vectorized(closes, window=100, embed_dim=3, delay=1,
                                     n_samples=20, max_dt=5):
    """
    Vectorized Lyapunov exponent for ALL bars at once.

    Replaces 8,900 sequential _compute_lyapunov_fast() calls with batch numpy
    operations — ~50-100× faster.

    Algorithm (same as original, just batched):
    1. Sliding windows of log returns → phase-space embeddings for ALL bars
    2. Batch distance computation (sampled reference points)
    3. Batch nearest-neighbor search (excluding temporal neighbors)
    4. Batch divergence tracking over dt=1..5
    5. Batch linear regression for Lyapunov slope

    Returns: np.ndarray of shape (n,), result[t] = Lyapunov at bar t.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(closes)
    result = np.zeros(n, dtype=np.float64)
    if n < window + 20:
        return result

    # --- 1. Full log-return series and sliding windows ---
    log_prices = np.log(np.maximum(closes, 1e-12))
    log_rets_full = np.diff(log_prices)  # shape: (n-1,)

    lr_len = window - 1               # 99 log returns per window
    embed_len = lr_len - embed_dim * delay  # 96 embedded points per window

    if embed_len < 20:
        return result

    # Sliding windows of log returns: shape (n_windows, lr_len)
    # Window t uses log_rets_full[t : t + lr_len]
    # Old code at idx uses c[idx-99:idx+1] → log_rets_full[idx-99:idx]
    # Window t matches old idx when t = idx - (window-1), so result[t + window-1] = old[idx]
    lr_windows = sliding_window_view(log_rets_full, lr_len)  # (n-window+1, 99)
    n_windows = len(lr_windows)
    if n_windows <= 0:
        return result

    # --- 2. Phase-space embedding (vectorized) ---
    # For embed_dim=3, delay=1: point[i] = [lr[i], lr[i+1], lr[i+2]]
    # This is just 3 overlapping slices of each window
    # embedded[w, i, d] = lr_windows[w, i + d * delay]
    embed_indices = np.arange(embed_len)[:, None] + np.arange(embed_dim)[None, :] * delay
    # embed_indices shape: (embed_len, embed_dim) = (96, 3)
    # Values: [[0,1,2], [1,2,3], ..., [95,96,97]]
    all_embedded = lr_windows[:, embed_indices]  # (n_windows, embed_len, embed_dim)

    # --- 3. Sample reference point indices (same as original) ---
    step = max(1, embed_len // n_samples)
    sample_idx = np.arange(0, embed_len - 10, step)  # e.g. [0,4,8,...,84] — match old code's full range
    n_samp = len(sample_idx)
    if n_samp == 0:
        return result

    # Pre-compute temporal exclusion mask (shared across chunks)
    all_j = np.arange(embed_len)[None, :]             # (1, E)
    samp_expanded = sample_idx[:, None]                 # (S, 1)
    temporal_mask_2d = np.abs(all_j - samp_expanded) <= 5  # (S, E)

    # --- Process in chunks to limit memory (~100 MB peak per chunk) ---
    CHUNK = 2000
    slopes_all = np.zeros(n_windows, dtype=np.float64)

    for c_start in range(0, n_windows, CHUNK):
        c_end = min(c_start + CHUNK, n_windows)
        C = c_end - c_start

        chunk_emb = all_embedded[c_start:c_end]       # (C, E, D)
        chunk_ref = chunk_emb[:, sample_idx, :]        # (C, S, D)

        # 4. Distance computation for this chunk
        diff = chunk_emb[:, None, :, :] - chunk_ref[:, :, None, :]  # (C, S, E, D)
        distances = np.sqrt(np.sum(diff * diff, axis=-1))  # (C, S, E)
        del diff

        # 5. Exclude temporal neighbors
        distances[:, temporal_mask_2d] = np.inf

        # 6. Nearest neighbor per (window, sample)
        nearest_idx = np.argmin(distances, axis=2)  # (C, S)
        cw = np.arange(C)[:, None]
        cs = np.arange(n_samp)[None, :]
        initial_dist = distances[cw, cs, nearest_idx]  # (C, S)
        del distances

        # 7. Divergence tracking for dt = 1..max_dt
        chunk_divs = []
        chunk_dts = []
        chunk_valids = []

        for dt in range(1, max_dt + 1):
            s_shifted = sample_idx[None, :] + dt  # (1, S)
            n_shifted = nearest_idx + dt           # (C, S)
            valid = (s_shifted < embed_len) & (n_shifted < embed_len) & (initial_dist > 1e-10)
            if not np.any(valid):
                continue
            s_safe = np.clip(s_shifted, 0, embed_len - 1)
            n_safe = np.clip(n_shifted, 0, embed_len - 1)
            pt_s = chunk_emb[cw, s_safe, :]    # (C, S, D)
            pt_n = chunk_emb[cw, n_safe, :]    # (C, S, D)
            cur_dist = np.sqrt(np.sum((pt_s - pt_n) ** 2, axis=-1))  # (C, S)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_div = np.log(cur_dist / initial_dist)
            valid = valid & (cur_dist > 1e-10) & np.isfinite(log_div) & (np.abs(log_div) < 5.0)
            for si in range(n_samp):
                mask = valid[:, si]
                if np.any(mask):
                    chunk_divs.append(log_div[:, si])
                    chunk_dts.append(np.full(C, dt, dtype=np.float64))
                    chunk_valids.append(mask)

        if not chunk_divs:
            continue

        # 8. Batch linear regression for this chunk
        ld = np.array(chunk_divs)     # (obs, C)
        td = np.array(chunk_dts)      # (obs, C)
        vm = np.array(chunk_valids)   # (obs, C)
        dm = np.where(vm, td, 0.0)
        ym = np.where(vm, ld, 0.0)
        cnt = vm.sum(axis=0).astype(np.float64)
        ok = cnt >= 10
        with np.errstate(divide='ignore', invalid='ignore'):
            d_mean = dm.sum(axis=0) / cnt
            y_mean = ym.sum(axis=0) / cnt
        dc = np.where(vm, td - d_mean[None, :], 0.0)
        yc = np.where(vm, ld - y_mean[None, :], 0.0)
        cov = (dc * yc).sum(axis=0)
        var = (dc * dc).sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            sl = cov / (var + 1e-15)
        slopes_all[c_start:c_end] = np.where(ok & np.isfinite(sl), sl, 0.0)

    # --- 9. Write results ---
    # Window t matches old idx = t + (window-1), so result[window-1 : window-1+n_windows]
    end_idx = min(window - 1 + n_windows, n)
    result[window - 1:end_idx] = slopes_all[:end_idx - (window - 1)]

    return result

def build_feature_matrix(df):
    """Build feature matrix with tracked column names (Fix #8: single source of truth)."""
    c,o,h,l = [df[x].values.astype(np.float64) for x in ["Close","Open","High","Low"]]
    atr=df["ATR"].values.astype(np.float64); n=len(c)
    As=np.maximum(atr,1e-8); Cs=np.maximum(c,1e-8)
    F_=np.zeros((n,NUM_FEATURES),dtype=np.float32)
    # Fix #8: Track names alongside assignments -- single source of truth
    _fn = []  # feature names in assignment order
    i = 0
    def _set(col, name):
        nonlocal i
        F_[:, i] = col; _fn.append(name); i += 1
    _set((c-df["SMA_20"].values)/As, "norm_close")
    _set((o-c)/As, "norm_open"); _set((h-c)/As, "norm_high"); _set((c-l)/As, "norm_low")
    _set((h-l)/As, "bar_range_atr"); _set(np.abs(c-o)/Cs*100, "body_pct")
    _set((h-np.maximum(c,o))/(h-l+1e-10), "upper_wick_pct")
    _set((np.minimum(c,o)-l)/(h-l+1e-10), "lower_wick_pct")
    g=(o-np.roll(c,1))/(np.roll(c,1)+1e-10)*100; g[0]=0; _set(g, "gap_pct")
    _set(df["RSI_14"].values/100-.5, "rsi_14"); _set(df["RSI_9"].values/100-.5, "rsi_9")
    _set(df["MACD_Hist"].values/As, "macd_hist_norm")
    mc=np.zeros(n); m=df["MACD"].values; ms=df["MACD_Sig"].values
    mc[1:]=np.sign(m[1:]-ms[1:])-np.sign(m[:-1]-ms[:-1]); _set(mc/2, "macd_signal_cross")
    _set(df["Stoch_K"].values/100-.5, "stoch_k"); _set(df["Stoch_D"].values/100-.5, "stoch_d")
    _set(df["WillR_14"].values/100+.5, "willr_14")
    _set(np.clip(df["CCI"].values/200,-1,1), "cci_norm")
    _set(np.clip(df["ROC_5"].values/10,-1,1), "roc_5")
    _set(np.clip(df["ROC_10"].values/10,-1,1), "roc_10")
    s20=df["SMA_20"].values; s50=df["SMA_50"].values
    _set(np.where(s20>s50,1.,-1.), "sma_cross_20_50")
    e9=df["EMA_9"].values; e21=df["EMA_21"].values
    _set(np.where(e9>e21,1.,-1.), "ema_cross_9_21")
    _set(np.clip(df["ADX"].values/50,0,1), "adx_norm")
    _set(np.clip(df["Plus_DI"].values/50,0,1), "plus_di_norm")
    _set(np.clip(df["Minus_DI"].values/50,0,1), "minus_di_norm")
    _set((c-s20)/As, "price_vs_sma_20"); _set((c-s50)/As, "price_vs_sma_50")
    _set(atr/Cs*100, "atr_pct")
    _set((h-l)/(atr+1e-10), "range_ratio")  # Tier 1: Intrabar range normalized by ATR
    bu=df["BB_Upper"].values; bl=df["BB_Lower"].values; br=bu-bl+1e-10
    _set((c-bl)/br-.5, "bb_position"); _set(br/Cs, "bb_width")
    _set(np.clip(df["Realized_Vol_20"].values,0,2), "realized_vol_20")
    _set(np.clip(df["Vol_Ratio"].values-1,-2,5), "vol_ratio")
    _set(np.clip(df["OBV_Slope"].values,-1,1), "obv_slope")
    _set(np.clip(df["Consec_Up"].values/5,0,1), "consec_up")
    _set(np.clip(df["Consec_Down"].values/5,0,1), "consec_down")
    _set(np.clip(df["Drop_From_High_5"].values/10,-1,0), "drop_from_high_5")
    _set(np.clip(df["Vol_Regime"].values-1,-1,2), "vol_regime")
    _set(np.clip(df["Trend_Strength"].values/3,-1,1), "trend_strength")
    _set(np.clip(df["Mean_Rev_Signal"].values,-1,1), "mean_reversion_signal")
    # Tier 1 & 3: Advanced regime and vol features
    _set(np.clip(df["Hurst_Exponent"].values-0.5,-0.5,0.5)*2, "hurst_exponent")  # Normalize to [-1, 1]
    _set(np.clip(df["Vol_of_Vol"].values,0,0.5)*2, "vol_of_vol")  # Normalize to [0, 1]
    _set(np.clip(df["ATR_Regime_Z"].values,-3,3)/3, "atr_regime_z")  # Normalize z-score to [-1, 1]
    # NEW: DD+Vol and Lyapunov features
    _set(df["DD_Duration_Norm"].values, "dd_duration_norm")  # Already normalized [0, 2]
    _set(np.clip(df["DD_Vol_Signal"].values,-3,3)/3, "dd_vol_signal")  # Normalize to [-1, 1]
    _set(np.clip(df["Lyapunov"].values,-0.1,0.1)*10, "lyapunov")  # Normalize to [-1, 1]
    # Regime detection one-hot: O(n) vectorized rolling approach
    vol_lb = 20; trend_lb = 50
    if n > trend_lb + 1:
        log_c = np.log(c + 1e-12)
        log_rets = np.diff(log_c, prepend=log_c[0])
        roll_vol = pd.Series(log_rets).rolling(vol_lb, min_periods=5).std().values
        roll_vol = np.nan_to_num(roll_vol, nan=0.0)
        trend_arr = np.zeros(n)
        trend_arr[trend_lb:] = c[trend_lb:] / (c[:-trend_lb] + 1e-12) - 1.0
        vol_q60 = pd.Series(roll_vol).expanding(min_periods=50).quantile(0.6).values
        vol_q60 = np.nan_to_num(vol_q60, nan=0.015)
        high_vol = roll_vol > vol_q60
        trending = np.abs(trend_arr) > 0.01
        regime_ids = np.where(~high_vol & ~trending, 0,
                     np.where(~high_vol & trending, 1,
                     np.where(high_vol & ~trending, 2, 3)))
        for rid_val in range(4):
            F_[regime_ids == rid_val, i + rid_val] = 1.0
    else:
        F_[:, i] = 1.0
    _fn.extend(["regime_0","regime_1","regime_2","regime_3"]); i += 4
    # Position state (filled at runtime by TradingEnv._obs)
    _fn.extend(["has_position","position_pnl_pct","bars_in_trade","position_direction"])
    # i += 4 not needed -- these are zero-initialized and written by env
    # Validate: _fn must match FEATURE_COLUMNS exactly
    assert _fn == FEATURE_COLUMNS, (
        f"Feature name mismatch!\n  Built: {_fn}\n  Expected: {FEATURE_COLUMNS}"
    )
    return np.nan_to_num(F_,nan=0.,posinf=1.,neginf=-1.)


# ============================================================================
# SYNTHETIC DATA
# ============================================================================
class SyntheticMarketGenerator:
    REGIMES=["trending_up","trending_down","mean_reverting","high_volatility"]
    def __init__(self,cfg,seed=42): self.cfg=cfg; self.rng=np.random.RandomState(seed)
    def generate(self,n=2000,p0=100.):
        prices=np.zeros((n,4)); vols=np.zeros(n); price=p0
        reg=self.rng.randint(0,4); bv=self.rng.uniform(*self.cfg.synthetic_vol_range)
        for i in range(n):
            if self.rng.random()<self.cfg.regime_transition_prob:
                reg=self.rng.randint(0,4); bv=self.rng.uniform(*self.cfg.synthetic_vol_range)
            r=self.REGIMES[reg]
            if r=="trending_up": dr,vol=self.rng.uniform(.0005,.003),bv*.8
            elif r=="trending_down": dr,vol=self.rng.uniform(-.003,-.0005),bv
            elif r=="mean_reverting": dr,vol=-.5*(price-p0)/p0*.01,bv*.6
            else: dr,vol=self.rng.uniform(-.001,.001),bv*2.5
            ret=dr+vol*self.rng.randn(); cl=price*(1+ret)
            iv=vol*self.rng.uniform(.5,2.)
            hi=max(price,cl)*(1+abs(iv*self.rng.randn()*.5))
            lo=min(price,cl)*(1-abs(iv*self.rng.randn()*.5))
            prices[i]=[price*(1+.001*self.rng.randn()),hi,lo,cl]
            vols[i]=max(1000,int(self.rng.lognormal(15,1))); price=cl
        df=pd.DataFrame(prices,columns=["Open","High","Low","Close"])
        df["Volume"]=vols.astype(int)
        df.index=pd.date_range(start="2020-01-01",periods=n,freq="D"); return df
    def generate_multiple(self,ns=10,nb=2000):
        d={}
        for i in range(ns):
            g=SyntheticMarketGenerator(self.cfg,seed=self.rng.randint(0,100000))
            d[f"SYN_{i:03d}"]=g.generate(nb,self.rng.uniform(20,500))
        return d


# ============================================================================
# DIFFERENTIAL SHARPE RATIO (Moody & Saffell, 2001)
# ============================================================================
class DifferentialSharpeReward:
    """
    Online incremental Sharpe ratio reward for RL.
    Computes the derivative of the Sharpe ratio w.r.t. the current return
    using exponential moving averages. More stable and theoretically grounded
    than batch-window Sharpe approximation.

    Reference: Moody & Saffell, "Learning to Trade via Direct Reinforcement", 2001
    """
    def __init__(self, eta=0.001):
        self.eta = eta
        self.A = 0.0   # EMA of returns
        self.B = 0.0   # EMA of squared returns
        self.n = 0

    def calculate(self, return_t):
        self.n += 1
        delta_A = return_t - self.A
        delta_B = (return_t ** 2) - self.B

        # Update EMAs
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        # Differential Sharpe: d(Sharpe)/d(R_t)
        # D_t = (B * delta_A - 0.5 * A * delta_B) / (B - A^2)^(3/2)
        variance = self.B - self.A ** 2
        if variance < 1e-12 or self.n < 5:
            return return_t  # Fallback to raw return until we have enough data

        denom = variance ** 1.5
        D_t = (self.B * delta_A - 0.5 * self.A * delta_B) / (denom + 1e-10)

        return float(np.clip(D_t, -2.0, 2.0))

    def reset(self):
        self.A = 0.0; self.B = 0.0; self.n = 0


# ============================================================================
# REGIME DETECTION (#4 -- lightweight, causal, no lookahead)
# ============================================================================
class SimpleRegimeDetector:
    """
    Classifies market regime from PAST data only (no lookahead).
    Uses rolling volatility + rolling trend to produce 4 regime states:
      0: low vol / mean-revert  (calm, choppy)
      1: low vol / trending     (smooth trend)
      2: high vol / mean-revert (volatile, choppy -- hedging regime)
      3: high vol / trending    (crisis or momentum)

    Output is a single integer regime_id suitable for one-hot or embedding.
    """
    def __init__(self, vol_lb=20, trend_lb=50):
        self.vol_lb = vol_lb
        self.trend_lb = trend_lb
        self.vol_history = []     # Running collection for quantile estimation
        self.vol_q60 = 0.015     # Default volatility quantile (updated online)

    def compute(self, closes):
        """
        closes: 1D array of recent close prices (at least trend_lb + 1 long).
        Returns: regime_id (int 0-3)
        """
        lb = max(self.vol_lb, self.trend_lb)
        if len(closes) < lb + 1:
            return 0  # Default calm regime for short histories

        x = closes[-(lb + 1):]
        lr = np.diff(np.log(x + 1e-12))

        # Rolling vol from recent window
        vol = float(np.std(lr[-self.vol_lb:]))
        self.vol_history.append(vol)
        # Update quantile estimate every 100 observations
        if len(self.vol_history) >= 100:
            self.vol_q60 = float(np.quantile(self.vol_history[-500:], 0.6))

        # Trend: total return over trend lookback
        trend = float(x[-1] / (x[-self.trend_lb] + 1e-12) - 1.0)

        high_vol = vol > self.vol_q60
        trending = abs(trend) > 0.01

        if not high_vol and not trending: return 0
        if not high_vol and trending:     return 1
        if high_vol and not trending:     return 2
        return 3

    def regime_onehot(self, regime_id, n_regimes=4):
        """Return one-hot vector for regime_id."""
        oh = np.zeros(n_regimes, dtype=np.float32)
        oh[min(regime_id, n_regimes - 1)] = 1.0
        return oh


# ============================================================================
# DRIFT MONITOR & KILL SWITCHES (#7 -- safety rails)
# ============================================================================
class DriftMonitor:
    """
    Monitors feature distribution for anomalous drift from training reference.
    Uses element-wise z-scores against reference mean/std.
    Triggers 'should_kill' when any feature exceeds z_threshold.
    """
    def __init__(self, ref_mean, ref_std, z_thresh=6.0):
        self.ref_mean = np.asarray(ref_mean, dtype=np.float64)
        self.ref_std = np.asarray(ref_std, dtype=np.float64) + 1e-12
        self.z_thresh = z_thresh
        self.max_z_seen = 0.0

    def score(self, x):
        """Compute max z-score across all features."""
        x = np.asarray(x, dtype=np.float64).ravel()
        # Only score the first len(ref_mean) features (obs may be flattened window)
        n = min(len(x), len(self.ref_mean))
        z = np.abs((x[:n] - self.ref_mean[:n]) / self.ref_std[:n])
        max_z = float(np.max(z))
        self.max_z_seen = max(self.max_z_seen, max_z)
        return max_z

    def should_kill(self, x):
        return self.score(x) > self.z_thresh

    @staticmethod
    def from_dataset(feat_array, z_thresh=6.0):
        """Create DriftMonitor from training feature array [N, obs_dim]."""
        ref_mean = np.mean(feat_array, axis=0)
        ref_std = np.std(feat_array, axis=0)
        return DriftMonitor(ref_mean, ref_std, z_thresh)


# ============================================================================
# ORDER SLICING -- VWAP/TWAP (#5 -- execution realism)
# ============================================================================
def slice_order(total_notional, n_slices=5, mode="TWAP"):
    """
    Split a large order into child orders for more realistic execution.
    TWAP: equal slices
    VWAP: front-loaded (approximates typical volume distribution)

    Returns: list of notional amounts per child slice.
    """
    if n_slices <= 1 or abs(total_notional) < 1.0:
        return [total_notional]

    if mode == "VWAP":
        # Front-load: more volume early (typical U-shaped intraday pattern)
        weights = np.linspace(1.3, 0.7, n_slices)
        weights = weights / weights.sum()
        return list(total_notional * weights)
    else:  # TWAP
        return [total_notional / n_slices] * n_slices


def execute_sliced_fill(fill_fn, side, mid, total_notional, n_slices=5, mode="TWAP",
                        impact_decay=0.8):
    """
    Execute a sliced order through fill_fn, with increasing impact per slice.
    Each child order's impact depends on cumulative participation.

    fill_fn: callable(side, ref_price, notional_abs) -> fill_price
    impact_decay: each slice sees decay * previous impact (mean-reverting)

    Returns: volume-weighted average fill price, total commission estimate.
    """
    slices = slice_order(abs(total_notional), n_slices, mode)
    total_qty = 0.0
    total_cost = 0.0
    cumulative_impact = 0.0

    for i, slice_notional in enumerate(slices):
        if slice_notional < 1.0:
            continue
        # Each subsequent slice faces slightly worse fills (cumulative impact)
        adjusted_mid = mid + cumulative_impact * (1.0 if side == "buy" else -1.0)
        fill = fill_fn(side, adjusted_mid, slice_notional)

        # Track for VWAP calculation
        qty = slice_notional / (mid + 1e-12)
        total_qty += qty
        total_cost += qty * fill

        # Impact accumulates but decays
        slice_impact = abs(fill - adjusted_mid)
        cumulative_impact = cumulative_impact * impact_decay + slice_impact

    if total_qty < 1e-12:
        return mid, 0.0

    vwap_fill = total_cost / total_qty
    return vwap_fill, total_qty
class TemporalAttention(nn.Module):
    def __init__(self,d,nh=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=d,num_heads=nh,batch_first=True,dropout=.1)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        o,_=self.attn(x,x,x); return self.norm(x+o)

class SharedBackbone(nn.Module):
    """Conv1d + Attention backbone (default)."""
    def __init__(self,nf,ws,hd=256,nl=3,do=.1,attn=True):
        super().__init__()
        self.ws=ws; self.nf=nf; self.hd=hd
        self.input_norm=nn.LayerNorm(nf)
        self.conv=nn.Sequential(
            nn.Conv1d(nf,hd,3,padding=1),nn.GELU(),nn.BatchNorm1d(hd),nn.Dropout(do),
            nn.Conv1d(hd,hd,3,padding=1),nn.GELU(),nn.BatchNorm1d(hd),nn.Dropout(do))
        self.use_attn=attn
        if attn: self.attention=TemporalAttention(hd,4)
        self.fc=nn.Sequential(nn.Linear(hd*ws,hd*2),nn.GELU(),nn.Dropout(do),nn.Linear(hd*2,hd),nn.GELU())
    def forward(self,x):
        B=x.shape[0]
        if x.dim()==2: x=x.view(B,self.ws,self.nf)
        x=self.input_norm(x)
        x=self.conv(x.permute(0,2,1)).permute(0,2,1)
        if self.use_attn: x=self.attention(x)
        return self.fc(x.reshape(B,-1))


class TransformerBackbone(nn.Module):
    """
    Full Transformer Encoder backbone with causal masking.
    Better at capturing long-range temporal dependencies in price sequences.
    Trade-off: ~2x more compute than Conv+Attn, but potentially better
    for longer windows or complex regime patterns.
    """
    def __init__(self, nf, ws, hd=256, nl=2, do=0.1, nh=4):
        super().__init__()
        self.ws = ws; self.nf = nf; self.hd = hd
        self.input_norm = nn.LayerNorm(nf)
        # Project features to model dimension
        self.embedding = nn.Linear(nf, hd)
        # Learnable positional encoding (more flexible than sinusoidal for finance)
        self.pos_encoding = nn.Parameter(torch.randn(1, ws, hd) * 0.02)
        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hd, nhead=nh, dim_feedforward=hd * 4,
            dropout=do, batch_first=True, activation='gelu',
            norm_first=True,  # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nl)
        self.norm = nn.LayerNorm(hd)
        # Project last hidden state
        self.fc = nn.Sequential(nn.Linear(hd, hd), nn.GELU(), nn.Dropout(do))
        # Cache causal mask
        self._causal_mask = None

    def _get_causal_mask(self, T, device):
        """Generate causal mask (prevent attending to future bars)."""
        if self._causal_mask is None or self._causal_mask.size(0) != T:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=device, dtype=torch.float32
            )
        return self._causal_mask

    def forward(self, x):
        B = x.shape[0]
        if x.dim() == 2:
            x = x.view(B, self.ws, self.nf)
        T = x.shape[1]
        # Normalize + Embed + positional
        x = self.input_norm(x)
        x = self.embedding(x) + self.pos_encoding[:, :T, :]
        # Causal mask
        mask = self._get_causal_mask(T, x.device)
        # Transformer encode
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.norm(x)
        # Use last timestep's representation (like GPT)
        last = x[:, -1, :]
        return self.fc(last)

class AlphaTradeNet(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; h=cfg.hidden_dim
        if cfg.backbone_type == "transformer":
            self.backbone=TransformerBackbone(NUM_FEATURES,cfg.window_size,h,
                                              cfg.num_layers,cfg.dropout)
        else:  # default: conv_attn
            self.backbone=SharedBackbone(NUM_FEATURES,cfg.window_size,h,
                                          cfg.num_layers,cfg.dropout,cfg.use_attention)
        self.policy_head=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,cfg.n_actions))
        self.value_head=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,1))
        self.risk_head=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,1),nn.Softplus())
        # Orthogonal init: small gain keeps initial outputs numerically safe
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.constant_(self.policy_head[-1].bias, 0.0)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.constant_(self.value_head[-1].bias, 0.0)
        nn.init.orthogonal_(self.risk_head[-2].weight, gain=0.01)  # -2: before Softplus
        nn.init.constant_(self.risk_head[-2].bias, 0.0)
    def forward(self,x):
        f=self.backbone(x)
        return self.policy_head(f),self.value_head(f).squeeze(-1),self.risk_head(f).squeeze(-1)
    def get_action_and_value(self,x,action=None):
        lo,v,r=self.forward(x)
        # NaN guard: if any env produced NaN obs, logits will be NaN and
        # Categorical will crash. Replace NaN logits with uniform (safe default).
        if not torch.isfinite(lo).all():
            lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
        d=Categorical(logits=lo)
        if action is None: action=d.sample()
        return action,d.log_prob(action),d.entropy(),v,r
    @torch.no_grad()
    def get_action_probs(self,x): lo,_,_=self.forward(x); return F.softmax(lo,dim=-1)
    @torch.no_grad()
    def get_value_only(self,x): return self.value_head(self.backbone(x)).squeeze(-1)

def build_network(cfg):
    net=AlphaTradeNet(cfg).to(DEVICE)
    if cfg.use_compile and HAS_COMPILE:
        try:
            net=torch.compile(net,mode="reduce-overhead")
            # Test forward pass to catch lazy Triton errors (esp. Windows)
            with torch.no_grad():
                dummy = torch.zeros(1, cfg.obs_dim, device=next(net.parameters()).device)
                net(dummy)
            tprint("torch.compile() ON","gpu")
        except Exception as e:
            tprint(f"torch.compile() failed ({e.__class__.__name__}), using eager","warn")
            net = unwrap_net(net)  # Remove compile wrapper if it was applied
    if cfg.use_multi_gpu and N_GPUS>1:
        net=nn.DataParallel(net); tprint(f"DataParallel: {N_GPUS} GPUs","gpu")
    return net

def unwrap_net(net):
    if isinstance(net,nn.DataParallel): return net.module
    if hasattr(net,"_orig_mod"): return net._orig_mod
    return net


# ============================================================================
# TRADING ENVIRONMENT -- Target-Position Action Space
# ============================================================================
# Actions = target exposure buckets: {-1.0, -0.5, 0.0, +0.5, +1.0}
# This enables: long, short, flat, continuous sizing -- all native.
# Transaction costs use a clean fill-price model applied consistently.
# Mark-to-market handles both long and short positions.
# ============================================================================

ACTION_NAMES = {0: "FULL_SHORT", 1: "HALF_SHORT", 2: "FLAT",
                3: "HALF_LONG", 4: "FULL_LONG"}

class TradingEnv(gym.Env):
    def __init__(self, feat, prices, cfg, sym="SYM", ev=False):
        super().__init__()
        self.feat = feat; self.prices = prices; self.cfg = cfg
        self.sym = sym; self.ev = ev; self.nb = len(feat); self.w = cfg.window_size
        self.action_targets = np.array(cfg.action_targets, dtype=np.float32)
        self.observation_space = spaces.Box(-10, 10, shape=(self.w * NUM_FEATURES,), dtype=np.float32)
        self.action_space = spaces.Discrete(cfg.n_actions)
        # #8: Auto-set bars_per_year from symbol timeframe
        parts = sym.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in cfg.bars_per_year_map:
            self.bars_per_year = cfg.bars_per_year_map[parts[1]]
        else:
            self.bars_per_year = cfg.bars_per_year
        # FIX Bug #11: Validate asymmetric stop config parameters
        if cfg.use_asymmetric_stops:
            if not (0 < cfg.loss_stop_pct < 0.5):
                raise ValueError(f"loss_stop_pct must be in (0, 0.5), got {cfg.loss_stop_pct}")
            if not (0 < cfg.profit_trail_pct < 1.0):
                raise ValueError(f"profit_trail_pct must be in (0, 1.0), got {cfg.profit_trail_pct}")
            if not (cfg.loss_stop_atr_mult > 0):
                raise ValueError(f"loss_stop_atr_mult must be positive, got {cfg.loss_stop_atr_mult}")
            if not (cfg.profit_trail_atr_mult > 0):
                raise ValueError(f"profit_trail_atr_mult must be positive, got {cfg.profit_trail_atr_mult}")
            if not (0.01 < cfg.vol_baseline < 10.0):
                raise ValueError(f"vol_baseline must be in (0.01, 10.0), got {cfg.vol_baseline}")
            if not (cfg.vol_max_adjustment > 0):
                raise ValueError(f"vol_max_adjustment must be positive, got {cfg.vol_max_adjustment}")
            if cfg.time_tighten_enabled and not (0 < cfg.time_tighten_factor < 1.0):
                raise ValueError(f"time_tighten_factor must be in (0, 1.0) to tighten stops, got {cfg.time_tighten_factor}")

        # Initialize asymmetric stop manager (will be set in _reset_state)
        self.asymmetric_stop_manager = None
        self._peak_pnl_pct = 0.0
        self._bars_in_current_trade = 0
        self._reset_state()

    def clone(self):
        """Lightweight clone for MCTS simulation. Shares immutable data, copies mutable state."""
        c = TradingEnv.__new__(TradingEnv)
        c.feat = self.feat; c.prices = self.prices; c.cfg = self.cfg
        c.sym = self.sym; c.ev = True; c.nb = self.nb; c.w = self.w
        c.action_targets = self.action_targets
        c.observation_space = self.observation_space
        c.action_space = self.action_space
        c.bars_per_year = self.bars_per_year
        # Copy mutable state
        c.cs = self.cs; c.cash = self.cash; c.peak_value = self.peak_value
        c.shares = self.shares; c.exposure = self.exposure
        c.entry_vwap = self.entry_vwap; c.entry_shares = self.entry_shares
        c.entry_step = self.entry_step
        c.prev_value = self.prev_value; c.max_dd = self.max_dd; c.turnover = self.turnover
        c.n_wins = self.n_wins; c.n_losses = self.n_losses
        c.gross_profit = self.gross_profit; c.gross_loss = self.gross_loss
        c.trade_log = []; c.trade_entries = []; c.return_history = list(self.return_history[-20:])
        c.step_rewards = []; c.np_random = self.np_random
        c.dsr = DifferentialSharpeReward(eta=self.cfg.dsr_eta)
        c.dsr.A = self.dsr.A; c.dsr.B = self.dsr.B; c.dsr.n = self.dsr.n
        # New state fields
        c.episode_pnl = self.episode_pnl
        c.episode_start_value = self.episode_start_value
        c.step_turnovers = deque(maxlen=100)
        c.kill_triggered = self.kill_triggered
        c.last_reward_components = {}
        c.regime_det = SimpleRegimeDetector(
            vol_lb=self.cfg.regime_vol_lookback,
            trend_lb=self.cfg.regime_trend_lookback
        )
        c.drift_monitor = getattr(self, 'drift_monitor', None)  # Share drift monitor
        # Trailing stop state
        c.trailing_stop_price = self.trailing_stop_price
        c.trailing_stop_entry_price = self.trailing_stop_entry_price
        # Asymmetric stop state - FIX Bug #5: Lightweight copy (not deepcopy) to avoid
        # shared mutable state while being fast enough for MCTS's ~4000 clones/symbol.
        if hasattr(self, 'asymmetric_stop_manager') and self.asymmetric_stop_manager is not None:
            from alphago_stop_loss import AsymmetricStopLoss
            asm = AsymmetricStopLoss.__new__(AsymmetricStopLoss)
            asm.config = self.asymmetric_stop_manager.config  # Immutable dataclass, safe to share
            asm.stops_hit = dict(self.asymmetric_stop_manager.stops_hit)
            asm.avg_loss_at_stop = []   # Stats not needed for simulation clones
            asm.avg_profit_at_stop = []
            c.asymmetric_stop_manager = asm
        else:
            c.asymmetric_stop_manager = None
        c._peak_pnl_pct = getattr(self, '_peak_pnl_pct', 0.0)
        c._bars_in_current_trade = getattr(self, '_bars_in_current_trade', 0)
        return c

    def _precompute_stochastic_cache(self, block_size=5, horizon=60):
        """Pre-compute regime data for efficient repeated stochastic cloning.

        Called ONCE per root env before MCTS rollouts. All subsequent
        stochastic_clone() calls reuse the cached regime info, block pool,
        and valid_starts — eliminating ~99.9% of redundant regime computation
        (was: recomputed identically for every clone, ~4000× per symbol).
        """
        lb = max(60, self.cfg.regime_trend_lookback + 1)
        start_idx = max(0, self.cs - lb)
        past_closes = self.prices[start_idx:self.cs + 1, 3].astype(np.float64)

        if len(past_closes) < 30:
            self._stochastic_cache = None
            return

        log_rets = np.diff(np.log(past_closes + 1e-12))
        if len(log_rets) < 20:
            self._stochastic_cache = None
            return

        # Regime classification
        recent_vol = float(np.std(log_rets[-20:]))
        overall_vol_med = float(np.median(np.abs(log_rets)))
        high_vol = recent_vol > overall_vol_med * 1.2

        # Adaptive block size
        r20 = log_rets[-20:]
        trend_strength = abs(float(r20.sum())) / (float(np.sum(np.abs(r20))) + 1e-10)
        if trend_strength > 0.40:
            block_size = max(block_size, 10)
        elif trend_strength < 0.15:
            block_size = max(3, block_size - 2)

        # Rolling vol — vectorized (was: Python for-loop over n_hist)
        n_hist = len(log_rets)
        roll_window = min(20, n_hist // 3)
        if roll_window < 5:
            self._stochastic_cache = None
            return

        import pandas as _pd
        hist_vol = _pd.Series(log_rets).rolling(roll_window, min_periods=1).std(ddof=0).values

        # Regime-matched block start indices
        if high_vol:
            regime_mask = hist_vol > np.median(hist_vol)
        else:
            regime_mask = hist_vol <= np.median(hist_vol)

        regime_indices = np.where(regime_mask)[0]
        if len(regime_indices) < block_size * 2:
            regime_indices = np.arange(n_hist)

        valid_starts = regime_indices[regime_indices <= n_hist - block_size]
        if len(valid_starts) == 0:
            valid_starts = np.arange(max(0, n_hist - block_size))

        self._stochastic_cache = {
            'log_rets': log_rets,
            'recent_vol': recent_vol,
            'block_size': block_size,
            'valid_starts': valid_starts,
            'last_close': float(past_closes[-1]),
        }

    def stochastic_clone(self, horizon=20, block_size=5):
        """
        Clone env with STOCHASTIC future prices for MCTS planning under uncertainty.

        Uses pre-computed regime cache (from _precompute_stochastic_cache) when
        available, falling back to full computation otherwise. The cache eliminates
        regime estimation + rolling vol computation (~4000 redundant calls per symbol).

        Synthetic OHLCV generation is fully vectorized (no Python for-loops).
        """
        c = self.clone()

        # Use cached regime data if available (pre-computed once per root env)
        cache = getattr(self, '_stochastic_cache', None)
        if cache is None:
            # No cache — compute from scratch (fallback for non-MCTS usage)
            self._precompute_stochastic_cache(block_size, horizon)
            cache = getattr(self, '_stochastic_cache', None)
            if cache is None:
                return c  # Too short for regime estimation

        log_rets = cache['log_rets']
        recent_vol = cache['recent_vol']
        block_size = cache['block_size']
        valid_starts = cache['valid_starts']
        last_close = cache['last_close']

        # Block bootstrap: sample blocks of consecutive returns
        n_blocks = (horizon + block_size - 1) // block_size  # ceil division
        starts = np.random.choice(valid_starts, size=n_blocks)
        # Gather all blocks at once (vectorized)
        blocks = [log_rets[s:s + block_size] for s in starts]
        sim_rets = np.concatenate(blocks)[:horizon]

        # Add noise proportional to estimation uncertainty
        sim_rets = sim_rets + np.random.normal(0, recent_vol * 0.1, size=len(sim_rets))

        # Vectorized synthetic OHLCV generation (was: Python for-loop over 60 bars)
        end_bar = min(self.cs + 1 + horizon, self.nb)
        n_synth = end_bar - (self.cs + 1)
        if n_synth <= 0:
            return c

        sim_rets = sim_rets[:n_synth]

        # Close prices via cumulative log returns
        cum_log_rets = np.cumsum(sim_rets)
        closes = last_close * np.exp(cum_log_rets)

        # Per-bar volatility for OHLC noise
        bar_vols = np.abs(sim_rets) + recent_vol * 0.3

        # Open prices: previous close * (1 + noise)
        prev_closes = np.empty(n_synth)
        prev_closes[0] = last_close
        prev_closes[1:] = closes[:-1]
        opens = prev_closes * (1.0 + np.random.normal(0, bar_vols * 0.2, size=n_synth))

        # High/Low from max/min of open,close + noise
        high_base = np.maximum(opens, closes)
        low_base = np.minimum(opens, closes)
        highs = high_base * (1.0 + np.abs(np.random.normal(0, bar_vols * 0.5, size=n_synth)))
        lows = low_base * (1.0 - np.abs(np.random.normal(0, bar_vols * 0.5, size=n_synth)))

        # Apply to prices array (copy only the modified slice for efficiency)
        synth_prices = self.prices.copy()
        bar_slice = slice(self.cs + 1, self.cs + 1 + n_synth)
        synth_prices[bar_slice, 0] = opens
        synth_prices[bar_slice, 1] = highs
        synth_prices[bar_slice, 2] = lows
        synth_prices[bar_slice, 3] = closes

        c.prices = synth_prices

        # FIX Bug #6: Reset peak PnL for stochastic clone since price path is synthetic
        if abs(c.shares) > 1e-9:
            c._peak_pnl_pct = 0.0
            # Note: _bars_in_current_trade remains unchanged - time in trade is still valid

        return c

    def _reset_state(self):
        self.cs = self.w                           # Current step (bar index)
        self.cash = self.cfg.starting_capital      # Cash balance
        self._realized_capital = self.cfg.starting_capital  # Capital base for sizing (realized mode)
        self.shares = 0.0                          # Signed shares (+ long, - short)
        self.exposure = 0.0                        # Current exposure fraction [-1, +1]
        self.entry_vwap = 0.0                      # Volume-weighted avg entry price
        self.entry_shares = 0.0                    # Shares at entry (for logging)
        self.entry_step = 0                        # Step when position was opened
        self.peak_value = self.cfg.starting_capital
        self.prev_value = self.cfg.starting_capital
        self.max_dd = 0.0                          # Max drawdown fraction
        self.turnover = 0.0                        # Cumulative dollar turnover
        self.trade_log = []                        # Completed round-trip trades (legacy - kept for compatibility)
        self.trade_entries = []                    # NEW: All entry/exit actions (every BUY/SELL)
        self.return_history = []                   # Per-step returns
        self.step_rewards = []
        self.n_wins = 0; self.n_losses = 0
        self.gross_profit = 0.0; self.gross_loss = 0.0
        self.dsr = DifferentialSharpeReward(eta=self.cfg.dsr_eta)
        # Kill switch tracking (#7)
        self.episode_pnl = 0.0                       # Episode PnL tracking (Fix #4: not truly daily without timestamps)
        self.episode_start_value = self.cfg.starting_capital
        self.step_turnovers = deque(maxlen=100)    # Rolling step turnover for spike detection
        self.kill_triggered = False                # Whether kill switch has fired
        # Reward decomposition (#8)
        self.last_reward_components = {}           # Audit trail for reward components
        # Per-step execution data (for pipeline L4 ingestion)
        self._last_step_mid = 0.0
        self._last_step_commission = 0.0
        self._last_step_traded = 0.0
        self._last_step_slippage_bps = 0.0
        # Regime detector (#4)
        self.regime_det = SimpleRegimeDetector(
            vol_lb=self.cfg.regime_vol_lookback,
            trend_lb=self.cfg.regime_trend_lookback
        )
        # FIX Â§4.4: Pipeline eval mode flag â€" prevents accidental future data access
        self._pipeline_eval_mode = False
        # FIX Â§4.1: Last cost breakdown from canonical model for diagnostics
        self._last_cost_breakdown = None
        # Trailing stop state
        self.trailing_stop_price = None            # Current stop loss price (None = no active stop)
        self.trailing_stop_entry_price = 0.0       # Price when position was entered (for stop calculation)

        # Asymmetric stop loss state
        if HAS_ASYMMETRIC_STOPS and self.cfg.use_asymmetric_stops:
            stop_config = StopLossConfig(
                loss_stop_pct=self.cfg.loss_stop_pct,
                loss_stop_atr_mult=self.cfg.loss_stop_atr_mult,
                profit_trail_pct=self.cfg.profit_trail_pct,
                profit_trail_atr_mult=self.cfg.profit_trail_atr_mult,
                vol_adjust_stops=self.cfg.vol_adjust_stops,
                vol_baseline=self.cfg.vol_baseline,
                vol_max_adjustment=self.cfg.vol_max_adjustment,
                time_tighten_enabled=self.cfg.time_tighten_enabled,
                time_tighten_bars=self.cfg.time_tighten_bars,
                time_tighten_factor=self.cfg.time_tighten_factor,
            )
            self.asymmetric_stop_manager = AsymmetricStopLoss(stop_config)
        else:
            self.asymmetric_stop_manager = None

        self._peak_pnl_pct = 0.0                  # Track peak P&L for trailing stops
        self._bars_in_current_trade = 0           # Track bars since entry

    def _get_mid_price(self):
        """Current close price as mid reference."""
        idx = min(self.cs, self.nb - 1)
        return float(self.prices[idx, 3])

    def _exec_price_ref(self):
        """Price reference for execution: current close or next open."""
        if self.cfg.trade_at_next_open:
            idx = min(self.cs + 1, self.nb - 1)
            return float(self.prices[idx, 0])  # Next bar open
        return self._get_mid_price()

    def _get_swing_low(self, lookback=5):
        """Calculate swing low over last N bars (for trailing stop)."""
        if self.cs < lookback:
            return None
        start_idx = max(0, self.cs - lookback)
        end_idx = self.cs
        lows = self.prices[start_idx:end_idx, 2]  # Column 2 = Low
        return float(np.min(lows)) if len(lows) > 0 else None

    def _get_swing_high(self, lookback=5):
        """Calculate swing high over last N bars (for trailing stop)."""
        if self.cs < lookback:
            return None
        start_idx = max(0, self.cs - lookback)
        end_idx = self.cs
        highs = self.prices[start_idx:end_idx, 1]  # Column 1 = High
        return float(np.max(highs)) if len(highs) > 0 else None

    def _check_trailing_stop(self, current_low, current_high):
        """
        Check if trailing stop was hit this bar.
        Returns (hit, exit_price) tuple.
        """
        if self.trailing_stop_price is None or self.shares == 0.0:
            return False, None

        if self.shares > 0:  # Long position
            if current_low <= self.trailing_stop_price:
                return True, self.trailing_stop_price
        else:  # Short position
            if current_high >= self.trailing_stop_price:
                return True, self.trailing_stop_price

        return False, None

    def _update_trailing_stop(self, lookback=5):
        """Update trailing stop to follow price (trail up for longs, down for shorts)."""
        if self.shares == 0.0:
            self.trailing_stop_price = None
            return

        if self.shares > 0:  # Long position - trail stop up
            swing_low = self._get_swing_low(lookback)
            if swing_low is not None:
                if self.trailing_stop_price is None:
                    self.trailing_stop_price = swing_low
                else:
                    self.trailing_stop_price = max(self.trailing_stop_price, swing_low)

        else:  # Short position - trail stop down
            swing_high = self._get_swing_high(lookback)
            if swing_high is not None:
                if self.trailing_stop_price is None:
                    self.trailing_stop_price = swing_high
                else:
                    self.trailing_stop_price = min(self.trailing_stop_price, swing_high)

    def _check_asymmetric_stop(self, current_price, atr, realized_vol=0.15):
        """
        Check if asymmetric stop loss should trigger.

        Args:
            current_price: Current market price
            atr: Average True Range
            realized_vol: Current realized volatility

        Returns:
            Dict with 'should_exit', 'stop_price', 'stop_type', 'pnl_at_stop'
        """
        if self.asymmetric_stop_manager is None or self.shares == 0.0:
            return {'should_exit': False}

        # Calculate current P&L
        if abs(self.entry_vwap) < 1e-9:
            return {'should_exit': False}

        is_long = self.shares > 0
        if is_long:
            pnl_pct = (current_price - self.entry_vwap) / self.entry_vwap
        else:
            pnl_pct = (self.entry_vwap - current_price) / self.entry_vwap

        # Update peak P&L
        if pnl_pct > self._peak_pnl_pct:
            self._peak_pnl_pct = pnl_pct

        # Compute stop level
        result = self.asymmetric_stop_manager.compute_stop(
            position_pnl_pct=pnl_pct,
            entry_price=self.entry_vwap,
            current_price=current_price,
            atr=atr,
            peak_pnl_pct=self._peak_pnl_pct,
            bars_in_trade=self._bars_in_current_trade,
            is_long=is_long,
            realized_vol=realized_vol
        )

        # If stop hit, record statistics
        if result['should_exit']:
            self.asymmetric_stop_manager.record_stop_hit(result['stop_type'], pnl_pct)

        return {
            'should_exit': result['should_exit'],
            'stop_price': result['stop_price'],
            'stop_type': result['stop_type'],
            'stop_distance_pct': result['stop_distance_pct'],
            'pnl_at_stop': pnl_pct,
        }

    def _reset_position_stats(self):
        """Reset position statistics when opening new position."""
        self._peak_pnl_pct = 0.0
        self._bars_in_current_trade = 0

    def _increment_bars_in_trade(self):
        """Increment bar counter for current trade."""
        if abs(self.shares) > 1e-9:
            self._bars_in_current_trade += 1

    def _fill_price(self, side, mid, notional_abs=0.0):
        """
        Fill model with optional spread + market impact + order slicing (#5).
        When use_impact_model=True: uses volume-based ADV and local volatility.
        When use_order_slicing=True: splits large orders into child fills.
        When False: simple flat slippage (backward compatible).
        """
        sign = 1.0 if side == "buy" else -1.0

        if not self.cfg.use_impact_model:
            return mid * (1.0 + sign * self.cfg.slippage_pct)

        # If order slicing is enabled and order is large enough, use VWAP/TWAP
        if self.cfg.use_order_slicing and notional_abs > 0 and self.cfg.n_slices > 1:
            vwap_fill, _ = execute_sliced_fill(
                self._single_fill, side, mid, notional_abs,
                n_slices=self.cfg.n_slices,
                mode=self.cfg.slice_mode
            )
            return vwap_fill

        return self._single_fill(side, mid, notional_abs)

    def _single_fill(self, side, mid, notional_abs):
        """Single fill with spread + impact (used by slicing or directly).

        FIX Â§4.1: Delegates to canonical fill_price_from_cost() with correct
        parameter names and handles (fill_price, CostBreakdown) tuple return.
        This ensures RL training, evaluation, and L3 all use identical cost math.
        """
        sign = 1.0 if side == "buy" else -1.0

        # ADV dollars (rolling volume)
        i0 = max(0, self.cs - self.cfg.adv_window)
        if self.prices.shape[1] > 4:  # Volume available
            vols = self.prices[i0:self.cs + 1, 4]
            adv_shares = float(np.mean(vols)) if len(vols) > 0 else 1e6
        else:
            adv_shares = 1e6  # Default if no volume
        adv_dollars = adv_shares * mid

        if HAS_COST_MODEL:
            # FIX Â§4.1: Use correct function name and parameter signature.
            # fill_price_from_cost returns (fill_price, CostBreakdown) tuple.
            fill_price, cost_breakdown = fill_price_from_cost(
                mid=mid,
                side=side,
                trade_notional=notional_abs,
                adv_dollars=adv_dollars,
                spread_bps=self.cfg.spread_bps,
                impact_coeff=self.cfg.impact_coef,
                fees_pct=0.0,  # FIX Â§5.3: Fees handled separately by _commission() â€” do NOT embed here
            )
            # Store last cost breakdown for diagnostics (L4 monitoring)
            self._last_cost_breakdown = cost_breakdown
            return fill_price
        else:
            # Fallback: inline formula matching canonical model exactly.
            # canonical: half_spread + impact_coeff * sqrt(participation) * notional
            participation = min(1.0, notional_abs / max(adv_dollars, 1.0))
            half_spread = mid * (self.cfg.spread_bps * 1e-4) * 0.5
            # FIX Â§4.1 Layer 3: Use canonical sqrt-impact formula (no local_vol multiplier)
            impact = self.cfg.impact_coef * math.sqrt(participation) * notional_abs
            # Convert dollar impact to price displacement
            if notional_abs > 0:
                adverse_frac = (half_spread * (notional_abs / mid) + impact) / notional_abs
            else:
                adverse_frac = 0.0
            return mid * (1.0 + sign * adverse_frac)

    def _commission(self, notional):
        """Commission = commission_pct * |notional traded|"""
        return self.cfg.commission_pct * abs(notional)

    def _portfolio_value(self):
        """Mark-to-market portfolio value. Works for long AND short."""
        mid = self._get_mid_price()
        # For long: shares > 0, mtm = shares * price (positive)
        # For short: shares < 0, mtm = shares * price (negative, offsets short proceeds in cash)
        mtm = self.shares * mid
        return self.cash + mtm

    def _obs(self):
        o = self.feat[self.cs - self.w:self.cs].copy()
        ps = NUM_FEATURES - 4  # Position state features
        if abs(self.shares) > 0:
            mid = self._get_mid_price()
            unrealized_pnl = (mid - self.entry_vwap) / (self.entry_vwap + 1e-10)
            if self.shares < 0:
                unrealized_pnl = -unrealized_pnl  # Short profits when price drops
            o[-1, ps] = np.clip(self.exposure, -1.0, 1.0)       # Current exposure [-1, +1]
            o[-1, ps + 1] = np.clip(unrealized_pnl, -0.5, 0.5)  # Unrealized PnL fraction
            o[-1, ps + 2] = min((self.cs - self.entry_step) / 20.0, 1.0)  # Holding time
            o[-1, ps + 3] = np.sign(self.shares)                 # Direction: +1 long, -1 short
        else:
            o[-1, ps:ps + 4] = 0.0
        obs = o.flatten().astype(np.float32)
        # NaN guard: price gaps, entry_vwap~=0, or feature edge cases can
        # produce NaN that propagates through the network into NaN logits,
        # crashing Categorical. Replace with zeros (neutral observation).
        if not np.isfinite(obs).all():
            np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def _execute_rebalance(self, target_exposure):
        """
        Trade from current exposure to target exposure.
        All fills use consistent fill-price model.
        Returns: realized PnL from any closed portion, total cost paid.
        """
        mid = self._get_mid_price()
        ref = self._exec_price_ref()  # May be next-open if configured
        if getattr(self.cfg, 'use_realized_capital_sizing', False):
            # Size on starting_capital + cumulative realized PnL only — unrealized moves don't affect sizing
            port_value = self._realized_capital
        else:
            port_value = self._portfolio_value()
        max_notional = port_value * self.cfg.max_position_pct * self.cfg.max_leverage

        # Desired signed notional
        target_notional = target_exposure * max_notional
        current_notional = self.shares * mid
        delta_notional = target_notional - current_notional

        if abs(delta_notional) < mid * 0.5:
            # Too small to trade (less than half a share)
            self.turnover_this_step = 0.0
            return 0.0, 0.0, 0.0

        delta_shares = delta_notional / mid
        realized_pnl = 0.0
        total_cost = 0.0
        self.turnover_this_step = 0.0  # Track $ traded this step

        # --- CLOSING portion (if reducing or flipping) ---
        if self.shares != 0 and np.sign(delta_shares) != np.sign(self.shares):
            close_shares = min(abs(delta_shares), abs(self.shares))
            # Remember direction BEFORE modifying shares (for logging)
            was_long = self.shares > 0
            close_notional = close_shares * ref

            if was_long:
                # Closing long: SELL shares
                exit_fill = self._fill_price("sell", ref, close_notional)
                proceeds = close_shares * exit_fill
                exit_comm = self._commission(close_shares * exit_fill)  # Fix #2: commission on executed notional
                self.cash += proceeds - exit_comm
                realized_pnl = (exit_fill - self.entry_vwap) * close_shares
                self.shares -= close_shares

                # NEW: Log this SELL action
                self.trade_entries.append({
                    "bar": self.cs,
                    "action": "SELL",
                    "shares": close_shares,
                    "price": exit_fill,
                    "cost": exit_comm,
                    "realized_pnl": realized_pnl,
                    "position_before": close_shares + self.shares,  # Before the sell
                    "position_after": self.shares,
                    "entry_vwap": self.entry_vwap,
                })
            else:
                # Closing short: BUY to cover
                exit_fill = self._fill_price("buy", ref, close_notional)
                cover_cost = close_shares * exit_fill
                exit_comm = self._commission(close_shares * exit_fill)  # Fix #2: commission on executed notional
                self.cash -= cover_cost + exit_comm
                realized_pnl = (self.entry_vwap - exit_fill) * close_shares
                self.shares += close_shares

                # NEW: Log this BUY-to-cover action
                self.trade_entries.append({
                    "bar": self.cs,
                    "action": "BUY_COVER",
                    "shares": close_shares,
                    "price": exit_fill,
                    "cost": exit_comm,
                    "realized_pnl": realized_pnl,
                    "position_before": self.shares - close_shares,  # Before the cover
                    "position_after": self.shares,
                    "entry_vwap": self.entry_vwap,
                })

            total_cost += exit_comm
            self.turnover += close_shares * mid
            self.turnover_this_step += close_shares * mid

            # Log completed trade if fully closed
            if abs(self.shares) < 0.01:
                self._log_trade(mid, realized_pnl, was_long, close_shares)
                self.shares = 0.0
                self.entry_vwap = 0.0
                # Update realized capital base: cash when flat = starting_capital + all realized PnL net of costs
                if getattr(self.cfg, 'use_realized_capital_sizing', False):
                    self._realized_capital = self.cash
                # Reset asymmetric stop stats when closing position
                if self.asymmetric_stop_manager is not None:
                    self._reset_position_stats()
        # Recompute remaining delta from current state
        current_notional = self.shares * mid
        delta_notional = target_notional - current_notional
        delta_shares_remaining = delta_notional / (mid + 1e-10)

        # --- OPENING portion (if adding or new position) ---
        if abs(delta_shares_remaining) > 0.5:
            open_shares = abs(delta_shares_remaining)
            open_notional = open_shares * ref
            if delta_shares_remaining > 0:
                # Opening/adding long: BUY
                fill = self._fill_price("buy", ref, open_notional)
                cost = open_shares * fill
                comm = self._commission(open_shares * fill)  # Fix #2: on executed notional
                if cost + comm > self.cash * 0.95:
                    open_shares = (self.cash * 0.95 - comm) / (fill + 1e-10)
                    open_shares = max(0, open_shares)
                    cost = open_shares * fill
                    comm = self._commission(open_shares * fill)
                if open_shares > 0.5:
                    position_before = self.shares
                    old_notional = abs(self.shares) * self.entry_vwap if self.shares > 0 else 0
                    new_notional = open_shares * fill
                    total_shares = (self.shares if self.shares > 0 else 0) + open_shares
                    self.entry_vwap = (old_notional + new_notional) / (total_shares + 1e-10)
                    self.entry_shares = total_shares
                    self.cash -= cost + comm
                    self.shares += open_shares
                    if self.entry_step == 0 or abs(self.shares - open_shares) < 0.01:
                        self.entry_step = self.cs
                        # Reset asymmetric stop stats for new position
                        if self.asymmetric_stop_manager is not None:
                            self._reset_position_stats()

                    # NEW: Log this BUY action
                    self.trade_entries.append({
                        "bar": self.cs,
                        "action": "BUY",
                        "shares": open_shares,
                        "price": fill,
                        "cost": comm,
                        "realized_pnl": 0.0,  # Entry has no realized P&L
                        "position_before": position_before,
                        "position_after": self.shares,
                        "entry_vwap": self.entry_vwap,
                    })
            else:
                # Opening/adding short: SELL (short)
                fill = self._fill_price("sell", ref, open_notional)
                proceeds = open_shares * fill
                comm = self._commission(open_shares * fill)  # Fix #2: on executed notional
                margin_req = open_shares * fill * 0.5  # Fix #2: margin on fill too
                if margin_req > self.cash * 0.95:
                    open_shares = (self.cash * 0.95) / (fill * 0.5 + 1e-10)
                    open_shares = max(0, open_shares)
                    proceeds = open_shares * fill
                    comm = self._commission(open_shares * fill)
                if open_shares > 0.5:
                    position_before = self.shares
                    old_notional = abs(self.shares) * self.entry_vwap if self.shares < 0 else 0
                    new_notional = open_shares * fill
                    total_shares = (abs(self.shares) if self.shares < 0 else 0) + open_shares
                    self.entry_vwap = (old_notional + new_notional) / (total_shares + 1e-10)
                    self.entry_shares = total_shares
                    self.cash += proceeds - comm
                    self.shares -= open_shares
                    if self.entry_step == 0 or abs(self.shares + open_shares) < 0.01:
                        self.entry_step = self.cs
                        # Reset asymmetric stop stats for new position
                        if self.asymmetric_stop_manager is not None:
                            self._reset_position_stats()

                    # NEW: Log this SELL-short action
                    self.trade_entries.append({
                        "bar": self.cs,
                        "action": "SELL_SHORT",
                        "shares": open_shares,
                        "price": fill,
                        "cost": comm,
                        "realized_pnl": 0.0,  # Entry has no realized P&L
                        "position_before": position_before,
                        "position_after": self.shares,
                        "entry_vwap": self.entry_vwap,
                    })

            total_cost += comm
            self.turnover += open_shares * mid
            self.turnover_this_step += open_shares * mid

        # Update exposure fraction
        port_value_new = self._portfolio_value()
        if port_value_new > 0 and max_notional > 0:
            self.exposure = (self.shares * mid) / (max_notional + 1e-10)
        else:
            self.exposure = 0.0

        return realized_pnl, total_cost, self.turnover_this_step

    def _log_trade(self, exit_price, pnl, was_long=True, trade_shares=0):
        """Log a completed round-trip trade.
        Direction and shares passed explicitly because self.shares - 0 at call time.
        """
        self.trade_log.append({
            "entry": self.entry_vwap, "exit": exit_price,
            "pnl": pnl, "bars": self.cs - self.entry_step,
            "direction": "LONG" if was_long else "SHORT",
            "shares": trade_shares if trade_shares > 0 else getattr(self, 'entry_shares', 0)
        })
        if pnl > 0:
            self.n_wins += 1; self.gross_profit += pnl
        else:
            self.n_losses += 1; self.gross_loss += abs(pnl)

    def _compute_reward(self, realized_pnl, total_cost, traded_notional):
        """
        Stationary reward = log-return base + proportional penalties.
        Includes cash yield on idle cash (#8) and reward decomposition logging.
        """
        port_value = self._portfolio_value()

        # --- Cash yield on idle cash (#8: prevents "always trade" bias) ---
        if self.cfg.cash_yield_bps_annual > 0 and abs(self.shares) < 0.01:
            per_bar_yield = (self.cfg.cash_yield_bps_annual * 1e-4) / max(self.bars_per_year, 1)  # Fix #7: timeframe-aware
            self.cash += self.cash * per_bar_yield
            port_value = self._portfolio_value()  # Recompute after yield

        # --- Base: log portfolio return (stationary, well-conditioned) ---
        log_ret = math.log(max(port_value, 1.0) / max(self.prev_value, 1.0))
        base_reward = log_ret * self.cfg.reward_scale

        # --- Proportional turnover penalty (actual $ traded / portfolio value) ---
        turnover_frac = abs(traded_notional) / max(port_value, 1e-9)
        turnover_penalty = self.cfg.reward_turnover_cost * turnover_frac

        # --- Continuous quadratic drawdown penalty ---
        self.peak_value = max(self.peak_value, port_value)
        dd = (self.peak_value - port_value) / (self.peak_value + 1e-10)
        self.max_dd = max(self.max_dd, dd)
        dd_penalty = self.cfg.reward_drawdown_penalty * dd * dd

        # --- CVaR tail risk penalty (Fix #7b: smooth, no dead zone) ---
        step_return = (port_value - self.prev_value) / (self.prev_value + 1e-10)
        self.return_history.append(step_return)
        cvar_penalty = 0.0
        if len(self.return_history) > 20:
            sorted_rets = np.sort(self.return_history[-100:])
            cutoff = max(1, int(len(sorted_rets) * self.cfg.cvar_quantile))
            cvar = -np.mean(sorted_rets[:cutoff])
            # Smooth quadratic penalty: no dead zone, scales naturally with timeframe
            if cvar > 0:
                cvar_penalty = cvar * cvar * self.cfg.reward_cvar_penalty

        # --- Holding bonus (scale-consistent) ---
        hold_bonus = self.cfg.reward_holding_bonus if turnover_frac < 0.001 else 0.0

        # --- Combine ---
        reward = base_reward - turnover_penalty - dd_penalty - cvar_penalty + hold_bonus

        # --- Optional DSR auxiliary signal ---
        dsr_signal = 0.0
        if self.cfg.use_dsr and len(self.return_history) > 5:
            dsr_signal = self.dsr.calculate(step_return)
            reward = 0.7 * reward + 0.3 * dsr_signal

        # --- Soft clip via tanh ---
        reward = math.tanh(reward / self.cfg.reward_soft_clip_scale) * self.cfg.reward_soft_clip_scale

        # --- Reward decomposition logging (#8: audit trail) ---
        self.last_reward_components = {
            "base": round(base_reward, 6),
            "turnover_pen": round(-turnover_penalty, 6),
            "dd_pen": round(-dd_penalty, 6),
            "cvar_pen": round(-cvar_penalty, 6),
            "hold_bonus": round(hold_bonus, 6),
            "dsr_signal": round(dsr_signal, 6),
            "total": round(reward, 6),
        }

        self.prev_value = port_value
        return float(reward)

    def get_risk_target(self):
        """
        Compute forward-looking realized volatility for risk head training (point #4).
        Target = std of log-returns over next N bars.

        FIX Â§4.4: IMPORTANT â€” This method intentionally uses FUTURE data.
        It is ONLY used as an auxiliary training target for the risk head during
        RL training. It must NEVER be called during pipeline evaluation (L1â†’L4).
        The FutureLeakGuard in alphago_architecture.py prevents accidental use
        in the pipeline path. In RL training, the risk head learns to PREDICT
        this target from PAST features only â€” the future data is the label, not
        a feature.
        """
        horizon = self.cfg.risk_horizon
        end = min(self.cs + horizon, self.nb - 1)
        if end - self.cs < 5:
            return 0.01  # Default low vol if near end

        # GUARD: Assert we are in RL training context, not pipeline eval
        if getattr(self, '_pipeline_eval_mode', False):
            raise RuntimeError(
                "get_risk_target() called during pipeline eval â€” future data leak! "
                "This method may only be used during RL training as an auxiliary label."
            )

        future_closes = self.prices[self.cs:end, 3]
        log_rets = np.diff(np.log(future_closes + 1e-10))
        return float(np.std(log_rets)) if len(log_rets) > 1 else 0.01

    def set_index(self, idx):
        """
        Reset env to a specific bar index with flat position.
        Used by MCTS stress-biased root sampling to jump to arbitrary timesteps
        (instead of random-walking from start, which biases toward early bars).
        """
        idx = max(self.w, min(idx, self.nb - 2))
        self._reset_state()
        self.cs = idx

    def _get_action_mask(self):
        """
        #8: Action masking when near ruin or max DD.
        Returns: array of allowed actions (1=allowed, 0=masked).
        Full action set: {-1.0, -0.5, 0.0, +0.5, +1.0} -- indices {0,1,2,3,4}
        Near DD threshold: only allow flat (2) and reduce-exposure actions.
        """
        mask = np.ones(self.cfg.n_actions, dtype=np.float32)
        if not self.cfg.use_action_masking:
            return mask

        flat_idx = int(np.argmin(np.abs(self.action_targets - 0.0)))

        # If kill triggered, only allow FLAT
        if self.kill_triggered:
            mask[:] = 0.0
            mask[flat_idx] = 1.0  # FLAT
            return mask

        # Use CURRENT drawdown (not historical max) so masking relaxes after recovery
        dd = (self.peak_value - self._portfolio_value()) / (self.peak_value + 1e-10)
        if dd >= self.cfg.dd_mask_threshold:
            # Restrict to FLAT + reduced positions only
            # Mask out extreme actions (first and last)
            mask[0] = 0.0   # No FULL_SHORT
            mask[-1] = 0.0  # No FULL_LONG
        if dd >= self.cfg.max_drawdown_pct:
            # Only allow FLAT
            mask[:] = 0.0
            mask[flat_idx] = 1.0
        return mask

    def step(self, action):
        # Dynamic flat action index (works with any action grid size)
        _flat_action = int(np.argmin(np.abs(self.action_targets - 0.0)))

        # #8: Action masking -- override action if masked
        if self.cfg.use_action_masking:
            mask = self._get_action_mask()
            if mask[action] < 0.5:
                # Action is masked -- force to FLAT (safest)
                action = _flat_action

        # Fix #1: Capture risk target BEFORE step changes cs or triggers terminal logic
        pre_step_risk_target = self.get_risk_target()

        # --- TRAILING STOP CHECK ---
        # Check if we're stopped out BEFORE processing the action
        stopped_out = False
        if self.cfg.use_trailing_stops and self.shares != 0.0:
            current_idx = min(self.cs, self.nb - 1)
            current_low = float(self.prices[current_idx, 2])   # Column 2 = Low
            current_high = float(self.prices[current_idx, 1])  # Column 1 = High

            hit, exit_price = self._check_trailing_stop(current_low, current_high)
            if hit:
                # Force flat - we've been stopped out
                stopped_out = True
                action = _flat_action  # Dynamic flat (works with any action grid)
                # Log stop exit in trade_entries
                self.trade_entries.append({
                    'step': self.cs,
                    'action': 'STOP_EXIT',
                    'side': 'SELL' if self.shares > 0 else 'BUY',
                    'shares': abs(self.shares),
                    'price': exit_price,
                    'notional': abs(self.shares) * exit_price,
                    'exposure_before': self.exposure,
                    'exposure_after': 0.0,
                    'stop_price': self.trailing_stop_price
                })

        # FIX Bug #4: Increment bars in trade counter BEFORE stop check to fix off-by-one error
        self._increment_bars_in_trade()

        # FIX Bug #10: Track stop info for logging after execution
        asymmetric_stop_info = None

        # --- ASYMMETRIC STOP LOSS CHECK ---
        # Check asymmetric stop if enabled (takes precedence over action)
        if self.cfg.use_asymmetric_stops and self.asymmetric_stop_manager is not None and abs(self.shares) > 1e-9:
            current_idx = min(self.cs, self.nb - 1)
            current_price = float(self.prices[current_idx, 3])  # Close price

            # Get ATR (from features if available) - FIX Bug #3
            atr = None
            try:
                if hasattr(self.feat, 'iloc'):  # DataFrame
                    if 'ATR' in self.feat.columns and current_idx < len(self.feat):
                        atr = float(self.feat.iloc[current_idx]['ATR'])
                elif hasattr(self.feat, 'shape'):  # Numpy array
                    # ATR is computed in compute_indicators, find its index in FEATURE_COLUMNS
                    # For now, compute manually since feature index mapping is complex
                    pass
            except:
                pass

            # Fallback: Compute ATR manually from recent price bars
            if atr is None or not np.isfinite(atr) or atr <= 0:
                lookback = 14
                start = max(self.w, current_idx - lookback)  # FIX Bug #12: Don't look before window
                if start < current_idx:
                    highs = self.prices[start:current_idx+1, 1]
                    lows = self.prices[start:current_idx+1, 2]
                    closes = self.prices[start:current_idx+1, 3]
                    if len(highs) > 1:
                        tr = np.maximum(highs[1:] - lows[1:],
                                       np.maximum(np.abs(highs[1:] - closes[:-1]),
                                                 np.abs(lows[1:] - closes[:-1])))
                        atr = float(np.mean(tr))

                # Final fallback if still invalid
                if atr is None or not np.isfinite(atr) or atr <= 0:
                    atr = current_price * 0.015  # 1.5% of price

            # Get realized vol (from features if available) - FIX Bug #3
            realized_vol = None
            try:
                if hasattr(self.feat, 'iloc'):  # DataFrame
                    if 'Realized_Vol_20' in self.feat.columns and current_idx < len(self.feat):
                        realized_vol = float(self.feat.iloc[current_idx]['Realized_Vol_20'])
            except:
                pass

            # Fallback: Compute realized vol manually from recent returns
            if realized_vol is None or not np.isfinite(realized_vol) or realized_vol <= 0:
                lookback = 20
                start = max(self.w, current_idx - lookback)  # Don't look before window
                if start < current_idx:
                    closes = self.prices[start:current_idx+1, 3]
                    if len(closes) > 1:
                        rets = np.diff(np.log(closes + 1e-12))
                        realized_vol = float(np.std(rets) * np.sqrt(252))

                # Final fallback if still invalid
                if realized_vol is None or not np.isfinite(realized_vol) or realized_vol <= 0:
                    realized_vol = 0.15  # Default 15% annualized

            stop_result = self._check_asymmetric_stop(current_price, atr, realized_vol)

            if stop_result['should_exit']:
                # Asymmetric stop triggered - force flat
                stopped_out = True
                action = _flat_action  # Dynamic flat (works with any action grid)
                # FIX Bug #10: Save stop info for logging AFTER execution completes
                asymmetric_stop_info = {
                    'triggered': True,
                    'shares_before': abs(self.shares),
                    'exposure_before': self.exposure,
                    'stop_result': stop_result
                }

        # (Bar counter increment moved to before stop check - see FIX Bug #4 above)

        old_exposure = self.exposure
        target_exp = float(self.action_targets[action])

        realized_pnl, total_cost, traded_notional = self._execute_rebalance(target_exp)

        # FIX Bug #10: Log asymmetric stop AFTER execution completes (not before)
        if asymmetric_stop_info is not None and asymmetric_stop_info['triggered']:
            stop_result = asymmetric_stop_info['stop_result']
            self.trade_entries.append({
                'step': self.cs,
                'action': f'ASYMMETRIC_STOP_{stop_result["stop_type"].upper()}',
                'side': 'SELL' if asymmetric_stop_info['shares_before'] > 0 else 'BUY',
                'shares': asymmetric_stop_info['shares_before'],
                'price': stop_result['stop_price'],
                'notional': asymmetric_stop_info['shares_before'] * stop_result['stop_price'],
                'exposure_before': asymmetric_stop_info['exposure_before'],
                'exposure_after': self.exposure,  # Actual exposure after execution (not hardcoded 0.0)
                'stop_price': stop_result['stop_price'],
                'stop_type': stop_result['stop_type'],
                'pnl_pct': stop_result['pnl_at_stop'] * 100
            })

        # --- Store per-step execution data for pipeline L4 ingestion ---
        # This surfaces REAL fill costs (not config estimates) so L4's
        # slippage/commission monitoring reflects actual execution quality.
        mid = self._get_mid_price()
        self._last_step_mid = mid
        self._last_step_commission = total_cost  # _execute_rebalance returns commission as total_cost
        self._last_step_traded = traded_notional
        # Slippage estimate: total_cost relative to traded notional
        if traded_notional > 1e-6:
            self._last_step_slippage_bps = (total_cost / traded_notional) * 1e4
        else:
            self._last_step_slippage_bps = 0.0

        reward = self._compute_reward(realized_pnl, total_cost, traded_notional)

        # NaN/inf guard
        if not np.isfinite(reward):
            reward = -1.0

        self.step_rewards.append(reward)

        # --- UPDATE TRAILING STOP ---
        # After executing the trade, update the trailing stop for the new position
        if self.cfg.use_trailing_stops:
            lookback = self.cfg.trailing_stop_lookback
            self._update_trailing_stop(lookback)

        self.cs += 1

        # --- Borrow cost scaled by timeframe (#8) ---
        if self.shares < -0.01:
            short_notional = abs(self.shares) * self._get_mid_price()
            bpy = self.bars_per_year
            per_bar = (self.cfg.borrow_bps_daily * 1e-4) * (252.0 / max(bpy, 1))
            self.cash -= short_notional * per_bar

        # --- Kill switch checks (#7) ---
        port_value = self._portfolio_value()

        if self.cfg.use_kill_switches and not self.kill_triggered:
            # Drift monitor check (Fix #3: actually used now)
            dm = getattr(self, 'drift_monitor', None)
            if dm is not None:
                obs_now = self._obs()
                if dm.should_kill(obs_now):
                    self.kill_triggered = True

            # Daily loss check
            self.episode_pnl = port_value - self.episode_start_value
            episode_loss_frac = -self.episode_pnl / max(self.episode_start_value, 1e-9)
            if episode_loss_frac > self.cfg.max_episode_loss_pct:
                self.kill_triggered = True

            # Turnover spike detection
            self.step_turnovers.append(traded_notional)
            if len(self.step_turnovers) > 10:
                avg_turnover = np.mean(list(self.step_turnovers)[:-1])
                if avg_turnover > 0 and traded_notional > self.cfg.max_turnover_spike * avg_turnover:
                    self.kill_triggered = True

        # If kill triggered, force flat next step
        if self.kill_triggered and abs(self.shares) > 0.01:
            self._execute_rebalance(0.0)
            port_value = self._portfolio_value()

        term = trunc = False

        # Terminal conditions
        if self.cs >= self.nb - 1:
            if abs(self.shares) > 0:
                self._execute_rebalance(0.0)
            term = True
        if port_value < self.cfg.starting_capital * 0.3:  # Ruin
            term = True
        if self.cs - self.w >= self.cfg.max_episode_steps:
            if abs(self.shares) > 0:
                self._execute_rebalance(0.0)
            trunc = True

        info = self._info()
        info["risk_target"] = pre_step_risk_target  # Fix #1: captured before state change
        return self._obs(), reward, term, trunc, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        if not self.ev and self.nb > self.w + 200:
            mx = self.nb - self.cfg.max_episode_steps - 10
            if mx > self.w:
                self.cs = self.np_random.integers(self.w, mx)
        return self._obs(), self._info()

    def _info(self):
        tt = self.n_wins + self.n_losses
        port_value = self._portfolio_value()
        pnl = port_value - self.cfg.starting_capital
        sharpe = 0.0
        if len(self.return_history) > 10:
            r = np.array(self.return_history)
            r_std = np.std(r)
            # FIX §SA-4: Guard against zero-trade edge case where constant
            # cash-yield returns produce std ≈ 0, causing Sharpe to blow up
            # to ~25 million. Require both meaningful variation AND at least
            # one completed trade for a non-zero Sharpe.
            if r_std > 1e-6 and (self.n_wins + self.n_losses) > 0:
                sharpe = np.mean(r) / (r_std + 1e-10) * np.sqrt(252)
            # else: sharpe stays 0.0 (constant returns or no trades = no alpha)
        # CVaR estimate for Lagrangian constraints
        cvar_est = 0.0
        if len(self.return_history) > 20:
            sorted_rets = np.sort(self.return_history[-100:])
            cutoff = max(1, int(len(sorted_rets) * self.cfg.cvar_quantile))
            cvar_est = float(-np.mean(sorted_rets[:cutoff]))

        # Annualized turnover (#8: normalized per time, not just cumulative)
        n_steps = max(len(self.return_history), 1)
        bpy = self.bars_per_year
        step_turnover_frac = self.turnover / max(self.cfg.starting_capital, 1e-9)
        annualized_turnover = step_turnover_frac * (bpy / max(n_steps, 1))

        return {
            "symbol": self.sym, "total_trades": tt,
            "wins": self.n_wins, "losses": self.n_losses,
            "win_rate": self.n_wins / max(tt, 1) * 100,
            "profit_factor": self.gross_profit / max(self.gross_loss, 1e-10),
            "net_pnl": pnl, "net_pnl_pct": pnl / self.cfg.starting_capital * 100,
            "max_drawdown": self.max_dd * 100, "portfolio_value": port_value,
            "sharpe_ratio": sharpe,
            "trade_log": self.trade_log,  # Legacy: closed round-trips only
            "trade_entries": self.trade_entries,  # NEW: all BUY/SELL actions
            "exposure": self.exposure, "turnover": self.turnover,
            # Lagrangian constraint signals
            "turnover_frac": step_turnover_frac,
            "annualized_turnover": annualized_turnover,
            "dd_frac": float(self.max_dd),
            "cvar_est": cvar_est,
            # Reward decomposition (#8)
            "reward_components": self.last_reward_components,
            # Kill switch status (#7)
            "kill_triggered": self.kill_triggered,
            # Per-step execution quality (for pipeline L4 ingestion)
            "step_mid_price": self._last_step_mid,
            "step_commission": self._last_step_commission,
            "step_traded_notional": self._last_step_traded,
            "step_slippage_bps": self._last_step_slippage_bps,
        }


# ============================================================================
# VECTORIZED MULTI-ENVIRONMENT (GPU-parallel rollouts)
# ============================================================================
class VectorizedEnvs:
    """Steps N environments in parallel -> single GPU forward pass per timestep."""
    def __init__(self, envs):
        self.envs=envs; self.n=len(envs); self.obs_dim=envs[0].observation_space.shape[0]
    def reset(self):
        ob=np.zeros((self.n,self.obs_dim),dtype=np.float32)
        for i,e in enumerate(self.envs): ob[i],_=e.reset()
        if not np.isfinite(ob).all():
            np.nan_to_num(ob, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return ob
    def step(self, actions):
        ob=np.zeros((self.n,self.obs_dim),dtype=np.float32)
        rw=np.zeros(self.n,dtype=np.float32); dn=np.zeros(self.n,dtype=bool); inf=[None]*self.n
        for i,(e,a) in enumerate(zip(self.envs,actions)):
            o,r,t,tr,info=e.step(int(a)); d=t or tr
            dn[i]=d; rw[i]=r; inf[i]=info  # Fix #1: always capture info (has risk_target, constraint signals)
            if d:
                o,_=e.reset()
            ob[i]=o
        # NaN guard: prevent any env's bad observation from crashing the batch
        if not np.isfinite(ob).all():
            np.nan_to_num(ob, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.nan_to_num(rw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return ob,rw,dn,inf


# ============================================================================
# GPU-VECTORIZED TRADING ENV (TorchVecEnv — RTX 3090 parallel stepping)
# ============================================================================
# Runs ALL N environments on GPU as batched tensor operations.
# One env.step() = one GPU kernel on (N, ...) tensors — 10,000 CUDA cores active.
# Training-only: evaluation uses full TradingEnv for accurate trade logging.

class TorchVecEnv:
    """GPU-native vectorized trading environment.

    All N environments step in parallel via batched PyTorch tensor operations.
    Replaces sequential Python TradingEnv.step() calls with GPU kernels.

    Drop-in replacement for VectorizedEnvs (returns numpy arrays).
    """

    def __init__(self, env_args_list, cfg):
        self.n = len(env_args_list)
        self.obs_dim = cfg.obs_dim
        self._cfg = cfg
        dev = torch.device('cuda' if HAS_CUDA else 'cpu')
        self._dev = dev
        N = self.n
        w = cfg.window_size
        F = NUM_FEATURES

        # --- Load features & prices onto GPU (one-time, padded to max_T) ---
        feat_list = [a[0] for a in env_args_list]  # (T_i, F) numpy arrays
        price_list = [a[1] for a in env_args_list]  # (T_i, 5) numpy arrays
        lengths = [f.shape[0] for f in feat_list]
        max_T = max(lengths)

        # Pad and stack to (N, max_T, F) and (N, max_T, 5)
        feat_pad = np.zeros((N, max_T, F), dtype=np.float32)
        price_pad = np.zeros((N, max_T, 5), dtype=np.float32)
        for i, (f, p) in enumerate(zip(feat_list, price_list)):
            feat_pad[i, :f.shape[0]] = f
            price_pad[i, :p.shape[0]] = p

        self._features = torch.from_numpy(feat_pad).to(dev)  # (N, max_T, F)
        self._prices = torch.from_numpy(price_pad).to(dev)    # (N, max_T, 5)
        self._lengths = torch.tensor(lengths, dtype=torch.long, device=dev)  # (N,)
        self._max_T = max_T
        self._w = w
        self._F = F

        # --- Config constants as tensors ---
        self._action_targets = torch.tensor(cfg.action_targets, dtype=torch.float32, device=dev)
        self._starting_capital = float(cfg.starting_capital)
        self._max_position_pct = float(cfg.max_position_pct)
        self._max_leverage = float(cfg.max_leverage)
        self._slippage_pct = float(cfg.slippage_pct)
        self._commission_pct = float(cfg.commission_pct)
        self._reward_scale = float(cfg.reward_scale)
        self._reward_turnover_cost = float(cfg.reward_turnover_cost)
        self._reward_dd_penalty = float(cfg.reward_drawdown_penalty)
        self._reward_cvar_penalty = float(cfg.reward_cvar_penalty)
        self._reward_hold_bonus = float(cfg.reward_holding_bonus)
        self._reward_soft_clip = float(cfg.reward_soft_clip_scale)
        self._cvar_quantile = float(cfg.cvar_quantile)
        self._max_episode_steps = int(cfg.max_episode_steps)
        self._borrow_cost_per_bar = float(cfg.borrow_bps_daily * 1e-4) * (252.0 / cfg.bars_per_year)
        self._cash_yield_per_bar = float(cfg.cash_yield_bps_annual * 1e-4) / cfg.bars_per_year

        # --- Window index template (reusable) ---
        self._widx = torch.arange(w, device=dev)  # (w,)
        self._eidx = torch.arange(N, device=dev)   # (N,)

        # --- Per-env state tensors ---
        self._cs = torch.zeros(N, dtype=torch.long, device=dev)
        self._cash = torch.full((N,), self._starting_capital, dtype=torch.float32, device=dev)
        self._shares = torch.zeros(N, dtype=torch.float32, device=dev)
        self._exposure = torch.zeros(N, dtype=torch.float32, device=dev)
        self._entry_vwap = torch.zeros(N, dtype=torch.float32, device=dev)
        self._entry_step = torch.zeros(N, dtype=torch.long, device=dev)
        self._peak_value = torch.full((N,), self._starting_capital, dtype=torch.float32, device=dev)
        self._prev_value = torch.full((N,), self._starting_capital, dtype=torch.float32, device=dev)
        self._max_dd = torch.zeros(N, dtype=torch.float32, device=dev)
        self._episode_start_cs = torch.zeros(N, dtype=torch.long, device=dev)
        # Rolling return buffer for CVaR (ring buffer)
        self._ret_buf = torch.zeros(N, 100, dtype=torch.float32, device=dev)
        self._ret_buf_ptr = torch.zeros(N, dtype=torch.long, device=dev)
        self._ret_buf_len = torch.zeros(N, dtype=torch.long, device=dev)

        # Initialize with random start positions
        self._reset_all()

    def _reset_all(self):
        """Reset all environments to random starting positions."""
        N = self.n; dev = self._dev; w = self._w
        mx = (self._lengths - self._max_episode_steps - 10).clamp(min=w + 1)
        self._cs = torch.randint(w, int(mx.min().item()), (N,), device=dev).clamp(min=w)
        for i in range(N):
            hi = max(w + 1, mx[i].item())
            self._cs[i] = torch.randint(w, hi, (1,), device=dev)
        self._episode_start_cs = self._cs.clone()
        self._cash.fill_(self._starting_capital)
        self._shares.zero_()
        self._exposure.zero_()
        self._entry_vwap.zero_()
        self._entry_step.zero_()
        self._peak_value.fill_(self._starting_capital)
        self._prev_value.fill_(self._starting_capital)
        self._max_dd.zero_()
        self._ret_buf.zero_()
        self._ret_buf_ptr.zero_()
        self._ret_buf_len.zero_()

    def _reset_envs(self, mask):
        """Reset specific environments (where mask is True)."""
        if not mask.any():
            return
        w = self._w
        idx = mask.nonzero(as_tuple=True)[0]
        mx = (self._lengths[idx] - self._max_episode_steps - 10).clamp(min=w + 1)
        for j, i in enumerate(idx):
            hi = max(w + 1, mx[j].item())
            self._cs[i] = torch.randint(w, hi, (1,), device=self._dev)
        self._episode_start_cs[idx] = self._cs[idx]
        self._cash[idx] = self._starting_capital
        self._shares[idx] = 0.0
        self._exposure[idx] = 0.0
        self._entry_vwap[idx] = 0.0
        self._entry_step[idx] = 0
        self._peak_value[idx] = self._starting_capital
        self._prev_value[idx] = self._starting_capital
        self._max_dd[idx] = 0.0
        self._ret_buf[idx] = 0.0
        self._ret_buf_ptr[idx] = 0
        self._ret_buf_len[idx] = 0

    @torch.no_grad()
    def _get_obs(self):
        """Batched observation construction on GPU. Returns (N, obs_dim) float32."""
        N = self.n; w = self._w; F = self._F; dev = self._dev
        # Gather feature windows: obs[i] = features[i, cs[i]-w : cs[i]]
        # Build index matrix: (N, w) where each row is [cs_i-w, cs_i-w+1, ..., cs_i-1]
        windows = self._widx.unsqueeze(0) + (self._cs - w).unsqueeze(1)  # (N, w)
        windows = windows.clamp(0, self._max_T - 1)
        # Advanced indexing: gather (N, w, F) from (N, max_T, F)
        ei = self._eidx.unsqueeze(1).expand(-1, w)  # (N, w)
        obs = self._features[ei, windows]  # (N, w, F)

        # Inject position state into last row (last 4 features: NUM_FEATURES-4 .. NUM_FEATURES-1)
        ps = F - 4  # position state start index
        has_pos = self._shares.abs() > 1e-8
        has_pos_f = has_pos.float()
        mid = self._prices[self._eidx, self._cs, 3]  # (N,) close prices

        # Unrealized P&L fraction
        safe_vwap = self._entry_vwap.clamp(min=1e-6)
        unreal_long = (mid - self._entry_vwap) / safe_vwap
        unreal_short = (self._entry_vwap - mid) / safe_vwap
        unreal_pnl = torch.where(self._shares > 0, unreal_long, unreal_short) * has_pos_f

        obs[:, -1, ps]     = self._exposure.clamp(-1.0, 1.0) * has_pos_f
        obs[:, -1, ps + 1] = unreal_pnl.clamp(-0.5, 0.5)
        obs[:, -1, ps + 2] = ((self._cs - self._entry_step).float() / 20.0).clamp(0.0, 1.0) * has_pos_f
        obs[:, -1, ps + 3] = self._shares.sign()

        # Flatten to (N, w*F)
        obs = obs.reshape(N, -1)
        # NaN safety
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    @torch.no_grad()
    def step(self, actions):
        """Step all N envs in parallel. actions: numpy (N,) int."""
        N = self.n; dev = self._dev

        # Convert actions to GPU tensor
        act_t = torch.from_numpy(actions.astype(np.int64)).to(dev)

        # --- Pre-step portfolio value ---
        mid = self._prices[self._eidx, self._cs, 3]  # (N,) close
        mid = mid.clamp(min=1e-6)
        port_value = self._cash + self._shares * mid

        # --- Execution: rebalance to target exposure ---
        target_exp = self._action_targets[act_t]  # (N,)
        max_notional = port_value * self._max_position_pct * self._max_leverage
        max_notional = max_notional.clamp(min=1.0)
        target_notional = target_exp * max_notional
        current_notional = self._shares * mid

        # Target shares
        buy_fill = mid * (1.0 + self._slippage_pct)
        sell_fill = mid * (1.0 - self._slippage_pct)
        buy_fill = buy_fill.clamp(min=1e-6)
        sell_fill = sell_fill.clamp(min=1e-6)

        # Desired target shares (positive = long, negative = short)
        target_shares = torch.where(
            target_exp >= 0,
            target_notional / buy_fill,
            target_notional / sell_fill
        )
        delta_shares_raw = target_shares - self._shares

        # Minimum trade threshold: skip tiny trades
        trade_mask = delta_shares_raw.abs() >= 0.5
        delta_shares_raw = delta_shares_raw * trade_mask.float()

        # --- Split into CLOSING and OPENING portions ---
        # Closing: reduces |shares| toward zero (always allowed, no constraint)
        is_reducing = (self._shares * delta_shares_raw) < 0  # opposite signs = reducing
        reduce_amount = torch.min(delta_shares_raw.abs(), self._shares.abs())
        close_shares = torch.where(is_reducing, delta_shares_raw.sign() * reduce_amount, torch.zeros_like(delta_shares_raw))

        # Opening: new position beyond zero (constrained by cash/margin)
        open_shares = delta_shares_raw - close_shares

        # Execute closing portion (always allowed, earns/spends cash)
        close_fill = torch.where(close_shares > 0, buy_fill, sell_fill)  # buy to cover or sell to close
        close_cost = self._commission_pct * (close_shares.abs() * close_fill)
        close_cash_delta = -close_shares * close_fill - close_cost  # selling gives cash, buying costs cash
        cash_after_close = self._cash + close_cash_delta

        # Constrain opening portion by available capital
        open_buy = open_shares > 1e-8   # new long
        open_sell = open_shares < -1e-8  # new short
        open_buy_cost = open_shares * buy_fill
        open_margin = open_shares.abs() * sell_fill * 0.5
        max_capital = (cash_after_close * 0.95).clamp(min=0.0)

        # Scale down if over budget
        buy_scale = torch.where(open_buy & (open_buy_cost > max_capital),
                                max_capital / open_buy_cost.clamp(min=1e-6),
                                torch.ones_like(open_shares))
        sell_scale = torch.where(open_sell & (open_margin > max_capital),
                                 max_capital / open_margin.clamp(min=1e-6),
                                 torch.ones_like(open_shares))
        open_scale = torch.where(open_buy, buy_scale, torch.where(open_sell, sell_scale, torch.ones_like(open_shares)))
        open_shares = open_shares * open_scale

        # Execute opening portion
        open_fill = torch.where(open_shares > 0, buy_fill, sell_fill)
        open_cost = self._commission_pct * (open_shares.abs() * open_fill)
        open_cash_delta = -open_shares * open_fill - open_cost

        # Total delta
        total_delta_shares = close_shares + open_shares
        total_cost = close_cost + open_cost
        traded_notional = total_delta_shares.abs() * mid

        # Update shares & cash
        old_shares = self._shares.clone()
        self._shares = self._shares + total_delta_shares
        self._cash = cash_after_close + open_cash_delta

        # Update exposure
        new_port_value = self._cash + self._shares * mid
        self._exposure = torch.where(
            max_notional > 1e-6,
            (self._shares * mid) / max_notional,
            torch.zeros_like(self._exposure)
        )

        # Update entry VWAP (simplified: set on new position open)
        opened = (old_shares.abs() < 1e-8) & (self._shares.abs() > 1e-8)
        open_fill_used = torch.where(self._shares > 0, buy_fill, sell_fill)
        self._entry_vwap = torch.where(opened, open_fill_used, self._entry_vwap)
        self._entry_step = torch.where(opened, self._cs, self._entry_step)

        # --- Short borrow cost ---
        is_short = self._shares < -1e-8
        borrow = self._borrow_cost_per_bar * self._shares.abs() * mid * is_short.float()
        self._cash = self._cash - borrow

        # --- Cash yield when flat ---
        is_flat = self._shares.abs() < 1e-8
        self._cash = self._cash + self._cash * self._cash_yield_per_bar * is_flat.float()

        # --- Reward (BEFORE advancing cs — matches TradingEnv timing) ---
        new_port_value = (self._cash + self._shares * mid).clamp(min=1.0)

        # Base: log return
        log_ret = torch.log(new_port_value / self._prev_value.clamp(min=1.0))
        base_reward = log_ret * self._reward_scale

        # Turnover penalty
        turnover_frac = traded_notional / port_value.clamp(min=1.0)
        turnover_pen = self._reward_turnover_cost * turnover_frac

        # Drawdown penalty (quadratic)
        dd = ((self._peak_value - new_port_value) / self._peak_value.clamp(min=1.0)).clamp(min=0.0)
        self._peak_value = torch.max(self._peak_value, new_port_value)
        self._max_dd = torch.max(self._max_dd, dd)
        dd_pen = self._reward_dd_penalty * dd * dd

        # CVaR penalty from rolling return buffer
        step_ret = (new_port_value - self._prev_value) / self._prev_value.clamp(min=1.0)
        # Update ring buffer
        ptr = self._ret_buf_ptr
        self._ret_buf[self._eidx, ptr] = step_ret
        self._ret_buf_ptr = (ptr + 1) % 100
        self._ret_buf_len = (self._ret_buf_len + 1).clamp(max=100)
        # Compute CVaR for envs with enough history
        cvar_pen = torch.zeros(N, device=dev)
        has_hist = self._ret_buf_len >= 20
        if has_hist.any():
            # Sort returns, take worst quantile
            sorted_rets, _ = self._ret_buf.sort(dim=1)  # (N, 100) sorted ascending
            cutoff = (self._ret_buf_len.float() * self._cvar_quantile).long().clamp(min=1)
            # Vectorized CVaR: mean of worst `cutoff` returns
            # Use a fixed cutoff approximation for batched efficiency
            k = max(1, int(100 * self._cvar_quantile))  # = 5 for 5% quantile
            worst_k = sorted_rets[:, :k]  # (N, k) worst returns
            cvar = (-worst_k.mean(dim=1)).clamp(min=0.0)
            cvar_pen = torch.where(has_hist, self._reward_cvar_penalty * cvar * cvar, cvar_pen)

        # Holding bonus
        hold_bonus = torch.where(
            turnover_frac < 0.001,
            torch.full((N,), self._reward_hold_bonus, device=dev),
            torch.zeros(N, device=dev)
        )

        reward = base_reward - turnover_pen - dd_pen - cvar_pen + hold_bonus
        # Soft tanh clip
        reward = torch.tanh(reward / self._reward_soft_clip) * self._reward_soft_clip
        # NaN safety
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        self._prev_value = new_port_value.clone()

        # --- Advance step (AFTER reward, matching TradingEnv) ---
        self._cs = self._cs + 1

        # --- Terminal / truncation detection ---
        at_end = self._cs >= (self._lengths - 1)
        ruin = new_port_value < (self._starting_capital * 0.3)
        ep_too_long = (self._cs - self._episode_start_cs) >= self._max_episode_steps
        done = at_end | ruin | ep_too_long

        # --- Observations (before reset so done envs get their final obs) ---
        obs = self._get_obs()

        # --- Auto-reset done envs ---
        self._reset_envs(done)

        # --- Build info dicts ---
        # Compact: risk_target, turnover_frac, dd_frac, cvar_est
        risk_tgt = torch.full((N,), 0.01, device=dev)  # default
        turnover_np = turnover_frac.cpu().numpy()
        dd_np = dd.cpu().numpy()
        cvar_np = cvar_pen.cpu().numpy()

        obs_np = obs.cpu().numpy()
        rw_np = reward.cpu().numpy().astype(np.float32)
        dn_np = done.cpu().numpy()
        inf = [{'risk_target': 0.01, 'turnover_frac': float(turnover_np[i]),
                'dd_frac': float(dd_np[i]), 'cvar_est': float(cvar_np[i])} for i in range(N)]

        return obs_np, rw_np, dn_np, inf

    def reset(self):
        """Reset all environments. Returns (N, obs_dim) numpy array."""
        self._reset_all()
        obs = self._get_obs()
        return obs.cpu().numpy()

    def close(self):
        """No-op: GPU tensors freed by garbage collection."""
        pass


# ============================================================================
# GPU ROLLOUT BUFFER (pinned memory + vectorized GAE)
# ============================================================================
class GPURolloutBuffer:
    def __init__(self,ns,ne,od,pin=True):
        self.ns=ns;self.ne=ne;self.od=od;self.pin=pin and HAS_CUDA;self.ptr=0
        self.states=np.zeros((ns,ne,od),dtype=np.float32)
        self.actions=np.zeros((ns,ne),dtype=np.int64)
        self.log_probs=np.zeros((ns,ne),dtype=np.float32)
        self.rewards=np.zeros((ns,ne),dtype=np.float32)
        self.values=np.zeros((ns,ne),dtype=np.float32)
        self.dones=np.zeros((ns,ne),dtype=np.float32)
        self.risk_targets=np.zeros((ns,ne),dtype=np.float32)
        # Lagrangian constraint signals
        self.turnover_fracs=np.zeros((ns,ne),dtype=np.float32)
        self.dd_fracs=np.zeros((ns,ne),dtype=np.float32)
        self.cvar_ests=np.zeros((ns,ne),dtype=np.float32)
        self.advantages=None; self.returns=None
    def add(self,s,a,lp,r,v,d,rt=None,turn_f=None,dd_f=None,cv_e=None):
        self.states[self.ptr]=s;self.actions[self.ptr]=a;self.log_probs[self.ptr]=lp
        self.rewards[self.ptr]=r;self.values[self.ptr]=v;self.dones[self.ptr]=d
        if rt is not None: self.risk_targets[self.ptr]=rt
        if turn_f is not None: self.turnover_fracs[self.ptr]=turn_f
        if dd_f is not None: self.dd_fracs[self.ptr]=dd_f
        if cv_e is not None: self.cvar_ests[self.ptr]=cv_e
        self.ptr+=1
    def compute_gae(self,lv,gamma=.99,lam=.95):
        adv=np.zeros_like(self.rewards); lg=np.zeros(self.ne)
        for t in reversed(range(self.ptr)):
            nv=lv if t==self.ptr-1 else self.values[t+1]
            nnt=1.-self.dones[t]; delta=self.rewards[t]+gamma*nv*nnt-self.values[t]
            lg=delta+gamma*lam*nnt*lg; adv[t]=lg
        self.advantages=adv[:self.ptr]; self.returns=self.advantages+self.values[:self.ptr]
    def to_gpu_batches(self,bs):
        T=self.ptr; N=T*self.ne
        sf=self.states[:T].reshape(N,self.od); af=self.actions[:T].reshape(N)
        lpf=self.log_probs[:T].reshape(N); rf=self.returns.reshape(N)
        advf=self.advantages.reshape(N); rtf=self.risk_targets[:T].reshape(N)
        tff=self.turnover_fracs[:T].reshape(N)
        ddf=self.dd_fracs[:T].reshape(N)
        cvf=self.cvar_ests[:T].reshape(N)
        advf=(advf-advf.mean())/(advf.std()+1e-8)
        def _t(a, dt=torch.float32):
            t = torch.from_numpy(a).to(dtype=dt)
            if self.pin: t=t.pin_memory()
            return t
        ts=_t(sf);ta=_t(af,torch.long);tl=_t(lpf);tr=_t(rf);tad=_t(advf);trt=_t(rtf)
        t_turn=_t(tff);t_dd=_t(ddf);t_cv=_t(cvf)
        idx=torch.randperm(N)
        for s in range(0,N,bs):
            e=min(s+bs,N); ix=idx[s:e]
            yield (ts[ix].to(DEVICE,non_blocking=True),ta[ix].to(DEVICE,non_blocking=True),
                   tl[ix].to(DEVICE,non_blocking=True),tr[ix].to(DEVICE,non_blocking=True),
                   tad[ix].to(DEVICE,non_blocking=True),trt[ix].to(DEVICE,non_blocking=True),
                   t_turn[ix].to(DEVICE,non_blocking=True),t_dd[ix].to(DEVICE,non_blocking=True),
                   t_cv[ix].to(DEVICE,non_blocking=True))
    def reset(self): self.ptr=0


# ============================================================================
# BATCHED MCTS (GPU-accelerated)
# ============================================================================
class MCTSNode:
    __slots__=['parent','action','prior','children','vc','tv','mv']
    def __init__(self,parent=None,action=None,prior=0.):
        self.parent=parent;self.action=action;self.prior=prior
        self.children={};self.vc=0;self.tv=0.;self.mv=0.
    def is_leaf(self): return len(self.children)==0
    def ucb(self,c=1.414):
        if self.vc==0: return 1e9 + np.random.random()  # Break ties randomly (was: inf -- always action 0)
        return self.mv+c*self.prior*math.sqrt(self.parent.vc)/(1+self.vc)

class BatchedMCTSPlanner:
    """
    GPU-batched MCTS with forward simulation, path reward accumulation,
    and risk-adjusted leaf evaluation.

    Planning realism (mcts_stochastic=True):
      Uses stochastic_clone() to generate synthetic future prices via
      regime-conditioned block bootstrap. Each rollout plans over a
      DIFFERENT plausible future, not the realized tape.
      This eliminates hindsight leakage in training targets.

    When mcts_stochastic=False: uses deterministic tape (original behavior).
    """
    def __init__(self, net, cfg):
        self.net = net; self.cfg = cfg; self.na = cfg.n_actions
        self.gamma = cfg.gamma

    @torch.no_grad()
    def batch_search(self, root_envs, n_rollouts=None):
        if n_rollouts is None:
            n_rollouts = self.cfg.mcts_rollouts

        B = len(root_envs)
        self.net.eval()

        # Get root policy priors
        root_obs = np.array([env._obs() for env in root_envs])
        root_tensor = torch.FloatTensor(root_obs).to(DEVICE)
        with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
            root_priors = self.net.get_action_probs(root_tensor).cpu().numpy()

        roots = []
        for b in range(B):
            # AlphaGo-style Dirichlet noise at root for exploration (4A)
            pri = root_priors[b].copy()
            eps = self.cfg.mcts_dirichlet_eps
            alpha = self.cfg.mcts_dirichlet_alpha
            if eps > 0 and alpha > 0:
                noise = np.random.dirichlet([alpha] * self.na)
                pri = (1 - eps) * pri + eps * noise
                pri = pri / (pri.sum() + 1e-10)
            r = MCTSNode(prior=1.)
            for a in range(self.na):
                r.children[a] = MCTSNode(parent=r, action=a, prior=float(pri[a]))
            roots.append(r)

        for _ in range(n_rollouts):
            leaf_obs_list = []
            leaf_meta = []  # (node, needs_expand, path_reward, depth)

            for b in range(B):
                # Stochastic rollouts: each rollout sees a DIFFERENT plausible future
                if self.cfg.mcts_stochastic:
                    sim_env = root_envs[b].stochastic_clone(
                        horizon=self.cfg.mcts_sim_horizon,
                        block_size=self.cfg.mcts_bootstrap_block
                    )
                else:
                    sim_env = root_envs[b].clone()  # Deterministic (original behavior)
                node = roots[b]
                path_reward = 0.0
                depth = 0

                # SELECTION with path reward accumulation
                while not node.is_leaf():
                    best_a = max(node.children.keys(),
                                 key=lambda a: node.children[a].ucb(self.cfg.mcts_exploration))
                    node = node.children[best_a]

                    # Step simulation and collect reward
                    if sim_env.cs < sim_env.nb - 1:
                        _, rew, term, trunc, _ = sim_env.step(best_a)
                        path_reward += (self.gamma ** depth) * rew
                        depth += 1
                        if term or trunc:
                            break

                future_obs = sim_env._obs()
                leaf_obs_list.append(future_obs)
                # Expand on FIRST visit (was: vc>0 check delayed expansion by 1 visit per node)
                needs_expand = len(node.children) == 0
                leaf_meta.append((node, needs_expand, path_reward, depth))

            # BATCHED GPU EVALUATION of future states
            if leaf_obs_list:
                leaf_tensor = torch.FloatTensor(np.array(leaf_obs_list)).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                    # Get value AND risk for risk-adjusted evaluation
                    logits, leaf_values, leaf_risks = self.net.forward(leaf_tensor)
                    leaf_values = leaf_values.cpu().numpy()
                    leaf_risks = leaf_risks.cpu().numpy()
                    leaf_priors = F.softmax(logits, dim=-1).cpu().numpy()

                for idx, (node, needs_expand, path_rew, depth) in enumerate(leaf_meta):
                    # Expand leaf
                    if needs_expand and len(node.children) == 0:
                        for a in range(self.na):
                            node.children[a] = MCTSNode(parent=node, action=a,
                                                         prior=leaf_priors[idx, a])

                    # Risk-adjusted leaf value: V - >>*risk
                    v_leaf = leaf_values[idx] - 0.1 * leaf_risks[idx]

                    # Total backup: R_path + gamma^T * V(leaf)
                    v_total = path_rew + (self.gamma ** depth) * v_leaf

                    # Backpropagation
                    while node is not None:
                        node.vc += 1
                        node.tv += v_total
                        node.mv = node.tv / node.vc
                        node = node.parent

        # Extract improved policies
        improved = np.zeros((B, self.na), dtype=np.float32)
        for b, root in enumerate(roots):
            visits = np.array([root.children[a].vc for a in range(self.na)], dtype=np.float32)
            if self.cfg.mcts_temperature > 0:
                visits = visits ** (1.0 / self.cfg.mcts_temperature)
            total = visits.sum()
            improved[b] = visits / total if total > 0 else root_priors[b]

        return improved


# ============================================================================
# GPU PPO TRAINER
# ============================================================================
class GPUPPOTrainer:
    """
    PPO with:
      - Risk head auxiliary loss (point #4: no longer dead code)
      - MCTS policy distillation via KL divergence (point #5: search--learn)
      - AMP, vectorized envs, pinned buffers, fused optimizer
    """
    def __init__(self, net, cfg):
        self.net = net; self.cfg = cfg
        fused = HAS_CUDA
        self.opt = optim.AdamW(net.parameters(), lr=cfg.learning_rate, weight_decay=1e-5, fused=fused)
        ts = cfg.n_iterations * cfg.total_timesteps_per_iter // cfg.n_steps
        if cfg.lr_schedule == "cosine_warm_restarts":
            t0 = cfg.lr_restart_period if cfg.lr_restart_period > 0 else max(cfg.total_timesteps_per_iter // cfg.n_steps, 1)
            self.sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.opt, T_0=t0, T_mult=cfg.lr_restart_mult, eta_min=1e-6)
        elif cfg.lr_schedule == "cosine":
            self.sched = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(ts, 1), eta_min=1e-6)
        else:
            self.sched = None
        self.use_amp = cfg.use_amp and HAS_AMP
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp, init_scale=1024) if self.use_amp else None
        self.gs = 0
        # Lagrangian multipliers (3A)
        self.lam_turn = 0.0
        self.lam_dd = 0.0
        self.lam_cvar = 0.0
        # Value/risk replay buffer (4B)
        self.replay_states = deque(maxlen=200_000)
        self.replay_risk = deque(maxlen=200_000)

    def collect_rollouts(self, venv, ns):
        self.net.eval(); ne = venv.n; od = venv.obs_dim
        buf = GPURolloutBuffer(ns, ne, od, self.cfg.pin_memory)
        obs = venv.reset(); ep_rw = defaultdict(float); done_rw = []; eids = list(range(ne)); neid = ne
        with torch.no_grad():
            for step in range(ns):
                ot = torch.FloatTensor(obs).to(DEVICE, non_blocking=self.cfg.pin_memory)
                # Sanitize observations before forward pass
                if not torch.isfinite(ot).all():
                    ot = torch.nan_to_num(ot, nan=0.0, posinf=1.0, neginf=-1.0)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    acts, lps, _, vals, _ = self.net.get_action_and_value(ot)

                # 6B: NaN guard on critic
                if not torch.isfinite(vals).all():
                    tprint("Non-finite values in critic output", "warn")
                    vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))

                an = acts.cpu().numpy(); ln = lps.cpu().numpy(); vn = vals.cpu().numpy()
                nobs, rw, dn, infs = venv.step(an)
                # Fix #1: Use info dict from step (captures pre-reset state for done envs)
                risk_tgts = np.zeros(ne, dtype=np.float32)
                turn_fr = np.zeros(ne, dtype=np.float32)
                dd_fr = np.zeros(ne, dtype=np.float32)
                cv_est = np.zeros(ne, dtype=np.float32)
                for i in range(ne):
                    inf_i = infs[i]
                    if inf_i is not None:
                        risk_tgts[i] = inf_i.get("risk_target", 0.01)
                        turn_fr[i] = inf_i.get("turnover_frac", 0.0)
                        dd_fr[i] = inf_i.get("dd_frac", 0.0)
                        cv_est[i] = inf_i.get("cvar_est", 0.0)
                    else:
                        # Fallback: use default (SubprocVecEnv has no .envs access)
                        risk_tgts[i] = 0.01
                buf.add(obs, an, ln, rw, vn, dn.astype(np.float32), risk_tgts, turn_fr, dd_fr, cv_est)
                # Replay buffer for value/risk head (4B) — subsample every 4th step
                if step % 4 == 0:
                    for i in range(ne):
                        self.replay_states.append(obs[i].copy())
                        self.replay_risk.append(float(risk_tgts[i]))
                for i in range(ne):
                    ep_rw[eids[i]] += rw[i]
                    if dn[i]:
                        done_rw.append(ep_rw.pop(eids[i], 0)); eids[i] = neid; neid += 1
                obs = nobs
            ot = torch.FloatTensor(obs).to(DEVICE, non_blocking=self.cfg.pin_memory)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                _, _, _, lv, _ = self.net.get_action_and_value(ot)
        buf.compute_gae(lv.cpu().numpy(), self.cfg.gamma, self.cfg.gae_lambda)
        return {"buffer": buf, "episode_rewards": done_rw,
                "mean_reward": np.mean(done_rw) if done_rw else 0,
                "n_episodes": len(done_rw), "steps": ns * ne}

    def update(self, buf):
        self.net.train()
        tpl = tvl = te = tkl = trl = 0; nu = 0; tgn = 0
        last_turn_f = last_dd_f = last_cv_e = None
        # Dynamic entropy coefficient — boost if entropy collapsed below floor
        ent_coef = self.cfg.ent_coef
        if hasattr(self, '_last_avg_entropy') and self._last_avg_entropy < self.cfg.ent_floor:
            ent_coef = self.cfg.ent_coef * self.cfg.ent_boost_factor
        for ep in range(self.cfg.n_epochs):
            for st, ac, olp, ret, adv, risk_tgt, turn_f, dd_f, cv_e in buf.to_gpu_batches(self.cfg.batch_size):
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    _, nlp, ent, val, risk_pred = self.net.get_action_and_value(st, ac)

                    # PPO policy loss (float16 inside autocast)
                    ratio = torch.exp(torch.clamp(nlp - olp, -5.0, 5.0))
                    s1 = ratio * adv
                    s2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * adv
                    pl = -torch.min(s1, s2).mean()
                    el = -ent.mean()

                # All losses in float32 (prevents AMP overflow during backward)
                pl = pl.float(); el = el.float()
                vl = F.smooth_l1_loss(val.float(), ret)  # Huber: bounded gradient for large errors
                risk_loss = F.mse_loss(risk_pred.float(), risk_tgt)

                loss = (pl + self.cfg.vf_coef * vl
                        + ent_coef * el
                        + self.cfg.risk_coef * risk_loss)

                # Lagrangian constraint penalties (3A)
                if self.cfg.use_lagrangian:
                    v_turn = torch.relu(turn_f - self.cfg.target_turnover_frac)
                    v_dd = torch.relu(dd_f - self.cfg.target_dd)
                    v_cv = torch.relu(cv_e - self.cfg.target_cvar)
                    lag_pen = (self.lam_turn * v_turn.mean()
                              + self.lam_dd * v_dd.mean()
                              + self.lam_cvar * v_cv.mean())
                    loss = loss + lag_pen
                last_turn_f = turn_f; last_dd_f = dd_f; last_cv_e = cv_e

                # Skip update if loss is non-finite (prevents NaN from corrupting weights)
                if not torch.isfinite(loss):
                    tprint("Non-finite PPO loss — skipping update", "warn")
                    continue

                self.opt.zero_grad(set_to_none=True)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    gn = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.opt); self.scaler.update()
                else:
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                    self.opt.step()

                with torch.no_grad():
                    akl = (olp - nlp).mean().item()
                tpl += pl.item(); tvl += vl.item(); te += ent.mean().item()
                tkl += akl; trl += risk_loss.item(); nu += 1
                tgn += gn.item() if torch.is_tensor(gn) else gn

            if nu > 0 and abs(tkl / nu) > self.cfg.target_kl:
                break

        # Track average entropy for dynamic boost
        self._last_avg_entropy = te / max(nu, 1)

        # Update Lagrangian multipliers (projected gradient ascent)
        if self.cfg.use_lagrangian and nu > 0 and last_turn_f is not None:
            with torch.no_grad():
                self.lam_turn = float(max(0.0, self.lam_turn + self.cfg.lagrangian_lr * (last_turn_f.mean().item() - self.cfg.target_turnover_frac)))
                self.lam_dd = float(max(0.0, self.lam_dd + self.cfg.lagrangian_lr * (last_dd_f.mean().item() - self.cfg.target_dd)))
                self.lam_cvar = float(max(0.0, self.lam_cvar + self.cfg.lagrangian_lr * (last_cv_e.mean().item() - self.cfg.target_cvar)))

        # Value/risk replay step (4B: cheap off-policy calibration)
        if len(self.replay_states) > 4096:
            idx_r = np.random.choice(len(self.replay_states), size=min(2048, len(self.replay_states)), replace=False)
            st_r = torch.FloatTensor(np.array([self.replay_states[i] for i in idx_r])).to(DEVICE)
            rt_r = torch.FloatTensor(np.array([self.replay_risk[i] for i in idx_r])).to(DEVICE)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                _, _, risk_pred_r = self.net.forward(st_r)
                aux_loss = self.cfg.risk_coef * F.mse_loss(risk_pred_r.float(), rt_r)
            self.opt.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(aux_loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.opt); self.scaler.update()
            else:
                aux_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

        if self.sched:
            self.sched.step()
        self.gs += buf.ptr * buf.ne
        cur_lr = self.opt.param_groups[0]['lr']
        return {"pl": tpl / max(nu, 1), "vl": tvl / max(nu, 1), "ent": te / max(nu, 1),
                "kl": tkl / max(nu, 1), "risk_loss": trl / max(nu, 1), "nu": nu,
                "grad_norm": tgn / max(nu, 1), "lr": cur_lr, "ent_coef_used": ent_coef}

    def distill_mcts(self, states, mcts_policies, n_distill_steps=5):
        """
        Distill MCTS search results into the policy network.
        Minimizes KL(-_mcts || -_net) so the network learns from planning.
        n_distill_steps scales with search depth (deeper search = more distillation).
        """
        if len(states) == 0:
            return 0.0
        self.net.train()
        st = torch.FloatTensor(states).to(DEVICE)
        target_raw = torch.FloatTensor(mcts_policies).to(DEVICE)

        # Temperature-soften MCTS targets
        temp = self.cfg.mcts_distill_temp
        if temp > 1.0:
            target_soft = target_raw.pow(1.0 / temp)
            target_soft = target_soft / (target_soft.sum(dim=-1, keepdim=True) + 1e-10)
            target = torch.clamp(target_soft, min=1e-6)
            target = target / target.sum(dim=-1, keepdim=True)
        else:
            target = torch.clamp(target_raw, min=1e-6)

        total_loss = 0.0; valid_steps = 0
        N = len(states)
        bs = min(self.cfg.batch_size, N)
        n_steps = min(n_distill_steps, max(1, N // bs + 1))
        for _ in range(n_steps):
            # Fix #6: Random minibatch instead of full forward pass
            ix = torch.randint(0, N, (bs,), device=DEVICE)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, _, _ = self.net.forward(st[ix])
                log_probs = F.log_softmax(logits, dim=-1)
                if not torch.isfinite(log_probs).all():
                    continue  # Skip batch with non-finite log_probs (extreme logits)
                kl_loss = F.kl_div(log_probs, target[ix], reduction='batchmean')
                if not torch.isfinite(kl_loss):
                    continue  # Skip non-finite KL loss
                loss = self.cfg.mcts_kl_coef * kl_loss
            self.opt.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.opt); self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
            total_loss += kl_loss.item(); valid_steps += 1
        return total_loss / max(valid_steps, 1)

    def train_iteration(self, venv, total_steps):
        rws = []; ls = []; done = 0; spc = self.cfg.n_steps
        while done < total_steps:
            ro = self.collect_rollouts(venv, spc)
            ui = self.update(ro["buffer"])
            rws.extend(ro["episode_rewards"]); ls.append(ui); done += ro["steps"]
        # Aggregate update metrics across all PPO updates in this iteration
        avg_metrics = {}
        if ls:
            for k in ls[0]:
                vals = [d[k] for d in ls if k in d]
                avg_metrics[k] = np.mean(vals) if vals else 0
        return {"mean_reward": np.mean(rws) if rws else 0, "n_episodes": len(rws),
                "steps": done, "update_metrics": avg_metrics}


# ============================================================================
# VALIDATION
# ============================================================================
class ValidationFramework:
    def __init__(self,cfg): self.cfg=cfg
    @torch.no_grad()
    def _eval(self,net,feat,prices,sym):
        if len(feat)<self.cfg.window_size+50: return {"sharpe_ratio":0,"net_pnl":0}
        # 5B: Optionally use next-open execution in validation
        cfg_local = copy.copy(self.cfg)
        if getattr(self.cfg, "validate_delay", False):
            cfg_local.trade_at_next_open = True
        env=TradingEnv(feat,prices,cfg_local,sym,True); obs,_=env.reset();done=False;net.eval()
        while not done:
            st=torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            a,_,_,_,_=net.get_action_and_value(st)
            obs,_,t,tr,info=env.step(a.item());done=t or tr
        return info
    @torch.no_grad()
    def _batched_eval(self, net, fold_data_list, sym):
        """Evaluate multiple (feat, prices) segments simultaneously via batched GPU inference."""
        cfg_local = copy.copy(self.cfg)
        if getattr(self.cfg, "validate_delay", False):
            cfg_local.trade_at_next_open = True
        envs = []
        for feat, prices in fold_data_list:
            if len(feat) < self.cfg.window_size + 50:
                continue
            envs.append(TradingEnv(feat, prices, cfg_local, sym, True))
        if not envs:
            return []
        venv = VectorizedEnvs(envs)
        obs = venv.reset()
        done_flags = np.zeros(venv.n, dtype=bool)
        results = [None] * venv.n
        net.eval()
        while not done_flags.all():
            ot = torch.FloatTensor(obs).to(DEVICE)
            acts, _, _, _, _ = net.get_action_and_value(ot)
            actions = acts.cpu().numpy()
            actions[done_flags] = 2  # FLAT for already-done envs
            obs, _, dones, infos = venv.step(actions)
            for i in range(venv.n):
                if dones[i] and results[i] is None:
                    results[i] = infos[i]
                    done_flags[i] = True
        for i in range(venv.n):
            if results[i] is None:
                results[i] = {"sharpe_ratio": 0, "net_pnl": 0}
        return results
    def walk_forward(self,net,feat,prices,sym="SYM"):
        n=len(feat);seg=n//(self.cfg.walk_forward_windows+1)
        fold_data=[]
        for w in range(self.cfg.walk_forward_windows):
            ts,te=seg*(w+1),min(seg*(w+2),n)
            if te-ts>=self.cfg.window_size+50:
                fold_data.append((feat[ts:te],prices[ts:te]))
        if not fold_data: return {"valid":False}
        # Batch all folds into single GPU pass per timestep
        fold_results=self._batched_eval(net,fold_data,sym)
        res=[]
        for r in fold_results:
            seg_pnl=r.get("net_pnl",0)
            res.append({"pnl":seg_pnl,"sharpe":r.get("sharpe_ratio",0)})
        if not res: return {"valid":False}
        pnls=[r["pnl"] for r in res]
        sharpes=[r["sharpe"] for r in res]
        # Segment returns (normalized PnL per segment) for permutation testing
        start_cap = getattr(self.cfg, "starting_capital", 100000)
        seg_returns = np.array(pnls, dtype=np.float64) / max(start_cap, 1e-9)
        return {"valid":True,"mean_pnl":np.mean(pnls),"mean_sharpe":np.mean(sharpes),
                "pnl_consistency":sum(1 for p in pnls if p>0)/len(pnls),
                "segment_returns":seg_returns}

    def deflated_sharpe_ratio(self, sharpe, n_obs, n_trials=1, skew=0.0, kurt=3.0):
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
        Tests whether observed Sharpe is statistically significant given
        the number of strategies/trials attempted (multiple testing correction).

        Returns: p-value (lower = more significant)
        """
        try:
            from scipy import stats as sp_stats
        except ImportError:
            return 0.5  # Can't compute without scipy

        if n_obs < 10 or sharpe <= 0:
            return 1.0

        # Expected maximum Sharpe under null (Euler-Mascheroni correction)
        euler_mascheroni = 0.5772156649
        e_max_sharpe = np.sqrt(2 * np.log(max(n_trials, 1))) - \
            (np.log(np.pi) + euler_mascheroni) / (2 * np.sqrt(2 * np.log(max(n_trials, 1))) + 1e-10)

        # Sharpe standard error with non-normality adjustment
        se = np.sqrt((1.0 + 0.5 * sharpe**2 - skew * sharpe +
                       ((kurt - 3) / 4.0) * sharpe**2) / (n_obs - 1 + 1e-10))

        # Test statistic: is observed Sharpe significantly above expected max?
        if se < 1e-10:
            return 1.0
        t_stat = (sharpe - e_max_sharpe) / se

        # One-sided p-value from t-distribution
        p_value = 1.0 - sp_stats.t.cdf(t_stat, df=max(n_obs - 1, 1))
        return float(p_value)

    def monte_carlo_permutation(self, returns, n_perms=100):
        """
        Sign-flip test for mean returns.
        Randomly flips the sign of each return and computes Sharpe.
        Unlike shuffling (which preserves mean+std -- Sharpe identical),
        sign-flipping tests whether the positive mean is statistically real.
        """
        if len(returns) < 5:
            return {"significant": False, "p_value": 1.0}

        returns = np.asarray(returns, dtype=np.float64)
        real_sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        count_above = 0
        for _ in range(n_perms):
            signs = np.random.choice([-1.0, 1.0], size=len(returns))
            flipped = returns * signs
            perm_sharpe = np.mean(flipped) / (np.std(flipped) + 1e-10) * np.sqrt(252)
            if perm_sharpe >= real_sharpe:
                count_above += 1

        p_value = count_above / max(n_perms, 1)
        return {"significant": p_value < self.cfg.perm_p_threshold,
                "p_value": round(p_value, 4), "real_sharpe": round(real_sharpe, 4)}

    def placebo_feature_shuffle_eval(self, net, feat, prices, sym="SYM"):
        """5B: Shuffle features to destroy temporal alignment -- should not profit."""
        feat_shuf = feat.copy()
        rng = np.random.RandomState(0)
        rng.shuffle(feat_shuf)  # Destroys time alignment
        info = self._eval(net, feat_shuf, prices, sym)
        return info

    def full_validation(self, net, feat, prices, sym="SYM", n_trials=10):
        """
        Run walk-forward + deflated Sharpe gate + Monte Carlo permutation gate
        + placebo test. All must pass for the model to be accepted.
        """
        wf = self.walk_forward(net, feat, prices, sym)
        passed = True; reasons = []

        if not wf.get("valid"):
            return {"walk_forward": wf, "passed": False, "fail_reasons": ["walk_forward invalid"]}

        # 1) Walk-forward consistency gate
        consistency = wf.get("pnl_consistency", 0)
        if consistency < self.cfg.wf_consistency_min:
            passed = False
            reasons.append(f"walk-forward consistency {consistency:.1%} < {self.cfg.wf_consistency_min:.1%}")

        # 2) Deflated Sharpe gate
        mean_sharpe = wf.get("mean_sharpe", 0)
        n_obs = len(feat)
        dsr_p = self.deflated_sharpe_ratio(mean_sharpe, n_obs, n_trials)
        wf["deflated_sharpe_p"] = round(dsr_p, 4)
        if mean_sharpe > 0 and dsr_p > self.cfg.deflated_sharpe_threshold:
            passed = False
            reasons.append(f"deflated Sharpe p={dsr_p:.3f} > {self.cfg.deflated_sharpe_threshold}")

        # 3) Monte Carlo sign-flip test gate
        seg_returns = wf.get("segment_returns", np.array([]))
        if len(seg_returns) >= 5:
            perm = self.monte_carlo_permutation(
                seg_returns, n_perms=self.cfg.monte_carlo_permutations
            )
            wf["perm_p"] = perm["p_value"]
            if not perm["significant"]:
                passed = False
                reasons.append(f"permutation test p={perm['p_value']:.3f}")

        # 4) Placebo test (5B): shuffled features should not profit
        if getattr(self.cfg, "placebo_feature_shuffle", False):
            pl = self.placebo_feature_shuffle_eval(net, feat, prices, sym)
            placebo_pnl = pl.get("net_pnl", 0)
            if placebo_pnl > self.cfg.placebo_fail_pnl_threshold:
                passed = False
                reasons.append(f"placebo(feature shuffle) pnl={placebo_pnl:.0f} > {self.cfg.placebo_fail_pnl_threshold}")

        return {"walk_forward": wf, "passed": passed, "fail_reasons": reasons}


# ============================================================================
# ALPHA-TRADE: GPU-PARALLEL SELF-PLAY SYSTEM
# ============================================================================
@dataclass
class SymbolDataset:
    symbol:str
    features_train:np.ndarray;features_val:np.ndarray;features_test:np.ndarray
    prices_train:np.ndarray;prices_val:np.ndarray;prices_test:np.ndarray
    n_train:int;n_val:int;n_test:int
    timestamps_train:Optional[np.ndarray]=None
    timestamps_val:Optional[np.ndarray]=None
    timestamps_test:Optional[np.ndarray]=None
    timeframe:str="1d"
    split_group:str="train"  # "train"|"val"|"holdout"

def _split_indices(n, cfg):
    """Time-based train/val/test split with embargo gaps to prevent leakage.
    Embargo scales with series length: min(configured, 5% of n) to avoid
    rejecting short intraday series where 200-bar embargo is too aggressive.
    Guarantees non-overlapping segments: train | embargo | val | embargo | test.
    """
    min_seg = cfg.window_size + 50  # Minimum usable segment length
    # Scale embargo: use configured value or 5% of series, whichever is smaller
    emb = min(int(cfg.embargo_bars), max(5, int(n * 0.03)))

    # Quick reject: need at least 3 segments + 2 embargos
    if n < 3 * min_seg + 2 * emb:
        return None

    # Available bars after subtracting 2 embargo gaps
    avail = n - 2 * emb

    # Allocate proportionally
    train_frac = 1.0 - cfg.val_ratio - cfg.holdout_ratio
    tr_len = int(avail * train_frac)
    va_len = int(avail * cfg.val_ratio)
    te_len = avail - tr_len - va_len  # Remainder to test

    # Enforce minimums (shrink train if needed to give val/test enough)
    if va_len < min_seg: va_len = min_seg
    if te_len < min_seg: te_len = min_seg
    tr_len = avail - va_len - te_len
    if tr_len < min_seg:
        return None  # Can't fit

    # Layout: [0..tr_end] [embargo] [va_start..va_end] [embargo] [te_start..n]
    tr_end = tr_len
    va_start = tr_end + emb
    va_end = va_start + va_len
    te_start = va_end + emb

    # Sanity checks
    if te_start + min_seg > n: return None
    return (0, tr_end), (va_start, va_end), (te_start, n)

def _assign_symbol_pools(datasets, cfg):
    """Assign disjoint symbol pools: train/val/holdout by base symbol."""
    if not cfg.split_by_symbol:
        return datasets
    rng = np.random.RandomState(cfg.holdout_seed)
    def base_sym(s):
        return s.rsplit("_", 1)[0] if "_" in s else s
    bases = sorted(list({base_sym(d.symbol) for d in datasets}))
    rng.shuffle(bases)
    n = len(bases)
    n_hold = max(1, int(n * cfg.holdout_ratio))
    n_val = max(1, int(n * cfg.val_ratio))
    hold = set(bases[:n_hold])
    val_set = set(bases[n_hold:n_hold + n_val])
    for d in datasets:
        b = base_sym(d.symbol)
        if b in hold: d.split_group = "holdout"
        elif b in val_set: d.split_group = "val"
        else: d.split_group = "train"
    return datasets

_PREPARE_VALID_TFS = {"1m","2m","5m","15m","30m","60m","1h","90m","1d","5d","1wk","1mo"}

def _process_one_symbol(args):
    """
    Top-level worker for parallel feature engineering.
    Must be module-level (not nested) so it is picklable by ProcessPoolExecutor.

    Returns: (sym, SymbolDataset | None, error_str | None)
    Data integrity guarantee: all processing is identical to the sequential path —
    compute_indicators → build_feature_matrix → _split_indices → SymbolDataset.
    No shared mutable state; each worker operates on its own copy of the DataFrame.
    """
    sym, df, cfg = args
    try:
        di = compute_indicators(df)
        f  = build_feature_matrix(di)
        # Volume: use real column or synthesize dummy (same as sequential path)
        if "Volume" in di.columns:
            p = di[["Open","High","Low","Close","Volume"]].values.astype(np.float32)
        else:
            ohlc      = di[["Open","High","Low","Close"]].values.astype(np.float32)
            dummy_vol = np.full((len(ohlc), 1), 1e6, dtype=np.float32)
            p         = np.hstack([ohlc, dummy_vol])
        ts    = np.array(di.index.astype(str))
        parts = sym.rsplit("_", 1)
        tf    = parts[1] if len(parts) == 2 and parts[1] in _PREPARE_VALID_TFS else "1d"
        n     = len(f)
        splits = _split_indices(n, cfg)
        if splits is None:
            return sym, None, f"too short ({n} bars, need ~{3*(cfg.window_size+50)}+embargo)"
        (tr0, tr1), (va0, va1), (te0, te1) = splits
        dataset = SymbolDataset(
            sym,
            f[tr0:tr1], f[va0:va1], f[te0:te1],
            p[tr0:tr1], p[va0:va1], p[te0:te1],
            tr1-tr0, va1-va0, te1-te0,
            ts[tr0:tr1], ts[va0:va1], ts[te0:te1],
            tf, split_group="train"
        )
        return sym, dataset, None
    except Exception as ex:
        return sym, None, f"error: {ex.__class__.__name__}: {str(ex)[:80]}"


def prepare_datasets(data, cfg):
    """
    Build SymbolDataset objects from raw price data.

    Parallelism: uses ProcessPoolExecutor when symbol count > 4 to run
    compute_indicators + build_feature_matrix across CPU cores.  Falls back
    to sequential for small batches where process-spawn overhead dominates.

    Data integrity:
    - Each worker receives its own copy of the DataFrame (no shared state).
    - Results are sorted by symbol name before _assign_symbol_pools so the
      seeded RNG produces the same train/val/holdout split regardless of the
      order futures complete.
    - Exceptions in workers are caught and reported as skipped symbols, not
      silently dropped.
    """
    st  = time.time()
    tot = len(data)
    skipped_reasons: dict = {}

    # Threshold: ProcessPoolExecutor has ~0.5-1s spawn overhead on Windows.
    # Not worth it for tiny symbol counts.
    n_workers = min(getattr(cfg, 'n_workers', 4), tot, os.cpu_count() or 1)
    use_parallel = (tot > 4) and (n_workers > 1)

    args_list = [(sym, df, cfg) for sym, df in data.items()]
    raw_results: list = []   # [(sym, dataset_or_None, error_or_None)]

    if use_parallel:
        tprint(f"Feature engineering: {tot} symbols × {n_workers} workers (parallel)", "info")
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one_symbol, args): args[0]
                       for args in args_list}
            for future in as_completed(futures):
                completed += 1
                try:
                    raw_results.append(future.result())
                except Exception as ex:
                    # Worker process itself crashed (OOM, segfault, etc.)
                    sym = futures[future]
                    raw_results.append((sym, None,
                        f"worker crash: {ex.__class__.__name__}: {str(ex)[:80]}"))
                if completed % 20 == 0 or completed == tot:
                    progress_bar(completed, tot, "Features", start_time=st)
    else:
        tprint(f"Feature engineering: {tot} symbols (sequential)", "info")
        for i, args in enumerate(args_list):
            raw_results.append(_process_one_symbol(args))
            if (i + 1) % 20 == 0 or i + 1 == tot:
                progress_bar(i + 1, tot, "Features", start_time=st)

    # Sort by symbol name so _assign_symbol_pools is deterministic regardless
    # of the order futures completed.
    raw_results.sort(key=lambda x: x[0])

    ds: list = []
    for sym, dataset, error in raw_results:
        if error:
            skipped_reasons[sym] = error
        elif dataset is not None:
            ds.append(dataset)

    # Diagnostic: show why datasets were skipped
    if skipped_reasons:
        if len(ds) == 0:
            tprint(f"All {len(skipped_reasons)} symbols skipped:", "warn")
        for sym, reason in list(skipped_reasons.items())[:5]:
            tprint(f"  {sym}: {reason}", "warn")

    ds = _assign_symbol_pools(ds, cfg)
    n_tr = sum(1 for d in ds if d.split_group == "train")
    n_va = sum(1 for d in ds if d.split_group == "val")
    n_ho = sum(1 for d in ds if d.split_group == "holdout")
    tprint(f"{len(ds)} datasets | pools: train={n_tr} val={n_va} holdout={n_ho}", "ok")
    tprint(f"  bars: train={sum(d.n_train for d in ds):,} "
           f"val={sum(d.n_val for d in ds):,} "
           f"test={sum(d.n_test for d in ds):,}", "info")
    return ds

class AlphaTradeSystem:
    def __init__(self,datasets,cfg):
        self.ds=datasets;self.cfg=cfg
        # 5A: Split pools -- train/val/holdout (disjoint if split_by_symbol)
        self.train_ds = [d for d in self.ds if d.split_group == "train"]
        self.val_ds = [d for d in self.ds if d.split_group == "val"]
        self.holdout_ds = [d for d in self.ds if d.split_group == "holdout"]
        tprint(f"Pools | train:{len(self.train_ds)} val:{len(self.val_ds)} holdout:{len(self.holdout_ds)}", "info")
        self.champ=build_network(cfg);self.cs=-np.inf;self.hist=[]
        self.bp=os.path.join(cfg.output_dir,"champion_model.pt")
        self.val_fw=ValidationFramework(cfg);self.syn=SyntheticMarketGenerator(cfg)
        raw=unwrap_net(self.champ); pc=sum(p.numel() for p in raw.parameters())
        tprint(f"System ready | {pc:,} params | AMP={'ON' if cfg.use_amp else 'OFF'} | "
               f"compile={'ON' if cfg.use_compile else 'OFF'} | n_envs={cfg.n_envs}","ok")
        # Fix #3: Build drift monitor from flattened obs vectors (representative of actual network input)
        self.drift_monitor = None
        if cfg.use_kill_switches and self.train_ds:
            try:
                obs_samples = []
                for d in self.train_ds[:5]:
                    env = TradingEnv(d.features_train, d.prices_train, cfg, d.symbol, ev=True)
                    o, _ = env.reset()
                    for _ in range(min(500, d.n_train - cfg.window_size - 10)):
                        obs_samples.append(o.copy())
                        o, _, t, tr, _ = env.step(2)  # FLAT to advance without trading
                        if t or tr: break
                if len(obs_samples) > 100:
                    obs_arr = np.array(obs_samples)
                    self.drift_monitor = DriftMonitor.from_dataset(obs_arr, cfg.drift_z_threshold)
                    tprint(f"DriftMonitor: {obs_arr.shape[1]} obs dims tracked (from {len(obs_samples)} samples)", "info")
            except Exception as e:
                tprint(f"DriftMonitor init skipped: {e}", "warn")

    def _vec(self,dss,ev=False):
        nt=self.cfg.n_envs if not ev else len(dss)
        env_args = []
        for i in range(nt):
            d=dss[i%len(dss)]
            features = d.features_test if ev else d.features_train
            prices = d.prices_test if ev else d.prices_train
            env_args.append((features, prices, self.cfg, d.symbol, ev))

        # GPU-vectorized env for training, full TradingEnv for evaluation
        if not ev and HAS_CUDA:
            try:
                return TorchVecEnv(env_args, self.cfg)
            except Exception as e:
                tprint(f"TorchVecEnv failed ({e}), falling back to sequential", "warn")

        # Fallback: sequential VectorizedEnvs (used for eval + CPU fallback)
        envs = [TradingEnv(*args) for args in env_args]
        for e in envs:
            e.drift_monitor = self.drift_monitor
        return VectorizedEnvs(envs)

    @torch.no_grad()
    def evaluate(self,net,datasets,label="eval"):
        net.eval();pnls=[];tds=[];wrs=[];pfs=[];dds=[];shs=[];ps={}
        act_counts = np.zeros(self.cfg.n_actions, dtype=np.int64)  # Action distribution tracking
        for d in datasets:
            env=TradingEnv(d.features_test,d.prices_test,self.cfg,d.symbol,True)
            obs,_=env.reset();done=False
            while not done:
                # NaN guard on observation
                if not np.isfinite(obs).all():
                    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
                st=torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                    # 6C: Deterministic eval (argmax) vs stochastic
                    if self.cfg.deterministic_eval:
                        logits, _, _ = net.forward(st)
                        if not torch.isfinite(logits).all():
                            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
                        a = torch.argmax(logits, dim=-1)
                    else:
                        a,_,_,_,_=net.get_action_and_value(st)
                act_val = a.item()
                act_counts[act_val] += 1
                obs,_,t,tr,info=env.step(act_val);done=t or tr
            pnls.append(info["net_pnl"]);tds.append(info["total_trades"])
            wrs.append(info["win_rate"]);pfs.append(info["profit_factor"])
            dds.append(info["max_drawdown"]);shs.append(info.get("sharpe_ratio",0))
            ps[d.symbol]={"pnl":info["net_pnl"],"trades":info["total_trades"],
                          "turnover":info.get("turnover",0.0)}

        # --- Robust risk-aware score ---
        start_cap = float(self.cfg.starting_capital)
        rets = np.array(pnls, dtype=np.float64) / max(start_cap, 1e-9)

        # Consistent aggregation: all symbols (not just active)
        ash_all = float(np.mean(shs)) if shs else 0.0
        dds_arr = np.array(dds, dtype=np.float64) if dds else np.array([100.0])

        # Breadth -- NO floor (a policy losing on 95% of symbols should score badly)
        sp = sum(1 for p in pnls if p > 0)
        br = sp / max(len(pnls), 1)

        # Winsorize returns to reduce one-symbol dominance
        if len(rets) >= 20:
            lo, hi = np.quantile(rets, [0.05, 0.95])
        else:
            lo, hi = (rets.min() if len(rets) else -1), (rets.max() if len(rets) else 1)
        rets_w = np.clip(rets, lo, hi) if len(rets) else rets

        # Robust central tendency: blend mean + median
        mean_ret = float(np.mean(rets_w)) if len(rets_w) else 0.0
        med_ret = float(np.median(rets_w)) if len(rets_w) else 0.0

        # Tail drawdown penalty (q75 = "bad typical", max = worst case)
        dd_q75 = float(np.quantile(dds_arr, 0.75))
        dd_max = float(np.max(dds_arr))

        # Turnover proxy from trade counts (higher trades = more friction)
        avg_trades = sum(tds) / max(len(tds), 1)
        turnover_pen = avg_trades * 0.01  # Heuristic cost per trade

        # Include per-symbol turnover if available
        total_turnover_from_info = sum(
            ps[s].get("turnover", 0.0) for s in ps
        )
        if total_turnover_from_info > 0:
            turnover_pen = (total_turnover_from_info / max(start_cap * len(ps), 1e-9)) * 0.05

        # Score = return*breadth + sharpe - tail_DD - turnover
        sc = (
            100.0 * (0.7 * mean_ret + 0.3 * med_ret) * br  # Return (no floor on br)
            + 0.20 * ash_all                                 # Sharpe (negative hurts)
            - 0.35 * (dd_q75 / 100.0)                       # Q75 drawdown
            - 0.25 * (dd_max / 100.0)                        # Max drawdown (worst symbol)
            - 0.10 * turnover_pen                            # Turnover cost
        )

        # Penalize no-trading behavior
        if sum(tds) == 0:
            sc -= 1.0

        ac = [i for i, t in enumerate(tds) if t > 0]
        aw = np.mean([wrs[i] for i in ac]) if ac else 0
        ap = np.mean([pfs[i] for i in ac]) if ac else 0

        # Action distribution summary
        total_acts = act_counts.sum()
        act_dist = act_counts / max(total_acts, 1)
        act_labels = [f"{t:+.1f}" for t in self.cfg.action_targets]  # e.g. "-1.0", "-0.5", "0.0", "+0.5", "+1.0"
        act_str = " ".join(f"{l}:{d:.0%}" for l, d in zip(act_labels, act_dist))
        dominant_act = int(np.argmax(act_counts))
        if act_dist[dominant_act] > 0.80:
            tprint(f"Action concentration WARNING: {act_labels[dominant_act]} = {act_dist[dominant_act]:.0%} "
                   f"(possible local optimum)", "warn")

        return {"label":label,"total_pnl":sum(pnls),"avg_pnl":np.mean(pnls) if pnls else 0,
                "total_trades":sum(tds),"avg_wr":aw,"avg_pf":ap,
                "avg_dd":float(np.mean(dds_arr)),"dd_q75":dd_q75,"dd_max":dd_max,
                "avg_sh":ash_all,"sp":sp,"st":len(datasets),"breadth":br*100,"score":sc,
                "per_sym":ps,"act_dist":act_str}

    def _pe(self,r,it):
        pc=C.GREEN if r["total_pnl"]>0 else C.RED
        print(f"\n  {C.BOLD}{C.CYAN}--- It {it} | {r['label']} ---{C.RESET}")
        if not getattr(self, '_pe_legend_shown', False):
            print(f"  {C.DIM}P&L=profit/loss | WR=win rate | PF=profit factor (wins/losses) | "
                  f"Sh=Sharpe (return per unit risk){C.RESET}")
            print(f"  {C.DIM}DD=max drawdown (worst dip) | Br=breadth (passed/total symbols) | "
                  f"Score=overall grade{C.RESET}")
            self._pe_legend_shown = True
        print(f"  P&L:{pc}${r['total_pnl']:>+12,.2f}{C.RESET} (${r['avg_pnl']:>+,.2f}/sym)")
        print(f"  WR:{r['avg_wr']:>5.1f}% PF:{r['avg_pf']:.2f} Sh:{r['avg_sh']:.2f} "
              f"DD:{r['avg_dd']:.1f}% Br:{r['sp']}/{r['st']}")
        print(f"  Score:{C.BOLD}{r['score']:>+.2f}{C.RESET}")
        if "act_dist" in r:
            print(f"  Actions: [{r['act_dist']}]")

    def train(self):
        print_box("ALPHA-TRADE v3.0 GPU-PARALLEL TRAINING",
                   f"iters={self.cfg.n_iterations} steps/it={self.cfg.total_timesteps_per_iter:,} "
                   f"n_envs={self.cfg.n_envs} AMP={'ON' if self.cfg.use_amp else 'OFF'} {DEVICE}")
        tprint("The AI is learning to trade by practicing on historical data (like a flight simulator).", "info")
        tprint("Each iteration: practice trades -> grade performance -> improve strategy -> repeat.", "info")
        np.random.seed(42);torch.manual_seed(42)
        if HAS_CUDA: torch.cuda.manual_seed_all(42)
        for it in range(1,self.cfg.n_iterations+1):
            t0=time.time(); print_divider(f"ITERATION {it}/{self.cfg.n_iterations}")
            if it == 1:
                tprint("Each iteration = one full round of practice. The AI plays thousands of trading", "info")
                tprint("scenarios, learns from wins and losses, then tries to beat its previous best.", "info")
            if HAS_CUDA: tprint(f"GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f}GB","gpu")
            # Clone champion
            raw=unwrap_net(self.champ); cnet=build_network(self.cfg)
            unwrap_net(cnet).load_state_dict(raw.state_dict())
            trainer=GPUPPOTrainer(cnet,self.cfg)
            # Plateau LR reset: bump LR back to initial to escape basin
            if getattr(self, '_plateau_lr_reset', False):
                for pg in trainer.opt.param_groups:
                    pg['lr'] = self.cfg.learning_rate
                tprint(f"LR reset to {self.cfg.learning_rate:.2e} (plateau escape)", "warn")
                self._plateau_lr_reset = False
            # Select envs (STANDARD PPO PRACTICE - Use ALL available datasets)
            # Research: PPO is on-policy and benefits from using all environments in parallel
            # The "static" behavior (same envs each iteration) is correct for PPO
            tds=(self.train_ds if self.train_ds else self.ds).copy()
            np.random.shuffle(tds)  # Shuffle for presentation diversity across iterations
            sel = tds[:min(16, len(tds))]  # Use all datasets (cap at 16 for memory)

            # Add synthetic regimes
            if self.cfg.n_synthetic_regimes>0:
                sd=self.syn.generate_multiple(self.cfg.n_synthetic_regimes,self.cfg.synthetic_bars_per_regime)
                sel.extend(prepare_datasets(sd,self.cfg)[:4])

            tprint(f"{len(sel)} envs x {self.cfg.n_envs} parallel = {len(sel)*self.cfg.n_envs} total","info")
            # Train
            tprint(f"PPO training ({self.cfg.total_timesteps_per_iter:,} steps)...","info")
            venv=self._vec(sel); res=trainer.train_iteration(venv,self.cfg.total_timesteps_per_iter)
            if hasattr(venv, 'close'): venv.close()  # Free subprocess resources
            tt=time.time()-t0; tp=res["steps"]/tt
            tprint(f"Done: {tt:.1f}s | {tp:,.0f} steps/s | reward:{res['mean_reward']:+.4f}","ok")
            # --- Per-iteration diagnostics dashboard ---
            um = res.get("update_metrics", {})
            if um:
                if it == 1:
                    tprint("Diagnostics key: ent=exploration level (higher=more adventurous), "
                           "grad_norm=learning speed,", "info")
                    tprint("  lr=step size, pl=policy loss (lower=better decisions), "
                           "vl=value loss (lower=better predictions)", "info")
                avg_ent = um.get("ent", 0)
                avg_gn = um.get("grad_norm", 0)
                cur_lr = um.get("lr", self.cfg.learning_rate)
                ent_coef_used = um.get("ent_coef_used", self.cfg.ent_coef)
                ent_status = "OK" if avg_ent >= self.cfg.ent_floor else "LOW"
                if ent_coef_used > self.cfg.ent_coef:
                    ent_status = "BOOSTED"
                tprint(f"Diagnostics: ent={avg_ent:.3f}[{ent_status}] grad_norm={avg_gn:.3f} "
                       f"lr={cur_lr:.2e} pl={um.get('pl',0):.4f} vl={um.get('vl',0):.4f} "
                       f"kl={um.get('kl',0):.4f}", "info")
            # MCTS -- Deep search + distillation into policy (AlphaGo-style)
            _mcts_wall_t0 = time.time()
            if self.cfg.mcts_rollouts>0 and it>1:
                # Iterative deepening: scale rollouts as value network improves
                if getattr(self.cfg, 'mcts_schedule_enabled', False):
                    progress = (it - 2) / max(self.cfg.n_iterations - 2, 1)
                    n_rollouts = int(self.cfg.mcts_schedule_base +
                                     progress * (self.cfg.mcts_schedule_max - self.cfg.mcts_schedule_base))
                    n_rollouts = 2 ** round(math.log2(max(n_rollouts, 8)))  # Round to power of 2
                else:
                    n_rollouts = self.cfg.mcts_rollouts
                tprint(f"MCTS deep search ({n_rollouts} rollouts, "
                       f"horizon={self.cfg.mcts_sim_horizon}) [PARALLEL]...","info")
                # Use parallel MCTS with deep search
                try:
                    from alphago_mcts_parallel import ParallelMCTSPlanner
                    mcts=ParallelMCTSPlanner(unwrap_net(cnet),self.cfg)
                    tprint("ParallelMCTSPlanner loaded (deep search: virtual loss + progressive widening)","gpu")
                except ImportError as e:
                    mcts=BatchedMCTSPlanner(unwrap_net(cnet),self.cfg)
                    tprint(f"Fallback to BatchedMCTSPlanner: {e}","warn")
                all_mcts_states=[]
                all_mcts_policies=[]
                # Scale symbol count with universe size: ~10% of available, minimum 3
                n_mcts_syms = min(len(sel), max(3, len(sel) // 10))
                for d in sel[:n_mcts_syms]:
                    env=TradingEnv(d.features_train,d.prices_train,self.cfg,d.symbol)
                    env.reset()
                    # Scale down roots inversely with rollout count to keep time budget
                    max_roots = min(self.cfg.mcts_batch_size, d.n_train - self.cfg.window_size - 5)
                    scale_factor = max(1, n_rollouts // 64)
                    n_roots = max(16, max_roots // scale_factor)
                    if n_roots < 2:
                        continue

                    # --- Stress-biased root sampling (point #5) ---
                    # Instead of random-walking from start (biases toward early bars),
                    # sample indices with bias toward high-volatility regimes.
                    valid_start = self.cfg.window_size
                    valid_end = d.n_train - 2
                    all_indices = np.arange(valid_start, valid_end)

                    # Compute local volatility (actual rolling std of log returns)
                    closes = d.prices_train[valid_start:valid_end, 3]
                    if len(closes) > 20:
                        log_rets = np.diff(np.log(closes + 1e-10))
                        # Actual rolling std: sqrt(rolling_mean(rets^2))
                        w = 20
                        if len(log_rets) >= w:
                            vol = np.sqrt(np.convolve(log_rets**2, np.ones(w)/w, mode='same'))
                        else:
                            vol = np.abs(log_rets)
                        vol = np.concatenate([[vol[0]], vol])  # Pad to match indices

                        # 50% from high-vol (top 30%), 50% from remainder
                        n_stress = n_roots // 2
                        n_uniform = n_roots - n_stress

                        hi_vol_threshold = np.quantile(vol, 0.70)
                        hi_vol_idx = all_indices[vol > hi_vol_threshold]

                        if len(hi_vol_idx) >= n_stress:
                            stress_idx = np.random.choice(hi_vol_idx, size=n_stress, replace=False)
                        else:
                            stress_idx = hi_vol_idx
                            n_uniform = n_roots - len(stress_idx)

                        # Sample uniform from REMAINING indices (avoid np.unique sort bias)
                        remaining = np.setdiff1d(all_indices, stress_idx, assume_unique=False)
                        if len(remaining) >= n_uniform:
                            uniform_idx = np.random.choice(remaining, size=n_uniform, replace=False)
                        else:
                            uniform_idx = remaining
                        sampled_idx = np.concatenate([stress_idx, uniform_idx])
                        np.random.shuffle(sampled_idx)  # Randomize order
                    else:
                        sampled_idx = np.random.choice(all_indices, size=min(n_roots, len(all_indices)), replace=False)

                    # Build root environments at sampled positions
                    root_envs = []
                    root_obs = []
                    for idx in sampled_idx:
                        e = env.clone()
                        e.set_index(int(idx))
                        # Pre-compute regime cache for stochastic cloning (~256× faster)
                        if self.cfg.mcts_stochastic:
                            e._precompute_stochastic_cache(
                                block_size=self.cfg.mcts_bootstrap_block,
                                horizon=self.cfg.mcts_sim_horizon)
                        root_envs.append(e)
                        root_obs.append(e._obs())

                    if root_envs:
                        _mcts_t0 = time.time()
                        improved = mcts.batch_search(root_envs, n_rollouts=n_rollouts)
                        _mcts_dt = time.time() - _mcts_t0
                        tprint(f"  MCTS {d.symbol}: {len(root_envs)} roots × {n_rollouts} rollouts "
                               f"= {_mcts_dt:.1f}s ({_mcts_dt/len(root_envs):.2f}s/root)", "info")
                        all_mcts_states.extend(root_obs)
                        all_mcts_policies.extend(improved)
                # Log tree depth statistics
                if hasattr(mcts, '_tree_depth') and hasattr(mcts, '_principal_variation_depth'):
                    try:
                        # Access the last search's roots via a quick re-check isn't possible,
                        # but we can log the rollout count and expected depth
                        pw_str = "PW=3" if getattr(self.cfg, 'mcts_progressive_widening', False) else "PW=off"
                        cont_str = f"+{getattr(self.cfg, 'mcts_continuation_steps', 0)}cont"
                        tprint(f"MCTS depth: {n_rollouts} rollouts, {pw_str}, {cont_str}, "
                               f"est. PV depth ~{int(np.log(n_rollouts)/np.log(1/0.4))+getattr(self.cfg,'mcts_continuation_steps',0)}","info")
                    except Exception:
                        pass
                # DISTILL: train policy to match MCTS-improved distribution
                if all_mcts_states:
                    states_arr=np.array(all_mcts_states)
                    policies_arr=np.array(all_mcts_policies)
                    # Filter out NaN/Inf policy rows before distillation
                    valid_mask=np.isfinite(policies_arr).all(axis=1) & np.isfinite(states_arr).all(axis=1)
                    if not valid_mask.all():
                        n_bad=(~valid_mask).sum()
                        tprint(f"MCTS: filtering {n_bad}/{len(valid_mask)} invalid rows","warn")
                        states_arr=states_arr[valid_mask]
                        policies_arr=policies_arr[valid_mask]
                    if len(states_arr)>0:
                        # Scale distillation steps with search depth
                        distill_steps = min(10, max(5, n_rollouts // 64))
                        kl_loss=trainer.distill_mcts(states_arr, policies_arr, n_distill_steps=distill_steps)
                        tprint(f"MCTS distilled: {len(states_arr)} states, KL={kl_loss:.4f} "
                               f"(distill_steps={distill_steps})","ok")
                    else:
                        tprint("MCTS: all policy rows invalid after filtering","warn")
                else:
                    tprint("MCTS: no states collected","warn")
                tprint(f"MCTS total: {time.time()-_mcts_wall_t0:.1f}s","info")
            # Eval
            er=self.evaluate(cnet,self.ds,f"Challenger (it {it})"); self._pe(er,it)
            # Validate -- multi-symbol with pass ratio (point #5)
            vp=True
            if it%3==0 or it==self.cfg.n_iterations:
                val_sel = list(self.val_ds) if self.val_ds else list(self.ds)
                np.random.shuffle(val_sel)
                val_sel = val_sel[:min(self.cfg.val_k + 3, len(val_sel))]

                # Validate on K symbols -- require pass ratio (batched + threaded)
                val_passes = 0
                val_tested = 0
                val_candidates = [d for d in val_sel if d.n_test > self.cfg.window_size + 100][:self.cfg.val_k]
                import time as _time; _val_t0=_time.time()
                if val_candidates:
                    unet = unwrap_net(cnet)
                    def _val_sym(d):
                        return (d.symbol, self.val_fw.full_validation(
                            unet, d.features_test, d.prices_test,
                            d.symbol, n_trials=max(it, 1)))
                    with ThreadPoolExecutor(max_workers=min(3, len(val_candidates))) as ex:
                        futs = [ex.submit(_val_sym, d) for d in val_candidates]
                        for fut in as_completed(futs):
                            try:
                                sym_name, vr = fut.result()
                            except Exception as e:
                                tprint(f"Val thread error: {e}", "warn")
                                continue
                            val_tested += 1
                            if vr.get("passed", False):
                                val_passes += 1
                            else:
                                reasons = vr.get("fail_reasons", [])
                                tprint(f"Val FAILED ({sym_name}): {'; '.join(reasons)}", "warn")
                tprint(f"Validation completed in {_time.time()-_val_t0:.1f}s ({val_tested} symbols)", "ok")

                if val_tested > 0:
                    pass_ratio = val_passes / val_tested
                    if pass_ratio < self.cfg.val_pass_ratio:
                        vp = False
                        tprint(f"Val gate: {val_passes}/{val_tested} passed ({pass_ratio:.0%} < {self.cfg.val_pass_ratio:.0%})", "warn")
                    else:
                        tprint(f"Val gate: {val_passes}/{val_tested} passed ({pass_ratio:.0%})", "ok")

                # Microstructure robustness -- multiple symbols, blocking gates (point #3)
                if vp:
                    micro_syms = [d for d in val_sel if d.n_test > self.cfg.window_size + 100]
                    micro_syms = micro_syms[:self.cfg.micro_n_symbols]
                    for d in micro_syms:
                        micro = self._microstructure_eval(
                            unwrap_net(cnet), d, n_seeds=self.cfg.micro_n_seeds
                        )
                        if micro is None:
                            continue
                        # Hard gates -- any failure blocks champion
                        if micro["sharpe_std"] > self.cfg.micro_sharpe_std_max:
                            vp = False
                            tprint(f"Micro FAIL ({d.symbol}): Sharpe std={micro['sharpe_std']:.2f} > {self.cfg.micro_sharpe_std_max}", "warn")
                        if micro["sharpe_min"] < self.cfg.micro_sharpe_min:
                            vp = False
                            tprint(f"Micro FAIL ({d.symbol}): Sharpe min={micro['sharpe_min']:.2f} < {self.cfg.micro_sharpe_min}", "warn")
                        if micro["pnl_min"] < 0 and micro["pnl_mean"] > 0:
                            if abs(micro["pnl_min"]) > 0.5 * micro["pnl_mean"]:
                                vp = False
                                tprint(f"Micro FAIL ({d.symbol}): worst-case PnL flip (min={micro['pnl_min']:.0f} vs mean={micro['pnl_mean']:.0f})", "warn")
                        if micro["dd_max"] > self.cfg.max_drawdown_pct * 100:
                            tprint(f"Micro WARN ({d.symbol}): dd_max={micro['dd_max']:.1f}% > threshold", "warn")
            # Champion -- ADDITIVE margin (point #6: multiplicative breaks on negatives)
            if er["score"]>self.cs+self.cfg.champion_margin and vp:
                tprint(f"NEW CHAMPION! {er['score']:+.2f} (was {self.cs:+.2f})","ok")
                self.champ=cnet;self.cs=er["score"]
                torch.save(unwrap_net(cnet).state_dict(),self.bp)
            else: tprint(f"Rejected: {er['score']:+.2f} vs {self.cs+self.cfg.champion_margin:+.2f}","warn")
            self.hist.append({"it":it,"score":er["score"],"champ":self.cs,"time":tt,"tput":tp,
                              **{k:v for k,v in er.items() if k!="per_sym"},
                              **({"ent": um.get("ent",0), "grad_norm": um.get("grad_norm",0),
                                  "lr": um.get("lr",0)} if um else {})})
            # --- Plateau detection + LR warm restart trigger ---
            pat = self.cfg.plateau_patience
            if len(self.hist) >= pat + 1:
                recent_champ_scores = [h["champ"] for h in self.hist[-(pat+1):]]
                if recent_champ_scores[-1] <= recent_champ_scores[0]:
                    tprint(f"Plateau detected: champion score unchanged for {pat} iterations "
                           f"({recent_champ_scores[0]:+.2f} -> {recent_champ_scores[-1]:+.2f})", "warn")
                    # Force LR reset on the NEXT iteration's trainer
                    self._plateau_lr_reset = True
            if HAS_CUDA: torch.cuda.empty_cache()
        self._save()
        return self.champ

    @torch.no_grad()
    def _microstructure_eval(self, net, dataset, n_seeds=5):
        """
        Point 6: Stochastic microstructure sweep.
        Runs policy over test data with randomized friction parameters
        to check if performance is fragile to cost assumptions.

        Returns dict with mean/std of Sharpe across seeds, or None if insufficient data.
        """
        if dataset.n_test < self.cfg.window_size + 100:
            return None

        net.eval()
        sharpes = []
        pnls = []
        drawdowns = []

        for seed in range(n_seeds):
            # Create modified config with randomized frictions
            cfg_copy = copy.copy(self.cfg)
            rng = np.random.RandomState(seed + 42)
            cfg_copy.slippage_pct = self.cfg.slippage_pct * rng.uniform(0.8, 2.5)
            cfg_copy.commission_pct = self.cfg.commission_pct * rng.uniform(0.8, 2.0)

            env = TradingEnv(dataset.features_test, dataset.prices_test,
                             cfg_copy, dataset.symbol, ev=True)
            obs, _ = env.reset()
            done = False
            while not done:
                st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                    a, _, _, _, _ = net.get_action_and_value(st)
                obs, _, t, tr, info = env.step(a.item())
                done = t or tr

            sharpes.append(info.get("sharpe_ratio", 0.0))
            pnls.append(info.get("net_pnl", 0.0))
            drawdowns.append(info.get("max_drawdown", 100.0))

        return {
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_min": float(np.min(sharpes)),
            "sharpe_max": float(np.max(sharpes)),
            "pnl_mean": float(np.mean(pnls)),
            "pnl_std": float(np.std(pnls)),
            "pnl_min": float(np.min(pnls)),
            "dd_mean": float(np.mean(drawdowns)),
            "dd_max": float(np.max(drawdowns)),
            "n_seeds": n_seeds,
        }

    def _save(self):
        import platform, hashlib
        def _hash_config(cfg):
            s = json.dumps({k:v for k,v in cfg.__dict__.items() if not callable(v)},
                           sort_keys=True, default=str).encode()
            return hashlib.sha256(s).hexdigest()[:12]
        r={"config":{"capital":self.cfg.starting_capital,"iters":self.cfg.n_iterations,
           "steps":self.cfg.total_timesteps_per_iter,"n_envs":self.cfg.n_envs,
           "amp":self.cfg.use_amp,"compile":self.cfg.use_compile,"device":str(DEVICE),"gpus":N_GPUS},
           "run_meta":{
               "timestamp": datetime.now().isoformat(),
               "python": sys.version,
               "platform": platform.platform(),
               "torch": torch.__version__,
               "numpy": np.__version__,
               "pandas": pd.__version__,
               "config_hash": _hash_config(self.cfg),
           },
           "champion_score":self.cs,"iterations":self.hist}
        p=os.path.join(self.cfg.output_dir,"training_report.json")
        with open(p,"w") as f: json.dump(r,f,indent=2,default=str)
        tprint(f"Report: {p}","ok")

    def final_eval(self):
        print_divider("FINAL EVALUATION")
        tprint("Final exam: testing the champion model on data it has NEVER seen before.", "info")
        # 5A: Evaluate on holdout set (never used in training or champion selection)
        eval_set = self.holdout_ds if self.holdout_ds else self.ds
        eval_label = "Champion (Holdout)" if self.holdout_ds else "Champion (Final)"
        r=self.evaluate(self.champ,eval_set,eval_label);self._pe(r,self.cfg.n_iterations)
        for d in eval_set[:5]:
            if d.n_test>self.cfg.window_size+200:
                vr=self.val_fw.walk_forward(unwrap_net(self.champ),d.features_test,d.prices_test,d.symbol)
                s="PASSED" if vr.get("pnl_consistency",0)>=.4 else "FAILED"
                tprint(f"  {d.symbol}: {s}","ok" if s=="PASSED" else "warn");break
        # Run detailed decision analysis
        print_divider("DECISION ANALYSIS")
        tprint("Replaying every trade to understand WHY the AI bought/sold (like a post-game review).", "info")
        self.run_detailed_analysis()
        return r

    # ================================================================
    # DECISION TRACKING & DETAILED ANALYSIS
    # ================================================================

    # Use module-level ACTION_NAMES: {0:FULL_SHORT, 1:HALF_SHORT, 2:FLAT, 3:HALF_LONG, 4:FULL_LONG}

    @torch.no_grad()
    def run_detailed_analysis(self):
        """
        Replay the champion agent on ALL test datasets, recording every
        decision: action chosen, action probabilities, value estimate,
        key indicator readings, position state, and portfolio value.

        Outputs:
          - decision_log.json    : Full structured trade-by-trade log
          - decision_log.csv     : Flat CSV of every trade entry/exit
          - step_decisions.csv   : Every step's action (optional, large)
          - Terminal summary     : Formatted decision table
        """
        net = unwrap_net(self.champ)
        net.eval()
        cfg = self.cfg

        all_trades = []       # Completed round-trip trades
        all_steps = []        # Per-step decisions (sampled)
        symbol_summaries = [] # Per-symbol summary

        tprint(f"Analyzing decisions for {len(self.ds)} datasets...", "info")
        analysis_start = time.time()

        for ds_idx, ds in enumerate(self.ds):
            if ds.n_test < cfg.window_size + 50:
                continue

            sym = ds.symbol
            feat = ds.features_test
            prices = ds.prices_test
            timestamps = ds.timestamps_test
            tf = ds.timeframe
            n_bars = len(feat)

            # Setup env
            env = TradingEnv(feat, prices, cfg, sym, ev=True)
            obs, _ = env.reset()
            done = False

            # Track per-step decisions
            step_log = []
            # Track open trade context
            open_trade = None
            trade_list = []
            step_idx = 0
            portfolio_values = [cfg.starting_capital]

            while not done:
                bar_idx = env.cs  # Current bar index in test data
                bar_ts = timestamps[bar_idx] if timestamps is not None and bar_idx < len(timestamps) else f"bar_{bar_idx}"
                close_price = prices[bar_idx, 3] if bar_idx < len(prices) else prices[-1, 3]
                open_price = prices[bar_idx, 0] if bar_idx < len(prices) else prices[-1, 0]
                high_price = prices[bar_idx, 1] if bar_idx < len(prices) else prices[-1, 1]
                low_price = prices[bar_idx, 2] if bar_idx < len(prices) else prices[-1, 2]

                # Get action probabilities, value estimate, risk estimate
                st = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                with torch.amp.autocast('cuda', enabled=cfg.use_amp):
                    logits, value, risk = net.forward(st)
                    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                    action = int(torch.argmax(logits, dim=-1).item())  # Greedy for analysis
                    val_est = value.item()
                    risk_est = risk.item()

                # Extract key indicators from feature window (last bar)
                fi = bar_idx - 1 if bar_idx > 0 else 0
                if fi < len(feat):
                    indicators = {
                        "rsi_14": float(feat[fi, FEATURE_COLUMNS.index("rsi_14")] * 100 + 50) if "rsi_14" in FEATURE_COLUMNS else 0,
                        "rsi_9": float(feat[fi, FEATURE_COLUMNS.index("rsi_9")] * 100 + 50) if "rsi_9" in FEATURE_COLUMNS else 0,
                        "macd_hist": float(feat[fi, FEATURE_COLUMNS.index("macd_hist_norm")]) if "macd_hist_norm" in FEATURE_COLUMNS else 0,
                        "bb_position": float(feat[fi, FEATURE_COLUMNS.index("bb_position")] + 0.5) if "bb_position" in FEATURE_COLUMNS else 0,
                        "adx": float(feat[fi, FEATURE_COLUMNS.index("adx_norm")] * 50) if "adx_norm" in FEATURE_COLUMNS else 0,
                        "vol_ratio": float(feat[fi, FEATURE_COLUMNS.index("vol_ratio")] + 1) if "vol_ratio" in FEATURE_COLUMNS else 0,
                        "trend_strength": float(feat[fi, FEATURE_COLUMNS.index("trend_strength")]) if "trend_strength" in FEATURE_COLUMNS else 0,
                        "vol_regime": float(feat[fi, FEATURE_COLUMNS.index("vol_regime")] + 1) if "vol_regime" in FEATURE_COLUMNS else 0,
                        "sma_cross": float(feat[fi, FEATURE_COLUMNS.index("sma_cross_20_50")]) if "sma_cross_20_50" in FEATURE_COLUMNS else 0,
                        "ema_cross": float(feat[fi, FEATURE_COLUMNS.index("ema_cross_9_21")]) if "ema_cross_9_21" in FEATURE_COLUMNS else 0,
                    }
                else:
                    indicators = {}

                # Derive human-readable reasoning from indicators + action
                reason = self._derive_reason(action, probs, indicators, env.shares, val_est)

                shares_before = env.shares
                obs, rew, term, trunc, info = env.step(action)
                done = term or trunc
                shares_after = env.shares

                # Portfolio value
                mk = info.get("portfolio_value", cfg.starting_capital)
                portfolio_values.append(mk)

                # Detect trade entry (was flat, now has position)
                if abs(shares_before) < 0.01 and abs(shares_after) > 0.01:
                    open_trade = {
                        "symbol": sym, "timeframe": tf,
                        "entry_time": bar_ts, "entry_bar": int(bar_idx),
                        "entry_price": float(close_price),
                        "entry_shares": float(abs(shares_after)),  # Store for PnL% calc
                        "entry_action": ACTION_NAMES.get(action, str(action)),
                        "entry_probs": {ACTION_NAMES[k]: round(float(probs[k]), 4) for k in range(5)},
                        "entry_value_est": round(float(val_est), 6),
                        "entry_risk_est": round(float(risk_est), 6),
                        "entry_indicators": {k: round(v, 4) for k, v in indicators.items()},
                        "entry_reason": reason,
                        "entry_portfolio": round(float(mk), 2),
                    }

                # Detect trade exit (had position, now flat)
                elif abs(shares_before) > 0.01 and abs(shares_after) < 0.01 and open_trade is not None:
                    pnl = info.get("net_pnl", 0)
                    # Use the last trade from trade_log to get exact PnL
                    last_trade_pnl = 0
                    if env.trade_log:
                        last_trade_pnl = env.trade_log[-1].get("pnl", 0)

                    open_trade.update({
                        "exit_time": bar_ts, "exit_bar": int(bar_idx),
                        "exit_price": float(close_price),
                        "exit_action": ACTION_NAMES.get(action, str(action)),
                        "exit_probs": {ACTION_NAMES[k]: round(float(probs[k]), 4) for k in range(5)},
                        "exit_value_est": round(float(val_est), 6),
                        "exit_risk_est": round(float(risk_est), 6),
                        "exit_indicators": {k: round(v, 4) for k, v in indicators.items()},
                        "exit_reason": reason,
                        "exit_portfolio": round(float(mk), 2),
                        "trade_pnl": round(float(last_trade_pnl), 2),
                        "trade_pnl_pct": round(float(last_trade_pnl / (open_trade["entry_price"] * open_trade.get("entry_shares", 1) + 1e-10) * 100), 4),
                        "holding_bars": int(bar_idx - open_trade["entry_bar"]),
                        "result": "WIN" if last_trade_pnl > 0 else "LOSS",
                    })
                    trade_list.append(open_trade)
                    all_trades.append(open_trade)
                    open_trade = None

                # Record step (sample every Nth to keep file manageable)
                if step_idx % max(1, n_bars // 200) == 0 or action != 0:
                    step_record = {
                        "symbol": sym, "timeframe": tf, "step": step_idx,
                        "bar": int(bar_idx), "timestamp": str(bar_ts),
                        "O": round(float(open_price), 4), "H": round(float(high_price), 4),
                        "L": round(float(low_price), 4), "C": round(float(close_price), 4),
                        "action": ACTION_NAMES.get(action, str(action)),
                        "confidence": round(float(max(probs)), 4),
                        "probs": {ACTION_NAMES[k]: round(float(probs[k]), 4) for k in range(5)},
                        "value_est": round(float(val_est), 6),
                        "risk_est": round(float(risk_est), 6),
                        "position": "LONG" if env.shares > 0.01 else ("SHORT" if env.shares < -0.01 else "FLAT"),
                        "portfolio": round(float(mk), 2),
                        "reward": round(float(rew), 6),
                        "reason": reason,
                    }
                    all_steps.append(step_record)

                step_idx += 1

            # Close any still-open trade
            if open_trade is not None:
                open_trade.update({
                    "exit_time": bar_ts, "exit_bar": int(bar_idx) if 'bar_idx' in dir() else 0,
                    "exit_price": float(prices[-1, 3]), "exit_action": "FORCED_CLOSE",
                    "exit_reason": "Episode ended", "result": "FORCED",
                    "holding_bars": int((bar_idx if 'bar_idx' in dir() else 0) - open_trade["entry_bar"]),
                    "trade_pnl": round(float(env.trade_log[-1]["pnl"]) if env.trade_log else 0, 2),
                })
                trade_list.append(open_trade)
                all_trades.append(open_trade)

            # Per-symbol summary
            if trade_list:
                wins = sum(1 for t in trade_list if t.get("result") == "WIN")
                losses = sum(1 for t in trade_list if t.get("result") == "LOSS")
                total_pnl = sum(t.get("trade_pnl", 0) for t in trade_list)
                avg_hold = np.mean([t.get("holding_bars", 0) for t in trade_list])
                symbol_summaries.append({
                    "symbol": sym, "timeframe": tf, "trades": len(trade_list),
                    "wins": wins, "losses": losses,
                    "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "avg_holding_bars": round(avg_hold, 1),
                    "peak_portfolio": round(max(portfolio_values), 2),
                    "min_portfolio": round(min(portfolio_values), 2),
                })

            if (ds_idx + 1) % 20 == 0 or ds_idx + 1 == len(self.ds):
                progress_bar(ds_idx + 1, len(self.ds), "Analyzing", start_time=analysis_start)

        # === SAVE REPORTS ===
        out_dir = cfg.output_dir

        # 1. Full trade decision log (JSON)
        decision_report = {
            "generated": datetime.now().isoformat(),
            "model": self.bp,
            "device": str(DEVICE),
            "total_symbols_analyzed": len(symbol_summaries),
            "total_trades": len(all_trades),
            "total_wins": sum(1 for t in all_trades if t.get("result") == "WIN"),
            "total_losses": sum(1 for t in all_trades if t.get("result") == "LOSS"),
            "total_pnl": round(sum(t.get("trade_pnl", 0) for t in all_trades), 2),
            "symbol_summaries": symbol_summaries,
            "trades": all_trades,
        }
        json_path = os.path.join(out_dir, "decision_log.json")
        with open(json_path, "w") as f:
            json.dump(decision_report, f, indent=2, default=str)
        tprint(f"Decision log: {json_path} ({len(all_trades)} trades)", "ok")

        # 2. Trade CSV (flat, easy to import to Excel/Sheets)
        if all_trades:
            csv_path = os.path.join(out_dir, "trade_decisions.csv")
            csv_rows = []
            for t in all_trades:
                row = {
                    "symbol": t.get("symbol", ""),
                    "timeframe": t.get("timeframe", ""),
                    "entry_time": t.get("entry_time", ""),
                    "entry_bar": t.get("entry_bar", ""),
                    "entry_price": t.get("entry_price", ""),
                    "entry_action": t.get("entry_action", ""),
                    "entry_reason": t.get("entry_reason", ""),
                    "entry_rsi": t.get("entry_indicators", {}).get("rsi_14", ""),
                    "entry_macd": t.get("entry_indicators", {}).get("macd_hist", ""),
                    "entry_bb_pos": t.get("entry_indicators", {}).get("bb_position", ""),
                    "entry_adx": t.get("entry_indicators", {}).get("adx", ""),
                    "entry_trend": t.get("entry_indicators", {}).get("trend_strength", ""),
                    "entry_vol_regime": t.get("entry_indicators", {}).get("vol_regime", ""),
                    "entry_value_est": t.get("entry_value_est", ""),
                    "exit_time": t.get("exit_time", ""),
                    "exit_bar": t.get("exit_bar", ""),
                    "exit_price": t.get("exit_price", ""),
                    "exit_action": t.get("exit_action", ""),
                    "exit_reason": t.get("exit_reason", ""),
                    "exit_rsi": t.get("exit_indicators", {}).get("rsi_14", ""),
                    "exit_macd": t.get("exit_indicators", {}).get("macd_hist", ""),
                    "exit_bb_pos": t.get("exit_indicators", {}).get("bb_position", ""),
                    "holding_bars": t.get("holding_bars", ""),
                    "trade_pnl": t.get("trade_pnl", ""),
                    "result": t.get("result", ""),
                }
                csv_rows.append(row)
            df_csv = pd.DataFrame(csv_rows)
            df_csv.to_csv(csv_path, index=False)
            tprint(f"Trade CSV: {csv_path}", "ok")

        # 3. Step-level decisions CSV (sampled)
        if all_steps:
            steps_path = os.path.join(out_dir, "step_decisions.csv")
            df_steps = pd.DataFrame(all_steps)
            # Flatten probs dict into columns
            for a_name in ACTION_NAMES.values():
                df_steps[f"prob_{a_name}"] = df_steps["probs"].apply(lambda x: x.get(a_name, 0))
            df_steps.drop(columns=["probs"], inplace=True, errors="ignore")
            df_steps.to_csv(steps_path, index=False)
            tprint(f"Step decisions: {steps_path} ({len(all_steps)} rows)", "ok")

        # === TERMINAL DECISION TABLE ===
        self._print_decision_table(all_trades, symbol_summaries)

    def _derive_reason(self, action, probs, ind, shares, val_est):
        """Generate human-readable reason for the decision based on indicators."""
        reasons = []
        rsi = ind.get("rsi_14", 50)
        macd = ind.get("macd_hist", 0)
        bb = ind.get("bb_position", 0.5)
        adx = ind.get("adx", 25)
        trend = ind.get("trend_strength", 0)
        vol_reg = ind.get("vol_regime", 1)
        sma_x = ind.get("sma_cross", 0)
        ema_x = ind.get("ema_cross", 0)

        target = self.cfg.action_targets[action]

        if target > 0:  # LONG targets
            if rsi < 35: reasons.append("RSI oversold")
            elif rsi < 45: reasons.append("RSI near oversold")
            if macd > 0: reasons.append("MACD bullish")
            if bb < 0.3: reasons.append("Near lower BB")
            if trend > 0.3: reasons.append("Uptrend")
            if sma_x > 0 and ema_x > 0: reasons.append("Moving avg bullish cross")
            if adx > 25: reasons.append("Strong trend (ADX)")
            if val_est > 0: reasons.append(f"Positive value est ({val_est:.4f})")
            if target >= 0.9: reasons.append("Full long conviction")
        elif target < 0:  # SHORT targets
            if rsi > 70: reasons.append("RSI overbought")
            elif rsi > 60: reasons.append("RSI elevated")
            if macd < 0: reasons.append("MACD bearish")
            if bb > 0.8: reasons.append("Near upper BB")
            if trend < -0.2: reasons.append("Downtrend signal")
            if vol_reg > 1.5: reasons.append("High volatility regime")
            if val_est < 0: reasons.append(f"Negative value est ({val_est:.4f})")
            if target <= -0.9: reasons.append("Full short conviction")
        else:  # FLAT target
            conf = max(probs)
            if abs(shares) > 0:
                reasons.append("Exiting position")
                if abs(trend) < 0.2: reasons.append("Trend fading")
            else:
                reasons.append("No clear signal")
                if conf < 0.4: reasons.append("Low confidence")
                if adx < 20: reasons.append("Weak trend (low ADX)")

        if not reasons:
            reasons.append("Policy network decision")

        return "; ".join(reasons[:4])

    def _print_decision_table(self, trades, summaries):
        """Print formatted decision summary to terminal."""
        if not trades:
            tprint("No trades recorded during analysis.", "warn")
            return

        # Header
        print(f"\n  {C.BOLD}{C.CYAN}{'='*96}{C.RESET}")
        print(f"  {C.BOLD}{C.WHITE}  TRADE DECISION LOG -- {len(trades)} Trades across {len(summaries)} Symbols{C.RESET}")
        print(f"  {C.BOLD}{C.CYAN}{'='*96}{C.RESET}")
        print(f"  {C.DIM}  Every trade the AI made, with its reasoning -- like reading a trader's journal.{C.RESET}")

        # Per-symbol summary table
        print(f"\n  {C.BOLD}  SYMBOL SUMMARY:{C.RESET}")
        print(f"  {C.DIM}TF=timeframe | Win%=percentage of profitable trades | "
              f"P&L=total profit/loss | AvgHold=average trade duration{C.RESET}")
        print(f"  {'Symbol':<18s} {'TF':>4s} {'Trades':>6s} {'Win%':>6s} {'P&L':>12s} {'AvgHold':>8s}")
        print(f"  {'-'*60}")
        for s in sorted(summaries, key=lambda x: x["total_pnl"], reverse=True):
            pnl_c = C.GREEN if s["total_pnl"] > 0 else C.RED
            print(f"  {s['symbol']:<18s} {s['timeframe']:>4s} {s['trades']:>6d} "
                  f"{s['win_rate']:>5.1f}% {pnl_c}${s['total_pnl']:>+10,.2f}{C.RESET} "
                  f"{s['avg_holding_bars']:>7.1f}b")

        # Top trades table
        sorted_trades = sorted(trades, key=lambda x: abs(x.get("trade_pnl", 0)), reverse=True)
        n_show = min(20, len(sorted_trades))
        print(f"\n  {C.BOLD}  TOP {n_show} TRADES (by absolute P&L):{C.RESET}")
        print(f"  {'#':>3s} {'Symbol':<15s} {'TF':>4s} {'Entry':>12s} {'Exit':>12s} "
              f"{'Action':>10s} {'P&L':>10s} {'Bars':>5s} {'Result':>6s} {'Entry Reason'}")
        print(f"  {C.DIM}  Entry/Exit=when trade started/ended | P&L=profit or loss | "
              f"Bars=how long held | Entry Reason=why the AI traded{C.RESET}")
        print(f"  {'-'*120}")

        for i, t in enumerate(sorted_trades[:n_show]):
            pnl = t.get("trade_pnl", 0)
            pnl_c = C.GREEN if pnl > 0 else C.RED
            result_c = C.GREEN if t.get("result") == "WIN" else C.RED

            # Format timestamps (show date or time depending on timeframe)
            entry_t = str(t.get("entry_time", ""))[:19]  # Trim to YYYY-MM-DD HH:MM:SS
            exit_t = str(t.get("exit_time", ""))[:19]
            # For daily, show just date; for intraday show time
            if t.get("timeframe", "1d") in ("1d", "5d", "1wk"):
                entry_t = entry_t[:10]  # YYYY-MM-DD
                exit_t = exit_t[:10]
            else:
                entry_t = entry_t[5:]  # MM-DD HH:MM:SS
                exit_t = exit_t[5:]

            entry_act = t.get("entry_action", "?")[:10]
            reason = t.get("entry_reason", "")[:45]

            print(f"  {i+1:>3d} {t.get('symbol',''):<15s} {t.get('timeframe',''):>4s} "
                  f"{entry_t:>12s} {exit_t:>12s} "
                  f"{entry_act:>10s} {pnl_c}${pnl:>+9,.2f}{C.RESET} "
                  f"{t.get('holding_bars',0):>5d} "
                  f"{result_c}{t.get('result','?'):>6s}{C.RESET} {reason}")

        # Aggregate stats
        wins = sum(1 for t in trades if t.get("result") == "WIN")
        losses = sum(1 for t in trades if t.get("result") == "LOSS")
        total_pnl = sum(t.get("trade_pnl", 0) for t in trades)
        avg_win = np.mean([t["trade_pnl"] for t in trades if t.get("result") == "WIN"]) if wins else 0
        avg_loss = np.mean([t["trade_pnl"] for t in trades if t.get("result") == "LOSS"]) if losses else 0
        avg_hold = np.mean([t.get("holding_bars", 0) for t in trades])

        # Reason frequency analysis
        all_entry_reasons = []
        for t in trades:
            r = t.get("entry_reason", "")
            all_entry_reasons.extend([x.strip() for x in r.split(";") if x.strip()])
        reason_counts = defaultdict(int)
        for r in all_entry_reasons:
            reason_counts[r] += 1
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:8]

        print(f"\n  {C.BOLD}  AGGREGATE STATS:{C.RESET}")
        pnl_c = C.GREEN if total_pnl > 0 else C.RED
        print(f"  Total P&L:     {pnl_c}${total_pnl:>+12,.2f}{C.RESET}")
        print(f"  Win/Loss:      {wins}W / {losses}L ({wins/(max(wins+losses,1))*100:.1f}%)")
        print(f"  Avg Win:       ${avg_win:>+,.2f}")
        print(f"  Avg Loss:      ${avg_loss:>+,.2f}")
        print(f"  Avg Hold:      {avg_hold:.1f} bars")
        if avg_loss != 0:
            print(f"  Reward/Risk:   {abs(avg_win/(avg_loss+1e-10)):.2f}")

        if top_reasons:
            print(f"\n  {C.BOLD}  TOP ENTRY REASONS:{C.RESET}")
            for reason, count in top_reasons:
                pct = count / max(len(trades), 1) * 100
                bar_len = int(pct / 3)
                print(f"  {'#' * bar_len:<20s} {count:>4d} ({pct:>5.1f}%) {reason}")

        print(f"\n  {C.CYAN}{'='*96}{C.RESET}")


# ============================================================================
# DATA LOADING
# ============================================================================
# WARNING: SURVIVORSHIP BIAS -- This list contains CURRENT S&P 500 constituents.
# Any backtest using these symbols is biased because it excludes firms that were
# delisted, acquired, or dropped from the index during the backtest period.
# For valid historical backtests, use a point-in-time index membership dataset.
# Estimated bias: +0.3 to +0.5 annualized Sharpe for US large-cap equities.
# Top 200 US stocks by market cap (Feb 2026), organized by GICS sector.
# SMA trend overlay performance note: works best on cyclical/growth sectors
# (Technology, Industrials, Financials) with persistent trends; weakest on
# defensive/stable sectors (Healthcare, Energy, Utilities) that mean-revert.
DEFAULT_SYMBOLS=[
    # --- Technology (45) -- Software, Semiconductors, Internet, Cybersecurity ---
    "AAPL",   # Apple -- Consumer electronics / software
    "MSFT",   # Microsoft -- Software / cloud (Azure)
    "NVDA",   # NVIDIA -- Semiconductors (AI/GPU)
    "AMZN",   # Amazon -- E-commerce / cloud (AWS)
    "GOOGL",  # Alphabet -- Internet / advertising
    "META",   # Meta Platforms -- Social media / advertising
    "AVGO",   # Broadcom -- Semiconductors / infrastructure software
    "ORCL",   # Oracle -- Enterprise software / cloud
    "CRM",    # Salesforce -- Cloud CRM software
    "AMD",    # AMD -- Semiconductors (CPU/GPU)
    "PLTR",   # Palantir -- AI / data analytics
    "ADBE",   # Adobe -- Creative / document software
    "CSCO",   # Cisco -- Networking equipment / software
    "ACN",    # Accenture -- IT consulting / services
    "IBM",    # IBM -- Enterprise IT / hybrid cloud
    "NOW",    # ServiceNow -- IT workflow automation
    "TXN",    # Texas Instruments -- Analog semiconductors
    "QCOM",   # Qualcomm -- Mobile semiconductors
    "KLAC",   # KLA -- Semiconductor process control
    "INTU",   # Intuit -- Financial software (TurboTax, QuickBooks)
    "AMAT",   # Applied Materials -- Semiconductor equipment
    "LRCX",   # Lam Research -- Semiconductor equipment
    "APH",    # Amphenol -- Electronic connectors / sensors
    "ANET",   # Arista Networks -- Cloud networking
    "PANW",   # Palo Alto Networks -- Cybersecurity
    "MU",     # Micron -- Memory semiconductors
    "INTC",   # Intel -- Semiconductors (CPU)
    "ADI",    # Analog Devices -- Analog/mixed-signal semiconductors
    "NFLX",   # Netflix -- Streaming media
    "UBER",   # Uber -- Ride-sharing / delivery platform
    "APP",    # AppLovin -- Mobile advertising / gaming
    "CRWD",   # CrowdStrike -- Cybersecurity (endpoint)
    "WDC",    # Western Digital -- Hard disk drives
    "SNDK",   # SanDisk -- Flash storage / SSD (spun off from WDC)
    "SNPS",   # Synopsys -- EDA / chip design software
    "CDNS",   # Cadence Design Systems -- EDA / chip design software
    "DELL",   # Dell Technologies -- PCs / servers / storage
    "MSI",    # Motorola Solutions -- Public safety communications
    "GLW",    # Corning -- Specialty glass / optical fiber
    "DASH",   # DoorDash -- Delivery platform
    "MRVL",   # Marvell Technology -- Semiconductors (networking/storage)
    "NET",    # Cloudflare -- Internet security / CDN
    "FTNT",   # Fortinet -- Cybersecurity (network)
    "SNOW",   # Snowflake -- Cloud data platform
    "MPWR",   # Monolithic Power Systems -- Power semiconductors

    # --- Financials (33) -- Banks, Insurance, Payments, Capital Markets ---
    "JPM",    # JPMorgan Chase -- Diversified banking
    "V",      # Visa -- Payment processing
    "MA",     # Mastercard -- Payment processing
    "BAC",    # Bank of America -- Diversified banking
    "GS",     # Goldman Sachs -- Investment banking / capital markets
    "MS",     # Morgan Stanley -- Investment banking / wealth management
    "WFC",    # Wells Fargo -- Diversified banking
    "C",      # Citigroup -- Diversified banking
    "AXP",    # American Express -- Payments / consumer finance
    "BLK",    # BlackRock -- Asset management
    "SPGI",   # S&P Global -- Financial data / ratings
    "SCHW",   # Charles Schwab -- Brokerage / wealth management
    "CB",     # Chubb -- Property & casualty insurance
    "MMC",    # Marsh McLennan -- Insurance brokerage
    "FI",     # Fiserv -- Financial technology / payments
    "BX",     # Blackstone -- Alternative asset management
    "KKR",    # KKR -- Private equity / alternative assets
    "CME",    # CME Group -- Derivatives exchange
    "ICE",    # Intercontinental Exchange -- Financial exchanges
    "PYPL",   # PayPal -- Digital payments
    "COF",    # Capital One -- Consumer banking / credit cards
    "PGR",    # Progressive -- Auto / property insurance
    "PNC",    # PNC Financial -- Regional banking
    "USB",    # U.S. Bancorp -- Regional banking
    "IBKR",   # Interactive Brokers -- Electronic brokerage
    "BK",     # BNY Mellon -- Custody / asset servicing
    "MCO",    # Moody's -- Credit ratings / analytics
    "APO",    # Apollo Global Management -- Private equity
    "HOOD",   # Robinhood -- Retail brokerage
    "TRV",    # Travelers -- Property / casualty insurance
    "TFC",    # Truist Financial -- Regional banking
    "AFL",    # Aflac -- Supplemental insurance
    "AJG",    # Arthur J. Gallagher -- Insurance brokerage

    # --- Healthcare (25) -- Pharma, MedTech, Insurance, Distribution ---
    "LLY",    # Eli Lilly -- Pharmaceuticals (GLP-1 / diabetes)
    "UNH",    # UnitedHealth -- Health insurance / Optum
    "JNJ",    # Johnson & Johnson -- Pharma / medical devices
    "ABBV",   # AbbVie -- Pharmaceuticals (immunology)
    "MRK",    # Merck -- Pharmaceuticals (oncology / vaccines)
    "TMO",    # Thermo Fisher -- Life sciences instruments
    "ABT",    # Abbott Labs -- Medical devices / diagnostics
    "ISRG",   # Intuitive Surgical -- Robotic surgery
    "SYK",    # Stryker -- Orthopedic / medical devices
    "VRTX",   # Vertex -- Biotech (cystic fibrosis)
    "GILD",   # Gilead Sciences -- Biotech (antiviral)
    "BSX",    # Boston Scientific -- Medical devices
    "AMGN",   # Amgen -- Biotech (oncology / bone)
    "PFE",    # Pfizer -- Pharmaceuticals
    "REGN",   # Regeneron -- Biotech (ophthalmology)
    "CI",     # Cigna -- Health insurance
    "BMY",    # Bristol-Myers Squibb -- Pharmaceuticals
    "MCK",    # McKesson -- Pharma distribution
    "DHR",    # Danaher -- Life sciences / diagnostics
    "CVS",    # CVS Health -- Pharmacy / health services
    "HCA",    # HCA Healthcare -- Hospital operator
    "ELV",    # Elevance Health -- Health insurance
    "COR",    # Cencora -- Pharma distribution
    "BDX",    # Becton Dickinson -- Medical devices / diagnostics
    "ZTS",    # Zoetis -- Animal health pharmaceuticals

    # --- Consumer Staples (12) -- Food, Beverages, Household ---
    "WMT",    # Walmart -- Retail (grocery / general merchandise)
    "COST",   # Costco -- Warehouse retail
    "PG",     # Procter & Gamble -- Household products
    "KO",     # Coca-Cola -- Beverages
    "PEP",    # PepsiCo -- Beverages / snacks
    "PM",     # Philip Morris -- Tobacco
    "MCD",    # McDonald's -- Quick-service restaurants
    "MDLZ",   # Mondelez -- Snacks (Oreo, Cadbury)
    "EL",     # Estee Lauder -- Beauty / cosmetics
    "MO",     # Altria -- Tobacco / nicotine
    "MNST",   # Monster Beverage -- Energy drinks
    "CL",     # Colgate-Palmolive -- Oral / personal care

    # --- Consumer Discretionary (20) -- Retail, Auto, Leisure, Media ---
    "TSLA",   # Tesla -- Electric vehicles / energy
    "HD",     # Home Depot -- Home improvement retail
    "LOW",    # Lowe's -- Home improvement retail
    "BKNG",   # Booking Holdings -- Online travel
    "TJX",    # TJX Companies -- Off-price retail
    "SBUX",   # Starbucks -- Coffee / restaurants
    "NKE",    # Nike -- Athletic apparel / footwear
    "BRK-B",  # Berkshire Hathaway -- Conglomerate (insurance-led)
    "DIS",    # Walt Disney -- Theme parks / streaming / media
    "MAR",    # Marriott -- Hotels / hospitality
    "RCL",    # Royal Caribbean -- Cruise lines / leisure
    "ORLY",   # O'Reilly Automotive -- Auto parts retail
    "ABNB",   # Airbnb -- Short-term rental platform
    "GM",     # General Motors -- Automobiles
    "HLT",    # Hilton -- Hotels / hospitality
    "CVNA",   # Carvana -- Online auto retail
    "ROST",   # Ross Stores -- Off-price retail
    "AZO",    # AutoZone -- Auto parts retail
    "F",      # Ford -- Automobiles
    "WBD",    # Warner Bros Discovery -- Media / entertainment

    # --- Industrials (30) -- Aerospace, Defense, Equipment, Transport ---
    "CAT",    # Caterpillar -- Construction / mining equipment
    "GE",     # GE Aerospace -- Jet engines / power
    "RTX",    # RTX (Raytheon) -- Aerospace / defense systems
    "DE",     # Deere & Co -- Agricultural / construction equipment
    "UNP",    # Union Pacific -- Railroad / freight transport
    "HON",    # Honeywell -- Aerospace / building tech / industrials
    "BA",     # Boeing -- Aerospace / defense
    "ADP",    # ADP -- Payroll / HR services
    "LMT",    # Lockheed Martin -- Defense / aerospace
    "HWM",    # Howmet Aerospace -- Aerospace components / fasteners
    "NOC",    # Northrop Grumman -- Defense / space systems
    "UPS",    # UPS -- Package delivery / logistics
    "GD",     # General Dynamics -- Defense / aerospace / marine
    "VRT",    # Vertiv -- Data center power / cooling infrastructure
    "WM",     # Waste Management -- Waste collection / recycling
    "FDX",    # FedEx -- Express delivery / logistics
    "MMM",    # 3M -- Diversified industrials / materials
    "ITW",    # Illinois Tool Works -- Diversified manufacturing
    "EMR",    # Emerson -- Automation / industrial software
    "PWR",    # Quanta Services -- Electric / infrastructure construction
    "CMI",    # Cummins -- Engines / power generation
    "CTAS",   # Cintas -- Workplace uniforms / facility services
    "TDG",    # TransDigm -- Aerospace components
    "NSC",    # Norfolk Southern -- Railroad / freight transport
    "CSX",    # CSX -- Railroad / freight transport
    "RSG",    # Republic Services -- Waste collection / recycling
    "PCAR",   # Paccar -- Heavy trucks (Kenworth, Peterbilt)
    "LHX",    # L3Harris -- Defense electronics / communications
    "PH",     # Parker-Hannifin -- Motion / control technologies
    "URI",    # United Rentals -- Equipment rental

    # --- Energy (14) -- Oil & Gas, Midstream, Refining, Services ---
    "XOM",    # ExxonMobil -- Oil & gas (integrated)
    "CVX",    # Chevron -- Oil & gas (integrated)
    "COP",    # ConocoPhillips -- Oil & gas (exploration & production)
    "WMB",    # Williams Companies -- Natural gas pipelines (midstream)
    "EPD",    # Enterprise Products -- NGL pipelines / midstream MLP
    "SLB",    # SLB (Schlumberger) -- Oilfield services / technology
    "KMI",    # Kinder Morgan -- Natural gas pipelines (midstream)
    "EOG",    # EOG Resources -- Oil & gas (exploration & production)
    "ET",     # Energy Transfer -- Midstream MLP (pipelines / terminals)
    "PSX",    # Phillips 66 -- Refining / midstream / chemicals
    "VLO",    # Valero Energy -- Refining
    "BKR",    # Baker Hughes -- Oilfield services / equipment
    "MPLX",   # MPLX -- Midstream MLP (Marathon Petroleum)
    "MPC",    # Marathon Petroleum -- Refining

    # --- Utilities (9) -- Electric, Gas, Renewables ---
    "NEE",    # NextEra Energy -- Electric utilities / renewables
    "SO",     # Southern Company -- Electric / gas utilities
    "DUK",    # Duke Energy -- Electric utilities
    "CEG",    # Constellation Energy -- Nuclear / clean energy
    "GEV",    # GE Vernova -- Power generation / grid equipment
    "AEP",    # American Electric Power -- Electric utilities
    "VST",    # Vistra -- Power generation / retail electricity
    "SRE",    # Sempra -- Electric / gas utilities
    "D",      # Dominion Energy -- Electric / gas utilities

    # --- Materials (7) -- Chemicals, Mining, Coatings ---
    "LIN",    # Linde -- Industrial gases
    "APD",    # Air Products -- Industrial gases
    "SCCO",   # Southern Copper -- Copper mining
    "NEM",    # Newmont -- Gold mining
    "FCX",    # Freeport-McMoRan -- Copper / gold mining
    "SHW",    # Sherwin-Williams -- Paints / coatings
    "ECL",    # Ecolab -- Water treatment / hygiene chemicals

    # --- Real Estate (7) -- REITs ---
    "PLD",    # Prologis -- Industrial REIT (warehouses / logistics)
    "WELL",   # Welltower -- Healthcare REIT (senior housing)
    "EQIX",   # Equinix -- Data center REIT
    "AMT",    # American Tower -- Cell tower REIT
    "SPG",    # Simon Property Group -- Retail mall REIT
    "DLR",    # Digital Realty -- Data center REIT
    "O",      # Realty Income -- Net lease retail REIT

    # --- Telecom (4) -- Wireless, Cable, Broadband ---
    "T",      # AT&T -- Telecom / wireless
    "VZ",     # Verizon -- Telecom / wireless
    "TMUS",   # T-Mobile -- Wireless telecom
    "CMCSA",  # Comcast -- Cable / broadband / media

    # --- ETF Benchmarks ---
    "SPY","QQQ","DIA",
    # Leveraged ETFs (2x bull)
    "SSO","QLD","DDM",
    # Leveraged ETFs (3x bull)
    "UPRO","TQQQ","UDOW",
    # Leveraged ETFs (2x inverse/bear)
    "SDS","QID","DXD",
    # Leveraged ETFs (3x inverse/bear)
    "SPXU","SQQQ","SDOW",
]

# Sector → symbol mapping (matches DEFAULT_SYMBOLS groupings above).
# Keys are lowercase; CLI does case-insensitive lookup via .lower().
SECTOR_MAP = {
    "technology":    ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","ORCL",
                      "CRM","AMD","PLTR","ADBE","CSCO","ACN","IBM","NOW","TXN",
                      "QCOM","KLAC","INTU","AMAT","LRCX","APH","ANET","PANW",
                      "MU","INTC","ADI","NFLX","UBER","APP","CRWD","WDC","SNDK",
                      "SNPS","CDNS","DELL","MSI","GLW","DASH","MRVL","NET",
                      "FTNT","SNOW","MPWR"],
    "financials":    ["JPM","V","MA","BAC","GS","MS","WFC","C","AXP","BLK",
                      "SPGI","SCHW","CB","MMC","FI","BX","KKR","CME","ICE",
                      "PYPL","COF","PGR","PNC","USB","IBKR","BK","MCO","APO",
                      "HOOD","TRV","TFC","AFL","AJG"],
    "healthcare":    ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","ISRG","SYK",
                      "VRTX","GILD","BSX","AMGN","PFE","REGN","CI","BMY","MCK",
                      "DHR","CVS","HCA","ELV","COR","BDX","ZTS"],
    "consumer_staples":       ["WMT","COST","PG","KO","PEP","PM","MCD","MDLZ",
                               "EL","MO","MNST","CL"],
    "consumer_discretionary": ["TSLA","HD","LOW","BKNG","TJX","SBUX","NKE","BRK-B",
                               "DIS","MAR","RCL","ORLY","ABNB","GM","HLT","CVNA",
                               "ROST","AZO","F","WBD"],
    "industrials":   ["CAT","GE","RTX","DE","UNP","HON","BA","ADP","LMT","HWM",
                      "NOC","UPS","GD","VRT","WM","FDX","MMM","ITW","EMR","PWR",
                      "CMI","CTAS","TDG","NSC","CSX","RSG","PCAR","LHX","PH","URI"],
    "energy":        ["XOM","CVX","COP","WMB","EPD","SLB","KMI","EOG","ET",
                      "PSX","VLO","BKR","MPLX","MPC"],
    "utilities":     ["NEE","SO","DUK","CEG","GEV","AEP","VST","SRE","D"],
    "materials":     ["LIN","APD","SCCO","NEM","FCX","SHW","ECL"],
    "real_estate":   ["PLD","WELL","EQIX","AMT","SPG","DLR","O"],
    "telecom":       ["T","VZ","TMUS","CMCSA"],
}


YF_INTRADAY_MAX_PERIOD={"1m":"7d","2m":"60d","5m":"60d","15m":"60d","30m":"60d",
    "60m":"730d","1h":"730d","90m":"60d","1d":"max","5d":"max","1wk":"max","1mo":"max"}

def download_data(symbols, cfg, cache_dir="data_cache", use_cache=False):
    """Download market data from Yahoo Finance.

    Args:
        symbols: List of ticker symbols
        cfg: Configuration object
        cache_dir: Directory for cached data (default: "data_cache")
        use_cache: Enable read/write cache (default: False, requires --cache flag)
    """
    if not HAS_YF: tprint("yfinance not installed","err");return {}

    import threading
    import pickle
    from pathlib import Path
    from datetime import datetime

    tfs=cfg.timeframes;data={};lock=threading.Lock()
    total=len(symbols)*len(tfs);done=[0];fail=[0];st=time.time()

    # Check cache only if explicitly enabled
    if use_cache:
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)
        cache_file = cache_path / f"market_data_{'_'.join(sorted(symbols)[:3])}_{len(symbols)}syms_{'_'.join(tfs)}.pkl"
        cache_meta_file = cache_path / f"{cache_file.stem}_meta.txt"

        if cache_file.exists() and cache_meta_file.exists():
            try:
                with open(cache_meta_file, 'r') as f:
                    cache_time = datetime.fromisoformat(f.read().strip())
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                tprint(f"Found cached data from {cache_time:%Y-%m-%d %H:%M:%S} ({age_hours:.1f}h ago)", "info")
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                tc=defaultdict(int);tb=defaultdict(int)
                for k,v in data.items(): t=k.rsplit("_",1)[-1];tc[t]+=1;tb[t]+=len(v)
                tprint(f"Loaded {len(data)} datasets from cache", "ok")
                for tf in tfs: tprint(f"  {tf:>4s}: {tc[tf]:>3d} syms {tb[tf]:>10,d} bars","info")
                return data
            except Exception as e:
                tprint(f"Cache read error: {e}, downloading fresh...", "warn")

    # Download data
    tprint(f"Downloading {len(symbols)} symbols x {len(tfs)} TFs = {total} jobs", "info")

    def _dl(sym,tf):
        try:
            per=YF_INTRADAY_MAX_PERIOD.get(tf,cfg.download_period)
            df=yf.Ticker(sym).history(period=per,interval=tf)
            if df is not None and not df.empty:
                if df.index.tz: df.index=df.index.tz_localize(None)
                df=df[["Open","High","Low","Close","Volume"]].dropna()
                mb=cfg.min_bars if tf=="1d" else max(60,cfg.min_bars//4)
                if len(df)>=mb:
                    with lock: data[f"{sym}_{tf}"]=df
        except:
            with lock: fail[0]+=1
        finally:
            with lock:
                done[0]+=1
                if done[0]%50==0 or done[0]==total: progress_bar(done[0],total,"Download",start_time=st)

    with ThreadPoolExecutor(max_workers=min(32,total)) as pool:
        futs=[pool.submit(_dl,s,t) for s in symbols for t in tfs]
        for f in as_completed(futs): pass

    # Save to cache only if explicitly enabled
    if use_cache and data:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            with open(cache_meta_file, 'w') as f:
                f.write(datetime.now().isoformat())
            tprint(f"Cached data to {cache_file}", "info")
        except Exception as e:
            tprint(f"Failed to cache: {e}", "warn")

    tc=defaultdict(int);tb=defaultdict(int)
    for k,v in data.items(): t=k.rsplit("_",1)[-1];tc[t]+=1;tb[t]+=len(v)
    tprint(f"{len(data)} datasets in {time.time()-st:.1f}s","ok")
    for tf in tfs: tprint(f"  {tf:>4s}: {tc[tf]:>3d} syms {tb[tf]:>10,d} bars","info")
    return data

def load_from_dir(dd,mb=200):
    data={};p=Path(dd)
    if not p.exists(): return data
    for fp in list(p.glob("*.csv"))+list(p.glob("*.parquet")):
        try:
            sym=fp.stem.upper()
            df=pd.read_csv(fp,index_col=0,parse_dates=True) if fp.suffix==".csv" else pd.read_parquet(fp)
            cm={}
            for col in df.columns:
                cl=col.lower().strip()
                if "close" in cl and "adj" not in cl: cm[col]="Close"
                elif cl=="open": cm[col]="Open"
                elif cl=="high": cm[col]="High"
                elif cl=="low": cm[col]="Low"
                elif cl in("volume","vol"): cm[col]="Volume"
            df=df.rename(columns=cm)
            if {"Open","High","Low","Close","Volume"}.issubset(df.columns):
                df=df[["Open","High","Low","Close","Volume"]].dropna()
                if len(df)>=mb: data[sym]=df
        except: pass
    tprint(f"Loaded {len(data)} symbols","ok"); return data

# ── Norgate Data loader (PIT-safe, reads from norgate_download_all.py output) ──
NORGATE_DIR = os.environ.get("NORGATE_ROOT", r"C:\ProgramData\NorgateData")
NORGATE_DATABASES = {
    "US_Equities":          {"has_volume": True,  "desc": "Active US stocks (PIT-adjusted)"},
    "US_Equities_Delisted": {"has_volume": True,  "desc": "Delisted US stocks (survivorship-bias-free)"},
    "US_Indices":           {"has_volume": True,  "desc": "US market indices (S&P 500, Nasdaq, etc.)"},
    "World_Indices":        {"has_volume": True,  "desc": "Global market indices"},
    "Continuous_Futures":   {"has_volume": True,  "desc": "Continuous futures contracts"},
    "Cash_Commodities":     {"has_volume": False, "desc": "Cash commodity indices"},
    "Forex_Spot":           {"has_volume": False, "desc": "Spot FX rates"},
    "Economic":             {"has_volume": False, "desc": "Economic indicators (CPI, GDP, etc.)"},
}

def load_from_norgate(norgate_dir=NORGATE_DIR, databases=None, symbols=None, mb=200):
    """Load data from local Norgate parquet files (downloaded by norgate_download_all.py).

    Args:
        norgate_dir: Root directory containing database subfolders.
        databases:   List of database folder names to load (default: all with volume).
        symbols:     Optional list of specific symbols to load (case-insensitive).
        mb:          Minimum bars required per symbol.
    Returns:
        dict of {symbol: DataFrame[OHLCV]} ready for prepare_datasets().
    """
    base = Path(norgate_dir)
    if not base.exists():
        tprint(f"Norgate dir not found: {norgate_dir}", "err")
        return {}

    # Default: only databases that have volume data (tradeable instruments)
    if databases is None:
        databases = [db for db, info in NORGATE_DATABASES.items() if info["has_volume"]]

    sym_filter = None
    if symbols:
        sym_filter = {s.upper() for s in symbols}

    data = {}
    st = time.time()
    total_files = 0
    total_loaded = 0
    total_skipped = 0
    skip_errors = []

    for db_name in databases:
        db_dir = base / db_name
        if not db_dir.exists():
            tprint(f"  {db_name}: folder not found, skipping", "warn")
            continue

        files = sorted(db_dir.glob("*.parquet"))
        total_files += len(files)
        db_loaded = 0

        for fp in files:
            sym = fp.stem.upper()
            if sym_filter and sym not in sym_filter:
                continue
            try:
                df = pd.read_parquet(fp)
                # Build OHLCV — fill Volume=0 for databases without it
                cols = {}
                for c in df.columns:
                    cl = c.lower().strip()
                    if cl == "open": cols[c] = "Open"
                    elif cl == "high": cols[c] = "High"
                    elif cl == "low": cols[c] = "Low"
                    elif cl == "close" and "unadj" not in cl.replace(" ", ""): cols[c] = "Close"
                    elif cl in ("volume", "vol"): cols[c] = "Volume"
                df = df.rename(columns=cols)

                need = {"Open", "High", "Low", "Close"}
                if not need.issubset(df.columns):
                    total_skipped += 1
                    continue

                if "Volume" in df.columns:
                    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                else:
                    df = df[["Open", "High", "Low", "Close"]].dropna()
                    df["Volume"] = 0.0

                if len(df) < mb:
                    total_skipped += 1
                    continue

                # Key format: SYM_1d (parsed by prepare_datasets to extract timeframe)
                key = f"{sym}_1d"
                if key in data:
                    # Collision: e.g. same ticker in US_Equities and US_Equities_Delisted
                    key = f"{sym}_{db_name}_1d"
                data[key] = df
                db_loaded += 1
            except Exception as e:
                total_skipped += 1
                skip_errors.append(f"{sym} ({type(e).__name__}: {e})")

        total_loaded += db_loaded
        tprint(f"  {db_name + ':':.<30s} {db_loaded:>5} symbols loaded ({len(files)} files)", "info")

    elapsed = time.time() - st

    # Report skip breakdown
    if skip_errors:
        tprint(f"  Skipped with errors ({len(skip_errors)}): "
               f"{', '.join(skip_errors[:5])}"
               f"{'...' if len(skip_errors) > 5 else ''}", "warn")

    # Verify all requested symbols were found
    if sym_filter:
        loaded_uids = {k.rsplit('_', 1)[0] for k in data}
        missing = sym_filter - loaded_uids
        if missing:
            tprint(f"  WARNING: {len(missing)}/{len(sym_filter)} requested symbols "
                   f"not loaded: {', '.join(sorted(missing)[:10])}"
                   f"{'...' if len(missing) > 10 else ''}", "warn")

    tprint(f"Norgate: {total_loaded} symbols from {len(databases)} databases "
           f"({total_skipped} skipped, {elapsed:.1f}s)", "ok")
    return data


def discover_midcap_symbols(
    norgate_dir: str = NORGATE_DIR,
    exclude_symbols: set = None,
    min_dollar_volume: float = 20_000_000,    # $20M/day — proxy for ~$1B+ market cap
    max_dollar_volume: float = 2_000_000_000, # $2B/day  — below mega large-cap territory
    min_avg_volume_shares: float = 300_000,   # 300K shares/day — float proxy (low-float gate)
    min_price: float = 10.0,                  # $10 minimum — eliminates sub-institutional stocks
    min_history_bars: int = 252,              # require at least 1 year of data
    lookback_bars: int = 63,                  # 3-month window for stable liquidity estimate
    database: str = "US_Equities",
    max_workers: int = 8,
) -> list:
    """
    Systematically discover mid-cap stock candidates from Norgate OHLCV data.

    No hand-curation. All criteria are objective and computed from OHLCV only
    (Norgate provides no market cap or float data — we use tradeable proxies):

      1. Avg daily dollar volume (close × volume, last 63 bars) in
         [min_dollar_volume, max_dollar_volume]:
           $20M/day lower bound  → proxy for ~$1B+ market cap at typical 1-2% daily turnover
           $2B/day upper bound   → below blue-chip large-caps not already in DEFAULT_SYMBOLS
      2. Avg daily volume in shares >= min_avg_volume_shares (default 300K/day):
           Float proxy — low-float stocks can hit $20M/day on a spike but have thin,
           erratic volume day-to-day. Consistent 300K shares/day implies adequate float
           for institutional-size entries/exits without excessive slippage.
      3. Close price >= min_price ($10)  → eliminates sub-institutional / penny stocks
      4. At least min_history_bars of data  → ensures reliable signal computation
      5. Not already in exclude_symbols  → avoids duplicating the large-cap universe

    Uses a thread pool for parallel I/O across potentially thousands of parquet files.
    Returns a sorted list of symbol strings (without _1d suffix).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    db_dir = Path(norgate_dir) / database
    if not db_dir.exists():
        tprint(f"  [Mid-Cap] Norgate database not found: {db_dir}", "warn")
        return []

    exclude = exclude_symbols or set()
    files = sorted(db_dir.glob("*.parquet"))

    def _check_file(fp):
        sym = fp.stem.upper()
        if sym in exclude:
            return None
        try:
            df = pd.read_parquet(fp)
            # Normalize column names (same logic as load_from_norgate)
            cols = {}
            for c in df.columns:
                cl = c.lower().strip()
                if cl == "close" and "unadj" not in cl.replace(" ", ""):
                    cols[c] = "Close"
                elif cl in ("volume", "vol"):
                    cols[c] = "Volume"
            df = df.rename(columns=cols)

            if "Close" not in df.columns or "Volume" not in df.columns:
                return None
            if len(df) < min_history_bars:
                return None

            tail = df.tail(lookback_bars)
            closes  = tail["Close"].values.astype(float)
            volumes = tail["Volume"].values.astype(float)

            if len(closes) == 0 or float(closes[-1]) < min_price:
                return None

            # Dollar volume gate: proxy for ~$1B+ market cap
            avg_dv = float(np.mean(closes * volumes))
            if avg_dv < min_dollar_volume or avg_dv > max_dollar_volume:
                return None

            # Share volume gate: float proxy — rules out low-float / erratic-volume stocks
            if min_avg_volume_shares > 0:
                avg_vol_shares = float(np.mean(volumes))
                if avg_vol_shares < min_avg_volume_shares:
                    return None

            return sym
        except Exception:
            return None

    candidates = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check_file, fp): fp for fp in files}
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                candidates.append(result)

    return sorted(candidates)


# ============================================================================
# CLI & MAIN
# ============================================================================
def parse_args():
    p=argparse.ArgumentParser(description="Alpha-Trade v3.0 GPU-Accelerated")
    p.add_argument("--data-dir",type=str,default="",
                   help="Load from a directory of CSV/Parquet files")
    p.add_argument("--norgate-dir",type=str,default="",
                   help="Override Norgate data path (default: D:\\Experiments\\norgate_data)")
    p.add_argument("--norgate-db",type=str,default="",
                   help="Comma-separated Norgate databases (e.g. US_Equities,US_Equities_Delisted)")
    p.add_argument("--yahoo",action="store_true",
                   help="Use Yahoo Finance instead of Norgate (survivorship-biased)")
    p.add_argument("--cache",action="store_true",
                   help="Enable Yahoo Finance data caching (disabled by default)")
    p.add_argument("--symbols",type=str,default="",
                   help="Comma-separated symbols to load (filters Norgate, or selects Yahoo symbols)")
    p.add_argument("--n-symbols",type=int,default=100,
                   help="Number of default symbols for --yahoo mode (default: 100)")
    p.add_argument("--timeframes",type=str,default="5m,15m,30m,1h,1d")
    p.add_argument("--iterations",type=int,default=3)        # v7.0 optimal: 3 iterations
    p.add_argument("--steps-per-iter",type=int,default=50_000)  # v7.0 optimal: 50k per iter (3 × 50k = 150k total)
    p.add_argument("--mcts-rollouts",type=int,default=256)  # Match Config design intent; schedule grows 64→512
    p.add_argument("--capital",type=float,default=100_000)
    p.add_argument("--hidden-dim",type=int,default=256)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--n-envs",type=int,default=16)
    p.add_argument("--no-amp",action="store_true")
    p.add_argument("--no-compile",action="store_true")
    p.add_argument("--no-dsr",action="store_true",help="Disable Differential Sharpe reward")
    p.add_argument("--backbone",type=str,default="conv_attn",choices=["conv_attn","transformer"],
                   help="Network backbone: conv_attn (default) or transformer")
    p.add_argument("--synthetic",action="store_true")
    p.add_argument("--n-synthetic",type=int,default=20)
    p.add_argument("--export-pine",action="store_true")
    p.add_argument("--verbose",type=int,default=1)
    return p.parse_args()

def main():
    args=parse_args()
    tfs=tuple(t.strip() for t in args.timeframes.split(",") if t.strip())
    vtf={"1m","2m","5m","15m","30m","60m","1h","90m","1d","5d","1wk","1mo"}
    for t in tfs:
        if t not in vtf: print(f"Invalid TF: {t}");return

    cfg=Config(starting_capital=args.capital,n_iterations=args.iterations,
        total_timesteps_per_iter=args.steps_per_iter,mcts_rollouts=args.mcts_rollouts,
        hidden_dim=args.hidden_dim,batch_size=args.batch_size,n_envs=args.n_envs,
        use_amp=not args.no_amp,use_compile=not args.no_compile,
        use_dsr=not args.no_dsr,backbone_type=args.backbone,
        timeframes=tfs,download_period="max",verbose=args.verbose)

    print_box("ALPHA-TRADE v3.0 - GPU-Accelerated AlphaGo Trading System",
              f"PPO+MCTS+Self-Play+AMP+VecEnv | {datetime.now():%Y-%m-%d %H:%M}")
    print_gpu_info()
    tprint(f"PyTorch {torch.__version__} | AMP:{'ON' if cfg.use_amp else 'OFF'} | "
           f"compile:{'ON' if cfg.use_compile else 'OFF'} | envs:{cfg.n_envs} | batch:{cfg.batch_size}","info")
    tprint(f"Backbone: {cfg.backbone_type} | DSR: {'ON' if cfg.use_dsr else 'OFF'} | "
           f"Deflated Sharpe threshold: p<{cfg.deflated_sharpe_threshold}","info")

    print_divider("DATA LOADING")
    # --- Data source: explicit flags > Norgate (default) ---
    # Priority: --yahoo > --data-dir > --synthetic > Norgate (default)
    # If none of the explicit flags are set, Norgate is used.
    # If Norgate data is missing, pipeline stops (no silent fallback).
    data = None
    data_source = "unknown"

    if args.yahoo:
        data_source = "Yahoo Finance (survivorship-biased)"
        if not HAS_YF:
            tprint("--yahoo requires yfinance: pip install yfinance", "err"); return
        syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols else DEFAULT_SYMBOLS[:args.n_symbols])
        tprint(f"Data source: {data_source}", "warn")
        tprint("  Backtest results will be upward-biased. Use Norgate PIT data for valid results.", "warn")
        data = download_data(syms, cfg, use_cache=args.cache)
    elif args.data_dir:
        data_source = f"Local directory: {args.data_dir}"
        tprint(f"Data source: {data_source}", "info")
        data = load_from_dir(args.data_dir, cfg.min_bars)
    elif args.synthetic:
        data_source = f"Synthetic ({args.n_synthetic} instruments)"
        tprint(f"Data source: {data_source}", "info")
        data = SyntheticMarketGenerator(cfg).generate_multiple(args.n_synthetic, 2000)
    else:
        norgate_dir = args.norgate_dir or NORGATE_DIR
        norgate_dbs = ([db.strip() for db in args.norgate_db.split(",") if db.strip()]
                       if args.norgate_db else None)
        norgate_syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                        if args.symbols else DEFAULT_SYMBOLS[:args.n_symbols])

        if not Path(norgate_dir).exists():
            tprint(f"Norgate data not found at: {norgate_dir}", "err")
            tprint("Run norgate_download_all.py first, or use --yahoo / --synthetic", "err")
            return

        db_list = norgate_dbs or [db for db, info in NORGATE_DATABASES.items() if info["has_volume"]]
        data_source = f"Norgate PIT data: {norgate_dir}"
        tprint(f"Data source: {data_source}", "ok")
        tprint(f"  Databases: {', '.join(db_list)}", "info")
        if norgate_syms:
            tprint(f"  Symbols filter: {', '.join(norgate_syms[:10])}"
                   f"{'...' if len(norgate_syms) > 10 else ''}", "info")
        data = load_from_norgate(norgate_dir, databases=norgate_dbs,
                                 symbols=norgate_syms, mb=cfg.min_bars)

    if not data:
        tprint(f"No data loaded from: {data_source}", "err")
        tprint("Check your data source or use --synthetic", "err"); return

    print_divider("FEATURE ENGINEERING")
    datasets=prepare_datasets(data,cfg)
    if not datasets: tprint("No datasets","err");return

    sys_=AlphaTradeSystem(datasets,cfg)
    champ=sys_.train(); res=sys_.final_eval()

    if args.export_pine:
        pc="//@version=6\nindicator(\"Alpha-Trade v3.0\",overlay=true)\n"
        fp=os.path.join(cfg.output_dir,"signal.pine")
        with open(fp,"w") as f: f.write(pc)
        tprint(f"Pine: {fp}","ok")

    rp=os.path.join(cfg.output_dir,"final_results.json")
    sr={k:v for k,v in res.items() if k!="per_sym"};sr["timeframes"]=list(tfs)
    with open(rp,"w") as f: json.dump(sr,f,indent=2,default=str)
    if HAS_CUDA: tprint(f"Peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB","gpu")
    print_box("COMPLETE",f"Score:{sys_.cs:+.2f} | TFs:{','.join(tfs)} | Model:{sys_.bp}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print(f"\n{C.YELLOW}  Terminated.{C.RESET}")
    except SystemExit: pass