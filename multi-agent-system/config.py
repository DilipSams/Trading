import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── Models ────────────────────────────────────────────────────────────────────
ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
RESEARCHER_MODEL   = "claude-sonnet-4-6"   # Upgraded: strategy analysis needs depth
CODER_MODEL        = "claude-sonnet-4-6"
REVIEWER_MODEL     = "claude-sonnet-4-6"
EXPERIMENTER_MODEL = "claude-sonnet-4-6"

# ── Execution limits ──────────────────────────────────────────────────────────
MAX_TURNS      = 50     # Complex multi-strategy task needs more turns
MAX_BUDGET_USD = 20.00  # Increased for multi-strategy research + experiments

# ── Paths ─────────────────────────────────────────────────────────────────────
TRADING_DIR  = r"D:\Experiments\Trading"
NORGATE_DIR  = r"D:\Experiments\norgate_data"

# ── Target strategies ─────────────────────────────────────────────────────────
# How to run each:
#   v3.0       — python alphago_layering.py --version v7 --n-symbols 50
#                (v3.0 results appear in the comparison table column "Base v3.0")
#   v7.0       — python alphago_layering.py --version v7 --n-symbols 50
#                (pipeline_results with use_sma=True)
#   v7.0 noSMA — same run; nosma_results column shows this
#   Add --skip-ablation to speed up runs during experimentation
TARGET_STRATEGIES = ["v3.0", "v7.0", "v7.0_nosma"]

# ── Performance target ────────────────────────────────────────────────────────
SPY_CAGR_BENCHMARK      = 17.58   # % — SPY CAGR over the OOS holdout period
TARGET_IMPROVEMENT_PCT  = 50.0    # % — must beat SPY by at least this much
TARGET_MIN_CAGR         = SPY_CAGR_BENCHMARK * (1 + TARGET_IMPROVEMENT_PCT / 100)
# → TARGET_MIN_CAGR = 26.37%

# ── Known baselines (from research report, OOS holdout Aug 2022–Feb 2026) ─────
BASELINE_SCORES = {
    "v3.0":        +1.993,
    "v7.0_nosma":  +14.502,
    "v7.0":        +17.5,
    "v8.0":        +68.128,  # Iter 9 best — reference ceiling
}

# ── Tool access per agent ─────────────────────────────────────────────────────
ORCHESTRATOR_TOOLS = ["Read", "Glob", "Grep", "Bash", "Write", "Edit", "Task"]
RESEARCHER_TOOLS   = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
CODER_TOOLS        = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
REVIEWER_TOOLS     = ["Read", "Glob", "Grep"]