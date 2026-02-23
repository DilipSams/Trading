"""
Trading Strategy Improvement System
=====================================
Multi-agent system targeting v3.0, v7.0, and v7.0 (no SMA).
Goal: achieve CAGR ≥ 26.4% (≥50% above SPY's 17.58%) on the OOS holdout.

Data source: Norgate PIT data at D:\\Experiments\\norgate_data
Pipeline:    D:\\Experiments\\Trading\\alphago_layering.py

Usage:
    python run_trading_analysis.py
"""

import asyncio
import os
import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
from config import (
    ORCHESTRATOR_MODEL,
    EXPERIMENTER_MODEL,
    MAX_TURNS,
    MAX_BUDGET_USD,
    RESEARCHER_TOOLS,
    CODER_TOOLS,
    REVIEWER_TOOLS,
    RESEARCHER_MODEL,
    CODER_MODEL,
    REVIEWER_MODEL,
    TRADING_DIR,
    NORGATE_DIR,
    SPY_CAGR_BENCHMARK,
    TARGET_MIN_CAGR,
    BASELINE_SCORES,
)

# ── How to run a backtest ─────────────────────────────────────────────────────
# A single run of alphago_layering.py with --version v7 produces ALL THREE
# comparison columns in its output table:
#   • Base v3.0       — Score column "Base v3.0"
#   • v7.0 (no SMA)  — nosma_results
#   • v7.0 (SMA)     — pipeline_results  ← primary target
#   • v8.0 reference — v8_results
#
# Fast experiment run (1 iter, 10k steps, skip ablation):
RUN_CMD_FAST = (
    f"cd /d {TRADING_DIR} && "
    "python alphago_layering.py --version v7 --n-symbols 50 "
    "--iterations 1 --steps-per-iter 10000 --skip-ablation"
)
# Full evaluation run (3 iters, 50k steps — matches research report):
RUN_CMD_FULL = (
    f"cd /d {TRADING_DIR} && "
    "python alphago_layering.py --version v7 --n-symbols 50 --skip-ablation"
)

# ── Agent Definitions ─────────────────────────────────────────────────────────

researcher = AgentDefinition(
    description=(
        "Research specialist. Use for reading and understanding the trading "
        "codebase, identifying which parameters drive v3.0 / v7.0 / v7.0-noSMA "
        "performance, and finding improvement opportunities."
    ),
    prompt=(
        "You are the Researcher for the ALPHA-TRADE improvement project.\n\n"
        "TARGET STRATEGIES:\n"
        "  • v3.0       — baseline, Score ~+1.993\n"
        "  • v7.0 noSMA — Score ~+14.502, Kill Switch fired\n"
        "  • v7.0       — Score ~+17.5, primary improvement target\n\n"
        "PERFORMANCE GOAL: All three strategies must achieve CAGR ≥ "
        f"{TARGET_MIN_CAGR:.1f}% (≥50% above SPY {SPY_CAGR_BENCHMARK}%).\n\n"
        "DATA SOURCE:\n"
        f"  Norgate PIT data: {NORGATE_DIR}\n"
        f"  Trading code:     {TRADING_DIR}\n\n"
        "YOUR JOB:\n"
        "1. Read ALL files in the trading directory thoroughly:\n"
        "   - alphago_architecture.py  (L1-L4 pipeline, ExecutionEngine, kill switches)\n"
        "   - alphago_layering.py      (entry point, v7/v8 mode, arg parser)\n"
        "   - alphago_trading_system.py (Config, DEFAULT_SYMBOLS, data loading)\n"
        "2. Document every tunable parameter in each strategy layer:\n"
        "   - L1: alpha signal weights (trend, MR, value, carry, seasonality, etc.)\n"
        "   - L2: meta-learner config (ridge alpha, coef cap, retrain freq)\n"
        "   - L3: position sizing (vol target, Kelly fraction, drawdown thresholds)\n"
        "   - L4: execution (kill switches, action_targets, no-trade threshold)\n"
        "   - SMA sizing block (base_pos values, trend_boost ceiling)\n"
        "3. Identify exactly WHY v3.0 and v7.0-noSMA underperform:\n"
        "   - v3.0: what is missing vs v7.0?\n"
        "   - v7.0-noSMA: why does the kill switch fire? what triggers it?\n"
        "4. Find 5-10 concrete parameter changes most likely to improve returns.\n"
        "5. Check the PIPELINE_V8_RESEARCH_REPORT.md for what already worked.\n\n"
        "Be extremely thorough. The coder and experimenter depend on your analysis."
    ),
    tools=RESEARCHER_TOOLS,
    model=RESEARCHER_MODEL,
)

reviewer = AgentDefinition(
    description=(
        "Code review specialist. Reviews trading code for bugs, look-ahead bias, "
        "data leakage, and logical errors that could inflate backtest results."
    ),
    prompt=(
        "You are the Reviewer for the ALPHA-TRADE improvement project.\n\n"
        "Focus specifically on these trading system files:\n"
        f"  {TRADING_DIR}\\alphago_architecture.py\n"
        f"  {TRADING_DIR}\\alphago_layering.py\n"
        f"  {TRADING_DIR}\\alphago_trading_system.py\n\n"
        "REVIEW CHECKLIST:\n"
        "1. Look-ahead bias — are future prices/returns used in signal generation?\n"
        "2. Data leakage — does train/val/holdout split use strictly disjoint data?\n"
        "3. Kill switch logic — do any kill switches fire prematurely on valid trades?\n"
        "4. SMA sizing correctness — do base_pos values snap to correct action_targets?\n"
        "5. action_targets discretization — is the grid complete? any walls?\n"
        "6. Off-by-one errors in rolling windows, SMA calculations\n"
        "7. Survivorship bias — does data loading filter delisted symbols correctly?\n"
        "8. Position sizing edge cases — zero vol, zero price, extreme leverage\n\n"
        "For each issue found, specify:\n"
        "  - File name and line number\n"
        "  - Severity: CRITICAL / WARNING / SUGGESTION\n"
        "  - Exact fix required\n"
    ),
    tools=REVIEWER_TOOLS,
    model=REVIEWER_MODEL,
)

coder = AgentDefinition(
    description=(
        "Code implementation specialist. Fixes bugs, applies parameter changes, "
        "and creates versioned experiment files for the trading system."
    ),
    prompt=(
        "You are the Coder for the ALPHA-TRADE improvement project.\n\n"
        "STRATEGIES TO IMPROVE:\n"
        "  • v3.0       (alphago_trading_system.py base logic)\n"
        "  • v7.0 noSMA (alphago_layering.py with use_sma=False)\n"
        "  • v7.0       (alphago_layering.py with use_sma=True)\n\n"
        "ARCHITECTURE OVERVIEW:\n"
        "  - alphago_architecture.py: AlphaConfig dataclass (all tunable params),\n"
        "    ExecutionEngine (action_targets, kill switches), SMA sizing block\n"
        "  - alphago_layering.py: CLI entry point, runs all three strategy modes\n"
        "  - alphago_trading_system.py: Config, data loading, DEFAULT_SYMBOLS\n\n"
        f"DATA: Always load from Norgate — {NORGATE_DIR}\n\n"
        "RULES:\n"
        "  - NEVER modify the original source files directly\n"
        "  - Create experiment copies: alphago_architecture_exp1.py, etc.\n"
        "  - OR use CLI args / config overrides when possible (preferred)\n"
        "  - Each change must be isolated and measurable\n"
        "  - Use type hints and comments explaining WHY each change is made\n"
        "  - The run command for a backtest is:\n"
        f"    {RUN_CMD_FAST}\n"
        "  - The comparison table in the output shows v3.0, v7.0-noSMA, v7.0, v8.0\n"
        "    columns — capture Score, P&L, Sharpe, Max DD, Kill Switch for each\n"
    ),
    tools=CODER_TOOLS,
    model=CODER_MODEL,
)

experimenter = AgentDefinition(
    description=(
        "Trading strategy experimenter. Designs and runs parameter experiments "
        "to improve v3.0, v7.0, and v7.0-noSMA returns by ≥50% above SPY."
    ),
    prompt=(
        "You are the Experimenter for the ALPHA-TRADE improvement project.\n\n"
        "MISSION: Push v3.0, v7.0-noSMA, and v7.0 CAGR to ≥"
        f"{TARGET_MIN_CAGR:.1f}% (currently ~17.58% for best v7.0).\n\n"
        "KNOWN BASELINES (OOS holdout Aug 2022 – Feb 2026):\n"
        "  v3.0:       Score +1.993\n"
        "  v7.0-noSMA: Score +14.502, P&L $464,977 (202 sym), Kill Switch fired\n"
        "  v7.0:       Score +17.5,   P&L ~$27K (15 sym baseline)\n"
        "  SPY:        CAGR 17.58%, P&L $7,729\n"
        "  v8.0 ref:   Score +68.128, P&L $156,310  ← ceiling to aim for\n\n"
        "RUN COMMANDS:\n"
        f"  Fast (1 iter):  {RUN_CMD_FAST}\n"
        f"  Full (3 iters): {RUN_CMD_FULL}\n\n"
        "EXPERIMENT AREAS (in priority order):\n"
        "1. Kill switch tuning — v7.0-noSMA fires prematurely:\n"
        "   kill_vol_breach_mult (try 5.0, 10.0), kill_dd_duration_bars (999),\n"
        "   kill_cost_spike_mult (999.0)\n"
        "2. action_targets grid — add 1.25x, 1.5x leverage steps:\n"
        "   (-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0,1.25,1.5)\n"
        "3. SMA base_pos values — ensure correct snap to action_targets:\n"
        "   golden_cross: 0.70→0.80, trend_boost ceiling: 0.30→0.60\n"
        "4. Kelly fraction tuning (0.3 vs 0.5 vs 0.7)\n"
        "5. target_annual_vol tuning (0.10, 0.15, 0.20, 0.25)\n"
        "6. n-symbols tuning (15, 30, 50, 100) — more symbols = more alpha\n"
        "7. Alpha signal weights — disable weak alphas (carry, seasonality)\n"
        "8. Drawdown scale thresholds (loosen to allow more trading in recovery)\n\n"
        "EXPERIMENT PROTOCOL:\n"
        "  1. Run FAST baseline first — record v3.0, v7.0-noSMA, v7.0 scores\n"
        "  2. Change ONE parameter at a time\n"
        "  3. Run FAST version, compare all three strategy columns\n"
        "  4. Promote best changes to FULL run for final validation\n"
        "  5. Log every experiment in trading_experiments_report.md:\n"
        "     | Exp | Change | v3.0 Score | v7.0-noSMA Score | v7.0 Score | Notes |\n\n"
        "SUCCESS CRITERIA:\n"
        "  All three strategies achieve Score > 26 AND P&L > SPY ($7,729 × 1.5)\n"
        "  Kill switches must NOT fire in the winning configuration\n"
    ),
    tools=CODER_TOOLS,
    model=EXPERIMENTER_MODEL,
)

# ── Master task prompt ────────────────────────────────────────────────────────

TASK = f"""\
Improve the ALPHA-TRADE trading system in {TRADING_DIR}.

OBJECTIVE
=========
Raise the returns of all three strategies to ≥ SPY +50%:
  Target Score: > 26   |   Target CAGR: ≥{TARGET_MIN_CAGR:.1f}% (SPY {SPY_CAGR_BENCHMARK}% × 1.5)

Target strategies (all three appear in a single pipeline run):
  1. v3.0       — base system,    Score ~+1.993   (CRITICAL: only 8 trades on top-100 — not investing)
  2. v7.0 noSMA — no SMA overlay, Score ~+14.502  (CRITICAL: kill switch fires, Score dropped to -4.39 on top-100)
  3. v7.0       — SMA overlay,    Score ~+17.5    (closest to target; reached +23.7 on top-100)

LATEST RUN RESULTS (top-100 curated stocks, OOS Aug 2022 – Feb 2026):
  v3.0:       Score +11.553, only 8 trades, Sharpe +8.4 — almost no capital deployed
  v7.0-noSMA: Score -4.390,  Max DD 37.3%, Breadth 24% — kill switch causing massive underperformance
  v7.0 (SMA): Score +23.708, Max DD 43.9%, Breadth 91% — best active strategy, close to target
  v8.0 (ref): Score +33.282, Profit Factor 6.70        — reference ceiling
  SPY:        CAGR 17.58%, P&L $7,729

KEY PROBLEMS IDENTIFIED:
  1. v3.0: Barely trades (8 trades in 3.5 years) — either no-trade threshold is too high,
     or position sizing collapses to zero. Need to find and fix what prevents trading.
  2. v7.0-noSMA: Kill switch fires prematurely. Max DD hits 37% then breadth collapses to 24%.
     kill_vol_breach_mult, kill_dd_duration_bars, or kill_cost_spike_mult need loosening.
  3. v7.0 (SMA): Score +23.7 needs to reach +26.4. Small improvements in kelly_fraction,
     target_annual_vol, or action_targets leverage steps may close the gap.

Data source: Norgate PIT data at {NORGATE_DIR}
Run command: cd /d {TRADING_DIR} && python alphago_layering.py --version v7 \\
             --n-symbols 50 --skip-ablation [--iterations N --steps-per-iter N]

A single run outputs all four columns: v3.0, v7.0-noSMA, v7.0, v8.0.
The v8.0 Iter 9 fixes ARE already in the code. Focus on v3.0, v7.0-noSMA, v7.0.

WORKFLOW
========

PHASE 1 — UNDERSTAND (researcher agent):
  - Read alphago_architecture.py, alphago_layering.py, alphago_trading_system.py
  - Read PIPELINE_V8_RESEARCH_REPORT.md (fix history for reference)
  - Find the EXACT code that causes v3.0 to produce only 8 trades:
    * What is the no_trade_threshold_pct? Is it too aggressive?
    * Does v3.0 use a different code path than v7.0?
    * What is the v3.0 action selection logic vs v7.0?
  - Find the EXACT kill switch parameters causing v7.0-noSMA to fire:
    * Which trigger fires first: vol_breach? cost_spike? dd_duration? max_loss?
    * What are the current threshold values?
    * What values would prevent premature firing while keeping catastrophic protection?
  - List 5-10 concrete parameter changes ranked by expected improvement.

PHASE 2 — REVIEW (reviewer agent):
  - Confirm: does v3.0 use the same ExecutionEngine / action_targets as v7.0?
  - Identify any logic in v3.0 path that forces inaction (no-trade zone, action=hold)
  - Review kill switch interaction: do multiple switches compound to cause early exit?
  - Check if kill_allow_auto_reenter=True is effective for recovery
  - Verify action_targets 9-element grid is active for all three strategies

PHASE 3 — FIX BUGS (coder agent):
  - Fix any CRITICAL bugs causing v3.0 inaction or v7.0-noSMA kill switch over-firing
  - Create versioned experiment files (never modify originals):
      alphago_architecture_exp1_v3_trading.py   (fix v3.0 no-trade issue)
      alphago_architecture_exp2_kill_loose.py   (loosen noSMA kill switches)
      alphago_architecture_exp3_leverage.py     (extend action_targets to 1.5×)
  - Verify the 9-element action_targets is active: (-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0,1.25,1.5)

PHASE 4 — EXPERIMENT (experimenter agent):
  - Run fast baseline first: record current v3.0, v7.0-noSMA, v7.0 scores
  - Run ≥6 targeted experiments:
    Exp 1: Fix v3.0 no-trade issue (lower no_trade_threshold_pct to 0.001 or 0.0)
    Exp 2: Loosen v7.0-noSMA kill switches (vol_breach_mult=15, dd_duration=999)
    Exp 3: Increase kelly_fraction 0.5→0.7 for v7.0
    Exp 4: Increase target_annual_vol 0.15→0.20
    Exp 5: Expand n-symbols 50→100 for more breadth
    Exp 6: Disable weak alphas (carry, seasonality) that may hurt IC
  - Log ALL results to {TRADING_DIR}\\trading_experiments_report.md
  - Promote top 2 winning combos to full 3-iteration runs

PHASE 5 — REPORT:
  - Final summary table: | Strategy | Baseline Score | Best Score | Delta | Kill SW |
  - Exact winning parameter set for each strategy
  - Flag any remaining gaps and what to try next
"""


async def main():
    print("\n" + "=" * 65)
    print("  ALPHA-TRADE IMPROVEMENT SYSTEM")
    print("=" * 65)
    print(f"  Strategies : v3.0  |  v7.0 (no SMA)  |  v7.0 (SMA)")
    print(f"  Data source: {NORGATE_DIR}")
    print(f"  Target CAGR: >={TARGET_MIN_CAGR:.1f}% (SPY {SPY_CAGR_BENCHMARK}% x 1.5)")
    print(f"  Budget     : ${MAX_BUDGET_USD}")
    print(f"  Max turns  : {MAX_TURNS}")
    print("=" * 65 + "\n")

    if not os.path.isdir(TRADING_DIR):
        print(f"ERROR: Trading directory not found: {TRADING_DIR}")
        return
    if not os.path.isdir(NORGATE_DIR):
        print(f"WARNING: Norgate directory not found: {NORGATE_DIR}")
        print("Ensure Norgate data is downloaded before running experiments.")

    agents = {
        "researcher":   researcher,
        "coder":        coder,
        "experimenter": experimenter,
        "reviewer":     reviewer,
    }

    options = ClaudeAgentOptions(
        model=ORCHESTRATOR_MODEL,
        allowed_tools=["Read", "Glob", "Grep", "Bash", "Write", "Edit", "Task"],
        agents=agents,
        max_turns=MAX_TURNS,
        max_budget_usd=MAX_BUDGET_USD,
        cwd=TRADING_DIR,
        permission_mode="acceptEdits",
    )

    async for message in query(prompt=TASK, options=options):
        if hasattr(message, "content") and isinstance(message.content, str):
            print(message.content, end="", flush=True)
        elif hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                if hasattr(block, "text"):
                    print(block.text, end="", flush=True)
                elif hasattr(block, "type") and block.type == "tool_use":
                    if block.name == "Task":
                        agent_type = block.input.get("subagent_type", "unknown")
                        desc = block.input.get("description", "")
                        print(f"\n  -> [{agent_type.upper()}]: {desc}")

    print("\n\n" + "=" * 65)
    print("  ANALYSIS COMPLETE")
    print(f"  Report: {TRADING_DIR}\\trading_experiments_report.md")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
