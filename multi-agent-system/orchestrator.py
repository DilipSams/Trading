"""
Orchestrator Agent
==================
Lead agent — breaks the improvement task into phases and delegates
to researcher, coder, reviewer, and experimenter.
"""
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
from config import (
    ORCHESTRATOR_MODEL,
    ORCHESTRATOR_TOOLS,
    MAX_TURNS,
    MAX_BUDGET_USD,
    TRADING_DIR,
    NORGATE_DIR,
    SPY_CAGR_BENCHMARK,
    TARGET_MIN_CAGR,
)
from agents.researcher import RESEARCHER_AGENT
from agents.coder import CODER_AGENT
from agents.reviewer import REVIEWER_AGENT
from experimenter import EXPERIMENTER_AGENT

ORCHESTRATOR_SYSTEM_PROMPT = f"""\
You are the Orchestrator — lead agent for the ALPHA-TRADE improvement project.

OBJECTIVE
=========
Improve three trading strategies to achieve CAGR ≥{TARGET_MIN_CAGR:.1f}%
(≥50% above SPY benchmark of {SPY_CAGR_BENCHMARK}%).

Strategies (all evaluated in one pipeline run):
  1. v3.0       — base system,     Score ~+1.993
  2. v7.0-noSMA — no SMA overlay,  Score ~+14.502 (kill switch fires)
  3. v7.0       — with SMA,        Score ~+17.5

Data: Norgate PIT files at {NORGATE_DIR}
Code: {TRADING_DIR}

AVAILABLE SPECIALIST AGENTS
============================
- researcher   : Reads/analyses code; finds tunable parameters; explains
                 WHY each strategy underperforms; ranks improvement ideas.
- reviewer     : Checks for look-ahead bias, kill switch bugs, discretization
                 walls, off-by-one errors, survivorship bias.
- coder        : Fixes bugs; creates versioned experiment files; applies
                 parameter changes.
- experimenter : Runs backtests (fast and full); logs results to
                 trading_experiments_report.md; identifies winning configs.

WORKFLOW
========
Phase 1 — UNDERSTAND  : researcher  → document all three strategies + params
Phase 2 — REVIEW      : reviewer    → find bugs suppressing returns
Phase 3 — FIX         : coder       → fix CRITICAL bugs, create exp files
Phase 4 — EXPERIMENT  : experimenter → ≥5 experiments, log all results
Phase 5 — REPORT      : you         → final summary + recommended param set

RULES
=====
- Always run experiments using Norgate data (not synthetic).
- Never modify original source files — always create versioned copies.
- A single alphago_layering.py run shows v3.0, v7.0-noSMA, v7.0, v8.0 columns.
- Use --skip-ablation and --iterations 1 --steps-per-iter 10000 for fast runs.
- The comparison table Score metric is the primary success criterion.
- SUCCESS = all three strategies Score > 26 AND kill switches do not fire.
"""


async def run(task, working_dir=TRADING_DIR):
    print(f"\n{'=' * 65}")
    print(f"  ALPHA-TRADE ORCHESTRATOR")
    print(f"{'=' * 65}")
    print(f"  Task:    {task[:80]}...")
    print(f"  CWD:     {working_dir}")
    print(f"  Target:  ≥{TARGET_MIN_CAGR:.1f}% CAGR across v3.0, v7.0-noSMA, v7.0")
    print(f"{'=' * 65}\n")

    agents = {
        "researcher":   RESEARCHER_AGENT,
        "coder":        CODER_AGENT,
        "reviewer":     REVIEWER_AGENT,
        "experimenter": EXPERIMENTER_AGENT,
    }

    options = ClaudeAgentOptions(
        model=ORCHESTRATOR_MODEL,
        allowed_tools=ORCHESTRATOR_TOOLS,
        agents=agents,
        max_turns=MAX_TURNS,
        max_budget_usd=MAX_BUDGET_USD,
        cwd=working_dir,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        permission_mode="acceptEdits",
    )

    async for message in query(prompt=task, options=options):
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

    print(f"\n\n{'=' * 65}")
    print(f"  ORCHESTRATOR — Task complete")
    print(f"  Report: {TRADING_DIR}\\trading_experiments_report.md")
    print(f"{'=' * 65}\n")
