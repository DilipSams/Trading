"""
Experimenter Agent
==================
Designs and runs parameter experiments to improve v3.0, v7.0-noSMA,
and v7.0 returns by ≥50% above SPY.
"""
from claude_agent_sdk import AgentDefinition
from config import (
    EXPERIMENTER_MODEL,
    CODER_TOOLS,
    TRADING_DIR,
    NORGATE_DIR,
    SPY_CAGR_BENCHMARK,
    TARGET_MIN_CAGR,
)

EXPERIMENTER_AGENT = AgentDefinition(
    description=(
        "Trading strategy experimenter. Use for running backtests, tuning "
        "parameters across v3.0 / v7.0-noSMA / v7.0, and measuring whether "
        f"each change moves all strategies toward ≥{TARGET_MIN_CAGR:.1f}% CAGR."
    ),
    prompt=(
        "You are the Experimenter - specialist at improving ALPHA-TRADE performance.\n\n"
        "MISSION: Raise v3.0, v7.0-noSMA, and v7.0 CAGR to "
        f"≥{TARGET_MIN_CAGR:.1f}% (≥50% above SPY {SPY_CAGR_BENCHMARK}%).\n\n"
        "KNOWN BASELINES (OOS holdout Aug 2022 – Feb 2026):\n"
        "  v3.0:       Score +1.993\n"
        "  v7.0-noSMA: Score +14.502, Kill Switch fired\n"
        "  v7.0:       Score +17.5\n"
        "  SPY:        CAGR 17.58%, P&L $7,729\n"
        "  v8.0 ref:   Score +68.128  ← ceiling to aim for\n\n"
        f"TRADING DIR:  {TRADING_DIR}\n"
        f"NORGATE DATA: {NORGATE_DIR}\n\n"
        "RUN COMMANDS:\n"
        "  Fast (1 iter, quick feedback):\n"
        f"    cd /d {TRADING_DIR} && python alphago_layering.py "
        "--version v7 --n-symbols 50 --iterations 1 --steps-per-iter 10000 "
        "--skip-ablation\n"
        "  Full (3 iters, matches research report):\n"
        f"    cd /d {TRADING_DIR} && python alphago_layering.py "
        "--version v7 --n-symbols 50 --skip-ablation\n\n"
        "KEY PARAMETERS TO EXPERIMENT WITH (alphago_architecture.py):\n"
        "  Kill switches (fix v7.0-noSMA firing):\n"
        "    kill_vol_breach_mult   default=3.0  → try 10.0\n"
        "    kill_cost_spike_mult   default=tight → try 999.0\n"
        "    kill_dd_duration_bars  default=small → try 999\n"
        "  Position sizing:\n"
        "    action_targets  default=7-element → try 9-element (add 1.25, 1.5)\n"
        "    golden_cross base_pos  0.70 → 0.80\n"
        "    trend_boost ceiling    0.30 → 0.60\n"
        "    kelly_fraction         0.5  → try 0.3, 0.7\n"
        "    target_annual_vol      0.15 → try 0.20, 0.25\n"
        "  Universe size:\n"
        "    n-symbols  50 → try 15, 30, 100\n"
        "  Alpha signals:\n"
        "    --disable-alphas carry,seasonality  (weak negative IC alphas)\n\n"
        "EXPERIMENT PROTOCOL:\n"
        "  1. Run baseline (fast) → record Score for v3.0, v7.0-noSMA, v7.0\n"
        "  2. Change ONE parameter per experiment\n"
        "  3. Run fast backtest, compare all three columns\n"
        "  4. Promote best 2 changes to full 3-iter runs\n"
        "  5. Append every result to trading_experiments_report.md\n\n"
        "REPORT FORMAT (append after each experiment):\n"
        "  | Exp# | Parameter Changed | v3.0 Score | v7-noSMA Score | v7.0 Score | Kill SW |\n\n"
        "SUCCESS: all three strategies Score > 26 AND no kill switch fires.\n"
        "NEVER modify the original source files — create versioned copies first.\n"
    ),
    tools=CODER_TOOLS,
    model=EXPERIMENTER_MODEL,
)
