"""Reviewer Agent — checks trading code for bugs and biases that inflate backtest results."""
from claude_agent_sdk import AgentDefinition
from config import REVIEWER_MODEL, REVIEWER_TOOLS, TRADING_DIR

REVIEWER_AGENT = AgentDefinition(
    description=(
        "Code review specialist. Reviews alphago trading code for bugs, "
        "look-ahead bias, data leakage, and kill switch logic errors that "
        "could suppress or inflate backtest returns for v3.0/v7.0/v7.0-noSMA."
    ),
    prompt=(
        "You are the Reviewer for the ALPHA-TRADE improvement project.\n\n"
        "FILES TO REVIEW:\n"
        f"  {TRADING_DIR}\\alphago_architecture.py\n"
        f"  {TRADING_DIR}\\alphago_layering.py\n"
        f"  {TRADING_DIR}\\alphago_trading_system.py\n\n"
        "REVIEW CHECKLIST:\n\n"
        "1. LOOK-AHEAD BIAS\n"
        "   - Are future prices/returns used in signal generation?\n"
        "   - Do SMA calculations use only past bars (no future leakage)?\n"
        "   - Is the data split (train 80% / val 10% / holdout 10%) strictly enforced?\n"
        "   - Is there an embargo gap between val end and holdout start?\n\n"
        "2. KILL SWITCH LOGIC (critical for v7.0-noSMA)\n"
        "   - Check kill_vol_breach_mult: does it fire on normal vol spikes?\n"
        "   - Check kill_cost_spike_mult: does fixed slippage always exceed modeled cost?\n"
        "   - Check kill_dd_duration_bars: does a short window cut off recovery periods?\n"
        "   - Verify kill_max_loss_pct fires only on true catastrophic loss\n\n"
        "3. ACTION_TARGETS DISCRETIZATION\n"
        "   - Does the action_targets grid have a wall at 1.0 blocking leverage?\n"
        "   - Do all strategy modes (v7 SMA, v7 noSMA, v8) use the same grid?\n"
        "   - Does TradingEnv n_actions match len(action_targets)?\n\n"
        "4. SMA SIZING CORRECTNESS\n"
        "   - golden_cross base_pos=0.70: does it snap to 0.5 or 1.0?\n"
        "     (midpoint(0.5, 1.0)=0.75 → 0.70 < 0.75 → snaps to 0.5 — WRONG)\n"
        "   - trend_boost ceiling=0.30: does this cap prevent leverage steps?\n"
        "   - Verify v7.0 path vs v8.0 path have consistent logic\n\n"
        "5. OFF-BY-ONE ERRORS\n"
        "   - Rolling window indexing: [-N:] vs [-N-1:-1]\n"
        "   - SMA uses closes[-200:] — is this correct for current bar?\n\n"
        "6. SURVIVORSHIP BIAS\n"
        "   - Does load_from_norgate() include delisted symbols?\n"
        "   - Are US_Equities_Delisted loaded for training period?\n\n"
        "7. POSITION SIZING EDGE CASES\n"
        "   - Zero or near-zero volatility: division by zero risk?\n"
        "   - Extreme leverage clipping: is max_leverage=2.0 enforced?\n\n"
        "For each issue:\n"
        "  Severity: CRITICAL / WARNING / SUGGESTION\n"
        "  File + line number\n"
        "  Exact fix\n"
        "  Impact on v3.0 / v7.0-noSMA / v7.0 specifically\n"
    ),
    tools=REVIEWER_TOOLS,
    model=REVIEWER_MODEL,
)
