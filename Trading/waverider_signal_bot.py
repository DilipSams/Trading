"""
WaveRider T5 MS BearVol2x — Automated Daily Signal Bot.

Generates portfolio signals after market close and sends them via Telegram.
Daily summary on hold days, full trade details on rebalance days.

Usage:
    python waverider_signal_bot.py                    # run signal + send
    python waverider_signal_bot.py --capital 250000   # custom capital
    python waverider_signal_bot.py --dry-run          # print only, don't send
    python waverider_signal_bot.py --setup            # first-time Telegram setup
"""
import argparse
import math
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
LOG_PATH = SCRIPT_DIR / "signal_log.txt"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_env() -> dict:
    """Load key=value pairs from .env file, fall back to os.environ."""
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def get_config(env: dict) -> dict:
    """Resolve config from .env + environment variables."""
    return {
        "bot_token": env.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", "")),
        "chat_id": env.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID", "")),
        "capital": float(env.get("CAPITAL", os.environ.get("CAPITAL", "100000"))),
    }


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(text: str, bot_token: str, chat_id: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    # Telegram message limit is 4096 chars; truncate if needed
    if len(text) > 4000:
        text = text[:3990] + "\n...(truncated)"
    for attempt in range(2):
        try:
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }, timeout=15)
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            # Markdown parse error — retry without formatting
            if attempt == 0 and "can't parse" in resp.text.lower():
                resp = requests.post(url, json={
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": True,
                }, timeout=15)
                return resp.status_code == 200 and resp.json().get("ok")
        except Exception:
            if attempt == 0:
                continue
    return False


def send_error_alert(msg: str, bot_token: str, chat_id: str):
    """Best-effort error alert via Telegram."""
    try:
        send_telegram(f"WaveRider ERROR:\n{msg}", bot_token, chat_id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

def get_current_price(prices: pd.DataFrame, uid: str) -> float:
    if uid in prices.columns:
        s = prices[uid].dropna()
        if len(s) > 0:
            return float(s.iloc[-1])
    return 0.0


def format_rebalance_message(signal, prices, uid_map, capital, result) -> str:
    """Full rebalance message with trade instructions."""
    from waverider import compute_nav_metrics

    n = len(signal.holdings_clean)
    effective = capital * signal.leverage
    per_stock = effective / n if n > 0 else 0
    m = compute_nav_metrics(result.nav_leveraged)

    lines = []
    lines.append("REBALANCE | " + signal.date.strftime("%Y-%m-%d"))
    lines.append("")

    # Trades
    if signal.buys or signal.sells:
        lines.append("TRADES:")
        for sym in signal.buys:
            uid = uid_map.get(sym, sym)
            price = get_current_price(prices, uid)
            shares = math.floor(per_stock / price) if price > 0 else 0
            alloc = shares * price
            lines.append(f"  BUY  {sym:<7s} {shares} sh @ ${price:.2f} = ${alloc:,.0f}")
        for sym in signal.sells:
            lines.append(f"  SELL {sym:<7s} (exit entire position)")
        lines.append("")

    # Portfolio
    weight_pct = 100.0 / n if n > 0 else 0
    lines.append(f"PORTFOLIO ({n} stocks, {weight_pct:.0f}% each):")
    total_alloc = 0
    for sym in signal.holdings_clean:
        uid = uid_map.get(sym, sym)
        price = get_current_price(prices, uid)
        shares = math.floor(per_stock / price) if price > 0 else 0
        alloc = shares * price
        total_alloc += alloc
        lines.append(f"  {sym:<7s} {shares:>5d} sh @ ${price:>8.2f}  ${alloc:>8,.0f}")
    cash = effective - total_alloc
    lines.append(f"  Cash remainder: ${cash:,.0f}")
    lines.append("")

    # Summary
    bear_str = "ON (SPY < SMA200)" if signal.bear_regime else "OFF"
    lines.append(f"Leverage: {signal.leverage:.2f}x | Exposure: ${effective:,.0f}")
    lines.append(f"Bear gate: {bear_str}")
    lines.append(f"Vol (21d): {signal.realized_vol:.2f} ann.")
    lines.append(f"CAGR: {m['cagr']*100:+.1f}% | Sharpe: {m['sharpe']:.2f}")

    return "\n".join(lines)


def format_daily_summary(signal, prices, uid_map, capital, result) -> str:
    """Brief daily status message."""
    from waverider import compute_nav_metrics

    effective = capital * signal.leverage
    m = compute_nav_metrics(result.nav_leveraged)

    # Estimate next rebalance: ~21 trading days from last signal date
    last_rebal = signal.date
    est_next = last_rebal + pd.tseries.offsets.BDay(21)

    # Compute daily P&L from NAV
    nav = result.nav_leveraged
    if len(nav) >= 2:
        daily_pnl = (nav.iloc[-1] / nav.iloc[-2] - 1) * 100
        daily_str = f"{daily_pnl:+.2f}%"
    else:
        daily_str = "n/a"

    holdings_str = ", ".join(signal.holdings_clean)
    bear_str = "ON" if signal.bear_regime else "OFF"

    lines = []
    lines.append("WaveRider Daily | " + datetime.now().strftime("%Y-%m-%d"))
    lines.append("")
    lines.append(f"Portfolio: {holdings_str}")
    lines.append(f"Leverage: {signal.leverage:.2f}x | Bear gate: {bear_str}")
    lines.append(f"Exposure: ${effective:,.0f} (${capital:,.0f} x {signal.leverage:.2f}x)")
    lines.append(f"Today: {daily_str}")
    lines.append(f"CAGR: {m['cagr']*100:+.1f}% | Sharpe: {m['sharpe']:.2f}")
    lines.append(f"Next rebalance: ~{est_next.strftime('%b %d')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_signal(signal, msg: str, sent_ok: bool):
    """Append to signal log file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"[{ts}] signal_date={signal.date.strftime('%Y-%m-%d')} "
        f"holdings={','.join(signal.holdings_clean)} "
        f"buys={','.join(signal.buys) or 'none'} "
        f"sells={','.join(signal.sells) or 'none'} "
        f"leverage={signal.leverage:.2f} "
        f"sent={'OK' if sent_ok else 'FAILED'}\n"
    )
    with open(LOG_PATH, "a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Setup wizard
# ---------------------------------------------------------------------------

def run_setup(capital: float):
    """Interactive first-time setup for Telegram bot."""
    print("=" * 60)
    print("  WaveRider Signal Bot — First-Time Setup")
    print("=" * 60)

    print("""
  Step 1: Create a Telegram bot
  ─────────────────────────────
  1. Open Telegram and search for @BotFather
  2. Send: /newbot
  3. Choose a name (e.g., "WaveRider Signals")
  4. Choose a username (e.g., "waverider_signal_bot")
  5. BotFather will give you a token like:
     1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
""")
    token = input("  Paste your bot token here: ").strip()
    if not token or ":" not in token:
        print("  Invalid token. Aborting.")
        return

    print("""
  Step 2: Get your Chat ID
  ────────────────────────
  1. Open Telegram and send ANY message to your new bot
  2. Press Enter here after you've sent a message...
""")
    input("  Press Enter after messaging the bot...")

    # Fetch chat ID from /getUpdates
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            timeout=10,
        )
        data = resp.json()
        if not data.get("ok") or not data.get("result"):
            print("  No messages found. Make sure you sent a message to the bot first.")
            print(f"  API response: {resp.text[:200]}")
            return
        chat_id = str(data["result"][-1]["message"]["chat"]["id"])
        chat_name = data["result"][-1]["message"]["chat"].get("first_name", "Unknown")
        print(f"  Found chat: {chat_name} (ID: {chat_id})")
    except Exception as e:
        print(f"  Failed to fetch chat ID: {e}")
        chat_id = input("  Enter your chat ID manually: ").strip()

    # Save .env
    env_content = (
        f"TELEGRAM_BOT_TOKEN={token}\n"
        f"TELEGRAM_CHAT_ID={chat_id}\n"
        f"CAPITAL={int(capital)}\n"
    )
    ENV_PATH.write_text(env_content)
    print(f"\n  Config saved to: {ENV_PATH}")

    # Send test message
    print("\n  Sending test message...", end="", flush=True)
    ok = send_telegram(
        "WaveRider Signal Bot connected!\nYou will receive daily portfolio signals here.",
        token, chat_id,
    )
    if ok:
        print(" sent! Check your Telegram.")
    else:
        print(" FAILED. Check your token and chat ID.")
        return

    # Offer to create scheduled task
    print("""
  Step 3: Schedule daily execution (optional)
  ────────────────────────────────────────────
  Create a Windows Task Scheduler task to run daily at 4:30 PM?
  This runs after market close so Norgate data is fresh.
""")
    create_task = input("  Create scheduled task? [y/N]: ").strip().lower()
    if create_task == "y":
        create_scheduled_task(capital)

    print("\n  Setup complete! Test with:")
    print(f"    python {Path(__file__).name} --dry-run")
    print()


def create_scheduled_task(capital: float):
    """Create a Windows Task Scheduler entry for daily 4:30 PM execution."""
    import subprocess

    script_path = Path(__file__).resolve()
    python_exe = sys.executable

    # Build command that runs in the script's directory
    cmd = f'cd /d "{SCRIPT_DIR}" && "{python_exe}" "{script_path}" --capital {int(capital)}'

    result = subprocess.run([
        "schtasks", "/create",
        "/tn", "WaveRider-DailySignal",
        "/tr", f'cmd /c "{cmd}"',
        "/sc", "weekly",
        "/d", "MON,TUE,WED,THU,FRI",
        "/st", "16:30",
        "/f",
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("  Scheduled task created: WaveRider-DailySignal")
        print("  Runs Mon-Fri at 4:30 PM local time.")
    else:
        print(f"  Failed to create task: {result.stderr.strip()}")
        print("  You may need to run this as Administrator.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WaveRider Daily Signal Bot")
    parser.add_argument("--capital", type=float, default=None,
                        help="Portfolio capital in dollars (overrides .env)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print message to terminal without sending")
    parser.add_argument("--setup", action="store_true",
                        help="Interactive first-time Telegram bot setup")
    args = parser.parse_args()

    # Load config
    env = load_env()
    cfg = get_config(env)
    capital = args.capital or cfg["capital"]
    bot_token = cfg["bot_token"]
    chat_id = cfg["chat_id"]

    # Setup mode
    if args.setup:
        run_setup(capital)
        return

    # Validate config
    if not args.dry_run and (not bot_token or not chat_id):
        print("ERROR: Telegram not configured. Run with --setup first.")
        sys.exit(1)

    try:
        # 1. Rebuild universe cache if Norgate data changed
        sys.path.insert(0, str(SCRIPT_DIR))
        from universe_builder import build_universe_cache
        print("  Checking universe cache...", end="", flush=True)
        build_universe_cache(force_rebuild=False)
        print(" done.")

        # 2. Generate signal
        from waverider import (
            WaveRiderStrategy, WaveRiderConfig, load_universe, load_spy,
            clean_uid, compute_nav_metrics,
        )
        print("  Loading data...", end="", flush=True)
        prices, rankings = load_universe()
        spy = load_spy()
        print(" done.")

        cfg_wr = WaveRiderConfig()
        strategy = WaveRiderStrategy(cfg_wr)

        print("  Computing signal...", end="", flush=True)
        signal = strategy.current_portfolio(prices, spy, rankings)
        result = strategy.backtest(prices, spy, rankings)
        print(" done.")

        # Build UID lookup
        uid_map = {}
        for uid in signal.holdings:
            uid_map[clean_uid(uid)] = uid

        # 3. Determine if today is a rebalance day with trades
        today = pd.Timestamp(datetime.now().date())
        is_rebalance = (signal.date.date() == today.date()) and (signal.buys or signal.sells)

        # 4. Format message
        if is_rebalance:
            msg = format_rebalance_message(signal, prices, uid_map, capital, result)
        else:
            msg = format_daily_summary(signal, prices, uid_map, capital, result)

        # 5. Send or print
        if args.dry_run:
            print("\n" + "=" * 50)
            print("  DRY RUN — message that would be sent:")
            print("=" * 50)
            print(msg)
            print("=" * 50)
            sent_ok = True
        else:
            print("  Sending Telegram...", end="", flush=True)
            sent_ok = send_telegram(msg, bot_token, chat_id)
            print(" sent!" if sent_ok else " FAILED!")

        # 6. Log
        log_signal(signal, msg, sent_ok)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n  ERROR: {e}")
        print(tb)

        # Log the error
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_PATH, "a") as f:
            f.write(f"[{ts}] ERROR: {e}\n{tb}\n")

        # Try to send error alert
        if not args.dry_run and bot_token and chat_id:
            send_error_alert(str(e), bot_token, chat_id)

        sys.exit(1)


if __name__ == "__main__":
    main()
