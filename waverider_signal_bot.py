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
import json
import math
import os
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress SSL verification warnings (verify=False used for EC2 compatibility)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
LOG_PATH = SCRIPT_DIR / "signal_log.txt"
LEDGER_PATH = SCRIPT_DIR / "portfolio_ledger.json"

# ---------------------------------------------------------------------------
# Portfolio Ledger — persistent trade state
# ---------------------------------------------------------------------------

class PortfolioLedger:
    """Persistent portfolio state tracking entry prices, shares, and closed trades.

    Positions are locked at the moment of entry (EOM rebalance day) and never
    recomputed from backtest data. This gives stable P&L regardless of daily
    Norgate price adjustments.

    JSON structure:
    {
      "version": 1,
      "capital": 100000,
      "last_rebalance_date": "2026-02-27",
      "positions": {
        "NEM": {"entry_date": "2026-01-30", "entry_price": 112.34, "shares": 160}
      },
      "closed_trades": [
        {"symbol": "GLDM", "entry_date": "...", "entry_price": 89.50,
         "exit_date": "...", "exit_price": 104.14, "shares": 95,
         "realized_pnl": 1389.30, "pnl_pct": 16.4}
      ]
    }
    """

    def __init__(self, capital: float, positions: dict, closed_trades: list,
                 last_rebalance_date: str):
        self.capital = capital
        self.positions = positions          # {sym: {entry_date, entry_price, shares}}
        self.closed_trades = closed_trades  # list of closed trade dicts
        self.last_rebalance_date = last_rebalance_date

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    @classmethod
    def load(cls):
        """Load from file. Returns None if file doesn't exist or is corrupt."""
        if not LEDGER_PATH.exists():
            return None
        try:
            data = json.loads(LEDGER_PATH.read_text())
            return cls(
                capital=float(data.get("capital", 100000)),
                positions=data.get("positions", {}),
                closed_trades=data.get("closed_trades", []),
                last_rebalance_date=data.get("last_rebalance_date", ""),
            )
        except Exception as e:
            print(f"  Warning: could not load ledger ({e}). Will re-bootstrap.")
            return None

    def save(self):
        """Write ledger to JSON."""
        data = {
            "version": 1,
            "capital": self.capital,
            "last_rebalance_date": self.last_rebalance_date,
            "positions": self.positions,
            "closed_trades": self.closed_trades,
        }
        LEDGER_PATH.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Bootstrap — initialise from backtest on first run
    # ------------------------------------------------------------------

    @classmethod
    def bootstrap(cls, signal, result, prices, uid_map, capital: float):
        """Initialise ledger from backtest state when no ledger file exists.

        Entry dates and prices come from Norgate closing prices at the detected
        entry date (continuous-streak walk). Shares are floor(per_stock / price)
        using the leverage that was active on the entry date (from result.leverage_series),
        not today's leverage — this gives accurate "as-traded" share counts.
        """
        n = len(signal.holdings_clean)

        positions = {}
        for sym in signal.holdings_clean:
            uid = uid_map.get(sym, sym)
            entry_date = find_entry_date(uid, result.holdings_log, result.rebalance_dates)
            entry_price = get_price_on_date(prices, uid, entry_date) if entry_date else 0.0

            # Use leverage at entry date (not current) for accurate share count
            if entry_date is not None and hasattr(result, "leverage_series"):
                lev_at_entry = float(result.leverage_series.asof(entry_date))
            else:
                lev_at_entry = signal.leverage
            per_stock = capital * lev_at_entry / n if n > 0 else 0

            shares = math.floor(per_stock / entry_price) if entry_price > 0 else 0
            positions[sym] = {
                "entry_date": entry_date.strftime("%Y-%m-%d") if entry_date else "",
                "entry_price": round(float(entry_price), 4),
                "shares": int(shares),
            }

        return cls(
            capital=capital,
            positions=positions,
            closed_trades=[],
            last_rebalance_date=signal.date.strftime("%Y-%m-%d"),
        )

    # ------------------------------------------------------------------
    # Rebalance update — called once on EOM day
    # ------------------------------------------------------------------

    def update_rebalance(self, signal, prices, uid_map):
        """Apply EOM rebalance: close sells, open buys, leave holds untouched."""
        n = len(signal.holdings_clean)
        effective = self.capital * signal.leverage
        per_stock = effective / n if n > 0 else 0
        today_str = signal.date.strftime("%Y-%m-%d")

        # --- Close sold positions ---
        for sym in signal.sells:
            if sym not in self.positions:
                continue
            pos = self.positions[sym]
            uid = uid_map.get(sym, sym)
            exit_price = get_current_price(prices, uid)
            entry_price = float(pos["entry_price"])
            shares = int(pos["shares"])
            realized_pnl = round(shares * (exit_price - entry_price), 2) if exit_price > 0 else 0.0
            pnl_pct = round((exit_price / entry_price - 1) * 100, 2) if entry_price > 0 and exit_price > 0 else 0.0
            self.closed_trades.append({
                "symbol": sym,
                "entry_date": pos["entry_date"],
                "entry_price": entry_price,
                "exit_date": today_str,
                "exit_price": round(float(exit_price), 4),
                "shares": shares,
                "realized_pnl": realized_pnl,
                "pnl_pct": pnl_pct,
            })
            del self.positions[sym]

        # --- Open new positions ---
        for sym in signal.buys:
            uid = uid_map.get(sym, sym)
            entry_price = get_current_price(prices, uid)
            shares = math.floor(per_stock / entry_price) if entry_price > 0 else 0
            self.positions[sym] = {
                "entry_date": today_str,
                "entry_price": round(float(entry_price), 4),
                "shares": int(shares),
            }

        self.last_rebalance_date = today_str

    # ------------------------------------------------------------------
    # Entry info accessor — drop-in replacement for build_entry_info()
    # ------------------------------------------------------------------

    def get_entry_info(self) -> dict:
        """Return {sym: (pd.Timestamp, entry_price, actual_shares)} for all positions.

        The 3-tuple is used by format_portfolio_table() to show locked share counts
        and stable P&L.
        """
        info = {}
        for sym, pos in self.positions.items():
            try:
                dt = pd.Timestamp(pos["entry_date"])
            except Exception:
                dt = None
            info[sym] = (dt, float(pos["entry_price"]), int(pos.get("shares", 0)))
        return info


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
    """Send a message via Telegram Bot API (HTML mode). Returns True on success."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    if len(text) > 4000:
        text = text[:3990] + "\n...(truncated)"
    for attempt in range(2):
        try:
            resp = requests.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }, timeout=15, verify=False)
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            # HTML parse error — retry without formatting
            if attempt == 0 and "can't parse" in resp.text.lower():
                import re
                plain = re.sub(r"<[^>]+>", "", text)
                resp = requests.post(url, json={
                    "chat_id": chat_id,
                    "text": plain,
                    "disable_web_page_preview": True,
                }, timeout=15, verify=False)
                return resp.status_code == 200 and resp.json().get("ok")
        except Exception:
            if attempt == 0:
                continue
    return False


def send_error_alert(msg: str, bot_token: str, chat_id: str):
    """Best-effort error alert via Telegram."""
    try:
        safe_msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        send_telegram(f"\u26A0\uFE0F <b>WaveRider ERROR</b>\n<pre>{safe_msg}</pre>", bot_token, chat_id)
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


def get_price_on_date(prices: pd.DataFrame, uid: str, date) -> float:
    if uid in prices.columns:
        val = prices[uid].asof(date)
        if pd.notna(val):
            return float(val)
    return 0.0


def find_entry_date(uid: str, holdings_log: dict, rebalance_dates: list):
    """Walk backwards to find when this UID first entered (continuous streak)."""
    entry = None
    for rd in reversed(rebalance_dates):
        if uid in holdings_log.get(rd, []):
            entry = rd
        else:
            break
    return entry


def build_entry_info(signal, result, prices, uid_map):
    """Build entry date + entry price for each current holding."""
    info = {}  # clean_sym -> (entry_date, entry_price)
    for uid in signal.holdings:
        sym = signal.holdings_clean[signal.holdings.index(uid)] if uid in signal.holdings else uid
        # Use clean_uid to get base symbol
        from waverider import clean_uid
        sym = clean_uid(uid)
        entry_date = find_entry_date(uid, result.holdings_log, result.rebalance_dates)
        entry_price = get_price_on_date(prices, uid, entry_date) if entry_date else 0.0
        info[sym] = (entry_date, entry_price)
    return info


def format_portfolio_table(signal, prices, uid_map, capital, entry_info) -> str:
    """Format a detailed portfolio table with entry dates and P&L."""
    n = len(signal.holdings_clean)
    effective = capital * signal.leverage
    per_stock = effective / n if n > 0 else 0

    # Build rows with P&L for sorting
    rows = []
    total_value = 0
    total_pnl = 0

    for sym in signal.holdings_clean:
        uid = uid_map.get(sym, sym)
        cur_price = get_current_price(prices, uid)

        # Use actual share count from ledger (3-tuple); fall back to computed
        raw = entry_info.get(sym, (None, 0, 0))
        ed, ep = raw[0], raw[1]
        actual_shares = raw[2] if len(raw) > 2 else 0
        shares = actual_shares if actual_shares > 0 else (math.floor(per_stock / cur_price) if cur_price > 0 else 0)

        value = shares * cur_price
        total_value += value

        ed_str = ed.strftime("%b %d") if ed else "n/a"

        if ep > 0:
            pnl_pct = (cur_price / ep - 1) * 100
            pnl_dollar = shares * (cur_price - ep)
            total_pnl += pnl_dollar
        else:
            pnl_pct = 0.0
            pnl_dollar = 0.0

        rows.append((sym, cur_price, shares, value, ed_str, ep, pnl_pct, pnl_dollar))

    # Sort by P&L% descending (best first)
    rows.sort(key=lambda r: r[6], reverse=True)

    lines = []
    lines.append(f"\U0001F4BC <b>PORTFOLIO</b> ({n} stocks, {100/n:.0f}% each, ${per_stock:,.0f}/pos)")
    lines.append("")
    lines.append("<pre>")

    for sym, cur_price, shares, value, ed_str, ep, pnl_pct, pnl_dollar in rows:
        icon = "\u2705" if pnl_pct >= 0 else "\u274C"
        lines.append(
            f"{icon} {sym:<6s} ${cur_price:>7.2f} x{shares:>4d}sh "
            f"${value:>7,.0f}  {ed_str:>6s}  {pnl_pct:>+6.1f}% ${pnl_dollar:>+7,.0f}"
        )

    cash = effective - total_value
    pnl_icon = "\U0001F4B0" if total_pnl >= 0 else "\U0001F534"
    lines.append(f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
    lines.append(f"\U0001F4B5 CASH {' ':>24s}${cash:>7,.0f}")
    lines.append(f"{pnl_icon} TOTAL{' ':>24s}${effective:>7,.0f}  P&L: ${total_pnl:>+7,.0f}")
    lines.append("</pre>")

    return "\n".join(lines)


def format_rebalance_message(signal, prices, uid_map, capital, result, ledger) -> str:
    """Full rebalance message with trade instructions."""
    from waverider import compute_nav_metrics

    entry_info = ledger.get_entry_info()
    n = len(signal.holdings_clean)
    effective = capital * signal.leverage
    per_stock = effective / n if n > 0 else 0
    m = compute_nav_metrics(result.nav_leveraged)

    lines = []
    lines.append(f"\U0001F514 <b>REBALANCE</b> | {signal.date.strftime('%Y-%m-%d')}")
    lines.append("")

    # Trades
    if signal.buys or signal.sells:
        lines.append("\U0001F4CB <b>TRADES:</b>")
        for sym in signal.buys:
            uid = uid_map.get(sym, sym)
            price = get_current_price(prices, uid)
            shares = math.floor(per_stock / price) if price > 0 else 0
            alloc = shares * price
            lines.append(f"  \U0001F7E2 BUY  <b>{sym}</b>  {shares} sh @ ${price:.2f} = ${alloc:,.0f}")
        for sym in signal.sells:
            lines.append(f"  \U0001F534 SELL <b>{sym}</b>  (exit entire position)")
        lines.append("")

    # Realized P&L for positions closed today
    today_str = signal.date.strftime("%Y-%m-%d")
    closed_today = [t for t in ledger.closed_trades if t.get("exit_date") == today_str]
    if closed_today:
        lines.append("\U0001F4B0 <b>CLOSED TODAY:</b>")
        for t in closed_today:
            icon = "\u2705" if t["pnl_pct"] >= 0 else "\u274C"
            lines.append(
                f"  {icon} {t['symbol']:<6s} "
                f"${t['entry_price']:.2f} \u2192 ${t['exit_price']:.2f}  "
                f"{t['pnl_pct']:+.1f}%  ${t['realized_pnl']:+,.0f}"
            )
        lines.append("")

    # Detailed portfolio table (new state after rebalance)
    lines.append(format_portfolio_table(signal, prices, uid_map, capital, entry_info))
    lines.append("")

    # Summary
    bear_icon = "\U0001F6A8" if signal.bear_regime else "\U0001F6E1"
    bear_str = "ON (SPY &lt; SMA200)" if signal.bear_regime else "OFF"
    lines.append(f"\u26A1 Leverage: <b>{signal.leverage:.2f}x</b> | Exposure: ${effective:,.0f}")
    lines.append(f"{bear_icon} Bear gate: {bear_str}")
    lines.append(f"\U0001F4CA Vol (21d): {signal.realized_vol:.2f} ann.")
    lines.append(f"\U0001F3AF CAGR: <b>{m['cagr']*100:+.1f}%</b> | Sharpe: {m['sharpe']:.2f}")

    return "\n".join(lines)


def format_daily_summary(signal, prices, uid_map, capital, result, ledger) -> str:
    """Daily status message with portfolio details."""
    from waverider import compute_nav_metrics

    entry_info = ledger.get_entry_info()
    m = compute_nav_metrics(result.nav_leveraged)

    # Estimate next rebalance: end of NEXT calendar month.
    # signal.date is the last trading day of its month (EOM), so +MonthEnd(1) gives
    # the last calendar day of the SAME month. We want the next month's end instead.
    last_rebal = signal.date
    next_month_first = (last_rebal + pd.DateOffset(months=1)).replace(day=1)
    est_next = next_month_first + pd.offsets.MonthEnd(0)

    # Compute daily P&L from NAV
    nav = result.nav_leveraged
    if len(nav) >= 2:
        daily_pnl = (nav.iloc[-1] / nav.iloc[-2] - 1) * 100
        daily_icon = "\U0001F7E2" if daily_pnl >= 0 else "\U0001F534"
        daily_str = f"{daily_pnl:+.2f}%"
    else:
        daily_icon = "\u2796"
        daily_str = "n/a"

    bear_icon = "\U0001F6A8" if signal.bear_regime else "\U0001F6E1"
    bear_str = "ON" if signal.bear_regime else "OFF"
    effective = capital * signal.leverage

    lines = []
    lines.append(f"\U0001F4CA <b>WaveRider Daily</b> | {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")

    # Detailed portfolio table
    lines.append(format_portfolio_table(signal, prices, uid_map, capital, entry_info))
    lines.append("")

    # Last signal summary
    lines.append(f"\U0001F4E1 <b>Last signal:</b> {signal.date.strftime('%Y-%m-%d')}")
    if signal.buys or signal.sells:
        n = len(signal.holdings_clean)
        per_stock = effective / n if n > 0 else 0
        for sym in signal.buys:
            uid = uid_map.get(sym, sym)
            price = get_price_on_date(prices, uid, signal.date)
            shares = math.floor(per_stock / price) if price > 0 else 0
            lines.append(f"  \U0001F7E2 BUY  <b>{sym}</b>  {shares} sh @ ${price:.2f} = ${shares*price:,.0f}")
        for sym in signal.sells:
            uid = uid_map.get(sym, sym)
            price = get_price_on_date(prices, uid, signal.date)
            lines.append(f"  \U0001F534 SELL <b>{sym}</b>  @ ${price:.2f}")
    else:
        lines.append("  \u2696 No trades (hold)")
    lines.append("")

    lines.append(f"\u26A1 Leverage: <b>{signal.leverage:.2f}x</b> | Exposure: ${effective:,.0f}")
    lines.append(f"{bear_icon} Bear gate: {bear_str} | {daily_icon} Today: <b>{daily_str}</b>")
    lines.append(f"\U0001F3AF CAGR: <b>{m['cagr']*100:+.1f}%</b> | Sharpe: {m['sharpe']:.2f}")
    lines.append(f"\U0001F4C5 Next rebalance: ~{est_next.strftime('%b %d')}")

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
            verify=False,
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
    parser.add_argument("--reset-ledger", action="store_true",
                        help="Re-bootstrap portfolio ledger from backtest (use after manual trades)")
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

        # 3. Load or bootstrap the persistent portfolio ledger
        if args.reset_ledger and LEDGER_PATH.exists():
            LEDGER_PATH.unlink()
            print("  Ledger reset.")

        ledger = PortfolioLedger.load()
        if ledger is None:
            print("  No portfolio ledger found — bootstrapping from backtest...", end="", flush=True)
            ledger = PortfolioLedger.bootstrap(signal, result, prices, uid_map, capital)
            ledger.save()
            print(f" done ({len(ledger.positions)} positions saved).")

        # 4. Determine if today is a rebalance day with trades
        today = pd.Timestamp(datetime.now().date())
        is_rebalance = (signal.date.date() == today.date()) and (signal.buys or signal.sells)

        # 5. On rebalance day, update ledger BEFORE formatting message
        if is_rebalance:
            ledger.update_rebalance(signal, prices, uid_map)
            ledger.save()

        # 6. Format message
        if is_rebalance:
            msg = format_rebalance_message(signal, prices, uid_map, capital, result, ledger)
        else:
            msg = format_daily_summary(signal, prices, uid_map, capital, result, ledger)

        # 7. Send or print
        if args.dry_run:
            # Ensure all pending text-mode output is flushed before we print the message
            sys.stdout.flush()
            print("\n" + "=" * 50)
            print("  DRY RUN -- message that would be sent:")
            print("=" * 50)
            # Reconfigure stdout to utf-8 so emoji prints cleanly on Windows
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, Exception):
                pass
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
            print("=" * 50)
            sent_ok = True
        else:
            print("  Sending Telegram...", end="", flush=True)
            sent_ok = send_telegram(msg, bot_token, chat_id)
            print(" sent!" if sent_ok else " FAILED!")

        # 8. Log
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
