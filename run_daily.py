"""
WaveRider Daily Orchestrator
============================
Scheduled via Windows Task Scheduler: Mon–Fri at 6 PM ET.

Flow:
  1. Trigger Norgate Data Updater (NDU) to fetch today's closing prices
  2. Wait until Norgate confirms data is fresh for today
  3. Load (or generate) the 10-year EOM rebalance calendar
  4. Run waverider_signal_bot.py:
       - EOM day  →  --rebalance  (full universe rebuild + new signal, ~3 min)
       - Hold day →  --hold-day   (fast path, fresh prices only,    ~10 sec)

Usage:
    python run_daily.py           # normal daily run
    python run_daily.py --dry-run # pass --dry-run through to the bot
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON     = sys.executable
BOT        = str(SCRIPT_DIR / "waverider_signal_bot.py")
CALENDAR_PATH = SCRIPT_DIR / "rebalance_calendar.json"
NDU        = r"C:\Program Files\Norgate Data Updater\bin\ndu.trigger.exe"


# ---------------------------------------------------------------------------
# Norgate trigger + freshness check
# ---------------------------------------------------------------------------

def trigger_norgate() -> None:
    """Fire NDU trigger to kick off a data update. NDU service must be running."""
    try:
        subprocess.run([NDU], timeout=15, capture_output=True)
        print("  NDU trigger sent.")
    except FileNotFoundError:
        print(f"  WARNING: NDU not found at {NDU}. Skipping trigger.")
    except Exception as e:
        print(f"  NDU trigger error: {e}")


def wait_for_fresh_data(timeout_minutes: int = 45) -> bool:
    """Poll norgatedata.last_price_update_time('SPY') until today's data is available.

    Returns True once data is confirmed fresh, False on timeout.
    Logs progress every minute.
    """
    try:
        import norgatedata
    except ImportError:
        print("  WARNING: norgatedata package not installed. Skipping freshness check.")
        return True

    today = date.today()
    deadline = time.time() + timeout_minutes * 60
    print(f"  Waiting for Norgate data (today = {today})...", end="", flush=True)

    while time.time() < deadline:
        try:
            t = norgatedata.last_price_update_time("SPY")
            if t and t.date() >= today:
                print(f" ready ({t.strftime('%H:%M:%S UTC')})")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(60)

    print(" TIMED OUT")
    return False


# ---------------------------------------------------------------------------
# EOM rebalance calendar
# ---------------------------------------------------------------------------

def _generate_eom_dates(start_year: int = 2026, end_year: int = 2035) -> list:
    """Return list of ISO date strings for the last NYSE trading day of each month.

    Uses pandas CustomBusinessMonthEnd with an NYSE holiday calendar.
    NYSE holidays included: New Year's Day, MLK Day, Presidents Day, Good Friday,
    Memorial Day, Juneteenth (from 2022), Independence Day, Labor Day,
    Thanksgiving Day, Christmas Day.
    """
    import pandas as pd
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar, GoodFriday, Holiday,
        USLaborDay, USMemorialDay, USMartinLutherKingJr,
        USPresidentsDay, USThanksgivingDay, nearest_workday,
    )
    from pandas.tseries.offsets import CustomBusinessMonthEnd

    class NYSECalendar(AbstractHolidayCalendar):
        rules = [
            Holiday("New Year's Day",   month=1,  day=1,  observance=nearest_workday),
            USMartinLutherKingJr,
            USPresidentsDay,
            GoodFriday,
            USMemorialDay,
            Holiday("Juneteenth",       month=6,  day=19, observance=nearest_workday,
                    start_date="2022-01-01"),
            Holiday("Independence Day", month=7,  day=4,  observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas",        month=12, day=25, observance=nearest_workday),
        ]

    cbme = CustomBusinessMonthEnd(calendar=NYSECalendar())
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq=cbme)
    return [d.strftime("%Y-%m-%d") for d in dates]


def load_or_generate_calendar() -> dict:
    """Load rebalance_calendar.json, generating it if missing or expired."""
    if CALENDAR_PATH.exists():
        data = json.loads(CALENDAR_PATH.read_text())
        dates = data.get("eom_dates", [])
        # Regenerate if last date is in the past (calendar expired)
        if dates and dates[-1] >= date.today().isoformat():
            return data
        print("  Rebalance calendar expired — regenerating...")

    print("  Generating EOM rebalance calendar (2026–2035)...")
    eom_dates = _generate_eom_dates(2026, 2035)
    data = {
        "generated": datetime.now().isoformat(),
        "note": "Last NYSE trading day of each month, 2026-2035",
        "eom_dates": eom_dates,
    }
    CALENDAR_PATH.write_text(json.dumps(data, indent=2))
    print(f"  Saved {len(eom_dates)} EOM dates → rebalance_calendar.json")
    return data


def is_eom_today(calendar: dict) -> bool:
    return date.today().isoformat() in calendar.get("eom_dates", [])


def next_eom_date(calendar: dict) -> str:
    today_iso = date.today().isoformat()
    return next((d for d in calendar.get("eom_dates", []) if d >= today_iso), "?")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WaveRider Daily Orchestrator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pass --dry-run to the bot (no Telegram send)")
    parser.add_argument("--skip-norgate-wait", action="store_true",
                        help="Skip NDU trigger and freshness check (for testing)")
    args = parser.parse_args()

    print(f"\n{'=' * 58}")
    print(f"  WaveRider Runner  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"{'=' * 58}")

    # 1. Trigger Norgate update
    if not args.skip_norgate_wait:
        print("\n[1/4] Triggering Norgate Data Updater...")
        trigger_norgate()
    else:
        print("\n[1/4] Skipping Norgate trigger (--skip-norgate-wait).")

    # 2. Wait for fresh data
    if not args.skip_norgate_wait:
        print("\n[2/4] Waiting for today's Norgate data...")
        fresh = wait_for_fresh_data(timeout_minutes=45)
        if not fresh:
            print("  WARNING: Data may be stale. Proceeding anyway (better than silence).")
    else:
        print("\n[2/4] Skipping freshness check.")

    # 3. Load / generate EOM calendar
    print("\n[3/4] Checking rebalance calendar...")
    calendar = load_or_generate_calendar()
    eom = is_eom_today(calendar)
    nxt = next_eom_date(calendar)
    mode = "EOM REBALANCE" if eom else "hold"
    print(f"  Today is a {mode} day.  Next EOM: {nxt}")

    # 4. Run the bot
    flag = "--rebalance" if eom else "--hold-day"
    cmd = [PYTHON, BOT, flag]
    if args.dry_run:
        cmd.append("--dry-run")

    print(f"\n[4/4] Running bot ({flag})...")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
