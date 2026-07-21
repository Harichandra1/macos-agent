"""
jobs.py — scheduled maintenance commands (Phase 5).

Run as a platform cron job in deployment (Render Cron), NOT an in-process
thread: a free-tier web service scales to zero when idle, so a background
timer inside the app would simply stop firing. A separate scheduled process
that shares the same DATABASE_URL is the durable design.

Commands:
  reset-weekly-credits — set every account's weekly allowance back to the
    configured limit and stamp the current ISO-week key. Idempotent and
    safe to run more than once per week: credits.get_account() ALSO resets
    lazily on first access, so this cron is a belt-and-braces refresh (and
    the thing that resets accounts of users who don't return that week, so
    reporting/analytics see a clean slate).

Usage:
  python -m app.jobs reset-weekly-credits
"""

import argparse
import sys

from sqlalchemy import update

from .config import get_settings
from .credits import week_key
from .db import get_sessionmaker, run_migrations
from .logging import get_logger
from .models import CreditAccount

logger = get_logger()


def reset_weekly_credits() -> int:
    """Refresh every account to the weekly limit for the current week.
    Returns the number of rows updated."""
    settings = get_settings()
    wk = week_key()
    with get_sessionmaker()() as db:
        result = db.execute(
            update(CreditAccount)
            .values(week_key=wk, weekly_left=settings.weekly_credit_limit))
        db.commit()
        n = result.rowcount or 0
    logger.info("weekly credits reset", extra={"extra_fields": {
        "week_key": wk, "accounts": n, "limit": settings.weekly_credit_limit}})
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduled maintenance jobs")
    parser.add_argument("command", choices=["reset-weekly-credits"])
    parser.add_argument("--no-migrate", action="store_true",
                        help="skip the startup migration check")
    args = parser.parse_args()

    # A cron container starts cold — make sure the schema exists before touching it.
    if not args.no_migrate:
        run_migrations()

    if args.command == "reset-weekly-credits":
        n = reset_weekly_credits()
        print(f"reset weekly credits for {n} account(s)")
    else:  # pragma: no cover — argparse already constrains choices
        parser.error(f"unknown command {args.command}")
        sys.exit(2)


if __name__ == "__main__":
    main()
