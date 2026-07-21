"""
credits.py — the hybrid credit system (Phase 3).

Two independent allowances per authenticated user:
  * weekly_left — N chat turns per ISO week (default 5), refreshed lazily:
    the first access in a new week resets the counter. Idempotent, so the
    Phase 5 deployment can ALSO run a real weekly cron without conflict.
  * month_cost_usd — cumulative provider cost this calendar month; once it
    reaches the cap (default $0.25) every turn is refused until the month
    rolls over. Cost comes from the per-turn UsageTracker (app/usage.py) and
    is written to the usage_log ledger alongside the aggregate.

A turn consumes 1 weekly credit UP FRONT (no racing a long stream), and its
cost lands at stream end. Credits require identity: when auth is disabled
there is no user row, enforcement is skipped, and usage is still logged with
user_id=NULL for observability.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from .config import get_settings
from .logging import get_logger
from .models import CreditAccount, UsageLog

logger = get_logger()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def week_key(now: Optional[datetime] = None) -> str:
    y, w, _ = (now or _now()).isocalendar()
    return f"{y}-W{w:02d}"


def month_key(now: Optional[datetime] = None) -> str:
    d = now or _now()
    return f"{d.year}-{d.month:02d}"


@dataclass
class CreditStatus:
    allowed: bool
    reason: Optional[str]      # None | "weekly" | "monthly"
    weekly_left: int
    weekly_limit: int
    month_cost_usd: float


WEEKLY_EXHAUSTED_MESSAGE = (
    "You've used all your free messages for this week — they refresh at the "
    "start of next week (UTC)."
)
MONTHLY_CAP_MESSAGE = (
    "This account reached its monthly usage cap — it resets when the new "
    "month starts."
)


def get_account(db: Session, user_id: str,
                now: Optional[datetime] = None) -> CreditAccount:
    """Fetch (or create) the user's credit row with lazy window resets applied.
    Mutations are flushed but not committed — callers own the transaction."""
    settings = get_settings()
    wk, mk = week_key(now), month_key(now)
    acct = db.get(CreditAccount, user_id)
    if acct is None:
        acct = CreditAccount(user_id=user_id, week_key=wk,
                             weekly_left=settings.weekly_credit_limit,
                             month_key=mk, month_cost_usd=0.0)
        db.add(acct)
    else:
        if acct.week_key != wk:        # new ISO week → refresh the counter
            acct.week_key = wk
            acct.weekly_left = settings.weekly_credit_limit
        if acct.month_key != mk:       # new month → cost cap resets
            acct.month_key = mk
            acct.month_cost_usd = 0.0
    db.flush()
    return acct


def check_and_consume(db: Session, user_id: str,
                      now: Optional[datetime] = None) -> CreditStatus:
    """Gate one chat turn: refuse on an exhausted window, else consume 1
    weekly credit and commit."""
    settings = get_settings()
    acct = get_account(db, user_id, now)

    if acct.month_cost_usd >= settings.monthly_cost_cap_usd:
        db.commit()   # persist any lazy reset even when refusing
        return CreditStatus(False, "monthly", acct.weekly_left,
                            settings.weekly_credit_limit, acct.month_cost_usd)
    if acct.weekly_left <= 0:
        db.commit()
        return CreditStatus(False, "weekly", 0,
                            settings.weekly_credit_limit, acct.month_cost_usd)

    acct.weekly_left -= 1
    db.commit()
    return CreditStatus(True, None, acct.weekly_left,
                        settings.weekly_credit_limit, acct.month_cost_usd)


def record_usage(db: Session, *, user_id: Optional[str], session_id: str,
                 llm_calls: int, input_tokens: int, output_tokens: int,
                 cost_usd: float, estimated: bool,
                 error_type: Optional[str] = None,
                 now: Optional[datetime] = None) -> None:
    """Append one ledger row and roll the cost into the user's monthly total."""
    db.add(UsageLog(user_id=user_id, session_id=session_id, llm_calls=llm_calls,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=cost_usd, estimated=estimated, error_type=error_type))
    if user_id is not None:
        acct = get_account(db, user_id, now)
        acct.month_cost_usd += cost_usd
    db.commit()
