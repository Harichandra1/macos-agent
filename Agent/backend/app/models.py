"""
models.py — SQLAlchemy ORM models (the app's single relational schema).

Phase 1 keeps the schema deliberately minimal: `users` only. Later phases add
their tables here (credits, cost ledger, feedback) and ship each change as an
Alembic migration in `migrations/versions/` — never by editing a released
migration.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _new_id() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """One row per Google-authenticated user (schema per the v2.0 plan)."""

    __tablename__ = "users"

    user_id:    Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    email:      Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                 default=_utcnow)


class CreditAccount(Base):
    """
    Per-user allowance state (Phase 3 hybrid credit system). Both windows use
    calendar KEYS ("2026-W30" / "2026-07") rather than timestamps: a row is
    stale exactly when its key differs from the current one, which makes the
    weekly refresh a lazy, idempotent reset on first access in the new window
    (no cron needed until the Phase 5 deployment adds one).
    """

    __tablename__ = "credit_accounts"

    user_id:        Mapped[str] = mapped_column(
        String(32), ForeignKey("users.user_id"), primary_key=True)
    week_key:       Mapped[str] = mapped_column(String(10), nullable=False)
    weekly_left:    Mapped[int] = mapped_column(Integer, nullable=False)
    month_key:      Mapped[str] = mapped_column(String(7), nullable=False)
    month_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class Feedback(Base):
    """One thumbs up/down per rated assistant answer (Phase 4). Keyed to the
    session + a client-supplied turn index so re-rating the same answer updates
    in place rather than piling up rows. No message content — just the signal
    the Grafana 'bad-feedback spikes' panel and offline evals consume."""

    __tablename__ = "feedback"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_index", name="uq_feedback_session_turn"),
    )

    id:         Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    user_id:    Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    rating:     Mapped[str] = mapped_column(String(4), nullable=False)   # "up" | "down"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                default=_utcnow)


class UsageLog(Base):
    """One row per /chat turn — the cost ledger (feeds the monthly cap now,
    Grafana in Phase 4). Token counts and cost only; never message content."""

    __tablename__ = "usage_log"

    id:            Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id:       Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    session_id:    Mapped[str] = mapped_column(String(64), nullable=False)
    created_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False,
                                                    default=_utcnow)
    llm_calls:     Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens:  Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd:      Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    estimated:     Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_type:    Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
