"""
db.py — engine, session factory, and migration runner for the app database.

One shared relational store (users now; credits/costs/feedback in later
phases). SQLite by default so dev needs zero infrastructure; the jump to
Postgres is a DATABASE_URL change — nothing in here or in the models is
SQLite-specific.

Schema is owned by Alembic (Agent/backend/migrations). `run_migrations()` is
called once at startup so a fresh checkout/container reaches the current
schema without a manual step; the same migrations run against Postgres in
deployment.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .logging import get_logger

logger = get_logger()

_BACKEND_ROOT = Path(__file__).resolve().parents[1]   # Agent/backend

_engine = None
_SessionLocal: sessionmaker | None = None


def _ensure_sqlite_dir(url: str) -> None:
    """SQLite won't create missing parent directories — do it for any file URL.
    Needed by BOTH engine creation and Alembic (which builds its own engine)."""
    if not url.startswith("sqlite"):
        return
    db_path = url.removeprefix("sqlite:///")
    if db_path and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _make_engine():
    settings = get_settings()
    url = settings.database_url
    # pool_pre_ping recycles connections a managed Postgres (or a laptop sleep)
    # silently dropped, so the first request after idle doesn't 500.
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        # FastAPI serves requests from multiple threads; SQLite's default
        # same-thread guard would reject sessions created off the main thread.
        kwargs["connect_args"] = {"check_same_thread": False}
        _ensure_sqlite_dir(url)
    else:
        # Managed Postgres free tiers cap total connections tightly — keep the
        # pool small and recycle hourly so we never exhaust the server's slots.
        kwargs.update(pool_size=settings.db_pool_size,
                      max_overflow=settings.db_max_overflow,
                      pool_recycle=1800)
    return create_engine(url, **kwargs)


def get_engine():
    global _engine
    if _engine is None:
        _engine = _make_engine()
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_db():
    """FastAPI dependency: one session per request, always closed."""
    session: Session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def run_migrations() -> None:
    """Bring the schema to head. Raises on failure — a half-migrated DB must
    not serve traffic silently; the caller decides whether that's fatal."""
    from alembic import command
    from alembic.config import Config

    url = get_settings().database_url
    _ensure_sqlite_dir(url)
    cfg = Config(str(_BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    logger.info("db migrated", extra={"extra_fields": {
        "url": get_settings().database_url.split("://")[0] + "://…"}})


def reset_for_tests(url: str) -> None:
    """Point the module at a fresh database (tests only)."""
    global _engine, _SessionLocal
    get_settings.cache_clear()
    import os
    os.environ["DATABASE_URL"] = url
    _engine = None
    _SessionLocal = None
