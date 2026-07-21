"""
deps.py — process-wide singletons.

The compiled LangGraph agent (with its Retriever, Qdrant client, and LLM
clients) is expensive to build, so we build it ONCE per process and reuse it
across requests. A single shared checkpointer gives every session its own
persistent conversation thread.
"""

import os

from .config import get_settings
from .logging import get_logger

_agent = None
_build_error: str | None = None
logger = get_logger()


def _export_env(settings) -> None:
    """
    The agent/retrieval modules read credentials from os.environ (via their own
    load_dotenv). Mirror settings into the environment so a single Settings
    source of truth drives everything, including values injected only as real
    container env vars.
    """
    mapping = {
        "OPENAI_API_KEY": settings.openai_api_key,
        "GROQ_API_KEY": settings.groq_api_key,
        "QDRANT_URL": settings.qdrant_url,
        "QDRANT_API_KEY": settings.qdrant_api_key,
        "QDRANT_COLLECTION": settings.qdrant_collection,
        "TAVILY_API_KEY": settings.tavily_api_key,
        # make_memory_checkpointer() reads this directly (env-driven so the
        # agent module stays usable standalone) — a Postgres URL here backs
        # conversation memory with the same DB as users/credits (V3 fix for
        # the checkpointer previously being in-memory-only in production).
        "DATABASE_URL": settings.database_url,
    }
    if settings.agent_checkpoint_db:
        mapping["AGENT_CHECKPOINT_DB"] = settings.agent_checkpoint_db
    for k, v in mapping.items():
        if v:
            os.environ.setdefault(k, v)


def build() -> None:
    """Attempt to build the agent; record (don't raise) any failure so the app
    can still start and report readiness via /health."""
    global _agent, _build_error
    settings = get_settings()

    # Load .env into os.environ so provider selection + provider keys
    # (SYNTH_PROVIDER, NVIDIA_API_KEY, …) are visible for a provider-aware check.
    from dotenv import load_dotenv
    load_dotenv(settings.model_config.get("env_file"))
    _export_env(settings)

    from app.agent.providers import required_key_envs, key_present
    infra_missing = settings.missing_agent_keys()                       # OpenAI + Qdrant
    provider_missing = [e for e in required_key_envs(["intake", "synth"])
                        if not key_present(e)]                          # Groq/NVIDIA/OpenRouter
    missing = infra_missing + [m for m in provider_missing if m not in infra_missing]
    if missing:
        _build_error = f"missing credentials: {', '.join(missing)}"
        logger.warning("agent not built", extra={"extra_fields": {"reason": _build_error}})
        return

    try:
        # Imported lazily so the module (and tests) load without heavy deps.
        from app.agent.graph import build_agent, make_memory_checkpointer
        _agent = build_agent(
            model=settings.agent_model,
            checkpointer=make_memory_checkpointer(),
        )
        _build_error = None
        logger.info("agent built", extra={"extra_fields": {"model": settings.agent_model}})
    except Exception as e:  # noqa: BLE001
        _build_error = repr(e)
        logger.error("agent build failed", extra={"extra_fields": {"error": _build_error}})


def get_agent():
    """Return the compiled agent, building on first use. None if unavailable."""
    if _agent is None and _build_error is None:
        build()
    return _agent


def readiness() -> dict:
    agent = get_agent()
    return {
        "ready": agent is not None,
        "reason": _build_error,
    }
