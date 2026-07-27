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
# AsyncConnectionPool backing the checkpointer, owned here so lifespan can close
# it. None when the in-memory tier is in use.
_checkpointer_pool = None
_checkpointer_tier: str = "unbuilt"
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


def _prepare() -> str | None:
    """Load credentials into os.environ and verify the required ones exist.
    Returns an error string, or None when the build may proceed."""
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
        return f"missing credentials: {', '.join(missing)}"
    return None


def _compile(checkpointer) -> None:
    """Compile the agent onto the given checkpointer; record (don't raise) any
    failure so the app can still start and report readiness via /health."""
    global _agent, _build_error
    settings = get_settings()
    try:
        # Imported lazily so the module (and tests) load without heavy deps.
        from app.agent.graph import build_agent
        _agent = build_agent(model=settings.agent_model, checkpointer=checkpointer)
        _build_error = None
        logger.info("agent built", extra={"extra_fields": {
            "model": settings.agent_model, "checkpointer": _checkpointer_tier,
        }})
    except Exception as e:  # noqa: BLE001
        _build_error = repr(e)
        logger.error("agent build failed",
                     extra={"extra_fields": {"error": _build_error}}, exc_info=True)


async def build_async() -> None:
    """
    Build the agent for the ASGI server. Must be awaited from inside a running
    event loop — AsyncPostgresSaver captures the loop at construction time.

    This is the path that matters in production: main.py drives the graph with
    astream(), which requires a checkpointer implementing the async interface.
    """
    global _build_error, _checkpointer_pool, _checkpointer_tier
    if (err := _prepare()):
        _build_error = err
        logger.warning("agent not built", extra={"extra_fields": {"reason": err}})
        return

    from app.agent.graph import make_async_checkpointer
    checkpointer, pool = await make_async_checkpointer()
    _checkpointer_pool = pool
    _checkpointer_tier = "postgres" if pool is not None else "memory"
    _compile(checkpointer)


def build(*, allow_sync_postgres: bool = True) -> None:
    """
    Build the agent for SYNC callers (CLI, evals, sync tests) that drive the
    graph with invoke(). The ASGI server uses build_async() instead.
    """
    global _build_error, _checkpointer_tier
    if (err := _prepare()):
        _build_error = err
        logger.warning("agent not built", extra={"extra_fields": {"reason": err}})
        return

    from app.agent.graph import make_memory_checkpointer
    checkpointer = make_memory_checkpointer(allow_sync_postgres=allow_sync_postgres)
    # Same vocabulary as build_async() so /health means one thing, not two.
    _checkpointer_tier = ("postgres" if type(checkpointer).__name__ == "PostgresSaver"
                          else "memory")
    _compile(checkpointer)


def get_agent():
    """Return the compiled agent, building on first use. None if unavailable."""
    if _agent is None and _build_error is None:
        # Unreachable in the server: lifespan awaits build_async() before any
        # request is accepted. Defensive only — and it refuses the sync Postgres
        # tier because this path has no running loop (so it cannot build the
        # async saver) and must never resurrect the astream NotImplementedError.
        build(allow_sync_postgres=False)
    return _agent


async def aclose() -> None:
    """Close the checkpointer pool. Called from the lifespan shutdown path."""
    global _checkpointer_pool
    if _checkpointer_pool is not None:
        await _checkpointer_pool.close()
        _checkpointer_pool = None


def readiness() -> dict:
    agent = get_agent()
    return {
        "ready": agent is not None,
        "reason": _build_error,
        # Surfaces a silent degrade: "memory" against a Postgres DATABASE_URL
        # means conversation history is being lost on every restart.
        "checkpointer": _checkpointer_tier,
    }
