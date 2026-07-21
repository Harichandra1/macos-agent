"""
providers.py — pluggable LLM provider factory.

Every LLM role in the agent (intake, smart-merge, synthesis) and in the benchmark
(judges) is built through `make_chat_model`, so each role can independently target
a free provider. Groq, NVIDIA NIM, and OpenRouter all expose OpenAI-compatible
endpoints, so a single `ChatOpenAI` with a swapped `base_url` covers them all —
plus OpenAI itself.

Per-role selection is read from the environment so the agent stays decoupled from
the FastAPI settings and works standalone (CLI, benchmark):

    SYNTH_PROVIDER=nvidia  SYNTH_MODEL=qwen/qwen2.5-72b-instruct
    INTAKE_PROVIDER=groq   INTAKE_MODEL=llama-3.1-8b-instant
    FILTER_PROVIDER=groq   FILTER_MODEL=llama-3.1-8b-instant

Defaults keep the original Groq behavior, so nothing regresses if these are unset.
"""

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

# provider -> (OpenAI-compatible base_url, env var holding the API key)
PROVIDERS: dict[str, tuple[str | None, str]] = {
    "groq":       ("https://api.groq.com/openai/v1",     "GROQ_API_KEY"),
    "nvidia":     ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1",        "OPENROUTER_API_KEY"),
    "openai":     (None,                                  "OPENAI_API_KEY"),  # default base_url
    "xai":        ("https://api.x.ai/v1",                 "XAI_API_KEY"),      # Grok (optional judge)
}

# role -> (default provider, default model)
ROLE_DEFAULTS: dict[str, tuple[str, str]] = {
    "intake":  ("groq", "llama-3.1-8b-instant"),
    "filter":  ("groq", "llama-3.1-8b-instant"),
    "planner": ("groq", "llama-3.3-70b-versatile"),  # decide ANSWER/CLARIFY/DIAGNOSE
    "synth":   ("groq", "llama-3.3-70b-versatile"),
}

# Per-role timeouts (seconds). Bounded so a hung provider can't hang the SSE
# stream, and short enough that the fallback chain gets its chance within a turn.
ROLE_TIMEOUTS: dict[str, float] = {
    "intake":  25.0,
    "filter":  25.0,
    "planner": 25.0,
    "synth":   60.0,
}
DEFAULT_TIMEOUT = 60.0

# Retries are executed by the OpenAI SDK with EXPONENTIAL BACKOFF + jitter,
# honoring a 429's Retry-After header when present — that's the Phase 2
# backoff requirement, so we tune the count rather than reimplement the loop.
# Terminal errors (daily quota, auth) still fail after retries, which is what
# hands the turn to the cross-provider fallback chain below.
DEFAULT_MAX_RETRIES = 2


def _default_max_retries() -> int:
    """LLM_MAX_RETRIES env override; SDK backoff stays exponential regardless."""
    try:
        return max(0, int(os.environ.get("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))))
    except ValueError:
        return DEFAULT_MAX_RETRIES


def classify_llm_error(exc: BaseException) -> str:
    """
    Coarse provider-error taxonomy: quota_exhausted | rate_limit | timeout |
    auth | other. Drives the user-facing SSE error message and the
    `error_type` field in structured logs (the Phase 4 metrics contract).
    String/type-name heuristics on purpose — the OpenAI-compatible providers
    (Groq/NVIDIA/OpenRouter/OpenAI) raise the same SDK types but differ in
    which condition maps to which status, and the message text is the only
    reliable discriminator across them.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    # Quota/billing BEFORE rate-limit: "insufficient_quota" errors arrive as
    # 429s too, but retrying/backoff can't fix them within a turn.
    if any(k in text for k in ("insufficient_quota", "quota", "billing",
                               "credit", "payment required")):
        return "quota_exhausted"
    if "ratelimit" in name or "rate limit" in text or "rate_limit" in text or "429" in text:
        return "rate_limit"
    if "timeout" in name or "timed out" in text or "timeout" in text:
        return "timeout"
    if ("auth" in name or "api key" in text or "api_key" in text
            or "401" in text or "403" in text or "permission" in text):
        return "auth"
    return "other"

# Same-class model on each alternative provider, per role tier. Failover swaps
# PROVIDERS for the same class of open model — it never introduces a second
# specialized model (CLAUDE.md single-model rule). Keyed by the role's default
# model size: small (8B-class) vs large (70B-class).
_FALLBACK_MODELS: dict[str, dict[str, str]] = {
    # provider -> model for the 70B-class roles (planner, synth)
    "large": {
        "groq":       "llama-3.3-70b-versatile",
        "nvidia":     "meta/llama-3.3-70b-instruct",
        "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
        "openai":     "gpt-4o-mini",   # last-resort paid fallback, tiny cost
    },
    # provider -> model for the 8B-class roles (intake, filter)
    "small": {
        "groq":       "llama-3.1-8b-instant",
        "nvidia":     "meta/llama-3.1-8b-instruct",
        "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
        "openai":     "gpt-4o-mini",
    },
}
_ROLE_CLASS = {"intake": "small", "filter": "small", "planner": "large", "synth": "large"}
_FALLBACK_ORDER = ("groq", "nvidia", "openrouter", "openai")


@dataclass(frozen=True)
class RoleConfig:
    provider: str
    model: str
    key_env: str


def resolve_role(role: str) -> RoleConfig:
    """Resolve a role to (provider, model, key_env) from env, falling back to defaults."""
    default_provider, default_model = ROLE_DEFAULTS.get(role, ("groq", "llama-3.3-70b-versatile"))
    provider = os.environ.get(f"{role.upper()}_PROVIDER", default_provider).strip().lower()
    model = os.environ.get(f"{role.upper()}_MODEL", default_model).strip()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}' for role '{role}'. "
                         f"choose from {sorted(PROVIDERS)}")
    return RoleConfig(provider=provider, model=model, key_env=PROVIDERS[provider][1])


def key_present(env_name: str) -> bool:
    v = os.environ.get(env_name, "")
    return bool(v) and v not in ("...", "changeme")


def required_key_envs(roles: list[str]) -> list[str]:
    """Distinct API-key env names needed to build the given roles."""
    seen, out = set(), []
    for r in roles:
        env = resolve_role(r).key_env
        if env not in seen:
            seen.add(env)
            out.append(env)
    return out


def _build_chat(provider: str, model: str, temperature: float,
                timeout: float, max_retries: int) -> ChatOpenAI:
    """Low-level ChatOpenAI construction for one provider+model."""
    base_url, key_env = PROVIDERS[provider]
    if not key_present(key_env):
        raise RuntimeError(f"missing {key_env} for provider '{provider}'")
    kwargs = {
        "model": model,
        "api_key": os.environ.get(key_env, ""),
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url
    # Report token usage on STREAMED responses too (final chunk carries it) —
    # the per-turn cost accounting (app/usage.py) reads it. Only for providers
    # verified to accept `stream_options.include_usage`; the others fall back
    # to the tracker's estimate path rather than risking a request error.
    if provider in ("groq", "openai"):
        kwargs["stream_usage"] = True
    return ChatOpenAI(**kwargs)


def _fallbacks_enabled() -> bool:
    return os.environ.get("LLM_FALLBACKS", "on").strip().lower() not in (
        "off", "0", "false", "no")


def make_chat_model(
    role: str | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,
    max_retries: int | None = None,
    with_fallbacks: bool = False,
):
    """
    Build a chat model for a role (env-resolved) or an explicit provider+model.

    All providers are OpenAI-compatible, so one ChatOpenAI covers them by base_url.
    Streaming is left to the caller/graph (LangGraph 'messages' mode captures tokens).

    with_fallbacks=True chains same-class models on every OTHER provider whose key
    is present (via LangChain `.with_fallbacks`), so a Groq 429/daily-cap or outage
    degrades to NVIDIA/OpenRouter instead of killing the turn. Kill-switch:
    LLM_FALLBACKS=off.
    """
    if role is not None:
        cfg = resolve_role(role)
        provider, model, key_env = cfg.provider, cfg.model, cfg.key_env
    else:
        if not provider or not model:
            raise ValueError("provide either role, or both provider and model")
        provider = provider.strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider '{provider}'")

    if timeout is None:
        timeout = ROLE_TIMEOUTS.get(role or "", DEFAULT_TIMEOUT)
    if max_retries is None:
        max_retries = _default_max_retries()

    primary = _build_chat(provider, model, temperature, timeout, max_retries)

    if not (with_fallbacks and _fallbacks_enabled()):
        return primary

    # Same-class models on the other providers, in preference order, keys present.
    tier = _ROLE_CLASS.get(role or "", "large")
    fallbacks = []
    for alt in _FALLBACK_ORDER:
        if alt == provider:
            continue
        alt_model = _FALLBACK_MODELS[tier].get(alt)
        if not alt_model or not key_present(PROVIDERS[alt][1]):
            continue
        try:
            # Fallbacks get fewer retries — fail over fast, don't stack delays.
            fallbacks.append(_build_chat(alt, alt_model, temperature, timeout, 1))
        except RuntimeError:
            continue
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
