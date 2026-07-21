"""
metrics.py — Prometheus instrumentation for the serving layer (Phase 4).

Complements the structured JSON logs (which Loki ingests) with numeric time
series Grafana can chart: request outcomes, tokens, cost, errors, credit
refusals, guardrail hits, and feedback. The ledger already has per-turn
truth (usage_log); these counters are the cheap, scrape-friendly aggregate.

Design:
  * One module owns every metric name, so the Grafana dashboard has a stable
    contract to query against.
  * Thin helper functions (record_*) are the ONLY way the app touches metrics,
    so call sites stay readable and the prometheus_client dependency is
    isolated here.
  * Import-safe: if prometheus_client is somehow unavailable, every helper
    becomes a no-op and /metrics returns empty — instrumentation must never
    take down the app it observes.
"""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    _ENABLED = True
except Exception:  # noqa: BLE001 — degrade to no-op instrumentation
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# Cost buckets span a single cheap turn (~$0.001) up to an outlier long paste
# ($0.05); latency buckets cover fast clarify turns to a slow refine loop.
_COST_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1)
_LATENCY_BUCKETS = (0.5, 1, 2, 4, 8, 15, 30, 60)


if _ENABLED:
    CHAT_REQUESTS = Counter(
        "macos_chat_requests_total",
        "Chat requests by terminal outcome.",
        ["outcome"],   # ok | error | abuse_blocked | credits_refused | agent_unavailable
    )
    CHAT_LLM_CALLS = Counter(
        "macos_chat_llm_calls_total", "LLM calls made across all turns.")
    CHAT_TOKENS = Counter(
        "macos_chat_tokens_total", "LLM tokens by direction.", ["direction"])  # input|output
    CHAT_COST_USD = Counter(
        "macos_chat_cost_usd_total",
        "Cumulative provider cost in USD (free tiers billed at list price).")
    CHAT_ERRORS = Counter(
        "macos_chat_errors_total", "Turn failures by provider-error class.",
        ["error_type"])   # rate_limit | quota_exhausted | timeout | auth | other
    CHAT_COST = Histogram(
        "macos_chat_turn_cost_usd", "Per-turn cost distribution (USD).",
        buckets=_COST_BUCKETS)
    CHAT_DURATION = Histogram(
        "macos_chat_turn_duration_seconds", "Wall-clock time per chat turn.",
        buckets=_LATENCY_BUCKETS)
    CREDIT_REFUSALS = Counter(
        "macos_credit_refusals_total", "Turns refused for exhausted allowance.",
        ["reason"])   # weekly | monthly
    PII_REDACTIONS = Counter(
        "macos_pii_redactions_total", "PII spans redacted, by kind.", ["kind"])
    ABUSE_BLOCKS = Counter(
        "macos_abuse_blocks_total", "Abusive requests blocked, by pattern.", ["kind"])
    FEEDBACK = Counter(
        "macos_feedback_total", "User answer ratings.", ["rating"])  # up | down


# --- recording helpers (no-ops when disabled) ------------------------------

def record_request(outcome: str) -> None:
    if _ENABLED:
        CHAT_REQUESTS.labels(outcome=outcome).inc()


def record_turn_usage(*, llm_calls: int, input_tokens: int, output_tokens: int,
                      cost_usd: float, duration_s: Optional[float] = None) -> None:
    if not _ENABLED:
        return
    CHAT_LLM_CALLS.inc(llm_calls)
    CHAT_TOKENS.labels(direction="input").inc(input_tokens)
    CHAT_TOKENS.labels(direction="output").inc(output_tokens)
    CHAT_COST_USD.inc(cost_usd)
    CHAT_COST.observe(cost_usd)
    if duration_s is not None:
        CHAT_DURATION.observe(duration_s)


def record_error(error_type: str) -> None:
    if _ENABLED:
        CHAT_ERRORS.labels(error_type=error_type).inc()


def record_credit_refusal(reason: str) -> None:
    if _ENABLED:
        CREDIT_REFUSALS.labels(reason=reason).inc()


def record_pii(counts: dict[str, int]) -> None:
    if _ENABLED:
        for kind, n in counts.items():
            PII_REDACTIONS.labels(kind=kind).inc(n)


def record_abuse(kind: str) -> None:
    if _ENABLED:
        ABUSE_BLOCKS.labels(kind=kind).inc()


def record_feedback(rating: str) -> None:
    if _ENABLED:
        FEEDBACK.labels(rating=rating).inc()


def exposition() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    if not _ENABLED:
        return (b"", CONTENT_TYPE_LATEST)
    return (generate_latest(), CONTENT_TYPE_LATEST)


def enabled() -> bool:
    return _ENABLED
