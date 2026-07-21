"""
main.py — FastAPI serving layer for the macOS troubleshooting agent.

Endpoints:
  GET  /health   — liveness + agent readiness (which credentials are missing).
  POST /chat     — Server-Sent Events stream of the diagnosis for one turn.

The /chat stream emits, in order:
  intake  → what we understood (macOS version, chip, already-tried)
  sources → the KB citations we grounded in (or web-fallback), with the path used
  token*  → the answer, streamed token by token
  done    → completion marker

Conversation memory is keyed by `session_id`: send the same id across turns and
the agent restores the accumulating case file so follow-ups escalate depth.

Run:
  cd Agent/backend
  ../../.venv/bin/uvicorn app.main:app --reload --port 8000
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from .agent.providers import classify_llm_error
from .auth import current_user, router as auth_router
from .config import get_settings
from .credits import (
    MONTHLY_CAP_MESSAGE, WEEKLY_EXHAUSTED_MESSAGE, CreditStatus,
    check_and_consume, record_usage,
)
from .db import get_sessionmaker, run_migrations
from .deps import build, get_agent, readiness
from .guardrails import ABUSE_MESSAGE, check_abuse, redact_text
from .logging import get_logger, timed
from . import metrics
from .models import Feedback, User
from .schemas import (
    ChatRequest, CreditsEvent, DiagnosticEvent, DoneEvent, ErrorEvent, FeedbackRequest,
    IntakeEvent, QuestionEvent, QuestionItem, Source, SourcesEvent, StatusEvent,
    TokenEvent, VerificationEvent,
)
from .usage import UsageTracker

settings = get_settings()
logger = get_logger()
limiter = Limiter(key_func=get_remote_address, default_limits=[])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    errors = settings.production_errors()
    if errors:
        raise RuntimeError("production configuration incomplete: " + ", ".join(errors))
    run_migrations()  # fail fast: a half-migrated schema must not serve traffic
    build()  # best-effort agent build; /health reports if credentials are missing.
    yield


app = FastAPI(title="macOS Troubleshooting Agent", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: EventSourceResponse(
        iter([{"event": "error", "data": json.dumps({
            "type": "error",
            "message": "You're sending messages quickly — give me a few seconds and try again.",
        })}])
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_host_list)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if settings.production_mode:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'self'; object-src 'none'; "
            "script-src 'self' 'unsafe-inline' https://accounts.google.com/gsi/client; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; "
            "connect-src 'self' https://accounts.google.com/gsi/; "
            "frame-src https://accounts.google.com/gsi/; frame-ancestors 'none'",
        )
    return response
app.include_router(auth_router)


@app.get("/health")
def health():
    status = readiness()
    if settings.production_mode and not status["ready"]:
        # Keep provider/database details out of a public liveness endpoint.
        status["reason"] = "agent unavailable"
    return {"status": "ok", **status}


@app.get("/metrics")
def prometheus_metrics(request: Request):
    """Prometheus scrape target (Phase 4). Text exposition of the counters/
    histograms in app.metrics — Grafana charts them via a Prometheus source."""
    if settings.production_mode:
        supplied = request.headers.get("authorization", "")
        if supplied != f"Bearer {settings.metrics_token}":
            raise HTTPException(status_code=401, detail="metrics authentication required")
    body, content_type = metrics.exposition()
    return Response(content=body, media_type=content_type)


def _sse(model) -> dict:
    """Serialize a pydantic event model into an SSE frame."""
    return {"event": model.type, "data": model.model_dump_json()}


# User-facing text per provider-error class (classify_llm_error). Most LLM
# failures degrade INSIDE the graph and never reach here; this is the last
# line, so the app fails with an honest, actionable sentence — not a bare 500.
_STREAM_ERROR_MESSAGES = {
    "rate_limit":      "The model provider is rate-limiting right now — wait a few "
                       "seconds and resend your message.",
    "quota_exhausted": "Today's free model quota is used up. The app recovers on its "
                       "own — try again in a little while.",
    "timeout":         "The model took too long to respond — please resend your message.",
    "auth":            "The server's model credentials look misconfigured — the operator "
                       "needs to check the provider API keys.",
    "other":           "internal error while generating the answer",
}


def _merged_to_source(m: dict) -> Source:
    return Source(
        origin=m.get("origin", "kb"),
        title=m.get("title", ""),
        url=m.get("url"),
        source=m.get("source", ""),
        category=m.get("category", ""),
        difficulty_tier=m.get("difficulty_tier"),
        score=m.get("score"),
        snippet=(m.get("text", "") or "")[:280],
    )


def _record_turn(tracker: UsageTracker, user_id: Optional[str],
                 session_id: str, error_type: Optional[str] = None) -> None:
    """Write the turn's ledger row + monthly aggregate. Accounting failures are
    logged, never allowed to break the user-facing stream."""
    try:
        with get_sessionmaker()() as db:
            record_usage(db, user_id=user_id, session_id=session_id,
                         llm_calls=tracker.llm_calls,
                         input_tokens=tracker.input_tokens,
                         output_tokens=tracker.output_tokens,
                         cost_usd=tracker.turn_cost_usd(),
                         estimated=tracker.estimated,
                         error_type=error_type)
    except Exception as e:  # noqa: BLE001
        logger.error("usage recording failed", extra={"extra_fields": {
            "session": session_id, "user_id": user_id, "error": repr(e)}})


async def _chat_events(agent, message: str, session_id: str,
                       user_id: Optional[str] = None,
                       credits: Optional[CreditStatus] = None):
    """Async generator mapping LangGraph stream output → SSE frames."""
    from langchain_core.messages import AIMessageChunk, HumanMessage

    # The tracker rides the callback bus into every LLM call this turn makes —
    # token/cost accounting at the provider boundary (Phase 3).
    tracker = UsageTracker()
    config = {"configurable": {"thread_id": session_id}, "callbacks": [tracker]}
    inp = {"messages": [HumanMessage(content=message)]}
    turn_start = time.perf_counter()

    streamed_any = False
    final_answer = ""
    n_sources = 0
    used_fallback = False
    pending_diag = None   # diagnostic dict captured from the decide node
    pending_qs = None     # structured questions captured from the decide node
    on_topic = True       # off-topic (task 2) suppresses sources/stage chrome

    def _stage(stage: str, label: str) -> dict:
        return _sse(StatusEvent(stage=stage, label=label))

    try:
        # Immediate progress signal — retrieval + planning take seconds and the
        # UI must never sit on a blank cursor.
        yield _stage("intake", "Reading your message…")

        async for mode, data in agent.astream(inp, config=config,
                                               stream_mode=["updates", "messages"]):
            if mode == "updates":
                for node, update in (data or {}).items():
                    if node == "decide":
                        pending_diag = update.get("diagnostic")
                        pending_qs = update.get("questions")
                        if (update.get("action") or "answer") == "answer":
                            yield _stage("synthesize", "Writing your fix…")
                    elif node == "ask_clarify":
                        # A clarify turn ends here — emit the structured
                        # questions (chips) + the readable message text.
                        msgs = update.get("messages") or []
                        q = str(getattr(msgs[-1], "content", "")) if msgs else ""
                        yield _sse(QuestionEvent(
                            question=q,
                            questions=[QuestionItem(text=i.get("text", ""),
                                                    options=i.get("options") or [])
                                       for i in (pending_qs or [])],
                        ))
                        if q:
                            yield _sse(TokenEvent(text=q))
                            streamed_any = True
                    elif node == "decline":
                        # Off-topic (task 2): a fixed message, no sources or
                        # verification — same short-circuit shape as
                        # ask_clarify/request_diagnostic.
                        msgs = update.get("messages") or []
                        body = str(getattr(msgs[-1], "content", "")) if msgs else ""
                        if body:
                            yield _sse(TokenEvent(text=body))
                            streamed_any = True
                    elif node == "request_diagnostic":
                        msgs = update.get("messages") or []
                        body = str(getattr(msgs[-1], "content", "")) if msgs else ""
                        d = pending_diag or {}
                        yield _sse(DiagnosticEvent(
                            command=d.get("command", ""),
                            rationale=d.get("rationale", ""),
                            look_for=d.get("look_for", ""),
                        ))
                        if body:
                            yield _sse(TokenEvent(text=body))
                            streamed_any = True
                    elif node == "intake":
                        intake = update.get("intake") or {}
                        on_topic = intake.get("on_topic", True)
                        yield _sse(IntakeEvent(
                            macos_version=intake.get("macos_version"),
                            mac_chip=intake.get("mac_chip"),
                            category=intake.get("category"),
                            already_tried=intake.get("already_tried") or [],
                        ))
                        if on_topic:
                            yield _stage("retrieve", "Searching the knowledge base…")
                    elif node == "smart_merge":
                        # Off-topic (task 2): kb_retrieve/web_search already
                        # short-circuited to empty — suppress the sources UI
                        # entirely rather than showing a hollow "0 sources" bar
                        # on a declined message.
                        if not on_topic:
                            continue
                        # Single sources event from the filtered KB∪web union.
                        merged = update.get("merged") or []
                        used_fallback = update.get("used_fallback", False)
                        n_sources = len(merged)
                        has_kb  = any(m.get("origin") == "kb" for m in merged)
                        has_web = any(m.get("origin") == "web" for m in merged)
                        path = ("hybrid" if has_kb and has_web
                                else "web_search" if has_web
                                else "knowledge_base" if has_kb else "none")
                        yield _sse(SourcesEvent(
                            used_fallback=used_fallback,
                            path=path,
                            sources=[_merged_to_source(m) for m in merged],
                        ))
                        yield _stage("decide", "Choosing the next step…")
                    elif node == "synthesize":
                        msgs = update.get("messages") or []
                        if msgs:
                            final_answer = str(getattr(msgs[-1], "content", "") or "")
                    elif node == "refine":
                        # Self-correction is about to re-synthesize → tell the UI
                        # to discard the ungrounded first draft before new tokens.
                        yield {"event": "reset", "data": "{}"}
                        yield _stage("refine", "Double-checking commands against sources…")
                        streamed_any = False
                        final_answer = ""
                    elif node == "verify":
                        v = update.get("verification") or {}
                        yield _sse(VerificationEvent(
                            grounded_command_ratio=v.get("ratio", 1.0),
                            total=v.get("total", 0),
                            grounded=v.get("grounded", 0),
                            ungrounded=v.get("ungrounded", []),
                        ))

            elif mode == "messages":
                chunk, meta = data
                # Only forward INCREMENTAL chunks from synthesis. A full
                # AIMessage here is a node-returned message LangGraph couldn't
                # dedupe against the token stream — forwarding it would render
                # the whole answer twice.
                if (meta.get("langgraph_node") == "synthesize"
                        and isinstance(chunk, AIMessageChunk)):
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        streamed_any = True
                        yield _sse(TokenEvent(text=text))

        # Safety net: if token streaming produced nothing, emit the whole answer.
        if not streamed_any and final_answer:
            yield _sse(TokenEvent(text=final_answer))

        _record_turn(tracker, user_id, session_id)
        metrics.record_turn_usage(
            llm_calls=tracker.llm_calls, input_tokens=tracker.input_tokens,
            output_tokens=tracker.output_tokens, cost_usd=tracker.turn_cost_usd(),
            duration_s=time.perf_counter() - turn_start)
        metrics.record_request("ok")
        if credits is not None:
            yield _sse(CreditsEvent(weekly_left=credits.weekly_left,
                                    weekly_limit=credits.weekly_limit))
        yield _sse(DoneEvent())
        logger.info("chat ok", extra={"extra_fields": {
            "session": session_id, "user_id": user_id, "n_sources": n_sources,
            "used_fallback": used_fallback, "streamed": streamed_any,
            "llm_calls": tracker.llm_calls, "input_tokens": tracker.input_tokens,
            "output_tokens": tracker.output_tokens,
            "cost_usd": round(tracker.turn_cost_usd(), 6),
            "cost_estimated": tracker.estimated}})

    except Exception as e:  # noqa: BLE001
        error_type = classify_llm_error(e)
        logger.error("chat stream failed", extra={"extra_fields": {
            "session": session_id, "user_id": user_id,
            "error_type": error_type, "error": repr(e)}})
        _record_turn(tracker, user_id, session_id, error_type=error_type)
        metrics.record_turn_usage(
            llm_calls=tracker.llm_calls, input_tokens=tracker.input_tokens,
            output_tokens=tracker.output_tokens, cost_usd=tracker.turn_cost_usd(),
            duration_s=time.perf_counter() - turn_start)
        metrics.record_error(error_type)
        metrics.record_request("error")
        yield _sse(ErrorEvent(message=_STREAM_ERROR_MESSAGES.get(
            error_type, _STREAM_ERROR_MESSAGES["other"])))
        yield _sse(DoneEvent())


@app.post("/chat")
@limiter.limit(settings.rate_limit)
async def chat(request: Request, body: ChatRequest,
               user: Optional[User] = Depends(current_user)):
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")
    if len(msg) > settings.max_message_chars:
        raise HTTPException(status_code=413,
                            detail=f"message exceeds {settings.max_message_chars} chars")

    # Guardrail: token-burner / injection shapes are refused before any
    # retrieval or LLM spend.
    abuse = check_abuse(msg)
    if abuse:
        logger.warning("abusive request blocked", extra={"extra_fields": {
            "session": body.session_id, "abuse_kind": abuse}})
        metrics.record_abuse(abuse)
        metrics.record_request("abuse_blocked")
        raise HTTPException(status_code=400, detail=ABUSE_MESSAGE)

    # Guardrail: PII never leaves the server — placeholders reach the LLM
    # providers and the conversation checkpoint, not the real values.
    if settings.pii_redaction:
        red = redact_text(msg)
        if red.redacted:
            logger.info("pii redacted", extra={"extra_fields": {
                "session": body.session_id, "pii_counts": red.counts}})
            metrics.record_pii(red.counts)
            msg = red.text

    agent = get_agent()
    if agent is None:
        metrics.record_request("agent_unavailable")
        raise HTTPException(status_code=503, detail=f"agent unavailable: {readiness()['reason']}")

    # Credits (Phase 3): 1 weekly credit consumed up front, monthly cost cap
    # checked — but only after the cheap validations and the agent-availability
    # check, so a doomed request never costs a credit. Identity required:
    # auth-disabled dev mode has no user, hence no enforcement.
    credit_status = None
    if user is not None:
        with get_sessionmaker()() as db:
            credit_status = check_and_consume(db, user.user_id)
        if not credit_status.allowed:
            logger.info("credits refused", extra={"extra_fields": {
                "session": body.session_id, "user_id": user.user_id,
                "reason": credit_status.reason,
                "month_cost_usd": round(credit_status.month_cost_usd, 6)}})
            metrics.record_credit_refusal(credit_status.reason or "unknown")
            metrics.record_request("credits_refused")
            raise HTTPException(
                status_code=429,
                detail=WEEKLY_EXHAUSTED_MESSAGE if credit_status.reason == "weekly"
                else MONTHLY_CAP_MESSAGE)

    return EventSourceResponse(_chat_events(
        agent, msg, body.session_id, user_id=user.user_id if user else None,
        credits=credit_status))


@app.post("/feedback")
def feedback(body: FeedbackRequest,
             user: Optional[User] = Depends(current_user)):
    """Record a thumbs up/down on one answer (Phase 4). Idempotent per
    (session, turn): re-rating updates in place. Feeds the Grafana
    bad-feedback panel and the offline eval loop."""
    user_id = user.user_id if user else None
    try:
        with get_sessionmaker()() as db:
            row = db.query(Feedback).filter_by(
                session_id=body.session_id, turn_index=body.turn_index).one_or_none()
            if row is None:
                db.add(Feedback(session_id=body.session_id, turn_index=body.turn_index,
                                user_id=user_id, rating=body.rating))
            else:
                row.rating = body.rating
                row.user_id = user_id
            db.commit()
    except Exception as e:  # noqa: BLE001
        logger.error("feedback write failed", extra={"extra_fields": {
            "session": body.session_id, "error": repr(e)}})
        raise HTTPException(status_code=500, detail="could not record feedback")

    metrics.record_feedback(body.rating)
    logger.info("feedback", extra={"extra_fields": {
        "session": body.session_id, "turn_index": body.turn_index,
        "rating": body.rating, "user_id": user_id}})
    return {"ok": True}


# --- Serve the static frontends if they are present (single-container deploy) -
# API routes are declared above these mounts.  Register /app first so the
# catch-all marketing site at / cannot claim the chat application's assets.
_FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"
_CHAT_FRONTEND = _FRONTEND_ROOT
_LANDING_FRONTEND = _FRONTEND_ROOT / "landing"
if _CHAT_FRONTEND.is_dir():
    @app.get("/app", include_in_schema=False)
    def chat_frontend_root():
        return RedirectResponse(url="/app/", status_code=307)

    app.mount("/app", StaticFiles(directory=str(_CHAT_FRONTEND), html=True),
              name="chat_frontend")
if _LANDING_FRONTEND.is_dir():
    @app.get("/privacy", include_in_schema=False)
    def privacy_page():
        return FileResponse(_LANDING_FRONTEND / "privacy.html")

    @app.get("/terms", include_in_schema=False)
    def terms_page():
        return FileResponse(_LANDING_FRONTEND / "terms.html")

    app.mount("/", StaticFiles(directory=str(_LANDING_FRONTEND), html=True),
              name="landing_frontend")
