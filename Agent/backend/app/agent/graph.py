"""
graph.py — macOS troubleshooting agent (LangGraph + Groq Llama 3.3 70B).

Architecture (from CLAUDE.md):
    User query
      → intake               (lightweight metadata extraction, merged into case file)
      → search_knowledge_base (vector search, metadata pre-filtered by case file)
          ↓ if score < threshold
      → web_search_apple     (Tavily, scoped to apple.com)
      → LLM synthesizes final answer, depth-triage enforced in system prompt

Design principles enforced here:
  1. Extract macOS version, chip, and what-already-tried from ONE user message.
  2. Never lead with Tier 1 solutions if the query signals they've been tried.
  3. RAG first — web search is the fallback, never the default path.
  4. Single LLM (Groq Llama 3.3 70B). No second model for command generation.
  5. Multi-turn: a persistent `case_file` accumulates version/chip/already-tried
     across turns so follow-ups escalate depth instead of restarting.

Usage:
    python graph.py "My wifi drops every 20 minutes on MacBook M2 Ventura"
    python graph.py  # interactive multi-turn session
"""

import json
import os
import re
import sys
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, HumanMessage, RemoveMessage, SystemMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

try:  # imported as app.agent.graph
    from .retrieval import Retriever
    from .providers import make_chat_model, required_key_envs, key_present, resolve_role
    from .taxonomy import normalize_category
    from .verify import check_grounding, context_text_from_merged, extract_commands
    from .prompts import (
        INTAKE_SYSTEM_PROMPT,
        DECIDE_SYSTEM_PROMPT,
        ANSWER_SYSTEM_PROMPT,
        FALLBACK_SYSTEM_PROMPT,
        POST_DIAGNOSTIC_PROMPT,
        POST_CLARIFY_PROMPT,
        SUMMARY_SYSTEM_PROMPT,
    )
except ImportError:  # standalone: python graph.py
    from retrieval import Retriever
    from providers import make_chat_model, required_key_envs, key_present, resolve_role
    from taxonomy import normalize_category
    from verify import check_grounding, context_text_from_merged, extract_commands
    from prompts import (
        INTAKE_SYSTEM_PROMPT,
        DECIDE_SYSTEM_PROMPT,
        ANSWER_SYSTEM_PROMPT,
        FALLBACK_SYSTEM_PROMPT,
        POST_DIAGNOSTIC_PROMPT,
        POST_CLARIFY_PROMPT,
        SUMMARY_SYSTEM_PROMPT,
    )

# ---------------------------------------------------------------------------
# Case file — the accumulating diagnostic memory across conversation turns.
#
# The single most important multi-turn behavior: what we learn in turn 1 (macOS
# version, chip, what the user already tried) must persist and *compound* over
# later turns, so a follow-up like "that didn't work, what else?" escalates the
# depth triage instead of restarting from Tier-1 basics.
# ---------------------------------------------------------------------------

# Sentinel: a case_file update sets pending_diagnostic to this to CLEAR it
# (None means "no change" in the improve-only reducer, so it can't clear).
CLEAR = "__clear__"


def merge_case_file(existing: Optional[dict], update: Optional[dict]) -> dict:
    """
    Improve-only reducer. Concrete new facts override; nulls never wipe a known
    value; `already_tried` is unioned (deduped, order-preserving). Runs whenever
    a node returns a `case_file` update.

    Topic lifecycle: an update carrying `new_problem: True` resets the
    per-problem state (topic, clarify/diagnostic budgets, already_tried,
    category) while keeping the machine facts (version, chip). Without this, a
    second problem in the same session would retrieve against the OLD topic and
    inherit exhausted budgets.
    """
    base = dict(existing or _empty_case_file())
    upd = update or {}

    # --- New problem → reset per-problem state, keep machine facts -----------
    if upd.get("new_problem"):
        base["last_query"]        = None      # let this update's topic take over
        base["category"]          = None
        base["already_tried"]     = []
        base["asked_clarify"]     = False
        base["diagnostic_count"]  = 0
        base["pending_diagnostic"] = None
        base["pending_questions"]  = None

    for key in ("macos_version", "mac_chip"):
        if upd.get(key):
            base[key] = upd[key]

    # last_query anchors on the FIRST substantive problem of the CURRENT topic
    # and is sticky — a terse follow-up reply ("macOS 14.5") must not overwrite
    # the actual topic, or multi-turn retrieval loses the problem entirely.
    # (A new_problem update cleared it above, so the new topic lands here.)
    if upd.get("last_query") and not base.get("last_query"):
        base["last_query"] = upd["last_query"]

    # category: a concrete (non-"general") value overrides; "general" never
    # clobbers an established category.
    new_cat = upd.get("category")
    if new_cat and new_cat != "general":
        base["category"] = new_cat

    # already_tried: union, preserving first-seen order.
    tried = list(base.get("already_tried") or [])
    seen = {t.lower() for t in tried}
    for t in (upd.get("already_tried") or []):
        if t.lower() not in seen:
            tried.append(t)
            seen.add(t.lower())
    base["already_tried"] = tried

    # Agentic budget (per problem): asked_clarify is sticky (at most ONE clarify);
    # diagnostic_count only grows. Both reset above on a new problem.
    base["asked_clarify"] = bool(base.get("asked_clarify")) or bool(upd.get("asked_clarify"))
    base["diagnostic_count"] = max(int(base.get("diagnostic_count") or 0),
                                   int(upd.get("diagnostic_count") or 0))

    # pending_diagnostic / pending_questions: what we asked the user for and are
    # now awaiting. value → set; CLEAR sentinel → clear; absent → keep.
    for key in ("pending_diagnostic", "pending_questions"):
        val = upd.get(key)
        if val == CLEAR:
            base[key] = None
        elif val:
            base[key] = val
    return base


def _empty_case_file() -> dict:
    return {
        "macos_version":      None,
        "mac_chip":           None,
        "category":           None,
        "already_tried":      [],
        "last_query":         None,
        "asked_clarify":      False,  # have we already asked a clarifying question?
        "diagnostic_count":   0,      # how many diagnostic requests we've made
        "pending_diagnostic": None,   # {command, look_for} we're awaiting output for
        "pending_questions":  None,   # [texts] we asked and are awaiting answers to
    }


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:       Annotated[list, add_messages]
    case_file:      Annotated[dict, merge_case_file]  # accumulates across turns
    # Rolling compression of turns older than the keep-window: the transcript's
    # narrative in 2-3 sentences, so long sessions don't grow the prompt
    # unboundedly (cost control). Facts live in case_file; this keeps the story.
    history_summary: str
    intake:         Optional[dict]          # extracted metadata from THIS turn
    # kb_retrieve and web_search run in PARALLEL, so they must write DISJOINT
    # channels (LangGraph forbids concurrent writes to a non-reducer channel).
    kb_results:     Optional[list[dict]]    # Qdrant hits           (written by kb_retrieve)
    kb_low_conf:    bool                     # KB confidence miss    (written by kb_retrieve)
    web_results:    Optional[list[dict]]    # Tavily deep hits      (written by web_search)
    web_ran:        bool                     # did we call Tavily?   (written by web_search)
    merged:         Optional[list[dict]]    # filtered KB∪web context (written by smart_merge)
    used_fallback:  bool                     # answer relies on web  (written by smart_merge)
    verification:   Optional[dict]           # command-grounding report (written by verify)
    # Agentic control (written by decide; consumed by routing + main.py SSE)
    action:         Optional[str]            # "answer" | "clarify" | "diagnose"
    questions:      Optional[list[dict]]     # [{text, options}] (CLARIFY, 1-3)
    diagnostic:     Optional[dict]           # {command, rationale, look_for} (DIAGNOSE)
    refine_count:   int                      # self-correction passes taken (verify loop)


# Continuation/failure signals in a follow-up turn — "what I suggested didn't
# work." Their presence means we should escalate depth rather than repeat.
_FOLLOWUP_FAILURE_SIGNALS = (
    "didn't work", "didnt work", "did not work", "still", "no luck",
    "same issue", "same problem", "same thing", "what else", "tried that",
    "that failed", "not working", "no change", "nothing changed", "already did",
)


def _count_user_turns(messages: list) -> int:
    return sum(1 for m in messages if isinstance(m, HumanMessage))


# Implied-already-tried inference: infer what a user has clearly done from HOW
# they phrase the problem, even when they don't say it. This is the depth-triage
# edge — a frontier model with no such reasoning wastes its first suggestions on
# steps the user is obviously past. Phrases deliberately embed Tier-2 keywords
# (terminal/log/nvram/plist) so the retrieval escalation in kb_retrieve fires.
_IMPLIED_TRIED_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("log show", "console.app", "system.log", "/var/log", "diagnosticreports",
      ".panic", "log stream", "read the log", "checked the log", "in the logs"),
     "read system logs in Terminal/Console (past the GUI checks)"),
    (("sudo ", "in terminal", "ran the command", "command line", "tccutil",
      "defaults write", "diskutil", "pmset", "launchctl", "killall"),
     "ran Terminal commands (past basic Settings toggles)"),
    (("reinstalled", "clean install", "reset nvram", "reset smc", "safe mode",
      "recovery mode", "reset the smc", "reset the nvram"),
     "did advanced resets/reinstall (past restart and toggles)"),
    (("plist", "preference file", "library/preferences", "preferences.plist"),
     "edited preference plists (past the Settings UI)"),
)


def _infer_implied_tried(text: str) -> list[str]:
    """Return implied already-tried steps inferred from the message's phrasing."""
    t = text.lower()
    return [phrase for signals, phrase in _IMPLIED_TRIED_RULES
            if any(s in t for s in signals)]


# ---------------------------------------------------------------------------
# Guardrail helpers — keep a turn alive/clean when an LLM call fails or the
# user pastes a wall of diagnostic output.
# ---------------------------------------------------------------------------

_VERSION_NAMES = {
    "tahoe": "26", "sequoia": "15", "sonoma": "14", "ventura": "13",
    "monterey": "12", "big sur": "11", "catalina": "10.15",
}
_VERSION_NUM_RE = re.compile(r"\b(?:macos|os x|osx|version)\s*(1[0-5]|2[0-6])(?:\.\d+){0,2}\b")
_CHIP_RE = re.compile(r"\bm[1-4]\b|\bapple silicon\b")
_CATEGORY_KEYWORDS = (
    ("wifi", ("wifi", "wi-fi", "wireless", "network drops")),
    ("bluetooth", ("bluetooth", "airpods", "magic mouse", "magic keyboard")),
    ("battery", ("battery", "won't sleep", "wont sleep", "drain", "power")),
    ("disk", ("disk", "storage", "ssd", "drive", "apfs", "mount")),
    ("performance", ("slow", "lag", "beachball", "spinning", "cpu", "memory pressure")),
    ("permissions", ("permission", "tcc", "privacy", "screen recording", "microphone", "camera access")),
)

# Deterministic on-topic backstop: if any of these appear in the raw message,
# it's about a Mac regardless of what the intake LLM's on_topic verdict says.
# A false positive here (wrongly declining a real macOS question) is worse
# than an occasional missed decline, so this only ever forces True, never False.
_ON_TOPIC_SIGNALS = (
    "mac", "macos", "os x", "osx", "imac", "macbook", "wifi", "wi-fi",
    "bluetooth", "airpods", "disk", "storage", "ssd", "apfs", "battery",
    "kernel", "panic", "spotlight", "finder", "safari", "icloud", "keychain",
    "permission", "tcc", "privacy", "terminal", "sudo", "nvram", "smc",
    "diskutil", "launchctl", "defaults", "plist", "crash", "freeze", "beachball",
    "sonoma", "sequoia", "ventura", "monterey", "catalina", "tahoe",
    "m1", "m2", "m3", "m4", "apple silicon", "intel chip", "system settings",
    "system preferences", "menu bar", "dock", "airdrop", "time machine",
)


def _is_on_topic_by_keyword(text: str) -> bool:
    t = text.lower()
    return any(sig in t for sig in _ON_TOPIC_SIGNALS)


def _regex_intake(text: str) -> dict:
    """No-LLM intake fallback so a provider outage never kills the turn."""
    t = text.lower()
    version = next((v for name, v in _VERSION_NAMES.items() if name in t), None)
    if not version:
        m = _VERSION_NUM_RE.search(t)
        version = m.group(1) if m else None
    chip = ("apple_silicon" if _CHIP_RE.search(t)
            else "intel" if "intel" in t else None)
    category = next((cat for cat, kws in _CATEGORY_KEYWORDS
                     if any(k in t for k in kws)), "general")
    return {
        "macos_version": version,
        "mac_chip":      chip,
        "category":      category,
        "already_tried": [],
        "clean_query":   text[:300],
        "new_problem":   False,
    }


# Salient-excerpt extraction for pasted diagnostic output. Embedding a 2-4k-char
# log paste dilutes the retrieval query into noise; the decisive lines (errors,
# denials, drops) plus a hard cap keep the signal.
_SALIENT_LINE_RE = re.compile(
    r"error|fail|panic|denied|deny|timeout|timed out|crash|warn|refus|invalid|"
    r"unable|missing|drop|disconnect|assert|prevent|blocked|corrupt|exceed|dfs|rssi",
    re.IGNORECASE,
)
_PASTE_THRESHOLD = 500   # messages longer than this are treated as pasted output
_EXCERPT_CAP = 300


def _salient_excerpt(text: str, cap: int = _EXCERPT_CAP) -> str:
    """Return the message as-is when short; else its most diagnostic lines."""
    text = text.strip()
    if len(text) <= _PASTE_THRESHOLD:
        return text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    salient = [ln for ln in lines if _SALIENT_LINE_RE.search(ln)]
    picked = salient[:6] if salient else lines[:3]
    return " ".join(picked)[:cap]


_CONTENT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{3,}")


def _looks_like_new_problem(text: str, topic: Optional[str]) -> bool:
    """
    Backstop heuristic for topic-shift when the intake LLM misses it: a
    substantive, prose-length message sharing NO content words with the current
    topic is a different problem. Deliberately conservative — pasted output and
    follow-up failure reports must not trigger it.
    """
    text = text.strip()
    if not topic or not (30 <= len(text) <= _PASTE_THRESHOLD):
        return False
    t = text.lower()
    if any(sig in t for sig in _FOLLOWUP_FAILURE_SIGNALS):
        return False
    topic_toks = set(_CONTENT_TOKEN_RE.findall(topic.lower()))
    text_toks = set(_CONTENT_TOKEN_RE.findall(t))
    return bool(text_toks) and not (topic_toks & text_toks)


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


# ---------------------------------------------------------------------------
# Rolling-summary history compression (cost control).
#
# Keep the last _HISTORY_KEEP messages verbatim; fold anything older into a
# 2-3 sentence running summary via the small intake LLM, then delete the folded
# messages from checkpointed state (RemoveMessage). The keep-window is >= 3
# user turns, so every turn-count threshold in this file (_MAX_TURNS_BEFORE_
# ANSWER, the depth escalation at 3 turns) saturates BEFORE compression can
# affect it — trimming never changes budget behavior.
# ---------------------------------------------------------------------------

_HISTORY_KEEP = 6          # messages kept verbatim (~3 user turns + replies)
_HISTORY_MIN_FOLD = 2      # don't bother summarizing fewer than this
_SUMMARY_MAX_CHARS = 600   # hard cap on the stored summary
_SUMMARY_MSG_CAP = 600     # per-message cap fed to the summarizer (pastes!)


def _compress_history(messages: list, existing_summary: str, llm) -> dict:
    """
    Return a state update folding old turns into `history_summary`, or {} when
    there's nothing to fold (or the summarizer fails — degrade by keeping the
    full transcript this turn and retrying next turn; never lose messages
    without a summary to replace them).
    """
    if len(messages) <= _HISTORY_KEEP + _HISTORY_MIN_FOLD - 1:
        return {}
    old, kept = messages[:-_HISTORY_KEEP], messages[-_HISTORY_KEEP:]
    # Only fold messages that made it into the checkpoint with ids — RemoveMessage
    # targets by id, and an id-less message could never be deleted.
    old = [m for m in old if getattr(m, "id", None)]
    if len(old) < _HISTORY_MIN_FOLD:
        return {}

    lines = []
    for m in old:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = str(m.content).strip()
        if len(text) > _SUMMARY_MSG_CAP:
            text = _salient_excerpt(text, cap=_SUMMARY_MSG_CAP)
        lines.append(f"{role}: {text}")

    prompt = (
        f"EXISTING SUMMARY:\n{existing_summary or '(none)'}\n\n"
        f"TURNS TO FOLD IN:\n" + "\n".join(lines)
    )
    try:
        resp = llm.invoke([
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        summary = str(resp.content).strip()[:_SUMMARY_MAX_CHARS]
    except Exception as e:  # noqa: BLE001 — degrade, never die
        print(f"  [history] summarization failed ({e!r}); keeping full transcript.",
              file=sys.stderr)
        return {}
    if not summary:
        return {}
    return {
        "history_summary": summary,
        "messages": [RemoveMessage(id=m.id) for m in old],
    }


# ---------------------------------------------------------------------------
# Node: intake — extract metadata from the latest turn, merged into the case file
# ---------------------------------------------------------------------------

def node_intake(state: AgentState, llm) -> dict:
    messages = state["messages"]
    user_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if not user_msg:
        return {"intake": {}}

    case = state.get("case_file") or _empty_case_file()
    latest_text = str(user_msg.content)

    # Cost control: fold turns older than the keep-window into the rolling
    # summary (and delete them from state) before this turn's prompts build.
    history_update = _compress_history(
        messages, state.get("history_summary") or "", llm)

    # Are we awaiting the output of a diagnostic we asked the user to run? Then
    # this message IS that output — downstream nodes switch to the post-diagnostic
    # flow (decide prefers ANSWER; synthesize states root cause from the output).
    pending_diag = case.get("pending_diagnostic")
    diagnostic_reply = bool(pending_diag)
    # Same for clarifying questions: this message is the ANSWERS to them.
    clarify_reply = bool(case.get("pending_questions")) and not diagnostic_reply

    # Give intake the prior case file as known context so it only has to extract
    # NEW facts from the latest message (and can resolve a terse follow-up like
    # "still broken" against the established topic). Big pastes are truncated —
    # metadata extraction doesn't need 16k chars of log output.
    known = (
        f"KNOWN SO FAR (from earlier turns — do not lose this):\n"
        f"  macos_version: {case.get('macos_version')}\n"
        f"  mac_chip: {case.get('mac_chip')}\n"
        f"  category: {case.get('category')}\n"
        f"  already_tried: {case.get('already_tried')}\n"
        f"  previous_problem: {case.get('last_query')}\n"
        f"  awaiting_diagnostic_output: {diagnostic_reply}\n"
        f"  awaiting_answers_to_our_questions: {clarify_reply}"
        + (f" (we asked: {case.get('pending_questions')})" if clarify_reply else "")
        + f"\n\nLATEST USER MESSAGE:\n{latest_text[:2000]}"
    )

    # Guardrail: an intake-LLM failure (rate limit, outage) must not kill the
    # turn — fall back to regex extraction and keep going.
    try:
        response = llm.invoke([
            SystemMessage(content=INTAKE_SYSTEM_PROMPT),
            HumanMessage(content=known),
        ])
        raw = re.sub(r"^```(?:json)?\s*", "", str(response.content).strip())
        raw = re.sub(r"\s*```$", "", raw)
        intake = json.loads(raw)
    except Exception as e:  # noqa: BLE001 — degrade, never die
        print(f"  [intake] LLM extraction failed ({e!r}); regex fallback.",
              file=sys.stderr)
        intake = _regex_intake(latest_text)

    # Fold IMPLIED already-tried (inferred from phrasing) into the explicit list.
    explicit_tried = intake.get("already_tried") or []
    implied_tried = _infer_implied_tried(latest_text)
    all_tried = explicit_tried + [t for t in implied_tried if t not in explicit_tried]
    intake["already_tried"] = all_tried   # surfaced in the SSE intake event too
    # The LLM occasionally invents off-vocabulary categories ("display") —
    # normalize into the controlled set so filters/chips stay consistent.
    if intake.get("category"):
        intake["category"] = normalize_category(str(intake["category"]))

    # On-topic classification (anti off-domain guardrail): default True (fail
    # open) when the field is missing — a malformed LLM response or the regex
    # fallback can't judge topic relevance, so never decline on a guess. The
    # keyword backstop can only push False -> True, never the reverse: wrongly
    # declining a real macOS question is worse than occasionally missing an
    # off-topic one.
    on_topic = bool(intake.get("on_topic", True))
    if not on_topic and _is_on_topic_by_keyword(latest_text):
        on_topic = True
    intake["on_topic"] = on_topic

    # Topic lifecycle. The zero-content-overlap HEURISTIC is suppressed while
    # awaiting diagnostic output / question answers (pasted output and terse
    # answers share no tokens with the topic by nature) — but the intake LLM's
    # EXPLICIT new_problem verdict wins even then: a user who ignores our
    # pending request and describes a different problem must not have it
    # misread as command output (stale pending state otherwise poisons the
    # whole turn). An off-topic aside (on_topic=False) never resets the topic
    # either — it's noise, not a new problem, and must not blow away an
    # in-progress diagnostic/clarify session.
    heuristic_new = _looks_like_new_problem(latest_text, case.get("last_query"))
    new_problem = on_topic and (bool(intake.get("new_problem")) or (
        (not diagnostic_reply) and (not clarify_reply) and heuristic_new))
    if new_problem:
        diagnostic_reply = False
        clarify_reply = False
    intake["new_problem"] = new_problem
    intake["diagnostic_reply"] = diagnostic_reply
    intake["clarify_reply"] = clarify_reply
    # Snapshot the questions being answered: decide clears pending_questions
    # this turn, but synthesize still needs them to pair Q with A.
    intake["answered_questions"] = (
        list(case.get("pending_questions") or []) if clarify_reply else [])

    # Build the case-file update. clean_query only overrides the running topic
    # when it's substantive — a terse follow-up shouldn't blank out the topic.
    clean_q = (intake.get("clean_query") or "").strip()
    cf_update = {
        "macos_version": intake.get("macos_version"),
        "mac_chip":      intake.get("mac_chip"),
        "category":      intake.get("category"),
        "already_tried": all_tried,
        "last_query":    clean_q if len(clean_q) >= 12 else None,
        "new_problem":   new_problem,
    }
    # Per-turn channel hygiene: clear last turn's control/verification state so
    # stale checkpointed values (a previous question/diagnostic/report — and
    # refine_count, which would otherwise disable self-correction for the rest
    # of the session) never leak into this turn.
    return {
        "intake": intake,
        "case_file": cf_update,
        "action": None,
        "questions": None,
        "diagnostic": None,
        "verification": None,
        "refine_count": 0,
        **history_update,
    }


# ---------------------------------------------------------------------------
# Node: kb_retrieve — vector search against Qdrant, driven by the case file.
# Runs in PARALLEL with web_search; writes only kb_results + kb_low_conf.
# ---------------------------------------------------------------------------

_TIER2_SIGNALS = ("terminal", "sudo", "defaults", "tccutil", "nvram",
                  "diskutil", "log show", "plist", "kext", "recovery")


def node_kb_retrieve(state: AgentState, retriever: Retriever) -> dict:
    case   = state.get("case_file") or _empty_case_file()
    intake = state.get("intake") or {}
    if not intake.get("on_topic", True):
        # Off-topic (task 2): decide will decline outright — skip the
        # embedding + Qdrant call entirely.
        return {"kb_results": [], "kb_low_conf": True}
    messages = state["messages"]
    user_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    latest_text = str(user_msg.content) if user_msg else ""
    topic = case.get("last_query")   # sticky: the original problem

    # Anchor retrieval on the established problem, folding in any new exact tokens
    # from this turn (command names, flags, error codes) so a follow-up reply like
    # "macOS 14.5" or pasted command output enriches the query rather than
    # replacing the topic. Pasted output is reduced to its salient lines first —
    # embedding a raw 2-4k-char log dump dilutes the query into noise.
    # First turn (no topic yet) → the user's own words.
    latest_salient = _salient_excerpt(latest_text)
    if topic:
        query = topic if latest_salient in topic else f"{topic} {latest_salient}".strip()
    elif len(latest_text.strip()) >= 20:
        query = latest_salient
    else:
        query = intake.get("clean_query") or latest_text

    # Post-diagnostic turn: fold the command we asked the user to run into the
    # query. Its output rarely names the tool, so without this the command's own
    # docs (which ground the FIX) wouldn't surface — the command-aware man-page
    # injection keys off the command name appearing in the query.
    pending = case.get("pending_diagnostic") or {}
    if pending.get("command") and pending["command"] not in query:
        query = f"{query} {pending['command']}".strip()
    macos_version = case.get("macos_version")
    category      = case.get("category")
    already_tried = case.get("already_tried") or []

    # --- Depth-triage escalation ------------------------------------------
    # (1) explicit tool-level signals in what the user has tried → skip Tier 1.
    min_tier = None
    if already_tried and any(s in " ".join(already_tried).lower() for s in _TIER2_SIGNALS):
        min_tier = 2

    # (2) multi-turn escalation: a follow-up that says "that didn't work" means
    # the prior suggestion was tried — climb a tier so we don't repeat it.
    user_turns = _count_user_turns(messages)
    is_followup_failure = user_turns > 1 and any(
        sig in latest_text.lower() for sig in _FOLLOWUP_FAILURE_SIGNALS
    )
    if is_followup_failure:
        min_tier = max(min_tier or 1, 2)
        if user_turns >= 3:
            min_tier = 3   # deep by the third failed attempt

    # Guardrail: an embeddings/Qdrant outage must not kill the turn — degrade to
    # "KB missed" and let the web/fallback path still produce an answer.
    try:
        result = retriever.retrieve(
            query=query,
            macos_version=macos_version,
            category=category,
            n=5,
            min_tier=min_tier,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [kb_retrieve] retrieval failed ({e!r}); continuing without KB.",
              file=sys.stderr)
        return {"kb_results": [], "kb_low_conf": True}

    # kb_low_conf is the retriever's confidence-miss signal — smart_merge decides
    # the final path. (Do NOT write used_fallback here: web_search runs in parallel.)
    return {
        "kb_results":  result["hits"],
        "kb_low_conf": result["fallback"],
    }


# ---------------------------------------------------------------------------
# Node: web_search — Tavily DEEP (advanced) search, run in PARALLEL with KB.
# Gated by WEB_SEARCH_MODE: smart (default) | always | off. Writes web_results
# + web_ran only.
# ---------------------------------------------------------------------------

# Signals that the answer likely needs current/live info beyond the KB's coverage:
# a recent macOS version or explicit recency language. On these, web pulls its
# weight the way a frontier model's browsing would.
_RECENCY_SIGNALS = (
    "latest", "newest", "just updated", "just installed", "after updating",
    "beta", "release candidate", "this version", "current version",
    "sequoia", "15.", "26.",   # Sequoia 15.x and the 2025 "26" naming
)
# KB coverage tops out around macOS 15; queries pinned to >= this lean on the web.
_KB_MAX_MAJOR = 15


def _web_worth_it(case: dict, latest_text: str) -> bool:
    """Smart gate: is a live web search likely to add value for this query?"""
    text = latest_text.lower()
    if any(sig in text for sig in _RECENCY_SIGNALS):
        return True
    ver = (case.get("macos_version") or "").split(".")[0]
    if ver.isdigit() and int(ver) >= _KB_MAX_MAJOR:
        return True
    return False


def node_web_search(state: AgentState) -> dict:
    mode = os.environ.get("WEB_SEARCH_MODE", "smart").strip().lower()
    if mode == "off":
        return {"web_results": [], "web_ran": False}

    case = state.get("case_file") or _empty_case_file()
    intake = state.get("intake") or {}
    if not intake.get("on_topic", True):
        # Off-topic (task 2): decide will decline outright — skip Tavily.
        return {"web_results": [], "web_ran": False}
    user_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    latest_text = str(user_msg.content) if user_msg else ""

    if mode == "smart" and not _web_worth_it(case, latest_text):
        return {"web_results": [], "web_ran": False}

    try:
        from tavily import TavilyClient
    except ImportError:
        return {"web_results": [], "web_ran": False}
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {"web_results": [], "web_ran": False}

    query = case.get("last_query") or intake.get("clean_query") or latest_text

    client = TavilyClient(api_key=api_key)
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",          # Tavily's deep search
            include_answer=False,
            include_raw_content=True,          # fuller page text for grounding
            include_domains=["apple.com", "discussions.apple.com", "support.apple.com"],
            max_results=6,
        )
        hits = [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                # prefer the deeper raw_content, fall back to the snippet
                "content": (r.get("raw_content") or r.get("content") or "")[:1500],
                "score":   r.get("score", 0),
            }
            for r in resp.get("results", [])
        ]
    except Exception as e:
        print(f"  [web_search] Error: {e}", file=sys.stderr)
        hits = []

    return {"web_results": hits, "web_ran": True}


# ---------------------------------------------------------------------------
# Node: smart_merge — fan-in of KB + web. When BOTH are present, an LLM
# relevance-filters and ranks the union (contextual compression) so synthesis
# gets clean, attributed context. When only one source has hits, it passes
# through (no extra LLM call). Writes merged + used_fallback only.
# ---------------------------------------------------------------------------

MERGE_CONTEXT_N = 6   # max sources fed to synthesis

MERGE_SYSTEM = """\
You are a retrieval relevance filter for a macOS troubleshooting assistant.
Given a USER PROBLEM and a numbered list of CANDIDATE SOURCES (from a knowledge
base and/or live web search), decide which sources are actually relevant to
solving THIS problem, and how relevant each is.

Reply with ONLY a JSON object:
{"ranking": [{"id": <int>, "relevance": <0-3>}, ...]}
  relevance 0 = irrelevant/off-topic, 1 = tangential, 2 = relevant,
            3 = directly addresses the problem.
Include every candidate id exactly once. No prose.
"""


def _kb_to_item(h: dict) -> dict:
    return {
        "origin": "kb",
        "title": h.get("title", ""),
        "url": h.get("url", ""),
        "text": h.get("text", "") or "",
        "source": h.get("source", ""),
        "category": h.get("category", ""),
        "difficulty_tier": h.get("difficulty_tier"),
        "score": h.get("score"),
    }


def _web_to_item(h: dict) -> dict:
    return {
        "origin": "web",
        "title": h.get("title", ""),
        "url": h.get("url", ""),
        "text": h.get("content", "") or "",
        "source": "web_search",
        "category": "",
        "difficulty_tier": None,
        "score": h.get("score"),
    }


def _llm_filter_merge(filter_llm, query: str, items: list[dict]) -> list[dict]:
    """LLM relevance-ranks the union of candidates; returns kept items (rel>=1)."""
    listing = "\n".join(
        f"[{i}] ({it['origin'].upper()}) {it['title']}\n{(it['text'] or '')[:400]}"
        for i, it in enumerate(items, 1)
    )
    prompt = f"USER PROBLEM:\n{query}\n\nCANDIDATE SOURCES:\n{listing}"
    try:
        resp = filter_llm.invoke([
            SystemMessage(content=MERGE_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = re.sub(r"^```(?:json)?\s*", "", str(resp.content).strip())
        raw = re.sub(r"\s*```$", "", raw)
        ranking = json.loads(raw).get("ranking", [])
        rel_by_id = {int(r["id"]): int(r.get("relevance", 0)) for r in ranking}
    except Exception:  # noqa: BLE001 — parse OR provider failure
        # Don't lose data: keep everything, unranked. A filter-LLM outage must
        # not kill the turn (the ranking is an optimization, not a requirement).
        return items

    kept = []
    for i, it in enumerate(items, 1):
        rel = rel_by_id.get(i, 1)
        if rel >= 1:
            kept.append({**it, "relevance": rel})
    # highest relevance first; stable within a tier preserves retrieval order.
    kept.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return kept or items   # never return empty if we had candidates


def node_smart_merge(state: AgentState, filter_llm) -> dict:
    kb_hits  = state.get("kb_results") or []
    web_hits = state.get("web_results") or []
    case     = state.get("case_file") or _empty_case_file()
    intake   = state.get("intake") or {}
    user_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = case.get("last_query") or intake.get("clean_query") or (
        str(user_msg.content) if user_msg else "")

    kb_items  = [_kb_to_item(h) for h in kb_hits]
    web_items = [_web_to_item(h) for h in web_hits]

    if kb_items and web_items:
        # Both paths fired → the extra LLM call earns its keep filtering the union.
        merged = _llm_filter_merge(filter_llm, query, kb_items + web_items)
    else:
        merged = kb_items or web_items   # single source → passthrough, no LLM

    merged = merged[:MERGE_CONTEXT_N]

    has_kb  = any(m["origin"] == "kb" for m in merged)
    has_web = any(m["origin"] == "web" for m in merged)
    # "Fallback" now means: no usable KB grounding → the answer leans on the web.
    used_fallback = (not has_kb) and has_web
    return {"merged": merged, "used_fallback": used_fallback}


# ---------------------------------------------------------------------------
# Shared: format the merged KB∪web sources into an attributed context block.
# ---------------------------------------------------------------------------

def _format_context(merged: Optional[list[dict]]) -> str:
    if not merged:
        return "(No relevant sources found.)"
    parts = []
    for i, m in enumerate(merged, 1):
        origin = "KB" if m["origin"] == "kb" else "WEB"
        tier   = m.get("difficulty_tier")
        tier_s = f", tier={tier}" if tier is not None else ""
        parts.append(
            f"[Source {i}] ({origin}, source={m.get('source', '')}{tier_s})\n"
            f"Title: {m.get('title', '')}\nURL: {m.get('url', '')}\n{m.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Node: decide — the agentic brain. Picks ANSWER / CLARIFY / DIAGNOSE.
# This is what makes the system beat a one-shot frontier model: it can ask the
# one key question or request the user's REAL system state instead of guessing.
# ---------------------------------------------------------------------------

_MAX_TURNS_BEFORE_ANSWER = 3
_MAX_DIAGNOSTICS = 2
_MAX_QUESTIONS = 3

# Raw-cosine floor for calling the retrieved context "weak": below this, the KB
# hits are topically adjacent at best and an answer built on them is a guess.
_WEAK_CONTEXT_SCORE = 0.55


def _context_quality(merged: list[dict]) -> str:
    """Deterministic context-quality signal for the planner: NONE/WEAK/GOOD."""
    if not merged:
        return "NONE"
    best = 0.0
    for m in merged:
        s = m.get("raw_score") or m.get("score") or 0.0
        try:
            best = max(best, float(s))
        except (TypeError, ValueError):
            continue
    return "WEAK" if best < _WEAK_CONTEXT_SCORE else "GOOD"


def _normalize_questions(raw) -> list[dict]:
    """Coerce the planner's questions into [{text, options}] (≤3, both clean)."""
    out = []
    for q in (raw or []):
        if isinstance(q, str):
            q = {"text": q}
        if not isinstance(q, dict):
            continue
        text = str(q.get("text") or "").strip()
        if not text:
            continue
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        out.append({"text": text, "options": options[:4]})
        if len(out) >= _MAX_QUESTIONS:
            break
    return out


# Category → the one user-observable question that best discriminates candidate
# causes. Used by the anti-hallucination override when the planner would answer
# with NO grounded context.
_CLARIFY_DISCRIMINATORS: dict[str, dict] = {
    "wifi":        {"text": "Does it happen only on this Wi-Fi network, or on every network?",
                    "options": ["Only this network", "Every network", "Not sure"]},
    "bluetooth":   {"text": "Does it affect one device or all Bluetooth devices?",
                    "options": ["One device", "All devices"]},
    "battery":     {"text": "Did this start after a macOS update or a new app install?",
                    "options": ["After an update", "After a new app", "No idea"]},
    "disk":        {"text": "Is this the internal drive or an external one?",
                    "options": ["Internal", "External"]},
    "performance": {"text": "Is it slow in every app, or one specific app?",
                    "options": ["Everywhere", "One specific app"]},
    "permissions": {"text": "Which app is affected, and which permission won't stick?",
                    "options": []},
}


def _template_questions(case: dict) -> list[dict]:
    """Fallback clarify questions when we must ask but the planner didn't."""
    qs: list[dict] = []
    disc = _CLARIFY_DISCRIMINATORS.get(case.get("category") or "")
    if disc:
        qs.append(dict(disc))
    qs.append({"text": "Paste the exact error message, or describe exactly what "
                       "you see when it happens.", "options": []})
    if not case.get("macos_version") or not case.get("mac_chip"):
        qs.append({"text": "Which macOS version and chip is this Mac on "
                           "(e.g. Sonoma on M2)?",
                   "options": ["Sequoia (15)", "Sonoma (14)", "Ventura (13)",
                               "Intel Mac"]})
    return qs[:_MAX_QUESTIONS]


def node_decide(state: AgentState, planner_llm, retriever=None) -> dict:
    case     = state.get("case_file") or _empty_case_file()
    intake   = state.get("intake") or {}
    merged   = state.get("merged") or []
    messages = state["messages"]
    user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    latest   = str(user_msg.content) if user_msg else ""

    # Off-topic guardrail (task 2): decided deterministically in intake, before
    # any retrieval/planner spend. A stale pending diagnostic/clarify is left
    # untouched (node_decline updates nothing) so a later relevant reply still
    # resumes correctly.
    if not intake.get("on_topic", True):
        return {"action": "decline"}

    turns = _count_user_turns(messages)
    # Budget: once we're deep in the conversation, stop asking and just answer.
    if turns >= _MAX_TURNS_BEFORE_ANSWER:
        return {"action": "answer"}

    # Conversation stage: the planner must know the ORIGINAL problem (the latest
    # message may be a terse reply or pasted output) and whether this turn IS the
    # output of a diagnostic we requested / the answers to our questions.
    topic = case.get("last_query")
    pending = case.get("pending_diagnostic") or {}
    pending_qs = case.get("pending_questions") or []
    consumed_questions = False
    if intake.get("diagnostic_reply") and pending:
        stage = (
            f"STAGE: The user just RAN the diagnostic we requested "
            f"(`{pending.get('command', '')}` — we were looking for: "
            f"{pending.get('look_for', 'n/a')}) and pasted its output as the "
            f"latest message. Interpret the output. Strongly prefer \"answer\" "
            f"(state the root cause + fix) unless the output clearly demands the "
            f"one remaining decisive diagnostic."
        )
    elif intake.get("clarify_reply") and pending_qs:
        consumed_questions = True
        qlist = " | ".join(pending_qs)
        stage = (
            f"STAGE: The user just ANSWERED our clarifying questions "
            f"({qlist}) — the latest message is their answers. You now have the "
            f"context you asked for: choose \"answer\" or \"diagnose\". Do NOT "
            f"ask again."
        )
    else:
        stage = "STAGE: normal turn."

    quality = _context_quality(merged)
    context = _format_context(merged)
    prompt = (
        f"ORIGINAL PROBLEM: {topic or latest}\n\n"
        f"LATEST USER MESSAGE: {_salient_excerpt(latest)}\n\n"
        f"{stage}\n"
        f"CONTEXT QUALITY: {quality}"
        + (" — the sources likely do NOT cover this problem; answering from them"
           " would be a guess. Prefer clarify (or a grounded diagnose)."
           if quality in ("NONE", "WEAK") else "") + "\n\n"
        f"KNOWN: macOS={case.get('macos_version')}, chip={case.get('mac_chip')}, "
        f"already_tried={case.get('already_tried')}, "
        f"asked_clarify={case.get('asked_clarify')}, "
        f"diagnostics_done={case.get('diagnostic_count')}\n\n"
        f"CONTEXT (a diagnostic command MUST come from here):\n{context}\n\n"
        f"Choose the next action. JSON only."
    )
    # The answers-received turn consumed pending_questions; clear regardless of
    # which action we take next (reducer CLEAR sentinel).
    consumed_cf = {"case_file": {"pending_questions": CLEAR}} if consumed_questions else {}
    try:
        resp = planner_llm.invoke([
            SystemMessage(content=DECIDE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = re.sub(r"^```(?:json)?\s*", "", str(resp.content).strip())
        raw = re.sub(r"\s*```$", "", raw)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start != -1 else {}
    except Exception:  # noqa: BLE001 — planner outage → just answer
        return {"action": "answer", **consumed_cf}

    action = (data.get("action") or "answer").lower().strip()

    # --- Anti-hallucination gate: the planner wants to ANSWER but retrieval
    # found NOTHING for this problem. An answer here is a pure guess — ask the
    # discriminating/template questions instead (turn 1 only, budget permitting).
    if (action == "answer" and quality == "NONE" and turns == 1
            and not case.get("asked_clarify")
            and not intake.get("diagnostic_reply")
            and not intake.get("clarify_reply")):
        return {"action": "clarify", "questions": _template_questions(case),
                **consumed_cf}

    # --- Guardrails: enforce the budget the LLM might ignore ---
    if action == "clarify":
        raw_qs = data.get("questions")
        if not raw_qs and data.get("question"):   # legacy single-question shape
            raw_qs = [data["question"]]
        questions = _normalize_questions(raw_qs)
        if case.get("asked_clarify") or not questions:
            return {"action": "answer", **consumed_cf}
        return {"action": "clarify", "questions": questions, **consumed_cf}

    if action == "diagnose":
        diag = data.get("diagnostic") or {}
        cmd = (diag.get("command") or "").strip()
        if case.get("diagnostic_count", 0) >= _MAX_DIAGNOSTICS or not cmd:
            return {"action": "answer"}
        # The diagnostic command MUST be grounded in the retrieved context — never
        # send the user a command we invented.
        out_extra: dict = {}
        report = check_grounding(f"```\n{cmd}\n```", context_text_from_merged(merged))
        if report.get("ungrounded") and retriever is not None:
            # RESCUE retrieval: the planner proposed the right decisive command
            # (it's trained on our DECIDE examples) but its doc didn't make the
            # top-5 for the symptom query. Fetch the command's own docs and
            # re-check, instead of silently downgrading to a blind answer —
            # this was the #1 eval failure (battery answered without
            # `pmset -g assertions`; disk never reached `tmutil`).
            try:
                res = retriever.retrieve(query=cmd, n=3)
                rescue = [_kb_to_item(h) for h in res.get("hits", [])]
            except Exception:  # noqa: BLE001
                rescue = []
            if rescue:
                report = check_grounding(
                    f"```\n{cmd}\n```", context_text_from_merged(merged + rescue))
                if not report.get("ungrounded"):
                    # Keep the rescued docs in context for this + later stages.
                    seen = {(m.get("title"), m.get("url")) for m in merged}
                    extra = [r for r in rescue
                             if (r.get("title"), r.get("url")) not in seen]
                    out_extra["merged"] = merged + extra
        if report.get("ungrounded"):
            return {"action": "answer"}
        return {"action": "diagnose", "diagnostic": {
            "command": cmd,
            "rationale": (diag.get("rationale") or "").strip(),
            "look_for": (diag.get("look_for") or "").strip(),
        }, **out_extra}

    return {"action": "answer"}


# ---------------------------------------------------------------------------
# Node: decline — the query has nothing to do with the macOS troubleshooting
# KB (task 2). Deterministic, no case_file update: any pending diagnostic or
# clarify state from before is left intact for the next, on-topic reply.
# ---------------------------------------------------------------------------

DECLINE_MESSAGE = (
    "I'm sorry, I can't answer that question as it isn't related to macOS "
    "troubleshooting knowledge base I have access to. Ask me about a "
    "specific macOS problem you're having, and I'll do my best to help."
)


def node_decline(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


# ---------------------------------------------------------------------------
# Node: ask_clarify — ask up to 3 targeted questions in ONE turn, end the turn.
# ---------------------------------------------------------------------------

def node_ask_clarify(state: AgentState) -> dict:
    questions = state.get("questions") or [
        {"text": "Could you share a bit more detail about the problem?",
         "options": []}]
    if len(questions) == 1:
        body = questions[0]["text"]
    else:
        body = ("To pin this down (so I can give you the exact fix, not a "
                "guess), a couple of quick questions:\n\n"
                + "\n".join(f"{i}. {q['text']}"
                            for i, q in enumerate(questions, 1)))
    return {
        "messages": [AIMessage(content=body)],
        "case_file": {
            "asked_clarify": True,
            # Remember what we asked — the NEXT user message is the answers,
            # which must not be mistaken for a new problem.
            "pending_questions": [q["text"] for q in questions],
        },
    }


# ---------------------------------------------------------------------------
# Node: request_diagnostic — ask the user to run a grounded command, end the turn.
# ---------------------------------------------------------------------------

def node_request_diagnostic(state: AgentState) -> dict:
    diag = state.get("diagnostic") or {}
    cmd = diag.get("command", "")
    look_for = diag.get("look_for", "")
    # Rationale/look_for are carried separately in the DiagnosticEvent and
    # rendered by the frontend as labeled PURPOSE/LOOKING FOR captions around
    # the code card — keep the streamed body to just the instruction + command
    # so they aren't duplicated as prose.
    body = "Run this and paste the output:\n\n```bash\n" + cmd + "\n```"
    case = state.get("case_file") or _empty_case_file()
    return {
        "messages": [AIMessage(content=body)],
        "case_file": {
            "diagnostic_count": int(case.get("diagnostic_count") or 0) + 1,
            # Remember what we asked for — the NEXT user message is its output,
            # which flips decide/synthesize into the post-diagnostic flow.
            "pending_diagnostic": {"command": cmd, "look_for": look_for},
        },
    }


# ---------------------------------------------------------------------------
# Routing after decide.
# ---------------------------------------------------------------------------

def route_after_decide(state: AgentState) -> str:
    return {"clarify": "ask_clarify", "diagnose": "request_diagnostic",
            "decline": "decline"}.get(state.get("action") or "answer", "synthesize")


# ---------------------------------------------------------------------------
# Node: synthesize — build final answer from KB or web context
# ---------------------------------------------------------------------------

def _apology_answer(merged: Optional[list[dict]]) -> str:
    """Honest degraded answer when every synthesis attempt failed."""
    lines = ["I hit a temporary problem generating your answer — please resend "
             "your message in a moment."]
    titles = [m.get("title") for m in (merged or []) if m.get("title")]
    if titles:
        lines.append("\nIn the meantime, these sources looked most relevant:")
        lines += [f"- {t}" for t in titles[:3]]
    return "\n".join(lines)


def node_synthesize(state: AgentState, llm) -> dict:
    case          = state.get("case_file") or _empty_case_file()
    intake        = state.get("intake") or {}
    merged        = state.get("merged")
    used_fallback = state.get("used_fallback", False)

    messages = state["messages"]
    user_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )

    # Build the context block from the merged KB∪web sources (attributed).
    context = _format_context(merged)

    diagnostic_reply = bool(intake.get("diagnostic_reply"))
    clarify_reply    = bool(intake.get("clarify_reply"))
    pending = case.get("pending_diagnostic") or {}
    answered_qs = intake.get("answered_questions") or []

    # Prompt selection: post-diagnostic / post-clarify turns get root-cause-
    # first prompts (their answers/output ARE the evidence — generic KB advice
    # that ignores them is the hallucination path); web-only grounding →
    # fallback prompt (URL citations); else the grounded prompt.
    if diagnostic_reply:
        system = POST_DIAGNOSTIC_PROMPT
    elif clarify_reply:
        system = POST_CLARIFY_PROMPT
    elif merged and used_fallback:
        system = FALLBACK_SYSTEM_PROMPT
    else:
        system = ANSWER_SYSTEM_PROMPT

    # Build the prompt with metadata context — sourced from the accumulated
    # case file so multi-turn context (version/chip/what's-been-tried) persists.
    version  = case.get("macos_version") or "unknown"
    chip     = case.get("mac_chip") or "unknown"
    tried    = case.get("already_tried") or []
    tried_str = (", ".join(tried)) if tried else "none mentioned"

    # Conversation context: the synthesizer must see the ORIGINAL problem (the
    # latest message may be pasted command output) and what we last told the
    # user, or the final answer can't connect symptom → evidence → fix.
    topic = case.get("last_query") or ""
    last_ai = _last_ai_text(messages)[:400]

    sections = [
        "## User context",
        f"macOS version: {version}",
        f"Mac chip: {chip}",
        f"Already tried: {tried_str}",
    ]
    if topic:
        sections += ["", "## Original problem", topic]
    summary = (state.get("history_summary") or "").strip()
    if summary:
        sections += ["", "## Earlier conversation (compressed)", summary]
    if last_ai:
        sections += ["", "## What we last told the user", last_ai]
    if diagnostic_reply and pending:
        sections += ["", "## Diagnostic the user just ran (latest message is its output)",
                     f"Command: {pending.get('command', '')}",
                     f"We were looking for: {pending.get('look_for', 'n/a')}"]
    if clarify_reply and answered_qs:
        sections += ["", "## Questions we asked (the latest message is their answers)"]
        sections += [f"{i}. {q}" for i, q in enumerate(answered_qs, 1)]
    sections += [
        "",
        "## Latest user message",
        str(user_msg.content) if user_msg else "",
        "",
        "## Relevant KB context",
        context,
    ]
    prompt = "\n".join(sections)

    # Guardrail: a synthesis failure must never bubble up as a bare 500 — the
    # provider fallback chain has already tried the alternates by the time we
    # get here, so degrade to an honest apology naming the sources found.
    try:
        # IMPORTANT: put the LLM's OWN response object into state (not a
        # re-wrapped AIMessage). Its message id matches the streamed chunks, so
        # LangGraph's "messages" stream mode dedupes it — a fresh AIMessage gets
        # RE-EMITTED as one giant chunk and the UI shows the answer twice.
        answer_msg = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
    except Exception as e:  # noqa: BLE001
        print(f"  [synthesize] all providers failed ({e!r}).", file=sys.stderr)
        answer_msg = AIMessage(content=_apology_answer(merged))

    out: dict = {"messages": [answer_msg]}
    if diagnostic_reply:
        # The diagnostic loop is closed — stop treating future messages as output.
        out["case_file"] = {"pending_diagnostic": CLEAR}
    return out


# ---------------------------------------------------------------------------
# Node: verify — check every shell command in the answer is grounded in the
# retrieved context (the anti-hallucination edge over a raw frontier model).
# ---------------------------------------------------------------------------

def node_verify(state: AgentState) -> dict:
    merged = state.get("merged") or []
    answer = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            answer = str(m.content)
            break
    report = check_grounding(answer, context_text_from_merged(merged))
    return {"verification": report}


# ---------------------------------------------------------------------------
# Node: refine — self-correction. When the answer contains ungrounded commands,
# do a targeted re-retrieval for those commands' docs and re-synthesize ONCE.
# ---------------------------------------------------------------------------

def node_refine(state: AgentState, retriever) -> dict:
    report     = state.get("verification") or {}
    ungrounded = report.get("ungrounded") or []
    merged     = list(state.get("merged") or [])

    if ungrounded:
        # Retrieve authoritative docs for the offending commands (command-aware
        # retrieval will inject the relevant man page).
        try:
            res = retriever.retrieve(query=" ".join(ungrounded)[:200], n=3)
            seen = {(m.get("title"), m.get("url")) for m in merged}
            for h in res.get("hits", []):
                item = _kb_to_item(h)
                key = (item.get("title"), item.get("url"))
                if key not in seen:
                    merged.append(item)
                    seen.add(key)
        except Exception:  # noqa: BLE001
            pass

    return {"merged": merged, "refine_count": int(state.get("refine_count") or 0) + 1}


def route_after_verify(state: AgentState) -> str:
    """Self-correct once if commands are ungrounded; otherwise finish."""
    report = state.get("verification") or {}
    if report.get("ungrounded") and int(state.get("refine_count") or 0) == 0:
        return "refine"
    return "end"


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_agent(model: str | None = None, checkpointer=None):
    """
    Build and return a compiled LangGraph agent.

    model : optional override for the SYNTH model (keeps the synth provider from
        env/defaults). Used by the benchmark/CLI. None → provider defaults.
    checkpointer : optional LangGraph checkpointer. When supplied, conversation
        state (messages + case_file) persists per `thread_id`, enabling
        multi-turn memory. Pass one for the API/CLI session; leave None for
        stateless single-shot use (benchmark, `diagnose`).

    Each LLM role (intake, synth) is provider-pluggable via providers.py — set
    e.g. SYNTH_PROVIDER=nvidia to run synthesis on a free NVIDIA model.
    """
    load_dotenv()

    # Provider-aware credential check: only the keys the chosen roles actually need.
    missing = [env for env in required_key_envs(["intake", "synth"]) if not key_present(env)]
    if missing:
        print(f"\n  ✗ Missing provider keys: {', '.join(missing)}. Add them to .env.\n")
        sys.exit(1)

    # Synthesis model — honour an explicit model override on the synth provider.
    # Every role gets the cross-provider fallback chain (with_fallbacks) so a
    # rate-limited/hung provider degrades instead of killing the turn.
    if model:
        synth = resolve_role("synth")
        llm = make_chat_model(provider=synth.provider, model=model, temperature=0,
                              with_fallbacks=True)
    else:
        llm = make_chat_model("synth", temperature=0, with_fallbacks=True)

    # Fast/cheap models for intake JSON extraction and the smart-merge filter.
    intake_llm  = make_chat_model("intake", temperature=0, with_fallbacks=True)
    filter_llm  = make_chat_model("filter", temperature=0, with_fallbacks=True)
    planner_llm = make_chat_model("planner", temperature=0, with_fallbacks=True)

    retriever = Retriever()
    return _build_graph(llm, intake_llm, filter_llm, planner_llm, retriever,
                        checkpointer)


def _build_graph(llm, intake_llm, filter_llm, planner_llm, retriever,
                 checkpointer=None):
    """Wire the agent graph from its components (test seam: inject fakes here)."""
    graph = StateGraph(AgentState)

    graph.add_node("intake",             lambda s: node_intake(s, intake_llm))
    graph.add_node("kb_retrieve",        lambda s: node_kb_retrieve(s, retriever))
    graph.add_node("web_search",         node_web_search)
    graph.add_node("smart_merge",        lambda s: node_smart_merge(s, filter_llm))
    graph.add_node("decide",             lambda s: node_decide(s, planner_llm, retriever))
    graph.add_node("decline",            node_decline)
    graph.add_node("ask_clarify",        node_ask_clarify)
    graph.add_node("request_diagnostic", node_request_diagnostic)
    graph.add_node("synthesize",         lambda s: node_synthesize(s, llm))
    graph.add_node("verify",             node_verify)
    graph.add_node("refine",             lambda s: node_refine(s, retriever))

    graph.set_entry_point("intake")
    # Fan-out: KB retrieval and web search run in PARALLEL from intake …
    graph.add_edge("intake", "kb_retrieve")
    graph.add_edge("intake", "web_search")
    # … fan-in: smart_merge waits for BOTH before the agentic decision.
    graph.add_edge("kb_retrieve", "smart_merge")
    graph.add_edge("web_search",  "smart_merge")
    graph.add_edge("smart_merge", "decide")
    # decide branches: answer → synthesize; else ask the user and end the turn.
    graph.add_conditional_edges("decide", route_after_decide, {
        "synthesize":        "synthesize",
        "ask_clarify":       "ask_clarify",
        "request_diagnostic": "request_diagnostic",
        "decline":           "decline",
    })
    graph.add_edge("ask_clarify", END)
    graph.add_edge("request_diagnostic", END)
    graph.add_edge("decline", END)
    graph.add_edge("synthesize", "verify")     # ground-check the answer's commands
    # Self-correct once if commands are ungrounded, else finish.
    graph.add_conditional_edges("verify", route_after_verify,
                                {"refine": "refine", "end": END})
    graph.add_edge("refine", "synthesize")

    return graph.compile(checkpointer=checkpointer)


def _postgres_checkpointer(database_url: str):
    """
    Build a PostgresSaver on a long-lived connection. Returns None if the
    driver is missing or the connection fails — the caller degrades.

    `PostgresSaver.from_conn_string` is a context-manager generator (same
    trap as SqliteSaver, see below) — construct the connection directly
    instead, with the exact kwargs that helper uses internally
    (autocommit=True, prepare_threshold=0, dict_row) so behavior matches.
    `.setup()` creates/upgrades the checkpointer's OWN tables (checkpoints,
    checkpoint_writes, …) — separate from the Alembic-managed schema — and is
    idempotent (tracks its own migration version), so calling it on every
    process start is safe and required ("MUST be called... the first time").
    """
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row

    # Raw psycopg wants a plain postgresql:// DSN — strip the SQLAlchemy
    # "+psycopg" driver suffix our settings layer adds for the ORM engine.
    dsn = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    conn = psycopg.connect(dsn, autocommit=True, prepare_threshold=0,
                           row_factory=dict_row)
    saver = PostgresSaver(conn)
    saver.setup()
    return saver


def make_memory_checkpointer():
    """
    Return a checkpointer for persistent conversation state.

    Precedence:
      1. Postgres, when DATABASE_URL points at one — the SAME database that
         already holds users/credits/feedback, so conversation memory
         survives restarts/redeploys and is shared across workers (the
         MemorySaver fallback below is per-process RAM only, which silently
         lost every session on every deploy).
      2. An on-disk SQLite store when AGENT_CHECKPOINT_DB is set — a dev
         convenience for persisting across restarts without Postgres.
      3. An in-process MemorySaver (single-instance dev server only).
    Any failure at a given tier degrades to the next rather than raising —
    a broken checkpoint store must never keep the app from serving turns.
    """
    database_url = os.environ.get("DATABASE_URL", "")
    if database_url.startswith("postgresql"):
        try:
            return _postgres_checkpointer(database_url)
        except Exception as e:  # noqa: BLE001
            print(f"  [checkpointer] Postgres unavailable ({e}); "
                  f"falling back.", file=sys.stderr)

    db_path = os.environ.get("AGENT_CHECKPOINT_DB")
    if db_path:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
            # NOTE: SqliteSaver.from_conn_string returns a CONTEXT MANAGER in
            # modern langgraph — construct the saver directly on a long-lived
            # connection instead. check_same_thread=False because FastAPI serves
            # turns from multiple threads.
            return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
        except Exception as e:  # noqa: BLE001
            print(f"  [checkpointer] SQLite unavailable ({e}); using in-memory.",
                  file=sys.stderr)
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def _fresh_state(query: str) -> dict:
    return {
        "messages":      [HumanMessage(content=query)],
        "case_file":     None,
        "history_summary": "",
        "intake":        None,
        "kb_results":    None,
        "kb_low_conf":   False,
        "web_results":   None,
        "web_ran":       False,
        "merged":        None,
        "used_fallback": False,
        "verification":  None,
        "action":        None,
        "questions":     None,
        "diagnostic":    None,
        "refine_count":  0,
    }


def _last_answer(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return "(No response generated)"


# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

def diagnose(query: str) -> str:
    """
    Stateless single-shot entry point. Returns the final answer text.
    Used by the benchmark and quick tests.
    """
    agent = build_agent()
    result = agent.invoke(_fresh_state(query))
    return _last_answer(result)


def chat(agent, query: str, thread_id: str) -> dict:
    """
    Multi-turn entry point. `agent` must have been built WITH a checkpointer.
    Only the new user turn is sent; prior messages + case_file are restored from
    the checkpoint keyed by `thread_id`. Returns the final state dict (answer +
    intake + kb_results + case_file), so callers can surface sources/metadata.
    """
    config = {"configurable": {"thread_id": thread_id}}
    # On a follow-up turn we must NOT resend case_file=None (that's an input the
    # reducer would fold in harmlessly, but we also must not reset messages);
    # sending just the new HumanMessage lets add_messages append it.
    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return {
        "answer":        _last_answer(result),
        "intake":        result.get("intake") or {},
        "case_file":     result.get("case_file") or _empty_case_file(),
        "kb_results":    result.get("kb_results") or [],
        "web_results":   result.get("web_results") or [],
        "merged":        result.get("merged") or [],
        "web_ran":       result.get("web_ran", False),
        "used_fallback": result.get("used_fallback", False),
        "verification":  result.get("verification") or {},
        "action":        result.get("action") or "answer",
        "questions":     result.get("questions"),
        "diagnostic":    result.get("diagnostic"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="macOS troubleshooting agent — type a problem, get a fix"
    )
    parser.add_argument("query", nargs="?", default=None, help="The problem to diagnose")
    parser.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model to use")
    parser.add_argument("--debug", action="store_true", help="Print intermediate state")
    args = parser.parse_args()

    load_dotenv()
    # Interactive CLI is a single continuous conversation → give it memory.
    agent = build_agent(model=args.model, checkpointer=make_memory_checkpointer())
    thread_id = "cli-session"

    def run_query(q: str):
        print(f"\nDiagnosing: {q}\n{'='*60}")
        out = chat(agent, q, thread_id=thread_id)

        if args.debug:
            print(f"\n[DEBUG] intake: {json.dumps(out['intake'], indent=2)}")
            print(f"[DEBUG] case_file: {json.dumps(out['case_file'], indent=2)}")
            print(f"[DEBUG] kb hits: {len(out['kb_results'])}  "
                  f"web hits: {len(out['web_results'])}  "
                  f"used_fallback: {out['used_fallback']}\n")

        print(out["answer"])

    if args.query:
        run_query(args.query)
    else:
        print("macOS Troubleshooting Agent — multi-turn session (Ctrl+C to exit)\n")
        while True:
            try:
                q = input("Problem: ").strip()
                if q:
                    run_query(q)
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break


if __name__ == "__main__":
    main()
