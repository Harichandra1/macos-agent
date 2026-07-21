"""
schemas.py — request/response + SSE event models for the chat API.

The /chat endpoint streams Server-Sent Events. Each event has a named `type`
and a JSON `data` payload; the frontend switches on `type`:

  intake  — extracted/accumulated case file (version, chip, already-tried).
  sources — retrieved KB citations (or web fallback), plus which path was used.
  token   — one incremental chunk of the answer text.
  done    — terminal marker; the answer is complete.
  error   — something failed; `message` explains.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's macOS problem or follow-up.")
    session_id: str = Field(..., min_length=1, description="Stable id tying turns into one conversation.")


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=0, description="0-based index of the rated assistant answer in this session.")
    rating: Literal["up", "down"]


class Source(BaseModel):
    origin: Literal["kb", "web"] = "kb"   # knowledge base vs live web
    title: str = ""
    url: Optional[str] = None
    source: str = ""            # apple_support, ask_different, web_search, …
    category: str = ""
    difficulty_tier: Optional[int] = None
    score: Optional[float] = None
    snippet: str = ""


class IntakeEvent(BaseModel):
    type: Literal["intake"] = "intake"
    macos_version: Optional[str] = None
    mac_chip: Optional[str] = None
    category: Optional[str] = None
    already_tried: list[str] = []


class SourcesEvent(BaseModel):
    type: Literal["sources"] = "sources"
    used_fallback: bool = False
    path: Literal["knowledge_base", "web_search", "hybrid", "none"] = "knowledge_base"
    sources: list[Source] = []


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class VerificationEvent(BaseModel):
    type: Literal["verification"] = "verification"
    grounded_command_ratio: float = 1.0   # 1.0 = every command found in sources
    total: int = 0
    grounded: int = 0
    ungrounded: list[str] = []            # commands NOT found in retrieved context


class QuestionItem(BaseModel):
    """One clarifying question, optionally with quick-reply choices."""
    text: str
    options: list[str] = []


class QuestionEvent(BaseModel):
    """The agent needs 1-3 clarifying facts before it can diagnose precisely.
    `question` keeps the joined text for back-compat; `questions` carries the
    structured items (the frontend renders `options` as quick-reply chips)."""
    type: Literal["question"] = "question"
    question: str = ""
    questions: list[QuestionItem] = []


class DiagnosticEvent(BaseModel):
    """The agent asks the user to run a grounded command and paste the output."""
    type: Literal["diagnostic"] = "diagnostic"
    command: str
    rationale: str = ""
    look_for: str = ""


class StatusEvent(BaseModel):
    """Progress heartbeat while the graph runs — the UI shows a stage line
    instead of a blank cursor (retrieval + planning take seconds)."""
    type: Literal["status"] = "status"
    stage: str          # machine-readable node name
    label: str          # human-readable, e.g. "Searching the knowledge base…"


class CreditsEvent(BaseModel):
    """Post-turn allowance snapshot (authenticated users only) — the UI shows
    a remaining-messages pill so exhaustion never comes as a surprise."""
    type: Literal["credits"] = "credits"
    weekly_left: int
    weekly_limit: int


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
