"""
guardrails.py — input guardrails applied at the /chat boundary (Phase 2).

Two concerns, both cheap and deterministic (no LLM, no external calls):

1. PII redaction (`redact_text`). The user's message is the ONLY free-form
   text we forward to external services (LLM providers, the embedding API,
   Tavily) — and macOS diagnostic pastes are full of identifiers: emails,
   MAC/BSSID addresses in `wdutil` output, IPs, `/Users/<name>` paths in every
   log line, hardware serials in system reports. Redaction runs BEFORE the
   graph sees the message, so placeholders — not the real values — reach the
   providers AND the conversation checkpoint. Regex deny-list by design:
   Presidio-class NER costs hundreds of MB of models — wrong trade for a
   free-tier deployment (v2.0 plan, Phase 2).

2. Prompt-abuse validation (`check_abuse`). Blocks the classic token-burner
   shapes (repeat-N-times, essay/HTML generation, ignore-your-instructions)
   with a clean 400 before any retrieval or LLM spend. Deliberately narrow:
   the intake off-topic classifier already declines general non-macOS asks,
   so this list only needs the requests designed to waste output tokens.
   False positives on real troubleshooting messages are worse than misses.

Both are pure functions over the message text — unit-testable without the app.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

# Order matters: MAC before IPv6 (a MAC is a valid IPv6-pattern match), and
# anchored serials before generic patterns so the anchor text survives.
_PII_PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("email",
     re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
     "[EMAIL]"),
    ("mac_address",
     re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
     "[MAC]"),
    ("ipv6",
     # 4+ hex groups or a '::' compression — 3 groups would swallow HH:MM:SS
     # timestamps that appear in every log line.
     re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){4,7}[0-9A-Fa-f]{1,4}\b"
                r"|\b[0-9A-Fa-f]{0,4}::(?:[0-9A-Fa-f]{1,4}:){0,5}[0-9A-Fa-f]{1,4}\b"),
     "[IPV6]"),
    ("ipv4",
     # All four octets required: macOS versions ("14.4.1", "10.15.7") are at
     # most three components and must NOT be redacted.
     re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"),
     "[IPV4]"),
    ("serial_number",
     # Anchored on a "serial ..." label — bare 10-12 char alphanumerics would
     # false-positive on flags, hashes, and error codes everywhere.
     re.compile(r"(?i)(serial\s*(?:number|no\.?|#)?\s*[:=]?\s+)([A-Z0-9]{8,17})\b"),
     r"\1[SERIAL]"),
    ("user_path",
     # /Users/<account> appears in nearly every pasted log line and names the
     # person. Keep the path shape so commands/answers stay coherent.
     re.compile(r"(/Users/)(?!Shared\b)([A-Za-z0-9._-]+)"),
     r"\1[USER]"),
)


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)   # kind -> occurrences

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


def redact_text(text: str) -> RedactionResult:
    """Replace PII with typed placeholders; report what was found (counts only —
    never log or return the original values)."""
    counts: dict[str, int] = {}
    for kind, pattern, replacement in _PII_PATTERNS:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[kind] = n
    return RedactionResult(text=text, counts=counts)


# ---------------------------------------------------------------------------
# Prompt-abuse validation
# ---------------------------------------------------------------------------

_ABUSE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("repeat_flood",
     re.compile(r"(?i)\b(?:repeat|say|print|write|type|output)\b.{0,50}"
                r"\b\d{3,}(?:,\d{3})*\s*times\b")),
    ("repeat_forever",
     re.compile(r"(?i)\b(?:repeat|say|print|write|output|loop)\b.{0,50}"
                r"\b(?:forever|endlessly|infinitely|non[- ]?stop|"
                r"in an? (?:infinite|endless) loop)\b")),
    ("bulk_prose",
     re.compile(r"(?i)\bwrite\b.{0,40}\b(?:essay|story|poem|song|novel|"
                r"screenplay|blog ?post|article)\b")),
    ("web_generation",
     re.compile(r"(?i)\b(?:generate|create|build|write|make)\b.{0,50}"
                r"\b(?:html|css|web ?site|web ?page|landing page|react app)\b")),
    ("injection",
     re.compile(r"(?i)\b(?:ignore|disregard|forget)\b.{0,30}"
                r"\b(?:previous|prior|above|all|your|earlier)\b.{0,30}"
                r"\b(?:instructions?|rules?|prompts?|system)\b")),
)

ABUSE_MESSAGE = (
    "I can only help with macOS troubleshooting — that request looks like "
    "it's asking me to generate unrelated or repeated content. Describe a "
    "specific Mac problem and I'll get to work."
)


def check_abuse(text: str) -> str | None:
    """Return the matched abuse kind, or None when the message is fine."""
    for kind, pattern in _ABUSE_PATTERNS:
        if pattern.search(text):
            return kind
    return None
