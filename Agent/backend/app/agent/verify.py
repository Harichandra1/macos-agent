"""
verify.py — command-grounding verification.

Frontier models' #1 failure on this domain is emitting confident but WRONG shell
commands / flags / paths. Our edge is that every command should be grounded in
retrieved context. This module extracts the shell commands and file paths from a
synthesized answer and checks each against the merged KB∪web context.

It does NOT call an LLM — grounding is a string/token problem (per CLAUDE.md:
"diagnose command-accuracy as a retrieval/grounding issue first"). The result is
a `grounded_command_ratio` sub-metric and a list of ungrounded commands the UI can
flag as unverified.
"""

import re

# macOS troubleshooting tools whose presence marks a line as a "command".
_TOOLS = (
    "sudo", "defaults", "tccutil", "diskutil", "pmset", "launchctl", "log",
    "xattr", "spctl", "killall", "nvram", "csrutil", "mdutil", "scutil",
    "networksetup", "systemsetup", "kextstat", "kmutil", "bless", "fsck",
    "dscl", "profiles", "softwareupdate", "plutil", "codesign", "wdutil",
    "system_profiler", "ioreg", "powermetrics", "tmutil", "mdfind", "hdiutil",
    "dscacheutil", "sysctl", "airport", "syslog", "osascript", "caffeinate",
)
_TOOL_RE = re.compile(r"\b(" + "|".join(_TOOLS) + r")\b")
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?(.*?)```", re.DOTALL)
_PATH_RE = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}|~/[A-Za-z0-9._/-]+")
# tokens to ignore when scoring overlap (shell noise / common words)
_STOP = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "with",
         "your", "you", "run", "then", "this", "that", "it", "is", "sudo"}


def extract_commands(answer: str) -> list[str]:
    """Pull candidate shell commands + file paths out of an answer."""
    candidates: list[str] = []

    # 1. lines inside fenced code blocks
    for block in _FENCE_RE.findall(answer):
        for line in block.splitlines():
            line = line.strip().lstrip("$ ").strip()
            if line and (_TOOL_RE.search(line) or _PATH_RE.search(line)):
                candidates.append(line)

    # 2. inline `code` spans and prose lines that name a tool
    answer_no_fences = _FENCE_RE.sub(" ", answer)
    for span in re.findall(r"`([^`]+)`", answer_no_fences):
        span = span.strip()
        if _TOOL_RE.search(span) or _PATH_RE.search(span):
            candidates.append(span)

    # dedupe, preserve order, cap length of each
    seen, out = set(), []
    for c in candidates:
        c = c[:200]
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def _tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9._/-]+", s.lower())
            if len(t) >= 3 and t not in _STOP]


def _is_grounded(command: str, context_lower: str) -> bool:
    """
    Grounded if the command's tool appears in context AND enough of its
    meaningful tokens also appear. Lenient on argument VALUES (bundle ids,
    device names vary) but strict enough to catch a hallucinated tool or flag.
    """
    # `sudo` is a prefix, not the command — find the actual tool after it.
    tools = [t for t in _TOOL_RE.findall(command) if t != "sudo"]
    # Word-boundary match: "log" must not count as grounded because the context
    # contains "dialog" or "login".
    if tools and not re.search(rf"\b{re.escape(tools[0])}\b", context_lower):
        return False  # named tool not present in sources → not grounded

    # Subcommand check: token overlap alone lets an INVENTED subcommand pass
    # when its words appear separately in context (e.g. `wdutil set 149` when
    # the context only ever says `wdutil info` and mentions channel 149). The
    # tool+subcommand bigram must appear verbatim.
    if tools:
        words = command.replace("sudo", " ").split()
        try:
            idx = words.index(tools[0])
            sub = words[idx + 1] if idx + 1 < len(words) else ""
        except ValueError:
            sub = ""
        if sub and re.fullmatch(r"[a-z][a-z-]*", sub):   # subcommand, not a flag/path
            ctx_norm = re.sub(r"\s+", " ", context_lower)
            if f"{tools[0]} {sub}" not in ctx_norm:
                return False

    toks = _tokens(command)
    if not toks:
        return True
    present = sum(1 for t in toks if t in context_lower)
    return (present / len(toks)) >= 0.5


def check_grounding(answer: str, context: str) -> dict:
    """
    Returns a verification report:
      {total, grounded, ungrounded: [commands], ratio}
    ratio is 1.0 when there are no commands to check (vacuously grounded).
    """
    commands = extract_commands(answer)
    ctx = (context or "").lower()
    if not commands:
        return {"total": 0, "grounded": 0, "ungrounded": [], "ratio": 1.0}

    ungrounded = [c for c in commands if not _is_grounded(c, ctx)]
    grounded = len(commands) - len(ungrounded)
    return {
        "total": len(commands),
        "grounded": grounded,
        "ungrounded": ungrounded,
        "ratio": round(grounded / len(commands), 3),
    }


def context_text_from_merged(merged: list[dict]) -> str:
    """Concatenate merged-source texts into one blob for grounding checks."""
    return "\n".join((m.get("text", "") or "") for m in (merged or []))
