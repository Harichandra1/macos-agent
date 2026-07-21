"""
Stack Overflow (macos tag) scraper for the RAG knowledge base.

Uses the Stack Exchange API v2.3 to fetch macos-tagged questions that have
accepted answers.  Produces Q&A pair documents in the same format as
ask_different_scraper.py (Problem / Solution sections).

Quality filters (CLAUDE.md Tier 3 spec):
  - Tagged 'macos'
  - Question Score >= 2
  - Has accepted answer (accepted=True search filter)
  - Accepted answer Score >= 1

Optional API key (raises daily quota from 300 to 10,000 requests/day):
  export SO_API_KEY=your_key_here
  Register at: https://stackapps.com/apps/oauth/register
  Check remaining quota in the output — the API returns it on every call.

Two-phase approach:
  Phase 1 — /search endpoint: tagged=macos, accepted=True, sort=votes
             → collect (question_id, title, body, tags, accepted_answer_id)
  Phase 2 — /answers/{ids} endpoint: batch-fetch accepted answer bodies

Source value: "stackoverflow"

Usage:
  python stackoverflow_scraper.py
  python main.py --step 7
"""

import html as _html_mod
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import requests
from bs4 import BeautifulSoup

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_Q_SCORE = 2
MIN_A_SCORE = 1
MAX_PAGES   = 50      # cap: 50 pages × 100 questions = up to 5 000 questions
DELAY       = 1.5     # seconds between API calls (generous — we want to share quota)

SE_API_BASE = "https://api.stackexchange.com/2.3"
SITE        = "stackoverflow"
TAG         = "macos"

# ---------------------------------------------------------------------------
# Category inference (same controlled vocab as all other scrapers)
# ---------------------------------------------------------------------------

_KEYWORD_CATEGORY: Dict[str, str] = {
    "bluetooth":        "bluetooth",
    "wifi":             "wifi", "wi-fi": "wifi", "network": "wifi",
    "ethernet":         "wifi", "vpn": "wifi", "socket": "wifi",
    "disk":             "disk", "apfs": "disk", "storage": "disk",
    "time machine":     "disk", "hfs": "disk", "ntfs": "disk",
    "battery":          "battery", "power": "battery", "sleep": "battery",
    "performance":      "performance", "memory": "performance", "cpu": "performance",
    "thread":           "performance", "hang": "performance",
    "permission":       "permissions", "privacy": "permissions",
    "tcc":              "permissions", "gatekeeper": "permissions",
    "code sign":        "permissions", "entitlement": "permissions",
    "keychain":         "permissions", "sandbox": "permissions",
    "crash":            "diagnostics", "log":     "diagnostics",
    "kernel":           "diagnostics", "panic":   "diagnostics",
    "diagnostic":       "diagnostics", "console": "diagnostics",
    "startup":          "system",  "boot":       "system",
    "launchctl":        "system",  "launchd":    "system",
    "defaults write":   "system",  "plist":      "system",
    "spotlight":        "system",  "finder":     "system",
}


def _infer_category(text: str, tags: List[str] = None) -> str:
    t = text.lower()
    # check SO tags first (they're curated)
    for tag in (tags or []):
        cat = _KEYWORD_CATEGORY.get(tag)
        if cat:
            return cat
    for kw, cat in _KEYWORD_CATEGORY.items():
        if kw in t:
            return cat
    return "general"


# ---------------------------------------------------------------------------
# macOS version extraction
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(
    r"macos\s+(\d+(?:\.\d+)*)"
    r"|(?:sequoia|sonoma|ventura|monterey|big\s*sur|catalina|mojave|high\s*sierra)",
    re.IGNORECASE,
)
_NAME_VERSION = {
    "sequoia": "15", "sonoma": "14", "ventura": "13", "monterey": "12",
    "bigsur": "11", "big-sur": "11",
    "catalina": "10.15", "mojave": "10.14", "highsierra": "10.13",
}

# SE tags that encode version (e.g. "macos-sonoma")
_SE_VERSION_TAGS: Dict[str, str] = {
    "macos-sequoia": "15", "macos-sonoma": "14", "macos-ventura": "13",
    "macos-monterey": "12", "macos-big-sur": "11", "macos-catalina": "10.15",
    "macos-mojave": "10.14", "macos-high-sierra": "10.13",
}


def _extract_versions(text: str, tags: List[str] = None) -> List[str]:
    versions: set = set()
    for tag in (tags or []):
        v = _SE_VERSION_TAGS.get(tag)
        if v:
            versions.add(v)
    for m in _VERSION_RE.finditer(text):
        if m.group(1):
            versions.add(m.group(1))
        else:
            name = m.group(0).lower().replace(" ", "").replace("-", "")
            v = _NAME_VERSION.get(name)
            if v:
                versions.add(v)
    return sorted(versions) if versions else ["all"]


# ---------------------------------------------------------------------------
# Difficulty tier inference (applied to accepted answer body)
# ---------------------------------------------------------------------------

_TIER3 = (
    "log show --predicate", "kernel extension", "kext",
    "recovery mode", "smc reset", "csrutil", "codesign",
    "entitlements", "/dev/", "iokit", "dtrace", "kernel panic",
)
_TIER2 = (
    "terminal", "sudo ", "defaults write", "defaults delete",
    "tccutil", "launchctl", "nvram ", "diskutil ", "pmset ",
    "/etc/", "/var/", "/usr/", "/Library/", "log show",
    "command line", "chmod ", "chown ",
)


def _infer_difficulty(answer_text: str) -> int:
    t = answer_text.lower()
    if any(s in t for s in _TIER3): return 3
    if any(s in t for s in _TIER2): return 2
    return 1


# ---------------------------------------------------------------------------
# HTML → plain text (same pattern as ask_different_scraper.py)
# ---------------------------------------------------------------------------

def _html_to_text(body_html: str) -> str:
    soup = BeautifulSoup(body_html, "html.parser")
    for tag in soup.find_all(["code", "pre"]):
        tag.replace_with(f" `{tag.get_text()}` ")
    return " ".join(soup.get_text(separator=" ", strip=True).split())


# ---------------------------------------------------------------------------
# Stack Exchange API client
# ---------------------------------------------------------------------------

class SEApiClient:
    """
    Thin wrapper around the SE API v2.3.

    Handles:
    - Optional API key injection (SO_API_KEY env var)
    - Gzip decompression (automatic via requests)
    - 'backoff' field compliance (the API can demand a wait)
    - Quota tracking with warnings
    """

    def __init__(self, api_key: Optional[str] = None, delay: float = DELAY):
        self.api_key  = api_key or os.environ.get("SO_API_KEY")
        self.delay    = delay
        self.session  = requests.Session()
        # SE API returns gzip; requests decodes automatically
        self.session.headers.update({"Accept-Encoding": "gzip, deflate"})
        self._quota_remaining: Optional[int] = None

    def get(self, endpoint: str, **params) -> dict:
        """GET {SE_API_BASE}/{endpoint} with automatic backoff and quota tracking."""
        params["site"] = SITE
        if self.api_key:
            params["key"] = self.api_key

        r = self.session.get(f"{SE_API_BASE}/{endpoint}", params=params, timeout=20)
        if r.status_code == 400:
            # SE API returns 400 when the search pagination window is exhausted
            # (~25 pages on the free tier). Treat as "no more results".
            return {"items": [], "has_more": False, "quota_remaining": self._quota_remaining}
        r.raise_for_status()
        data = r.json()

        # Honour 'backoff' field — the API demands we wait before next call
        backoff = data.get("backoff", 0)
        if backoff:
            print(f"  [SE API] backoff requested: {backoff}s")
            time.sleep(backoff)

        self._quota_remaining = data.get("quota_remaining")
        if self._quota_remaining is not None and self._quota_remaining < 20:
            print(f"  ⚠ SE API quota almost exhausted: {self._quota_remaining} remaining")

        return data

    def iter_questions(
        self,
        max_pages: int = MAX_PAGES,
    ) -> Iterator[dict]:
        """
        Yield macos-tagged questions with accepted answers, sorted by votes.
        Uses /search?accepted=True so every returned question has
        an accepted_answer_id.
        """
        page = 1
        while page <= max_pages:
            data = self.get(
                "search",
                tagged=TAG,
                accepted="True",
                sort="votes",
                order="desc",
                min=MIN_Q_SCORE,
                filter="withbody",    # includes .body and accepted_answer_id
                pagesize=100,
                page=page,
            )
            items = data.get("items", [])
            if not items:
                break

            for q in items:
                if q.get("accepted_answer_id"):
                    yield q

            if not data.get("has_more"):
                break

            page += 1
            time.sleep(self.delay)

    def fetch_answers(self, answer_ids: List[int]) -> Dict[int, dict]:
        """
        Batch-fetch answer bodies for a list of answer IDs.
        SE API allows up to 100 IDs per request (semicolon-separated).
        Paginates with pagesize=100 so a single 100-ID batch always
        resolves in one request (or two if very long answers trigger backoff).
        Returns {answer_id: answer_dict}.
        """
        answers: Dict[int, dict] = {}
        for i in range(0, len(answer_ids), 100):
            batch   = answer_ids[i : i + 100]
            ids_str = ";".join(str(a) for a in batch)
            page    = 1
            while True:
                data = self.get(
                    f"answers/{ids_str}",
                    filter="withbody",
                    sort="creation",
                    order="asc",
                    pagesize=100,
                    page=page,
                )
                for ans in data.get("items", []):
                    answers[ans["answer_id"]] = ans
                if not data.get("has_more"):
                    break
                page += 1
                time.sleep(self.delay)
            time.sleep(self.delay)
        return answers


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_kb_document(question: dict, answer: dict) -> Optional[Dict]:
    """Convert a (question, accepted answer) pair into a KB article."""
    q_id       = question["question_id"]
    title      = _html_mod.unescape(question.get("title", ""))
    tags       = question.get("tags", [])
    q_body_html = question.get("body", "")
    a_body_html = answer.get("body", "")

    q_text = _html_to_text(q_body_html)
    a_text = _html_to_text(a_body_html)

    if not q_text or not a_text:
        return None

    combined   = f"{title} {q_text} {a_text}"
    category   = normalize_category(_infer_category(combined, tags))
    versions   = _extract_versions(combined, tags)
    difficulty = _infer_difficulty(a_body_html)

    embed_text = normalize_embed_text(
        f"{title}. {q_text} {a_text}",
        title=title,
        category=category,
    )
    if len(embed_text) < 100:
        return None

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "id":               f"so_{q_id}",
        "article_id":       f"SO_{q_id}",
        "locale":           "en-us",
        "title":            title,
        "url":              question.get("link", f"https://stackoverflow.com/q/{q_id}"),
        "scraped_at":       now,
        "last_modified":    None,
        "affected_devices": ["Mac"],
        "macos_versions":   versions,
        "difficulty_tier":  difficulty,
        "category":         category,
        "categories":       ["Stack Overflow", "Developer Q&A"],
        "summary":          q_text[:300],
        "sections": [
            {"heading": "Problem",  "body": q_text},
            {"heading": "Solution", "body": a_text},
        ],
        "steps":      [],
        "embed_text": embed_text,
        "source":     "stackoverflow",
    }


# ---------------------------------------------------------------------------
# Scraper class — pipeline integration
# ---------------------------------------------------------------------------

class StackOverflowScraper:
    """
    Fetches macos-tagged SO questions with accepted answers via the SE API
    and writes Q&A pair documents to the KnowledgeBase.

    Interface: scrape_all(kb) → int (saved count).
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        delay:      float         = DELAY,
        max_pages:  int           = MAX_PAGES,
        min_q_score: int          = MIN_Q_SCORE,
        min_a_score: int          = MIN_A_SCORE,
    ):
        self.min_a_score = min_a_score
        self.max_pages   = max_pages
        self.client      = SEApiClient(api_key=api_key, delay=delay)

        # Override module-level threshold if caller specifies
        global MIN_Q_SCORE
        MIN_Q_SCORE = min_q_score

    def scrape_all(self, kb: KnowledgeBase) -> int:
        using_key = bool(self.client.api_key)
        print(
            f"  SE API key: {'yes (10,000 req/day quota)' if using_key else 'no (300 req/day quota)'}"
        )
        if not using_key:
            print(
                "  Tip: set SO_API_KEY for a much higher rate limit.\n"
                "  Register at https://stackapps.com/apps/oauth/register"
            )

        # Phase 1 — collect questions
        print("\n  [Phase 1] Fetching questions (macos, accepted=True, score≥2)...")
        questions:    Dict[int, dict] = {}
        answer_ids:   List[int]       = []

        for q in self.client.iter_questions(max_pages=self.max_pages):
            q_id  = q["question_id"]
            a_id  = q["accepted_answer_id"]
            aid   = f"SO_{q_id}"

            if kb.already_scraped(aid):
                continue

            questions[q_id]  = q
            answer_ids.append(a_id)

        print(f"  → {len(questions):,} new questions, {len(answer_ids):,} answers to fetch")

        if not questions:
            print("  Nothing new to save.")
            return 0

        if self.client._quota_remaining is not None:
            print(f"  Quota remaining after Phase 1: {self.client._quota_remaining}")

        # Phase 2 — batch-fetch accepted answer bodies
        print("\n  [Phase 2] Batch-fetching accepted answers...")
        answers = self.client.fetch_answers(answer_ids)
        print(f"  → {len(answers):,} answers retrieved")

        if self.client._quota_remaining is not None:
            print(f"  Quota remaining after Phase 2: {self.client._quota_remaining}")

        # Build documents
        saved = skipped = failed = 0
        # Map accepted_answer_id back to question
        answer_id_to_q = {q["accepted_answer_id"]: q for q in questions.values()}

        for a_id, answer in answers.items():
            if answer.get("score", 0) < self.min_a_score:
                skipped += 1
                continue

            q = answer_id_to_q.get(a_id)
            if not q:
                failed += 1
                continue

            doc = build_kb_document(q, answer)
            if not doc:
                failed += 1
                continue

            kb.save_article(doc)
            saved += 1
            if saved % 500 == 0:
                print(f"  ... {saved:,} saved")

        print(f"\n  Stack Overflow: {saved:,} saved, {skipped:,} skipped, {failed:,} failed")
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Stack Overflow (macos tag) into the RAG knowledge base"
    )
    parser.add_argument(
        "--max-pages", type=int, default=MAX_PAGES,
        help=f"Max API pages to fetch (100 q/page, default: {MAX_PAGES})",
    )
    parser.add_argument(
        "--min-score", type=int, default=MIN_Q_SCORE,
        help=f"Minimum question score (default: {MIN_Q_SCORE})",
    )
    args = parser.parse_args()

    kb      = KnowledgeBase()
    scraper = StackOverflowScraper(max_pages=args.max_pages, min_q_score=args.min_score)
    scraper.scrape_all(kb)

    stats = kb.stats()
    print(f"\n  KB: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")


if __name__ == "__main__":
    main()
