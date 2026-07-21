"""
GitHub Issues scraper for the RAG knowledge base.

Sources : Homebrew/brew, mas-cli/mas, and other macOS-adjacent repos.
Purpose : captures macOS compatibility issues, post-upgrade breakage
          patterns, and tool-specific troubleshooting context that does
          not appear in Apple's official documentation.

Quality filters (CLAUDE.md Tier 4 spec):
  - Closed issues only (resolved or explicitly closed as not reproducible)
  - Issue body >= 200 chars (real problem description, not a one-liner)
  - Comments >= 3 (discussion and resolution actually happened)
  - macOS keywords in title, body (first 500 chars), or labels

"Solution" selection (GitHub has no accepted-answer mechanism):
  1. Highest-reaction comment with >= MIN_REACTIONS and >= 100 chars
     (community-validated fix — reactions are GitHub's upvote proxy)
  2. Comment containing explicit resolution keywords ("fixed it", "solved")
  3. Last substantive comment (>= 100 chars) before close
  If no suitable comment is found, the issue is skipped — a problem-only
  document without a solution is not useful for this KB.

Auth (optional — increases rate limit from 60 → 5,000 req/hour):
  export GITHUB_TOKEN=ghp_...
  Register at: https://github.com/settings/tokens (no special scopes needed
  for public repo access)

Source value    : "github_issues"
Difficulty tier : inferred from content (same TIER2/TIER3 keyword scan)

Usage:
  python github_scraper.py
  python main.py --step 8
"""

import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_BODY_CHARS = 200    # minimum issue body length
MIN_COMMENTS   = 3      # minimum comment count before fetching comments
MIN_REACTIONS  = 3      # reaction threshold to consider a comment community-validated
MAX_PAGES      = 30     # 30 × 100 = up to 3,000 closed issues per repo
DELAY          = 1.0    # seconds between API calls

GH_API_BASE = "https://api.github.com"

# Repositories to scrape — (owner, repo) tuples.
# Homebrew: macOS package manager — rich source of macOS compat/permission issues.
# mas-cli: Mac App Store CLI — Apple ID, login, and macOS-version breakage issues.
TARGET_REPOS: List[Tuple[str, str]] = [
    ("Homebrew", "brew"),
    ("mas-cli",  "mas"),
]

USER_AGENT = (
    "MacOS-RAG-KB/1.0 (github.com/researchbot; educational use; "
    "contact via github issues)"
)

# At least one of these must appear in title or first 500 chars of body
# for the issue to be considered macOS-relevant.
MACOS_KEYWORDS = frozenset({
    "macos", "mac os", "osx", "os x",
    "monterey", "ventura", "sonoma", "sequoia", "big sur", "catalina", "mojave",
    "apple silicon", "m1", "m2", "m3", "intel mac",
    "homebrew", "brew install", "brew link", "brew upgrade",
    "permission denied", "privacy", "tcc", "gatekeeper", "notarization",
    "system integrity", "sip", "codesign",
    "bluetooth", "wifi", "wi-fi", "network",
    "disk", "apfs", "hfs", "storage",
    "battery", "power management", "pmset",
    "crash", "kernel panic",
    "launchd", "launchctl", "daemon",
    "certificate", "signing",
})

# Labels that unconditionally mark an issue as macOS-relevant
MACOS_LABELS = frozenset({
    "macos", "macos", "osx", "bug", "regression", "compatibility",
    "apple-silicon", "m1", "m2", "ventura", "sonoma", "monterey",
})

# ---------------------------------------------------------------------------
# Category inference  (controlled vocab — same as all scrapers)
# ---------------------------------------------------------------------------

_KEYWORD_CATEGORY: Dict[str, str] = {
    "bluetooth":      "bluetooth",
    "wifi":           "wifi",    "wi-fi":    "wifi",   "network":   "wifi",
    "socket":         "wifi",    "dns":      "wifi",   "vpn":       "wifi",
    "disk":           "disk",    "apfs":     "disk",   "storage":   "disk",
    "volume":         "disk",    "ntfs":     "disk",   "hfs":       "disk",
    "battery":        "battery", "power":    "battery","sleep":     "battery",
    "slow":           "performance", "hang": "performance", "cpu": "performance",
    "memory":         "performance", "thread": "performance",
    "crash":          "diagnostics", "log":   "diagnostics",
    "kernel panic":   "diagnostics", "console": "diagnostics",
    "permission":     "permissions", "privacy": "permissions",
    "tcc":            "permissions", "gatekeeper": "permissions",
    "codesign":       "permissions", "signing":    "permissions",
    "keychain":       "permissions", "sandbox":    "permissions",
    "entitlement":    "permissions",
    "startup":        "system",  "boot":     "system", "launchd":   "system",
    "defaults write": "system",  "plist":    "system", "spotlight": "system",
}


def _infer_category(text: str) -> str:
    t = text.lower()
    for kw, cat in _KEYWORD_CATEGORY.items():
        if kw in t:
            return cat
    return "general"


# ---------------------------------------------------------------------------
# Difficulty tier inference
# ---------------------------------------------------------------------------

_TIER3 = (
    "log show --predicate", "kernel extension", "kext", "recovery mode",
    "smc reset", "csrutil", "codesign --verify", "entitlements",
    "/dev/", "iokit", "dtrace", "kernel panic", "coredump",
)
_TIER2 = (
    "terminal", "sudo ", "defaults write", "tccutil", "launchctl",
    "nvram ", "diskutil ", "pmset ", "/etc/", "/var/", "/usr/",
    "/Library/", "log show", "chmod ", "chown ", "command line",
)


def _infer_difficulty(text: str) -> int:
    t = text.lower()
    if any(s in t for s in _TIER3):
        return 3
    if any(s in t for s in _TIER2):
        return 2
    return 1


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
    "bigsur": "11",  "big-sur": "11",
    "catalina": "10.15", "mojave": "10.14", "highsierra": "10.13",
}


def _extract_versions(text: str) -> List[str]:
    versions: set = set()
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
# Text utilities
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown syntax and collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"```[a-z]*\n?", " ", text)
    text = re.sub(r"```", " ", text)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"\*\*?|__?|~~|#{1,6}\s", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return " ".join(text.split())


def _is_macos_relevant(issue: dict) -> bool:
    title  = (issue.get("title") or "").lower()
    body   = (issue.get("body")  or "")[:500].lower()
    labels = {(l.get("name") or "").lower() for l in (issue.get("labels") or [])}
    combined = f"{title} {body}"
    return (
        any(kw in combined for kw in MACOS_KEYWORDS)
        or bool(labels & MACOS_LABELS)
    )


# ---------------------------------------------------------------------------
# GitHub API client
# ---------------------------------------------------------------------------

class GitHubClient:
    """
    Thin GitHub REST API v3 wrapper.
    Uses a personal access token when set via GITHUB_TOKEN env var;
    falls back to unauthenticated (60 req/hour) otherwise.
    """

    def __init__(self, token: Optional[str] = None, delay: float = DELAY):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept":     "application/vnd.github.v3+json",
            "User-Agent": USER_AGENT,
        })
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"

    def get(self, path: str, **params):
        url = f"{GH_API_BASE}{path}"
        r   = self.session.get(url, params=params, timeout=20)

        if r.status_code == 403:
            reset_at = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(int(reset_at - time.time()), 1)
            if wait > 300:
                print(f"  [GitHub] Rate limited — waiting {wait}s (~{wait//60}min). Set GITHUB_TOKEN to avoid this.")
            else:
                print(f"  [GitHub] Rate limited — waiting {wait}s")
            time.sleep(wait)
            return self.get(path, **params)

        if r.status_code == 422:
            # GitHub pagination hard limit: only 1000 results (10 pages) per list endpoint
            return []

        r.raise_for_status()
        return r.json()

    def iter_issues(
        self, owner: str, repo: str, max_pages: int = MAX_PAGES
    ) -> Iterator[dict]:
        """Yield closed issues (not PRs), newest-updated first."""
        for page in range(1, max_pages + 1):
            items = self.get(
                f"/repos/{owner}/{repo}/issues",
                state="closed",
                per_page=100,
                page=page,
                sort="updated",
                direction="desc",
            )
            if not items:
                break
            for issue in items:
                if not issue.get("pull_request"):  # skip PRs
                    yield issue
            if len(items) < 100:
                break
            time.sleep(self.delay)

    def get_comments(self, owner: str, repo: str, number: int) -> List[dict]:
        try:
            return self.get(
                f"/repos/{owner}/{repo}/issues/{number}/comments",
                per_page=100,
            )
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Solution extraction
# ---------------------------------------------------------------------------

_RESOLUTION_KWS = (
    "this fixed it", "fixed by", "worked for me", "solved by",
    "this works", "resolved by", "turns out", "the solution is",
    "closing this because", "closing as", "fix is to",
    "switching to", "downgrading to", "upgrading to",
)


def _pick_solution_comment(comments: List[dict]) -> Optional[str]:
    """
    Return the text of the best 'solution' comment, or None.

    Priority: highest-reaction → resolution keywords → last substantive.
    """
    def body_text(c: dict) -> str:
        return _strip_markdown(c.get("body") or "")

    substantive = [c for c in comments if len(body_text(c)) >= 100]
    if not substantive:
        return None

    def reaction_count(c: dict) -> int:
        r = c.get("reactions", {})
        return sum(r.get(k, 0) for k in ("+1", "laugh", "hooray", "heart", "rocket", "eyes"))

    best = max(substantive, key=reaction_count)
    if reaction_count(best) >= MIN_REACTIONS:
        return body_text(best)

    for c in substantive:
        if any(kw in (c.get("body") or "").lower() for kw in _RESOLUTION_KWS):
            return body_text(c)

    return body_text(substantive[-1])


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_kb_document(
    issue:    dict,
    comments: List[dict],
    owner:    str,
    repo:     str,
) -> Optional[Dict]:
    """Convert a GitHub issue + comments into a KB article."""
    number    = issue["number"]
    title     = (issue.get("title") or "").strip()
    body      = _strip_markdown(issue.get("body") or "")
    url       = issue.get("html_url", f"https://github.com/{owner}/{repo}/issues/{number}")

    if len(body) < MIN_BODY_CHARS:
        return None

    solution = _pick_solution_comment(comments)
    if not solution:
        return None

    combined   = f"{title} {body} {solution}"
    category   = normalize_category(_infer_category(combined))
    versions   = _extract_versions(combined)
    difficulty = _infer_difficulty(combined)

    embed_text = normalize_embed_text(
        f"{title}. {body[:600]} {solution[:600]}",
        title=title,
        category=category,
    )
    if len(embed_text) < 100:
        return None

    now       = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    repo_slug = f"{owner}_{repo}".upper()

    return {
        "id":               f"gh_{repo_slug}_{number}".lower(),
        "article_id":       f"GH_{repo_slug}_{number}",
        "locale":           "en-us",
        "title":            title,
        "url":              url,
        "scraped_at":       now,
        "last_modified":    None,
        "affected_devices": ["Mac"],
        "macos_versions":   versions,
        "difficulty_tier":  difficulty,
        "category":         category,
        "categories":       [f"GitHub/{owner}/{repo}", "Open Source Issues"],
        "summary":          body[:300],
        "sections": [
            {"heading": "Problem",  "body": body[:1000]},
            {"heading": "Solution", "body": solution[:1000]},
        ],
        "steps":      [],
        "embed_text": embed_text,
        "source":     "github_issues",
    }


# ---------------------------------------------------------------------------
# Scraper class — pipeline integration
# ---------------------------------------------------------------------------

class GitHubScraper:
    """
    Scrapes closed macOS-relevant issues from target GitHub repositories.
    Interface: scrape_all(kb) → int (saved count).
    """

    def __init__(
        self,
        repos:     List[Tuple[str, str]] = TARGET_REPOS,
        token:     Optional[str]          = None,
        max_pages: int                    = MAX_PAGES,
        delay:     float                  = DELAY,
    ):
        self.repos     = repos
        self.max_pages = max_pages
        self.client    = GitHubClient(token=token, delay=delay)

    def scrape_all(self, kb: KnowledgeBase) -> int:
        using_token = bool(self.client.token)
        print(
            f"  Auth: {'token (5,000 req/hour)' if using_token else 'unauthenticated (60 req/hour)'}"
        )
        if not using_token:
            print("  Tip: set GITHUB_TOKEN for a much higher rate limit.")

        saved = skipped = failed = filtered = 0

        for owner, repo in self.repos:
            repo_slug  = f"{owner}_{repo}".upper()
            print(f"\n  Repo: {owner}/{repo}")
            issue_count = 0

            for issue in self.client.iter_issues(owner, repo, self.max_pages):
                issue_count += 1
                number     = issue["number"]
                article_id = f"GH_{repo_slug}_{number}"

                if kb.already_scraped(article_id):
                    skipped += 1
                    continue

                # Quality pre-filter before fetching comments (saves API quota)
                if not _is_macos_relevant(issue):
                    filtered += 1
                    continue
                if issue.get("comments", 0) < MIN_COMMENTS:
                    filtered += 1
                    continue
                if len(_strip_markdown(issue.get("body") or "")) < MIN_BODY_CHARS:
                    filtered += 1
                    continue

                comments = self.client.get_comments(owner, repo, number)
                time.sleep(self.client.delay)

                doc = build_kb_document(issue, comments, owner, repo)
                if not doc:
                    failed += 1
                    continue

                kb.save_article(doc)
                saved += 1
                if saved % 100 == 0:
                    print(f"  ... {saved:,} saved")

            print(f"  → {issue_count:,} issues scanned")

        print(
            f"\n  GitHub: {saved:,} saved, {skipped:,} skipped, "
            f"{failed:,} no-solution, {filtered:,} filtered"
        )
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape macOS-relevant GitHub issues into the RAG knowledge base"
    )
    parser.add_argument(
        "--repos", nargs="+", metavar="OWNER/REPO",
        default=[f"{o}/{r}" for o, r in TARGET_REPOS],
        help="GitHub repos to scrape (default: Homebrew/brew mas-cli/mas)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=MAX_PAGES,
        help=f"Max pages per repo (100 issues/page, default: {MAX_PAGES})",
    )
    args = parser.parse_args()

    repos = [tuple(r.split("/", 1)) for r in args.repos]
    kb    = KnowledgeBase()
    scraper = GitHubScraper(repos=repos, max_pages=args.max_pages)
    scraper.scrape_all(kb)

    stats = kb.stats()
    print(f"\n  KB: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")


if __name__ == "__main__":
    main()
