"""
Apple Developer Forums scraper for the RAG knowledge base.

Scrapes macOS-relevant threads that have a "Correct Answer" marked reply.
Requires Playwright for JavaScript-rendered content.

Install:
    pip install playwright && playwright install chromium

Quality filters
---------------
  - Thread has at least one reply marked "Correct" (by OP or Apple)
  - Thread is in a macOS-relevant category/tag
  - Thread has at least MIN_REPLY_COUNT replies (indicates engagement)

Document shape
--------------
  One document per thread: question/OP post + correct answer body as
  "Problem" / "Solution" sections.  Matches the problem→solution chunking
  contract in CLAUDE.md.

Source value: "apple_developer_forums"

Standalone usage:
    python apple_devforums_scraper.py
Via pipeline:
    python main.py --step 5
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

FORUM_BASE = "https://developer.apple.com/forums"

# macOS-relevant tags to crawl on the Apple Developer Forums.
# Each resolves to: https://developer.apple.com/forums/tags/<tag>
MACOS_FORUM_TAGS = [
    "macos",
    "macos-sequoia",
    "macos-sonoma",
    "macos-ventura",
    "macos-monterey",
    "macos-big-sur",
    "bluetooth",
    "networking",
    "disk-management",
    "security",
    "privacy",
    "performance",
    "diagnostics",
    "launchd",
    "permissions",
]

# Threads must have at least this many replies to be worth indexing.
MIN_REPLY_COUNT = 2

# Max threads to collect per tag page (guards against crawling too deep).
MAX_THREADS_PER_TAG = 50

# ---------------------------------------------------------------------------
# Category inference from thread tags / title / content
# ---------------------------------------------------------------------------

_KEYWORD_CATEGORY: Dict[str, str] = {
    "bluetooth":        "bluetooth",
    "wifi":             "wifi", "wi-fi": "wifi", "network": "wifi",
    "ethernet":         "wifi", "vpn": "wifi",
    "disk":             "disk", "apfs": "disk", "storage": "disk",
    "time machine":     "disk", "diskutil": "disk",
    "battery":          "battery", "power": "battery", "sleep": "battery",
    "performance":      "performance", "memory": "performance", "cpu": "performance",
    "permission":       "permissions", "privacy": "permissions",
    "tcc":              "permissions", "gatekeeper": "permissions",
    "keychain":         "permissions", "codesign": "permissions",
    "crash":            "diagnostics", "log":     "diagnostics",
    "console":          "diagnostics", "kernel":  "diagnostics",
    "diagnostic":       "diagnostics",
    "startup":          "system", "boot": "system", "launch": "system",
    "spotlight":        "system", "finder": "system",
}


def _infer_category(text: str) -> str:
    t = text.lower()
    for keyword, cat in _KEYWORD_CATEGORY.items():
        if keyword in t:
            return cat
    return "general"


# ---------------------------------------------------------------------------
# macOS version extraction from thread content
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(
    r"macos\s+(\d+(?:\.\d+)*)"
    r"|(?:sequoia|sonoma|ventura|monterey|big\s*sur|catalina|mojave|high\s*sierra)",
    re.IGNORECASE,
)

_NAME_VERSION = {
    "sequoia": "15", "sonoma": "14", "ventura": "13", "monterey": "12",
    "big sur": "11", "bigsur": "11", "catalina": "10.15", "mojave": "10.14",
    "high sierra": "10.13", "highsierra": "10.13",
}


def _extract_versions(text: str) -> List[str]:
    versions = set()
    for m in _VERSION_RE.finditer(text):
        if m.group(1):
            versions.add(m.group(1))
        else:
            name = m.group(0).lower().replace(" ", "")
            v = _NAME_VERSION.get(name) or _NAME_VERSION.get(m.group(0).lower())
            if v:
                versions.add(v)
    return sorted(versions) if versions else ["all"]


# ---------------------------------------------------------------------------
# Difficulty tier inference
# ---------------------------------------------------------------------------

_TIER3 = (
    "log show --predicate", "kernel extension", "kext",
    "recovery mode", "smc reset", "sip", "csrutil",
    "codesign", "entitlements", "/dev/", "iokit",
    "kernel panic", "dtrace",
)
_TIER2 = (
    "terminal", "sudo ", "defaults write", "tccutil",
    "launchctl", "nvram ", "diskutil ", "pmset ",
    "/etc/", "/var/", "/usr/", "/Library/",
    "log show", "command line",
)


def _infer_difficulty(text: str) -> int:
    t = text.lower()
    if any(s in t for s in _TIER3):
        return 3
    if any(s in t for s in _TIER2):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Apple Developer Forums Playwright scraper
# ---------------------------------------------------------------------------

class AppleDevForumsScraper:
    """
    Discovers macOS threads with a Correct-marked answer on the Apple Developer
    Forums and saves them to the KnowledgeBase.

    Uses Playwright because the forums are a JavaScript-rendered SPA.
    Follows the same scrape_all(kb) → int interface as other scrapers.
    """

    def __init__(
        self,
        tags:          List[str] = MACOS_FORUM_TAGS,
        delay_seconds: float     = 2.5,
        max_per_tag:   int       = MAX_THREADS_PER_TAG,
    ):
        self.tags          = tags
        self.delay         = delay_seconds
        self.max_per_tag   = max_per_tag

    # ------------------------------------------------------------------
    # Playwright helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _discover_thread_urls(page, tag: str, max_threads: int) -> List[str]:
        """
        Load the tag page and collect thread URLs.
        Returns up to max_threads unique thread URLs.
        """
        tag_url = f"{FORUM_BASE}/tags/{tag}"
        try:
            await page.goto(tag_url, wait_until="networkidle", timeout=30_000)
            # Give JS a moment to render the thread list
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"  ✗ Tag page load failed ({tag}): {e}")
            return []

        thread_urls: List[str] = await page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                const threadLinks = links
                    .map(a => a.href)
                    .filter(h => /\\/forums\\/thread\\/\\d+/.test(h));
                // deduplicate, strip query params
                return [...new Set(threadLinks.map(h => h.split('?')[0]))];
            }
        """)
        return thread_urls[:max_threads]

    @staticmethod
    async def _scrape_thread(page, url: str) -> Optional[Dict]:
        """
        Load a thread page and extract the OP + Accepted-marked answer.
        Returns None if no Accepted answer is found.

        Selector notes (verified against live Apple Dev Forums DOM 2026-06):
          - Posts:          .content-post
          - Post body text: .post-content  (child of .content-post)
          - Accepted badge: .top-answer-badge.solved  (child of accepted .content-post)
          - Thread title:   .post-header-title  (on first .content-post)
          - Tags:           .tag-list elements or data in page config JSON
        """
        try:
            await page.goto(url, wait_until="networkidle", timeout=30_000)
            await page.wait_for_timeout(1500)
        except Exception as e:
            print(f"  ✗ Thread load failed ({url}): {e}")
            return None

        result = await page.evaluate("""
            () => {
                // ── Title ──────────────────────────────────────────────────
                // The thread title is in .post-header-title on the first post.
                const titleEl = document.querySelector('.post-header-title, h1');
                const title = titleEl ? titleEl.innerText.trim() : '';

                // ── All content posts ──────────────────────────────────────
                const allPosts = Array.from(document.querySelectorAll('.content-post'));

                // ── Original post (first .content-post) ───────────────────
                const opEl = allPosts[0] || null;
                const opContent = opEl ? opEl.querySelector('.post-content') : null;
                const opText = (opContent || opEl)
                    ? (opContent || opEl).innerText.trim()
                    : '';

                // ── Reply count ────────────────────────────────────────────
                // Posts count minus 1 (the original post itself)
                const replyCount = Math.max(allPosts.length - 1, 0);

                // ── Accepted answer ────────────────────────────────────────
                // Accepted answer badge class: top-answer-badge + solved
                // It lives inside the .content-post that is the accepted answer.
                let correctText = '';
                for (const post of allPosts) {
                    const badge = post.querySelector('.top-answer-badge.solved, [class*="top-answer-badge"][class*="solved"]');
                    if (badge) {
                        const body = post.querySelector('.post-content');
                        correctText = (body || post).innerText.trim();
                        break;
                    }
                }

                // ── Tags ───────────────────────────────────────────────────
                const tagEls = document.querySelectorAll('[class*="tag-list"] a, .tags a, [class*="topic-tag"]');
                const tags = Array.from(tagEls)
                    .map(e => e.innerText.trim().toLowerCase())
                    .filter(Boolean);

                return { title, opText, correctText, tags, replyCount };
            }
        """)

        title       = result.get("title", "").strip()
        op_text     = result.get("opText", "").strip()
        correct_text = result.get("correctText", "").strip()
        tags        = result.get("tags", [])
        reply_count = result.get("replyCount", 0)

        # Require a non-empty correct answer and a minimum reply count
        if not correct_text or reply_count < MIN_REPLY_COUNT:
            return None
        if not title and not op_text:
            return None

        return {
            "url":         url,
            "title":       title or op_text[:80],
            "op_text":     op_text,
            "correct_text": correct_text,
            "tags":        tags,
            "reply_count": reply_count,
        }

    def _build_document(self, thread: Dict) -> Optional[Dict]:
        title       = thread["title"]
        op_text     = thread["op_text"]
        answer_text = thread["correct_text"]
        tags        = thread["tags"]
        url         = thread["url"]

        combined = f"{title} {op_text} {answer_text}"
        category   = normalize_category(_infer_category(combined))
        versions   = _extract_versions(combined)
        difficulty = _infer_difficulty(answer_text)

        embed_text = normalize_embed_text(
            f"{title}. {op_text} {answer_text}",
            title=title,
            category=category,
        )
        if len(embed_text) < 100:
            return None

        # Stable article ID from the thread number in the URL
        thread_num = re.search(r"/thread/(\d+)", url)
        thread_id  = thread_num.group(1) if thread_num else url.split("/")[-1]
        now        = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "id":               f"devforum_{thread_id}",
            "article_id":       f"DEVFORUM_{thread_id}",
            "locale":           "en-us",
            "title":            title,
            "url":              url,
            "scraped_at":       now,
            "last_modified":    None,
            "affected_devices": ["Mac"],
            "macos_versions":   versions,
            "difficulty_tier":  difficulty,
            "category":         category,
            "categories":       ["Apple Developer Forums"],
            "summary":          op_text[:300],
            "sections": [
                {"heading": "Problem",  "body": op_text},
                {"heading": "Solution", "body": answer_text},
            ],
            "steps":      [],
            "embed_text": embed_text,
            "source":     "apple_developer_forums",
        }

    # ------------------------------------------------------------------
    # Async pipeline
    # ------------------------------------------------------------------

    async def _run(self, kb: KnowledgeBase) -> int:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print(
                "\n[Apple Dev Forums] playwright not installed.\n"
                "  Run: pip install playwright && playwright install chromium\n"
            )
            return 0

        seen_urls: set = set()
        saved = skipped = failed = 0

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page    = await browser.new_page()
            await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

            for tag in self.tags:
                print(f"\n  Tag: {tag}")
                thread_urls = await self._discover_thread_urls(page, tag, self.max_per_tag)
                print(f"  → {len(thread_urls)} thread URLs")

                for url in thread_urls:
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    thread_num = re.search(r"/thread/(\d+)", url)
                    if not thread_num:
                        continue
                    article_id = f"DEVFORUM_{thread_num.group(1)}"

                    if kb.already_scraped(article_id):
                        skipped += 1
                        continue

                    thread = await self._scrape_thread(page, url)
                    if not thread:
                        failed += 1
                        await asyncio.sleep(self.delay)
                        continue

                    doc = self._build_document(thread)
                    if not doc:
                        failed += 1
                    else:
                        kb.save_article(doc)
                        print(f"  ✓ {article_id}  {thread['title'][:55]}")
                        saved += 1

                    await asyncio.sleep(self.delay)

            await browser.close()

        print(f"\n  Apple Dev Forums: {saved} saved, {skipped} skipped, {failed} no-correct-answer")
        return saved

    def scrape_all(self, kb: KnowledgeBase) -> int:
        return asyncio.run(self._run(kb))


# ---------------------------------------------------------------------------
# Entry point (standalone)
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape Apple Developer Forums into the RAG knowledge base"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=MACOS_FORUM_TAGS,
        help="Forum tags to crawl (default: full macOS tag list)",
    )
    parser.add_argument(
        "--max-per-tag",
        type=int,
        default=MAX_THREADS_PER_TAG,
        help="Max threads to collect per tag (default: 50)",
    )
    args = parser.parse_args()

    kb      = KnowledgeBase()
    scraper = AppleDevForumsScraper(tags=args.tags, max_per_tag=args.max_per_tag)
    scraper.scrape_all(kb)

    stats = kb.stats()
    print(f"\n  KB: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")


if __name__ == "__main__":
    main()
