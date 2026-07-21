"""
WWDC session transcript/description scraper for the RAG knowledge base.

Fetches macOS-relevant WWDC sessions from Apple's developer website.
These documents add architectural context — explaining *why* a macOS
permission system, API, or security model works the way it does.
That "why" context is what makes the agent's explanations authoritative
rather than just quoting symptom-fix steps.

Examples of sessions that belong in the KB:
  - "What's New in Privacy" (WWDC19-WWDC24) — explains TCC, entitlements
  - "Advances in Networking" — explains Network.framework, privacy proxies
  - "Discover Log Analytics" — explains log predicates, OSLog internals
  - "Meet the New Diagnostics" — crash report architecture
  - "What's New in App Sandbox" — sandbox rules, inherited permissions

Approach (two strategies, tried in order):
  1. Apple's JSON documentation API — same format used for TN3xxx, fast
     and structured. URL: /tutorials/data/documentation/wwdc{year}.json
     If the index returns content, individual sessions are fetched via
     /tutorials/data/documentation/wwdc{year}/sessions/{id}.json.
  2. HTML scraping fallback — requests-based (no Playwright), parses
     the session listing page and individual session pages for title,
     og:description, and any visible transcript content.

Source value    : "wwdc_transcripts"
Difficulty tier : 3 (architectural developer content, always)
macos_versions  : derived from WWDC year (WWDC2023 → macOS "14")

Usage:
  python wwdc_scraper.py
  python wwdc_scraper.py --years 2022 2023 2024
  python main.py --step 9
"""

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

DELAY = 1.5  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# WWDC year → macOS version released at that WWDC
WWDC_YEAR_TO_MACOS: Dict[int, str] = {
    2019: "10.15",   # Catalina
    2020: "11",      # Big Sur
    2021: "12",      # Monterey
    2022: "13",      # Ventura
    2023: "14",      # Sonoma
    2024: "15",      # Sequoia
}

TARGET_YEARS: List[int] = list(WWDC_YEAR_TO_MACOS.keys())

# Apple JSON API base (same as TN3xxx scraper)
APPLE_JSON_API   = "https://developer.apple.com/tutorials/data"
# SSR'd all-videos aggregator — 1,692 session links, no JS required
ALL_VIDEOS_URL   = "https://developer.apple.com/videos/all-videos/"

# Session title or description must contain at least one of these for ingestion.
MACOS_SESSION_KEYWORDS = frozenset({
    "macos", " mac ", "privacy", "security", "permissions",
    "network", "bluetooth", "disk", "apfs", "battery", "power",
    "performance", "memory", "diagnostic", "crash", " log",
    "launchd", "daemon", "sandbox", "gatekeeper", "notariz",
    "signing", "certificate", "keychain", "tcc", "system extension",
    "kernel extension", "endpoint security", "filesystem", "xpc",
    "app lifecycle", "background task", "spotlight", "metadata",
    "nsuserdefaults", "preferences", "oslog", "core bluetooth",
    "network framework",
})

# ---------------------------------------------------------------------------
# Category inference
# ---------------------------------------------------------------------------

_KEYWORD_CATEGORY: Dict[str, str] = {
    "bluetooth":     "bluetooth", "core bluetooth": "bluetooth",
    "wifi":          "wifi",      "wi-fi":          "wifi",
    "network":       "wifi",      "networking":     "wifi",
    "disk":          "disk",      "apfs":           "disk",
    "filesystem":    "disk",      "storage":        "disk",
    "battery":       "battery",   "power":          "battery",
    "energy":        "battery",
    "performance":   "performance", "memory":        "performance",
    "cpu":           "performance",
    "permission":    "permissions", "privacy":       "permissions",
    "tcc":           "permissions", "gatekeeper":    "permissions",
    "codesign":      "permissions", "signing":       "permissions",
    "sandbox":       "permissions", "entitlement":   "permissions",
    "crash":         "diagnostics", "log":           "diagnostics",
    "oslog":         "diagnostics", "diagnostic":    "diagnostics",
    "console":       "diagnostics",
    "startup":       "system",    "boot":            "system",
    "launchd":       "system",    "spotlight":       "system",
    "xpc":           "system",    "daemon":          "system",
}


def _infer_category(text: str) -> str:
    t = text.lower()
    for kw, cat in _KEYWORD_CATEGORY.items():
        if kw in t:
            return cat
    return "general"


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

def _is_macos_relevant(title: str, description: str) -> bool:
    combined = f"{title} {description}".lower()
    return any(kw in combined for kw in MACOS_SESSION_KEYWORDS)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    return " ".join(soup.get_text(separator=" ", strip=True).split())


# ---------------------------------------------------------------------------
# WWDC scraper
# ---------------------------------------------------------------------------

class WWDCScraper:
    """
    Fetches macOS-relevant WWDC session content from Apple's developer
    website and saves it to the KnowledgeBase.

    Tries the JSON API first (structured, fast); falls back to HTML scraping.
    """

    def __init__(self, delay: float = DELAY, years: List[int] = TARGET_YEARS):
        self.delay = delay
        self.years = years
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_json(self, url: str) -> Optional[dict]:
        try:
            r = self.session.get(url, timeout=20)
            if r.ok:
                return r.json()
        except Exception:
            pass
        return None

    def _get_html(self, url: str) -> Optional[BeautifulSoup]:
        try:
            r = self.session.get(url, timeout=20)
            if r.ok:
                return BeautifulSoup(r.text, "html.parser")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Strategy 1: Apple JSON API (same system as TN3xxx)
    # ------------------------------------------------------------------

    def _iter_via_json(self, year: int) -> Iterator[Dict]:
        """
        Try the Apple documentation JSON API for WWDC {year}.
        URL: /tutorials/data/documentation/wwdc{year}.json
        Returns session dicts if the index is available, nothing otherwise.
        """
        index_url = f"{APPLE_JSON_API}/documentation/wwdc{year}.json"
        data = self._get_json(index_url)
        if not data:
            return

        refs = data.get("references", {})
        for section in data.get("topicSections", []):
            for ident in section.get("identifiers", []):
                ref   = refs.get(ident, {})
                slug  = ref.get("url", "").lstrip("/")
                title = ref.get("title", "")
                abstract_nodes = ref.get("abstract", [])
                abstract = " ".join(
                    p.get("text", "") for p in abstract_nodes
                    if isinstance(p, dict) and p.get("type") == "text"
                )

                if not title or not _is_macos_relevant(title, abstract):
                    continue

                # Fetch full session JSON to get body content
                content = abstract
                article_data = self._get_json(f"{APPLE_JSON_API}/{slug}.json")
                if article_data:
                    extracted = self._extract_text(article_data)
                    if extracted:
                        content = extracted

                session_id = slug.split("/")[-1]
                # Build the canonical Apple session URL from the session ID
                # Numeric IDs go to /videos/play/wwdc{year}/{id}/;
                # slug-style IDs go to /documentation/ path.
                if re.match(r"^\d+$", session_id.replace("session_", "")):
                    numeric_id = session_id.replace("session_", "")
                    video_url = f"https://developer.apple.com/videos/play/wwdc{year}/{numeric_id}/"
                else:
                    video_url = f"https://developer.apple.com/documentation/wwdc{year}/{session_id}"

                yield {
                    "id":          session_id,
                    "title":       title,
                    "description": abstract,
                    "content":     content,
                    "year":        year,
                    "url":         video_url,
                }
                time.sleep(self.delay * 0.3)

    def _extract_text(self, data: dict) -> str:
        """Flatten Apple JSON documentation content nodes to plain text."""
        parts: List[str] = []
        for p in data.get("abstract", []):
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        for section in data.get("primaryContentSections", []):
            for node in section.get("content", []):
                self._walk_node(node, parts)
        return " ".join(" ".join(parts).split())

    def _walk_node(self, node: dict, parts: List[str]):
        if not isinstance(node, dict):
            return
        ntype = node.get("type", "")
        if ntype in ("paragraph", "heading"):
            for inline in node.get("inlineContent", []):
                if isinstance(inline, dict) and inline.get("type") == "text":
                    parts.append(inline.get("text", ""))
        elif ntype in ("unorderedList", "orderedList"):
            for item in node.get("items", []):
                for child in item.get("content", []):
                    self._walk_node(child, parts)
        elif ntype == "codeListing":
            code = node.get("code", [])
            if code:
                parts.append("`" + " ".join(code[:10]) + "`")

    # ------------------------------------------------------------------
    # Strategy 2: HTML scraping — discover via all-videos page
    # ------------------------------------------------------------------

    def _discover_session_ids(self, years: List[int]) -> Dict[int, List[str]]:
        """
        Fetch the SSR'd all-videos aggregator page and return
        {year: [session_id, ...]} for the requested years only.

        The all-videos page is a single 4+ MB HTML response with every
        session link in raw markup — no JavaScript needed.
        """
        r = self.session.get(ALL_VIDEOS_URL, timeout=30)
        if not r.ok:
            print(f"  ✗ all-videos page returned {r.status_code}")
            return {}

        year_set = set(years)
        sessions_by_year: Dict[int, set] = {}
        pattern = re.compile(r'/videos/play/wwdc(\d{4})/(\d+)/')

        for m in pattern.finditer(r.text):
            year, sid = int(m.group(1)), m.group(2)
            if year in year_set:
                sessions_by_year.setdefault(year, set()).add(sid)

        result = {y: sorted(ids) for y, ids in sessions_by_year.items()}
        for year, ids in result.items():
            print(f"    WWDC{year}: {len(ids)} sessions discovered")
        return result

    def _iter_via_session_ids(self, year: int, session_ids: List[str]) -> Iterator[Dict]:
        """Scrape individual session pages by ID — fallback when JSON API yields nothing."""
        for session_id in session_ids:
            url = f"https://developer.apple.com/videos/play/wwdc{year}/{session_id}/"
            session_data = self._scrape_session_page(session_id, url, year)
            if session_data:
                yield session_data
            time.sleep(self.delay)

    def _scrape_session_page(
        self, session_id: str, url: str, year: int
    ) -> Optional[Dict]:
        soup = self._get_html(url)
        if not soup:
            return None

        # Title — prefer <h1> over og:title (og can be truncated)
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        if not title:
            og = soup.find("meta", property="og:title")
            if og:
                title = (og.get("content") or "").strip()

        # Description — og:description or name=description
        description = ""
        for selector in (
            {"name": "description"},
            {"property": "og:description"},
        ):
            tag = soup.find("meta", selector)
            if tag:
                description = (tag.get("content") or "").strip()
                if description:
                    break

        if not title or not _is_macos_relevant(title, description):
            return None

        # Transcript — may be in a div with "transcript" in the class
        content = description
        transcript_div = soup.find(class_=re.compile(r"transcript", re.I))
        if transcript_div:
            t = _clean_html(str(transcript_div))
            if len(t) > len(description):
                content = t

        return {
            "id":          session_id,
            "title":       title,
            "description": description,
            "content":     content,
            "year":        year,
            "url":         url,
        }

    # ------------------------------------------------------------------
    # Document builder
    # ------------------------------------------------------------------

    def _build_document(self, session: Dict) -> Optional[Dict]:
        title       = session["title"]
        description = session["description"]
        content     = session["content"]
        year        = session["year"]
        url         = session["url"]
        session_id  = str(session["id"])

        # content may equal description if no extra text was found
        full_text = f"{description} {content}".strip() if content != description else description
        category  = normalize_category(_infer_category(f"{title} {full_text}"))
        macos_ver = WWDC_YEAR_TO_MACOS.get(year, "all")

        embed_text = normalize_embed_text(
            f"WWDC{year}: {title}. {full_text}",
            title=f"WWDC{year}: {title}",
            category=category,
        )
        if len(embed_text) < 100:
            return None

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "id":               f"wwdc_{year}_{session_id}",
            "article_id":       f"WWDC_{year}_{session_id.upper()}",
            "locale":           "en-us",
            "title":            f"WWDC{year}: {title}",
            "url":              url,
            "scraped_at":       now,
            "last_modified":    None,
            "affected_devices": ["Mac"],
            "macos_versions":   [macos_ver],
            "difficulty_tier":  3,
            "category":         category,
            "categories":       [f"WWDC{year}", "Developer Sessions"],
            "summary":          description[:300],
            "sections": [
                {"heading": "Session Overview", "body": description},
                {"heading": "Content",          "body": content[:1500]},
            ],
            "steps":      [],
            "embed_text": embed_text,
            "source":     "wwdc_transcripts",
        }

    # ------------------------------------------------------------------
    # Main scrape loop
    # ------------------------------------------------------------------

    def scrape_all(self, kb: KnowledgeBase) -> int:
        saved = skipped = failed = 0

        # Discover all session IDs in a single fetch (all-years at once)
        print("  Discovering session IDs from Apple all-videos page...")
        sessions_by_year = self._discover_session_ids(self.years)
        total = sum(len(v) for v in sessions_by_year.values())
        print(f"  → {total} sessions across {len(sessions_by_year)} years")

        for year in self.years:
            macos_label = WWDC_YEAR_TO_MACOS.get(year, "?")
            print(f"\n  WWDC{year} → macOS {macos_label}...")

            # Strategy 1: JSON API (fast, structured)
            sessions = list(self._iter_via_json(year))
            strategy = "JSON API"

            # Strategy 2: per-session HTML using discovered IDs
            if not sessions:
                year_ids = sessions_by_year.get(year, [])
                sessions = list(self._iter_via_session_ids(year, year_ids))
                strategy = f"HTML ({len(year_ids)} IDs)"

            print(f"  → {len(sessions)} macOS-relevant sessions ({strategy})")

            for session in sessions:
                session_id = str(session["id"])
                article_id = f"WWDC_{year}_{session_id.upper()}"

                if kb.already_scraped(article_id):
                    skipped += 1
                    continue

                doc = self._build_document(session)
                if not doc:
                    failed += 1
                    continue

                kb.save_article(doc)
                saved += 1

            time.sleep(self.delay)

        print(f"\n  WWDC: {saved} saved, {skipped} skipped, {failed} failed")
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape WWDC session content into the RAG knowledge base"
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=TARGET_YEARS,
        help=f"WWDC years to scrape (default: {TARGET_YEARS})",
    )
    args = parser.parse_args()

    kb      = KnowledgeBase()
    scraper = WWDCScraper(years=args.years)
    scraper.scrape_all(kb)

    stats = kb.stats()
    print(f"\n  KB: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")


if __name__ == "__main__":
    main()
