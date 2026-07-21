"""
Apple Developer documentation scraper for the RAG knowledge base.

Three sources (all Tier 1 — authoritative Apple developer content):

  1. developer.apple.com/library/archive  — static HTML Tech Notes (TN2xxx)
     No extra dependencies; these explain *why* macOS things break.

  2. developer.apple.com/documentation    — current framework guide pages
     Requires `playwright` (headless Chromium) because the site is a JS SPA.
     Used selectively for the 8 frameworks that map to real user problems.

  3. developer.apple.com/documentation/technotes  — modern Tech Notes (TN3xxx)
     Post-2020 tech notes served via Apple's JSON documentation API.
     Uses requests (no Playwright needed); falls back to a warning if the
     JSON API changes.

Output: articles saved to KnowledgeBase (same schema as SiteMap.py output).
"""

import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text

# Framework guide URL slug → controlled category. Each framework maps to the
# real user-facing problem it explains.
FRAMEWORK_CATEGORY = {
    "oslog":               "diagnostics",
    "corebluetooth":       "bluetooth",
    "network":             "wifi",
    "security":            "permissions",
    "diskarbitration":     "disk",
    "iokit":               "battery",
    "systemconfiguration": "wifi",
    "xpc":                 "system",
}

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

ARCHIVE_BASE    = "https://developer.apple.com/library/archive"
APPLE_JSON_API  = "https://developer.apple.com/tutorials/data"

# Verified via HEAD probing 2050–2459.  URL pattern:
#   https://developer.apple.com/library/archive/technotes/tn{id}/index.html
# The old index page (technotes/index.html) no longer exists (404).
KNOWN_TN_IDS = [
    2050, 2056, 2057, 2058, 2060, 2062, 2063, 2064, 2065,
    2075, 2078, 2079, 2080, 2081, 2083, 2084, 2085,
    2091, 2093, 2095, 2096, 2097,
    2102, 2104, 2105, 2108, 2109, 2111, 2112, 2113, 2115,
    2120, 2124, 2125, 2127, 2130, 2131, 2133, 2138, 2139,
    2140, 2143, 2144, 2145, 2147, 2148, 2149, 2150, 2152,
    2153, 2154, 2155, 2157, 2161, 2162, 2163, 2166, 2169,
    2173, 2174, 2175, 2178, 2179, 2185, 2187, 2188, 2190,
    2198, 2199, 2200, 2201, 2203, 2204, 2206, 2207, 2213,
    2218, 2219, 2220, 2223, 2227, 2228, 2229, 2232, 2235,
    2236, 2237, 2239, 2241, 2242, 2244, 2247, 2248, 2250,
    2255, 2257, 2258, 2259, 2264, 2265, 2266, 2267, 2271,
    2273, 2276, 2277, 2280, 2283, 2284, 2285, 2286, 2287,
    2288, 2289, 2291, 2293, 2294, 2295, 2296, 2298, 2300,
    2302, 2307, 2310, 2311, 2312, 2313, 2315, 2318, 2319,
    2321, 2322, 2325, 2326, 2328, 2329, 2331, 2332, 2334,
    2335, 2336, 2339, 2347, 2348, 2350, 2351,
    2404, 2406, 2407, 2408, 2409, 2413, 2415, 2416, 2417,
    2418, 2420, 2428, 2429, 2432, 2434, 2435, 2436,
]

# Framework guide pages worth scraping with Playwright.
# Chosen because they explain system behaviour users encounter as problems.
FRAMEWORK_GUIDE_URLS = [
    "https://developer.apple.com/documentation/oslog",           # unified logging → log show advice
    "https://developer.apple.com/documentation/corebluetooth",   # bluetooth issues
    "https://developer.apple.com/documentation/network",         # wifi/network drops
    "https://developer.apple.com/documentation/security",        # permissions denied
    "https://developer.apple.com/documentation/diskarbitration", # disk errors
    "https://developer.apple.com/documentation/iokit",           # battery/power (IOPMLib)
    "https://developer.apple.com/documentation/systemconfiguration",  # network config
    "https://developer.apple.com/documentation/xpc",             # app launch / ipc failures
]

# Slug → category mapping used by FrameworkDocScraper
FRAMEWORK_SLUGS = list(FRAMEWORK_CATEGORY.keys())


# ---------------------------------------------------------------------------
# Shared JSON content-node extractor (used by both FrameworkDocScraper and
# ModernTechNoteScraper so neither has to import from the other)
# ---------------------------------------------------------------------------

def _extract_text_from_content_nodes(nodes: List[Dict]) -> str:
    """
    Recursively flatten Apple docs JSON content nodes into plain text.
    Handles paragraph, heading, codeListing, unorderedList, orderedList.
    """
    parts: List[str] = []
    for node in nodes:
        kind = node.get("type", "")
        if kind == "text":
            parts.append(node.get("text", ""))
        elif kind == "paragraph":
            parts.append(_extract_text_from_content_nodes(node.get("inlineContent", [])))
        elif kind == "heading":
            parts.append(node.get("text", ""))
        elif kind == "codeListing":
            code_lines = node.get("code", [])
            parts.append(" ".join(code_lines))
        elif kind in ("unorderedList", "orderedList"):
            for item in node.get("items", []):
                for item_content in item.get("content", []):
                    parts.append(
                        _extract_text_from_content_nodes(
                            item_content.get("inlineContent", [item_content])
                        )
                    )
        elif kind == "inlineHead":
            parts.append(node.get("text", ""))
    return " ".join(p for p in parts if p).strip()


# ---------------------------------------------------------------------------
# Source 1 — Tech Notes from the static archive
# ---------------------------------------------------------------------------

class TechNoteScraper:
    """
    Scrapes Apple Tech Notes (TN2xxx series) from the static archive library.
    No headless browser required — these are plain HTML pages.
    """

    def __init__(self, delay_seconds: float = 1.2):
        self.delay = delay_seconds
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def build_tn_list(self) -> List[Dict]:
        """
        Return {url, tn_id} for every known Tech Note.
        Uses KNOWN_TN_IDS (verified by HEAD probing) — the old index page no
        longer exists at developer.apple.com.
        """
        base = f"{ARCHIVE_BASE}/technotes"
        return [
            {
                "url":   f"{base}/tn{n}/_index.html",
                "tn_id": f"TN{n}",
            }
            for n in KNOWN_TN_IDS
        ]

    @staticmethod
    def _classify_category(text: str) -> str:
        """Map a Tech Note to the controlled vocabulary by keyword; default general."""
        t = text.lower()
        if "bluetooth" in t:
            return "bluetooth"
        if any(k in t for k in ["wi-fi", "wifi", "network", "ethernet", "socket"]):
            return "wifi"
        if any(k in t for k in ["battery", "power management", "iopmlib", "sleep/wake"]):
            return "battery"
        if any(k in t for k in ["disk", "volume", "apfs", "file system", "disk arbitration"]):
            return "disk"
        if any(k in t for k in ["permission", "privacy", "tcc", "code sign", "gatekeeper", "entitlement"]):
            return "permissions"
        if any(k in t for k in ["log", "oslog", "unified logging", "crash", "diagnostic"]):
            return "diagnostics"
        if any(k in t for k in ["performance", "cpu", "memory", "thread"]):
            return "performance"
        return "general"

    def scrape_technote(self, url: str, tn_id: str) -> Optional[Dict]:
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            if len(resp.text) < 2000:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # title
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else tn_id

            # last modified from meta
            last_modified = None
            for meta in soup.find_all("meta"):
                if meta.get("name", "").lower() in ("date", "revised", "lastmodified"):
                    last_modified = meta.get("content")
                    break

            # content root — prefer id="content", fall back to body
            root = soup.find(id="content") or soup.find("body") or soup

            sections: List[Dict] = []
            current_heading = "Overview"
            parts: List[str] = []

            for tag in root.find_all(["h2", "h3", "h4", "p", "li", "pre", "code"]):
                if tag.name in ("h2", "h3", "h4"):
                    if parts:
                        sections.append({
                            "heading": current_heading,
                            "body": " ".join(parts).strip(),
                        })
                    current_heading = tag.get_text(strip=True)
                    parts = []
                else:
                    text = tag.get_text(separator=" ", strip=True)
                    if text:
                        parts.append(text)

            if parts:
                sections.append({"heading": current_heading, "body": " ".join(parts).strip()})

            # skip stub pages (only one tiny section)
            if not sections or (len(sections) == 1 and len(sections[0]["body"]) < 100):
                return None

            section_text = " ".join(s["body"] for s in sections)
            category = normalize_category(self._classify_category(f"{title} {section_text}"))

            return {
                "id":               f"tn_{tn_id.lower()}",
                "article_id":       tn_id,
                "locale":           "en-us",
                "title":            title,
                "url":              url,
                "scraped_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_modified":    last_modified,
                "affected_devices": ["Mac"],   # Tech Notes are Mac/macOS focused
                "macos_versions":   ["all"],
                "difficulty_tier":  3,          # Tech Notes are always advanced
                "category":         category,
                "categories":       ["Developer", "Tech Notes"],
                "summary":          sections[0]["body"][:300] if sections else "",
                "sections":         sections,
                "steps":            [],
                "embed_text":       normalize_embed_text(
                    f"{title}. {section_text}", title=title, category=category,
                ),
                "source":           "apple_technotes",
            }

        except Exception as e:
            print(f"  ✗ {url}: {e}")
            return None

    def scrape_all(self, kb: KnowledgeBase, limit: Optional[int] = None) -> int:
        links = self.build_tn_list()
        print(f"  {len(links)} Tech Notes in known list")

        if limit:
            links = links[:limit]

        saved = failed = skipped = 0
        for item in links:
            tn_id = item["tn_id"]
            if not tn_id:
                continue
            if kb.already_scraped(tn_id):
                print(f"  — Already in KB: {tn_id}")
                skipped += 1
                continue

            doc = self.scrape_technote(item["url"], tn_id)
            if doc:
                kb.save_article(doc)
                print(f"  ✓ {tn_id:<12} {doc['title'][:55]}")
                saved += 1
            else:
                failed += 1

            time.sleep(self.delay)

        print(f"\n  Tech Notes: {saved} saved, {skipped} skipped, {failed} failed")
        return saved


# ---------------------------------------------------------------------------
# Source 2 — Framework guide pages via Apple JSON API
# ---------------------------------------------------------------------------

class FrameworkDocScraper:
    """
    Fetches Apple framework documentation via the same JSON API used by
    ModernTechNoteScraper. No Playwright required — all 8 target frameworks
    serve structured JSON at /tutorials/data/documentation/{slug}.json.
    """

    def __init__(self, slugs: List[str] = FRAMEWORK_SLUGS, delay_seconds: float = 2.0):
        self.slugs = slugs
        self.delay = delay_seconds
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _fetch_json(self, slug: str) -> Optional[dict]:
        url = f"{APPLE_JSON_API}/documentation/{slug}.json"
        try:
            r = self.session.get(url, timeout=20)
            return r.json() if r.ok else None
        except Exception as e:
            print(f"  ✗ {slug}: {e}")
            return None

    def _build_sections(self, data: dict) -> List[Dict]:
        sections: List[Dict] = []
        heading = "Overview"
        parts: List[str] = []

        for content_section in data.get("primaryContentSections", []):
            if content_section.get("kind") != "content":
                continue
            for node in content_section.get("content", []):
                if node.get("type") == "heading":
                    if parts:
                        sections.append({"heading": heading, "body": " ".join(parts).strip()})
                    heading = node.get("text", heading)
                    parts = []
                else:
                    text = _extract_text_from_content_nodes([node])
                    if text:
                        parts.append(text)

        if parts:
            sections.append({"heading": heading, "body": " ".join(parts).strip()})

        # Fall back to abstract if no primary content sections
        if not sections:
            abstract_nodes = data.get("abstract", [])
            abstract = _extract_text_from_content_nodes(abstract_nodes)
            if abstract:
                sections.append({"heading": "Overview", "body": abstract})

        return sections

    def _build_document(self, slug: str, data: dict) -> Optional[Dict]:
        title    = data.get("metadata", {}).get("title") or slug
        doc_url  = f"https://developer.apple.com/documentation/{slug}"
        sections = self._build_sections(data)

        if not sections:
            return None

        full_text = " ".join(s["body"] for s in sections)
        category  = normalize_category(FRAMEWORK_CATEGORY.get(slug, "system"))
        embed_text = normalize_embed_text(
            f"{title}. {full_text}", title=title, category=category,
        )
        if len(embed_text) < 100:
            return None

        return {
            "id":               f"devdoc_{slug}",
            "article_id":       f"DEVDOC_{slug.upper()}",
            "locale":           "en-us",
            "title":            title,
            "url":              doc_url,
            "scraped_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_modified":    None,
            "affected_devices": ["Mac"],
            "macos_versions":   ["all"],
            "difficulty_tier":  3,
            "category":         category,
            "categories":       ["Developer", "Framework Documentation"],
            "summary":          sections[0]["body"][:300] if sections else "",
            "sections":         sections,
            "steps":            [],
            "embed_text":       embed_text,
            "source":           "apple_developer_docs",
        }

    def scrape_all(self, kb: KnowledgeBase) -> int:
        saved = skipped = failed = 0
        for slug in self.slugs:
            article_id = f"DEVDOC_{slug.upper()}"
            if kb.already_scraped(article_id):
                print(f"  — Already in KB: {article_id}")
                skipped += 1
                continue

            print(f"  Fetching {slug} via JSON API...")
            data = self._fetch_json(slug)
            if not data:
                print(f"  ✗ No data: {slug}")
                failed += 1
                continue

            doc = self._build_document(slug, data)
            if doc:
                kb.save_article(doc)
                print(f"  ✓ {article_id:<28} {doc['title'][:50]}")
                saved += 1
            else:
                print(f"  ✗ No content: {slug}")
                failed += 1

            time.sleep(self.delay)

        print(f"\n  Framework docs: {saved} saved, {skipped} skipped, {failed} failed")
        return saved


# ---------------------------------------------------------------------------
# Source 3 — Modern Tech Notes (TN3xxx) via Apple docs JSON API
# ---------------------------------------------------------------------------

class ModernTechNoteScraper:
    """
    Scrapes TN3xxx Tech Notes from developer.apple.com/documentation/technotes.

    These are post-2020 tech notes — separate from the TN2xxx static archive
    that TechNoteScraper already handles.  Uses Apple's public JSON
    documentation API (no Playwright required).

    Article IDs are prefixed TN3_ to avoid collisions with TN2xxx entries.
    Source value: "apple_technotes" (same bucket — both are Apple-authored).
    Difficulty tier: 3 (all tech notes are advanced by definition).
    """

    # Top-level JSON index for all TN3xxx entries
    INDEX_JSON = "https://developer.apple.com/tutorials/data/documentation/technotes.json"
    # Per-article JSON template
    ARTICLE_JSON = "https://developer.apple.com/tutorials/data/documentation/technotes/{slug}.json"

    def __init__(self, delay_seconds: float = 1.5):
        self.delay   = delay_seconds
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------------
    # Index discovery
    # ------------------------------------------------------------------

    def _fetch_index(self) -> List[Dict]:
        """
        Fetch the JSON index and return a list of {slug, title, abstract} dicts
        for every TN3xxx entry listed under topicSections.
        """
        try:
            r = self.session.get(self.INDEX_JSON, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  ✗ JSON index fetch failed: {e}")
            return []

        # The Apple docs JSON format stores child pages in "references" keyed
        # by their doc:// identifier, and lists them in topicSections.
        refs = data.get("references", {})
        topic_sections = data.get("topicSections", [])

        article_ids: List[str] = []
        for section in topic_sections:
            article_ids.extend(section.get("identifiers", []))

        entries: List[Dict] = []
        for doc_id in article_ids:
            ref = refs.get(doc_id, {})
            url = ref.get("url", "")  # e.g. /documentation/technotes/tn3167-...
            if not url:
                continue
            slug = url.rstrip("/").split("/")[-1]
            if not slug.startswith("tn3"):
                continue
            abstract_nodes = ref.get("abstract", [])
            abstract = " ".join(
                node.get("text", "")
                for node in abstract_nodes
                if node.get("type") == "text"
            ).strip()
            entries.append({
                "slug":     slug,
                "title":    ref.get("title", slug),
                "abstract": abstract,
                "doc_url":  f"https://developer.apple.com{url}",
            })

        return entries

    # ------------------------------------------------------------------
    # Per-article fetch + parse
    # ------------------------------------------------------------------

    def _fetch_article(self, slug: str) -> Optional[Dict]:
        url = self.ARTICLE_JSON.format(slug=slug)
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"  ✗ {slug}: {e}")
            return None

    @staticmethod
    def _extract_text_from_content_nodes(nodes: List[Dict]) -> str:
        """Delegate to module-level helper (kept for backward compatibility)."""
        return _extract_text_from_content_nodes(nodes)

    def _build_sections(self, data: Dict) -> List[Dict]:
        """
        Convert Apple docs JSON primaryContentSections into KB section dicts.
        Each heading in the content creates a new section.
        """
        sections: List[Dict] = []
        heading = "Overview"
        parts:   List[str] = []

        for content_section in data.get("primaryContentSections", []):
            if content_section.get("kind") != "content":
                continue
            for node in content_section.get("content", []):
                if node.get("type") == "heading":
                    if parts:
                        sections.append({
                            "heading": heading,
                            "body":    " ".join(parts).strip(),
                        })
                    heading = node.get("text", heading)
                    parts   = []
                else:
                    text = self._extract_text_from_content_nodes([node])
                    if text:
                        parts.append(text)

        if parts:
            sections.append({"heading": heading, "body": " ".join(parts).strip()})

        return sections

    @staticmethod
    def _classify_category(text: str) -> str:
        t = text.lower()
        if "bluetooth" in t:
            return "bluetooth"
        if any(k in t for k in ["wi-fi", "wifi", "network", "ethernet"]):
            return "wifi"
        if any(k in t for k in ["battery", "power", "iopmlib", "sleep"]):
            return "battery"
        if any(k in t for k in ["disk", "volume", "apfs", "file system"]):
            return "disk"
        if any(k in t for k in ["permission", "privacy", "tcc", "gatekeeper", "codesign"]):
            return "permissions"
        if any(k in t for k in ["log", "oslog", "crash", "diagnostic"]):
            return "diagnostics"
        if any(k in t for k in ["performance", "cpu", "memory"]):
            return "performance"
        return "general"

    def _build_document(self, entry: Dict, data: Dict) -> Optional[Dict]:
        slug     = entry["slug"]
        title    = data.get("metadata", {}).get("title") or entry["title"]
        doc_url  = entry["doc_url"]
        sections = self._build_sections(data)

        if not sections:
            return None

        full_text = " ".join(s["body"] for s in sections)
        category  = normalize_category(self._classify_category(f"{title} {full_text}"))
        embed_text = normalize_embed_text(
            f"{title}. {full_text}", title=title, category=category,
        )
        if len(embed_text) < 100:
            return None

        # Extract macOS versions from platforms metadata if present
        platforms = data.get("metadata", {}).get("platforms", [])
        versions: List[str] = []
        for plat in platforms:
            if plat.get("name", "").lower() in ("macos", "mac"):
                intro = plat.get("introducedAt", "")
                if intro:
                    versions.append(intro)
        if not versions:
            versions = ["all"]

        return {
            "id":               f"tn3_{slug}",
            "article_id":       f"TN3_{slug.upper()}",
            "locale":           "en-us",
            "title":            title,
            "url":              doc_url,
            "scraped_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_modified":    None,
            "affected_devices": ["Mac"],
            "macos_versions":   versions,
            "difficulty_tier":  3,
            "category":         category,
            "categories":       ["Developer", "Tech Notes", "TN3xxx"],
            "summary":          sections[0]["body"][:300] if sections else entry["abstract"][:300],
            "sections":         sections,
            "steps":            [],
            "embed_text":       embed_text,
            "source":           "apple_technotes",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_all(self, kb: KnowledgeBase, limit: Optional[int] = None) -> int:
        entries = self._fetch_index()
        if not entries:
            print("  ✗ Could not fetch TN3xxx index — Apple JSON API may have changed.")
            return 0

        print(f"  {len(entries)} TN3xxx entries in index")
        if limit:
            entries = entries[:limit]

        saved = skipped = failed = 0
        for entry in entries:
            slug       = entry["slug"]
            article_id = f"TN3_{slug.upper()}"

            if kb.already_scraped(article_id):
                print(f"  — Already in KB: {article_id}")
                skipped += 1
                continue

            data = self._fetch_article(slug)
            if not data:
                failed += 1
                time.sleep(self.delay)
                continue

            doc = self._build_document(entry, data)
            if doc:
                kb.save_article(doc)
                print(f"  ✓ {article_id:<30} {doc['title'][:45]}")
                saved += 1
            else:
                print(f"  ✗ Empty content: {slug}")
                failed += 1

            time.sleep(self.delay)

        print(f"\n  Modern Tech Notes (TN3xxx): {saved} saved, {skipped} skipped, {failed} failed")
        return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    run_technotes: bool = True,
    run_framework_docs: bool = True,
    run_modern_technotes: bool = True,
    technote_limit: Optional[int] = None,
):
    kb = KnowledgeBase()

    print("=" * 60)
    print("APPLE DEVELOPER DOCS SCRAPER — RAG KNOWLEDGE BASE")
    print("=" * 60)

    total_saved = 0

    if run_technotes:
        print("\n── Source 1: Tech Notes TN2xxx (static archive) ───────────")
        scraper = TechNoteScraper(delay_seconds=1.2)
        total_saved += scraper.scrape_all(kb, limit=technote_limit)

    if run_framework_docs:
        print("\n── Source 2: Framework Guide Pages (Playwright) ────────────")
        scraper2 = FrameworkDocScraper(delay_seconds=2.0)
        total_saved += scraper2.scrape_all(kb)

    if run_modern_technotes:
        print("\n── Source 3: Modern Tech Notes TN3xxx (JSON API) ──────────")
        scraper3 = ModernTechNoteScraper(delay_seconds=1.5)
        total_saved += scraper3.scrape_all(kb, limit=technote_limit)

    stats = kb.stats()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Saved this run : {total_saved}")
    print(f"  Total articles : {stats['total_articles']}")
    print(f"  Total chunks   : {stats['total_chunks']}")
    print(f"  KB root        : {stats['kb_root']}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Apple Developer docs into the RAG knowledge base")
    parser.add_argument("--no-technotes",        action="store_true", help="Skip TN2xxx Tech Notes (archive)")
    parser.add_argument("--no-framework-docs",   action="store_true", help="Skip framework guide pages (Playwright)")
    parser.add_argument("--no-modern-technotes", action="store_true", help="Skip TN3xxx Tech Notes (JSON API)")
    parser.add_argument("--limit",               type=int, default=None, help="Max Tech Notes to scrape (default: all)")
    args = parser.parse_args()

    main(
        run_technotes=not args.no_technotes,
        run_framework_docs=not args.no_framework_docs,
        run_modern_technotes=not args.no_modern_technotes,
        technote_limit=args.limit,
    )
