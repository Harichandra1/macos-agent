import re
import requests
import time
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import KnowledgeBase, normalize_category, normalize_embed_text


DEVICE_KEYWORDS = [
    "iPhone", "iPad", "Mac", "MacBook", "Apple Watch",
    "AirPods", "Apple TV", "HomePod", "Vision Pro",
]

MACOS_VERSIONS = {
    "sequoia":  "15",
    "sonoma":   "14",
    "ventura":  "13",
    "monterey": "12",
    "big sur":  "11",
    "catalina": "10.15",
}

# Matches Apple Support article paths:  /en-us/HT201232  or  /en-us/102640
ARTICLE_PATH_RE = re.compile(r"^/[a-z]{2}-[a-z]+/(HT\d+|\d{5,})$")


class AppleSupportScraper:
    """
    Discovers and scrapes Apple Support articles.

    URL discovery (two sources):
      1. Category pages from /sitemap  → numeric-ID articles  (/en-us/102640)
      2. XML sitemaps from robots.txt  → legacy HT articles   (/en-us/HT201232)

    All returned article dicts match the KnowledgeBase schema.
    """

    BLOCKED_PATTERNS = [
        "/kb/index?",
        "src=support_app",
        "/docs/product/",
        "MANUALS/",
        "/guide/",       # user guides are a separate content type
    ]

    # Category slugs to crawl (from /sitemap)
    CATEGORIES = [
        "iphone", "ipad", "mac", "watch", "airpods",
        "apple-vision-pro", "homepod", "tv", "icloud",
        "apple-account", "apple-pay", "apple-card", "billing",
        "accessibility", "safari", "messages", "mail",
        "photos", "music", "keynote", "pages", "numbers",
    ]

    def __init__(self, delay_seconds: float = 1.0):
        self.base_url = "https://support.apple.com"
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    # ------------------------------------------------------------------
    # robots.txt compliance check
    # ------------------------------------------------------------------

    def check_robots_txt(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/robots.txt", timeout=10)
            if response.status_code == 200:
                print("✓ robots.txt OK — article pages are not in any Disallow path")
                return True
        except Exception as e:
            print(f"⚠ Could not fetch robots.txt: {e}")
        return False

    # ------------------------------------------------------------------
    # Source 1: Category page crawl (main source for numeric-ID articles)
    # ------------------------------------------------------------------

    def discover_urls_from_categories(self) -> List[str]:
        """
        Crawl /sitemap and each product/topic category page.
        Collect all en-us article links matching ARTICLE_PATH_RE.
        Returns deduplicated list of full URLs.
        """
        found: Set[str] = set()

        for slug in self.CATEGORIES:
            cat_url = f"{self.base_url}/{slug}"
            try:
                resp = self.session.get(cat_url, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")

                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    # Normalise to absolute URL
                    if href.startswith("/"):
                        href = urljoin(self.base_url, href)
                    if not href.startswith(self.base_url):
                        continue
                    path = urlparse(href).path.rstrip("/")
                    if ARTICLE_PATH_RE.match(path) and "/en-us/" in path:
                        found.add(href.split("?")[0])  # strip query strings

                print(f"  {slug:<22} → {len(found)} total so far")
                time.sleep(self.delay_seconds)

            except Exception as e:
                print(f"  ✗ {slug}: {e}")

        return sorted(found)

    # ------------------------------------------------------------------
    # Source 2: XML sitemaps from robots.txt (legacy HT articles)
    # ------------------------------------------------------------------

    def discover_english_sitemap_indexes(self) -> List[str]:
        """Parse robots.txt and return all en-* sitemap index URLs."""
        response = self.session.get(f"{self.base_url}/robots.txt", timeout=10)
        response.raise_for_status()
        indexes = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line.startswith("Sitemap:"):
                continue
            url = line.split("Sitemap:", 1)[1].strip()
            locale = urlparse(url).path.lstrip("/").split("/")[0]
            if locale.startswith("en-"):
                indexes.append(url)
        return indexes

    def _expand_sitemap_index(self, index_url: str) -> List[str]:
        try:
            r = self.session.get(index_url, timeout=10)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
            locs = root.findall(".//ns:loc", ns) or root.findall(".//loc")
            return [loc.text for loc in locs if loc.text]
        except Exception as e:
            print(f"  ✗ Skipped index {index_url}: {e}")
            return []

    def discover_urls_from_xml_sitemaps(self) -> List[str]:
        """Collect HT article URLs from all English XML sitemaps."""
        indexes = self.discover_english_sitemap_indexes()
        all_child_sitemaps: List[str] = []
        for idx_url in indexes:
            all_child_sitemaps.extend(self._expand_sitemap_index(idx_url))
            time.sleep(0.2)

        found: Set[str] = set()
        for sm_url in all_child_sitemaps:
            try:
                r = self.session.get(sm_url, timeout=10)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
                locs = root.findall(".//ns:loc", ns) or root.findall(".//loc")
                for loc in locs:
                    if loc.text and ARTICLE_PATH_RE.match(urlparse(loc.text).path.rstrip("/")):
                        found.add(loc.text.split("?")[0])
            except Exception as e:
                print(f"  ✗ {sm_url}: {e}")

        return sorted(found)

    # ------------------------------------------------------------------
    # Deduplication across locales
    # ------------------------------------------------------------------

    @staticmethod
    def _article_id_from_url(url: str) -> str:
        """Extract article ID (e.g. '102640' or 'HT201232') from URL."""
        return urlparse(url).path.rstrip("/").split("/")[-1]

    @staticmethod
    def deduplicate_by_article_id(urls: List[str]) -> List[str]:
        """
        Keep one URL per article ID, preferring en-us over other locales.
        This prevents the same article from being scraped 50 times (once
        per English locale).
        """
        best: Dict[str, str] = {}
        for url in urls:
            art_id = urlparse(url).path.rstrip("/").split("/")[-1]
            if art_id not in best or "/en-us/" in url:
                best[art_id] = url
        return list(best.values())

    # ------------------------------------------------------------------
    # Article scraping and extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def detect_macos_versions(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = [num for name, num in MACOS_VERSIONS.items()
                 if name in text_lower or f"macos {num}" in text_lower]
        return found or ["all"]

    @staticmethod
    def _classify_difficulty(text: str) -> int:
        text = text.lower()
        if any(k in text for k in ["nvram", "smc", "recovery mode", "kext", "csrutil"]):
            return 3
        if any(k in text for k in ["terminal", "sudo", "command", "log show", "defaults write"]):
            return 2
        return 1

    @staticmethod
    def _classify_category(text: str) -> str:
        """
        Map an article to the controlled category vocabulary by keyword.
        Order matters — first match wins, most-specific topics first.
        Falls back to "general".
        """
        t = text.lower()
        if "bluetooth" in t or "airpods" in t:
            return "bluetooth"
        if any(k in t for k in ["wi-fi", "wifi", "wireless", "network", "ethernet", "hotspot"]):
            return "wifi"
        if any(k in t for k in ["battery", "charge", "charging", "power adapter", "low power"]):
            return "battery"
        if any(k in t for k in ["disk", "storage", "startup disk", "apfs", "volume", "format", "disk utility"]):
            return "disk"
        if any(k in t for k in ["slow", "performance", "memory pressure", "beach ball", "unresponsive", "high cpu"]):
            return "performance"
        if any(k in t for k in ["permission", "privacy", "gatekeeper", "full disk access", "accessibility access"]):
            return "permissions"
        if any(k in t for k in ["console", "log show", "diagnostic", "crash report", "sysdiagnose"]):
            return "diagnostics"
        if any(k in t for k in ["macos", "system settings", "update", "reset", "restart", "preference"]):
            return "system"
        return "general"

    def is_url_allowed(self, url: str) -> bool:
        return not any(p in url for p in self.BLOCKED_PATTERNS)

    @staticmethod
    def _locale_from_url(url: str) -> str:
        segment = urlparse(url).path.lstrip("/").split("/")[0]
        return segment if "-" in segment else "en-us"

    def extract_article_data(self, html: str, url: str) -> Optional[Dict]:
        """
        Parse an HT or numeric-ID article page and return a KB-schema dict.
        Handles both URL formats:
          /en-us/HT201232   (legacy)
          /en-us/102640     (current)
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            locale = self._locale_from_url(url)
            article_id = self._article_id_from_url(url)
            if not article_id:
                return None
            doc_id = f"{locale}_{article_id}"

            # title
            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else "N/A"

            # last_modified from <meta>
            last_modified = None
            for meta in soup.find_all("meta"):
                if meta.get("name", "").lower() in ("date", "lastmodified", "revised"):
                    last_modified = meta.get("content")
                    break

            # affected devices
            page_text = soup.get_text()
            affected_devices = sorted(
                {kw for kw in DEVICE_KEYWORDS if kw.lower() in page_text.lower()}
            ) or ["Unknown"]

            # categories from breadcrumb
            categories: List[str] = []
            breadcrumb = soup.find(class_=lambda c: c and "breadcrumb" in c.lower())
            if breadcrumb:
                categories = [
                    a.get_text(strip=True)
                    for a in breadcrumb.find_all("a")
                    if a.get_text(strip=True)
                ]

            # summary — first paragraph with real content
            summary = ""
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 40:
                    summary = text
                    break

            # sections — walk heading + following content
            sections: List[Dict] = []
            content_root = (
                soup.find("div", class_=lambda c: c and "content" in c.lower())
                or soup.find("article")
                or soup.find("main")
                or soup
            )
            current_heading = "Overview"
            body_parts: List[str] = []

            for tag in content_root.find_all(["h2", "h3", "p", "ul", "ol"]):
                if tag.name in ("h2", "h3"):
                    if body_parts:
                        sections.append({
                            "heading": current_heading,
                            "body": " ".join(body_parts).strip(),
                        })
                    current_heading = tag.get_text(strip=True)
                    body_parts = []
                else:
                    text = tag.get_text(separator=" ", strip=True)
                    if text:
                        body_parts.append(text)

            if body_parts:
                sections.append({
                    "heading": current_heading,
                    "body": " ".join(body_parts).strip(),
                })

            # numbered steps from <ol> tags
            steps = [
                li.get_text(strip=True)
                for ol in soup.find_all("ol")
                for li in ol.find_all("li")
                if li.get_text(strip=True)
            ]

            # RAG-specific fields
            section_text = " ".join(s["body"] for s in sections)
            macos_versions  = self.detect_macos_versions(page_text)
            difficulty_tier = self._classify_difficulty(page_text)
            category        = self._classify_category(f"{title} {page_text}")
            embed_text      = normalize_embed_text(
                f"{title}. {summary}. {section_text}",
                title=title, category=category,
            )

            return {
                "id":               doc_id,
                "article_id":       article_id,
                "locale":           locale,
                "title":            title,
                "url":              url,
                "scraped_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_modified":    last_modified,
                "affected_devices": affected_devices,
                "macos_versions":   macos_versions,
                "difficulty_tier":  difficulty_tier,
                "category":         category,
                "categories":       categories,
                "summary":          summary,
                "sections":         sections,
                "steps":            steps,
                "embed_text":       embed_text,
                "source":           "apple_support",
            }

        except Exception as e:
            print(f"  ⚠ Extraction error for {url}: {e}")
            return None

    def scrape_article(self, url: str, retries: int = 3) -> Optional[Dict]:
        if not self.is_url_allowed(url):
            print(f"  ⊘ Blocked: {url}")
            return None
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                if len(response.text) < 500:  # empty / error page served as 200
                    return None
                article = self.extract_article_data(response.text, url)
                if article:
                    print(f"  ✓ {article['article_id']:<12} {article['title'][:60]}")
                time.sleep(self.delay_seconds)
                return article
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"  ✗ Attempt {attempt+1}/{retries} for {url}: {e}. Retry in {wait}s")
                time.sleep(wait)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    scraper = AppleSupportScraper(delay_seconds=1.0)
    kb = KnowledgeBase()

    print("=" * 60)
    print("APPLE SUPPORT SCRAPER — RAG KNOWLEDGE BASE")
    print("=" * 60)

    # 1 — robots.txt check
    print("\n[1] Checking robots.txt...")
    scraper.check_robots_txt()

    # 2a — discover article URLs from category pages (numeric-ID articles)
    print("\n[2a] Crawling category pages for article URLs...")
    category_urls = scraper.discover_urls_from_categories()
    print(f"  → {len(category_urls)} URLs from category pages")

    # 2b — discover HT article URLs from XML sitemaps
    print("\n[2b] Fetching XML sitemaps (HT articles)...")
    sitemap_urls = scraper.discover_urls_from_xml_sitemaps()
    print(f"  → {len(sitemap_urls)} URLs from XML sitemaps")

    # 3 — merge and deduplicate by article ID (prefer en-us)
    all_urls = list(dict.fromkeys(category_urls + sitemap_urls))
    deduped_urls = scraper.deduplicate_by_article_id(all_urls)
    print(f"\n[3] After dedup by article ID: {len(deduped_urls)} unique articles")

    # 4 — scrape (set LIMIT=None for a full run)
    LIMIT = 20
    urls_to_scrape = deduped_urls[:LIMIT]
    print(f"\n[4] Scraping {len(urls_to_scrape)} articles...")

    scraped = skipped = failed = 0
    for url in urls_to_scrape:
        art_id = scraper._article_id_from_url(url)
        if kb.already_scraped(art_id):
            print(f"  — Already in KB: {art_id}")
            skipped += 1
            continue
        article = scraper.scrape_article(url)
        if article:
            kb.save_article(article)
            scraped += 1
        else:
            failed += 1

    # 5 — record run metadata
    kb.save_run_metadata({
        "run_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "category_urls":   len(category_urls),
        "sitemap_urls":    len(sitemap_urls),
        "unique_articles": len(deduped_urls),
        "scraped":         scraped,
        "skipped":         skipped,
        "failed":          failed,
    })

    # Summary
    stats = kb.stats()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Discovered     : {len(deduped_urls)} unique articles")
    print(f"  Articles in KB : {stats['total_articles']}")
    print(f"  Chunks in KB   : {stats['total_chunks']}")
    print(f"  This run       : {scraped} scraped, {skipped} skipped, {failed} failed")
    print(f"\nKnowledge base  : {stats['kb_root']}/")
    print("  articles/       — full article JSON per file")
    print("  chunks.jsonl    — RAG-ready chunks for embedding")
    print("  index.json      — fast article lookup")
    print("  run_metadata    — scrape history")


if __name__ == "__main__":
    main()
