import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# On-disk layout
#
#   knowledge_base/
#   ├── articles/          one JSON file per article  (full structured content)
#   ├── chunks.jsonl       RAG-ready chunks, one per line (append-only)
#   ├── index.json         article_id → lightweight metadata for fast lookup
#   └── run_metadata.json  stats for every scrape run
# ---------------------------------------------------------------------------

# Anchored to the package's data/ dir so paths resolve regardless of CWD.
KB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base")
KB_ARTICLES_DIR = os.path.join(KB_ROOT, "articles")
KB_CHUNKS_FILE = os.path.join(KB_ROOT, "chunks.jsonl")
KB_INDEX_FILE = os.path.join(KB_ROOT, "index.json")
KB_RUN_META_FILE = os.path.join(KB_ROOT, "run_metadata.json")

CHUNK_MAX_CHARS = 1000  # target character length per RAG chunk

EMBED_MIN_CHARS = 100   # embed_text shorter than this is too sparse to retrieve well
EMBED_MAX_CHARS = 2000  # ceiling — keeps embeddings cheap and focused


# ---------------------------------------------------------------------------
# Controlled category vocabulary (shared by every scraper)
#
# ChromaDB metadata pre-filtering depends on category being one of these exact
# values. Scrapers must run their raw category through normalize_category().
# ---------------------------------------------------------------------------

ALLOWED_CATEGORIES = {
    "bluetooth", "wifi", "disk", "battery", "performance",
    "permissions", "system", "diagnostics", "general",
}

# Off-vocabulary values seen in the wild → the controlled term they map onto.
_CATEGORY_ALIASES = {
    "network": "wifi",        "networking": "wifi",   "wi-fi": "wifi",
    "wireless": "wifi",       "ethernet": "wifi",
    "security": "permissions", "privacy": "permissions",
    "preferences": "system",  "prefs": "system",      "config": "system",
    "spotlight": "diagnostics", "logs": "diagnostics", "logging": "diagnostics",
    "processes": "performance", "process": "performance",
    "cpu": "performance",     "memory": "performance",
    "storage": "disk",        "power": "battery",
}


def normalize_category(value: Optional[str]) -> str:
    """Coerce an arbitrary category string into the controlled vocabulary."""
    if not value:
        return "general"
    v = value.strip().lower()
    if v in ALLOWED_CATEGORIES:
        return v
    return _CATEGORY_ALIASES.get(v, "general")


def normalize_embed_text(text: str, *, title: str = "", category: str = "",
                         min_chars: int = EMBED_MIN_CHARS,
                         max_chars: int = EMBED_MAX_CHARS) -> str:
    """
    Flatten an embed_text to a single clean string, enforce the length window.

    - Collapses all whitespace runs (kills stray newlines / HTML-extraction noise).
    - If shorter than min_chars, pads with title + category context so the chunk
      still carries enough signal to embed (rather than dropping the document).
    - Truncates to max_chars.
    """
    text = " ".join((text or "").split())
    if len(text) < min_chars:
        context = " ".join(p for p in (title, category) if p).strip()
        if context:
            text = f"{text} {context}".strip()
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Source inference + category classification (used by rebuild_from_articles)
# ---------------------------------------------------------------------------

def _infer_source(article_id: str) -> str:
    """Infer source value from article_id prefix — used by rebuild_from_articles only."""
    aid = (article_id or "").upper()
    if aid.startswith("TN2") or aid.startswith("TN3_"):
        return "apple_technotes"
    if aid.startswith("DEVDOC_"):
        return "apple_developer_docs"
    if aid.startswith("DEVFORUM_"):
        return "apple_developer_forums"
    if aid.startswith("SO_"):
        return "stackoverflow"
    if aid.startswith("REDDIT_"):
        return "reddit"
    if aid.startswith("GH_"):
        return "github_issues"
    if aid.startswith("WWDC_"):
        return "wwdc_transcripts"
    if "MANPAGE" in aid or aid in ("LOG", "DEFAULTS", "TCCUTIL", "PMSET"):
        return "macos_man_pages"
    return "apple_support"


# Keyword → category, grouped by controlled-vocabulary term. Matched with word
# boundaries and SCORED (most keyword hits wins) rather than first-match, so an
# incidental mention of "network" in a battery question doesn't hijack the label.
# Keep these phrases specific — generic tokens ("log", "sleep") cause false hits.
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "bluetooth": [
        "bluetooth", "airpods", "airpod", "magic mouse", "magic keyboard",
        "magic trackpad", "headphone", "headphones", "handsfree", "a2dp", "hci",
    ],
    "wifi": [
        "wi-fi", "wifi", "wireless", "airport", "802.11", "ssid", "hotspot",
        "ethernet", "vpn", "dns", "dhcp", "router", "network", "networking",
        "bonjour", "captive portal",
    ],
    "disk": [
        "apfs", "hfs+", "filesystem", "file system", "disk utility", "diskutil",
        "time machine", "fusion drive", "hard drive", "hard disk", "partition",
        "unmount", "external drive", "fsck", "volume", "ntfs", "exfat", "format disk",
    ],
    "battery": [
        "battery", "charging", "charger", "magsafe", "power adapter",
        "energy saver", "hibernate", "hibernatemode", "pmset", "wake from sleep",
        "sleep wake", "standby", "amperage", "cycle count",
    ],
    "performance": [
        "beachball", "spinning wheel", "kernel_task", "high cpu", "cpu usage",
        "memory pressure", "swap", "sluggish", "freezing", "activity monitor",
        "overheating", "fans spinning", "slow performance", "running slow",
    ],
    "permissions": [
        "tccutil", "tcc", "privacy settings", "gatekeeper", "codesign",
        "code signing", "notariz", "quarantine", "keychain", "sandbox",
        "csrutil", "full disk access", "accessibility access", "chmod", "chown",
        "permission denied", "operation not permitted",
    ],
    "diagnostics": [
        "kernel panic", "crash report", "crashreport", "log show", "console.app",
        "diagnosticreports", "spindump", "sysdiagnose", "stack trace",
        "backtrace", "syslog", "panic log",
    ],
    "system": [
        "launchd", "launchctl", "launchagent", "launchdaemon", "plist",
        "defaults write", "nvram", "recovery mode", "safe mode", "spotlight",
        "mdfind", "finder", "system preferences", "system settings", "homebrew",
        "boot", "startup", "login items",
    ],
}

# Flatten into compiled (regex, category) pairs once at import time.
_CATEGORY_PATTERNS = [
    (re.compile(r"\b" + re.escape(kw) + r"\b"), cat)
    for cat, kws in _CATEGORY_KEYWORDS.items()
    for kw in kws
]

# Tie-break priority: more specific categories beat broader ones on equal score.
_CATEGORY_PRIORITY = [
    "bluetooth", "battery", "disk", "wifi",
    "diagnostics", "permissions", "performance", "system",
]


def _classify_text_category(text: str) -> str:
    """
    Score a text blob against every category's keyword set and return the
    best-supported controlled-vocabulary category, or 'general' if nothing hits.
    Scored (not first-match) so the dominant topic wins; ties break by specificity.
    """
    t = (text or "").lower()
    scores: Dict[str, int] = {}
    for pattern, cat in _CATEGORY_PATTERNS:
        if pattern.search(t):
            scores[cat] = scores.get(cat, 0) + 1
    if not scores:
        return "general"
    best = max(scores.values())
    winners = [c for c, s in scores.items() if s == best]
    if len(winners) == 1:
        return winners[0]
    for cat in _CATEGORY_PRIORITY:   # deterministic tie-break by specificity
        if cat in winners:
            return cat
    return winners[0]


# ---------------------------------------------------------------------------
# macOS version extraction from free text (shared by scrapers + migration)
# ---------------------------------------------------------------------------

# Pattern → canonical version string. Names and numbers both map to the same
# value so "Ventura" and "macOS 13.2" both yield "13".
_VERSION_PATTERNS: List[tuple] = [
    (re.compile(r"\bsequoia\b|\bmac\s?os\s*15(?:\.\d+)?\b"),      "15"),
    (re.compile(r"\bsonoma\b|\bmac\s?os\s*14(?:\.\d+)?\b"),       "14"),
    (re.compile(r"\bventura\b|\bmac\s?os\s*13(?:\.\d+)?\b"),      "13"),
    (re.compile(r"\bmonterey\b|\bmac\s?os\s*12(?:\.\d+)?\b"),     "12"),
    (re.compile(r"\bbig\s*sur\b|\bmac\s?os\s*11(?:\.\d+)?\b"),    "11"),
    (re.compile(r"\bcatalina\b|\b10\.15(?:\.\d+)?\b"),            "10.15"),
    (re.compile(r"\bmojave\b|\b10\.14(?:\.\d+)?\b"),              "10.14"),
    (re.compile(r"\bhigh\s*sierra\b|\b10\.13(?:\.\d+)?\b"),       "10.13"),
    (re.compile(r"\bsierra\b|\b10\.12(?:\.\d+)?\b"),              "10.12"),
    (re.compile(r"\bel\s*capitan\b|\b10\.11(?:\.\d+)?\b"),        "10.11"),
    (re.compile(r"\byosemite\b|\b10\.10(?:\.\d+)?\b"),            "10.10"),
]


def extract_versions_from_text(text: str) -> List[str]:
    """
    Pull macOS version mentions out of free text. Returns a sorted list of
    canonical version strings (e.g. ['13','14']) or ['all'] if none are found.
    Never returns an empty list (an empty list silently excludes a doc from
    every version-filtered query — see CLAUDE.md data-cleaning checklist).
    """
    t = (text or "").lower()
    found = {ver for pattern, ver in _VERSION_PATTERNS if pattern.search(t)}
    if not found:
        return ["all"]
    # Sort numerically-ish: newest first is fine; keep deterministic order.
    return sorted(found, key=lambda v: float(v) if v.replace(".", "").isdigit() else 0,
                  reverse=True)


# ---------------------------------------------------------------------------
# Article schema
# ---------------------------------------------------------------------------
#
# {
#   "id":               "en-us_HT201065",       unique across locales
#   "article_id":       "HT201065",
#   "locale":           "en-us",
#   "title":            "...",
#   "url":              "https://support.apple.com/en-us/HT201065",
#   "scraped_at":       "2024-01-01T00:00:00Z",
#   "last_modified":    "2024-01-01",            from <meta> if present
#   "affected_devices": ["iPhone", "Mac"],
#   "categories":       ["Security", "Privacy"],
#   "summary":          "First meaningful paragraph",
#   "sections": [
#       {"heading": "Before you begin", "body": "..."}
#   ],
#   "steps": ["Step 1 text", "Step 2 text"]
# }

# ---------------------------------------------------------------------------
# Chunk schema  (one JSON object per line in chunks.jsonl)
# ---------------------------------------------------------------------------
#
# {
#   "chunk_id":        "en-us_HT201065_0",
#   "article_id":      "HT201065",
#   "locale":          "en-us",
#   "title":           "...",
#   "url":             "...",
#   "section_heading": "Before you begin",
#   "text":            "...",                    the actual text to embed
#   "affected_devices": [...],
#   "scraped_at":      "2024-01-01T00:00:00Z"
# }


class KnowledgeBase:
    """
    Manages on-disk storage for scraped Apple Support articles.

    Responsibilities
    ----------------
    - Save full article JSON to articles/{article_id}.json
    - Split each article into overlapping text chunks and append to chunks.jsonl
    - Maintain index.json for duplicate detection and fast metadata lookup
    - Record per-run statistics in run_metadata.json
    """

    def __init__(self, root: str = KB_ROOT):
        global KB_ROOT, KB_ARTICLES_DIR, KB_CHUNKS_FILE, KB_INDEX_FILE, KB_RUN_META_FILE
        KB_ROOT = root
        KB_ARTICLES_DIR = os.path.join(root, "articles")
        KB_CHUNKS_FILE = os.path.join(root, "chunks.jsonl")
        KB_INDEX_FILE = os.path.join(root, "index.json")
        KB_RUN_META_FILE = os.path.join(root, "run_metadata.json")

        os.makedirs(KB_ARTICLES_DIR, exist_ok=True)
        self._index: Dict[str, dict] = self._load_index()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, dict]:
        if os.path.exists(KB_INDEX_FILE):
            with open(KB_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _flush_index(self):
        with open(KB_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)

    def already_scraped(self, article_id: str) -> bool:
        """Return True if this article_id is already in the knowledge base."""
        return article_id in self._index

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_article(self, article: Dict):
        """
        Persist a scraped article:
          1. Write full JSON to articles/{article_id}.json
          2. Append RAG chunks to chunks.jsonl
          3. Update index.json
        """
        article_id = article["article_id"]

        # 1. Full article
        article_path = os.path.join(KB_ARTICLES_DIR, f"{article_id}.json")
        with open(article_path, "w", encoding="utf-8") as f:
            json.dump(article, f, indent=2, ensure_ascii=False)

        # 2. Chunks (append-only so restarts don't overwrite previous data)
        chunks = self._chunk_article(article)
        with open(KB_CHUNKS_FILE, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # 3. Index entry
        self._index[article_id] = {
            "title":            article["title"],
            "url":              article["url"],
            "locale":           article["locale"],
            "affected_devices": article["affected_devices"],
            "macos_versions":   article.get("macos_versions", ["all"]),
            "difficulty_tier":  article.get("difficulty_tier", 1),
            "category":         normalize_category(article.get("category")),
            "source":           article.get("source", "unknown"),
            "categories":       article.get("categories", []),
            "scraped_at":       article["scraped_at"],
            "chunk_count":      len(chunks),
        }
        self._flush_index()

    def save_run_metadata(self, meta: Dict):
        """Append a run-summary dict to run_metadata.json."""
        history: List[dict] = []
        if os.path.exists(KB_RUN_META_FILE):
            with open(KB_RUN_META_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        history.append(meta)
        with open(KB_RUN_META_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def was_run_today(self, step: Optional[int] = None) -> bool:
        """
        Return True if a scrape run was already recorded today.
        Pass step=1/2/3 to check a specific pipeline step, or omit for any step.
        Prevents duplicate scraping on the same calendar day (UTC).
        """
        if not os.path.exists(KB_RUN_META_FILE):
            return False
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(KB_RUN_META_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for run in history:
            run_date = run.get("run_at", "")[:10]
            if run_date == today:
                if step is None or run.get("step") == step:
                    return True
        return False

    def stats(self) -> Dict:
        total_chunks = 0
        if os.path.exists(KB_CHUNKS_FILE):
            with open(KB_CHUNKS_FILE, "r", encoding="utf-8") as f:
                total_chunks = sum(1 for _ in f)
        return {
            "total_articles": len(self._index),
            "total_chunks":   total_chunks,
            "kb_root":        KB_ROOT,
        }

    def get_article(self, article_id: str) -> Optional[Dict]:
        """Load and return a full article dict, or None if not found."""
        path = os.path.join(KB_ARTICLES_DIR, f"{article_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def search_index(self, device: Optional[str] = None,
                     locale: Optional[str] = None,
                     macos_version: Optional[str] = None,
                     max_difficulty: Optional[int] = None) -> List[Dict]:
        """
        Metadata pre-filter over the index — run this before semantic search
        to narrow the candidate set.

        Args:
            device:         e.g. "Mac", "iPhone"
            locale:         e.g. "en-us"
            macos_version:  e.g. "14"  (matches "all" entries too)
            max_difficulty: 1=beginner, 2=intermediate, 3=advanced
        """
        results = []
        for article_id, meta in self._index.items():
            if device and device not in meta.get("affected_devices", []):
                continue
            if locale and meta.get("locale") != locale:
                continue
            if macos_version:
                versions = meta.get("macos_versions", ["all"])
                if "all" not in versions and macos_version not in versions:
                    continue
            if max_difficulty and meta.get("difficulty_tier", 1) > max_difficulty:
                continue
            results.append({"article_id": article_id, **meta})
        return results

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_article(self, article: Dict) -> List[Dict]:
        """
        Chunk by semantic unit, not by character count.

        Strategy
        --------
        - Summary  → 1 chunk  (high-signal intro, kept separate)
        - Section  → 1 chunk per section heading (problem+solution stays together)
          If a section body exceeds CHUNK_MAX_CHARS it is split at sentence
          boundaries, never mid-sentence or mid-paragraph.
        - Steps    → 1 chunk for the full procedure (splitting a numbered list
          destroys its meaning for retrieval)

        Each chunk is self-contained: the heading is prepended to the body so
        the chunk makes sense without surrounding context.
        Metadata fields needed for ChromaDB pre-filtering are carried on every chunk.
        """
        base_meta = {
            "article_id":      article["article_id"],
            "locale":          article["locale"],
            "title":           article["title"],
            "url":             article["url"],
            "affected_devices": article["affected_devices"],
            "macos_versions":  article.get("macos_versions", ["all"]),
            "difficulty_tier": article.get("difficulty_tier", 1),
            "category":        normalize_category(article.get("category")),
            "source":          article.get("source", "unknown"),
            "scraped_at":      article["scraped_at"],
        }

        chunks: List[Dict] = []
        idx = 0

        # 1. Summary — short, high-signal; kept as a single standalone chunk
        if article.get("summary"):
            chunk = self._build_chunk(
                idx, "Summary", article["summary"], base_meta, article["id"]
            )
            if chunk:
                chunks.append(chunk)
                idx += 1

        # 2. One chunk per section — preserve the problem→solution unit
        for section in article.get("sections", []):
            body = section.get("body", "").strip()
            if not body:
                continue
            heading = section.get("heading", "")
            # Prefix with heading so the chunk is self-contained when retrieved
            full_text = f"{heading}: {body}" if heading else body

            if len(full_text) <= CHUNK_MAX_CHARS:
                chunk = self._build_chunk(idx, heading, full_text, base_meta, article["id"])
                if chunk:
                    chunks.append(chunk)
                    idx += 1
            else:
                for sub in self._split_at_sentences(full_text):
                    chunk = self._build_chunk(idx, heading, sub, base_meta, article["id"])
                    if chunk:
                        chunks.append(chunk)
                        idx += 1

        # 3. Steps — the whole procedure is one semantic unit; never split it
        if article.get("steps"):
            steps_text = "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(article["steps"])
            )
            full_text = f"How to: {article['title']}\n{steps_text}"
            chunk = self._build_chunk(idx, "Steps", full_text, base_meta, article["id"])
            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def _split_at_sentences(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
        """Split a long text at sentence boundaries, keeping each piece ≤ max_chars."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: List[str] = []
        buffer = ""
        for sent in sentences:
            if len(buffer) + len(sent) > max_chars and buffer:
                result.append(buffer.strip())
                buffer = sent
            else:
                buffer = f"{buffer} {sent}".strip() if buffer else sent
        if buffer:
            result.append(buffer.strip())
        return result or [text[:max_chars]]

    @staticmethod
    def _build_chunk(index: int, heading: str, text: str,
                     base_meta: Dict, doc_id: str) -> Optional[Dict]:
        text = normalize_embed_text(
            text,
            title=base_meta.get("title", ""),
            category=base_meta.get("category", ""),
        )
        if len(text) < EMBED_MIN_CHARS:
            return None
        return {
            "chunk_id":        f"{doc_id}_{index}",
            "section_heading": heading,
            "text":            text,
            **base_meta,
        }

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild_from_articles(self) -> int:
        """
        Rebuild chunks.jsonl and index.json from the article JSON files on disk.
        Patches missing source/category fields in-place and writes them back.
        Makes no network requests — safe to run after a schema change.
        """
        import glob as _glob

        files = sorted(_glob.glob(os.path.join(KB_ARTICLES_DIR, "*.json")))
        if not files:
            print("  No article files found.")
            return 0

        print(f"  Found {len(files):,} article files — rebuilding...")

        # Truncate chunks and clear index before rebuilding
        open(KB_CHUNKS_FILE, "w").close()
        self._index = {}
        self._flush_index()

        saved = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                article = json.load(f)

            patched = False

            if not article.get("source") or article.get("source") == "unknown":
                article["source"] = _infer_source(article.get("article_id", ""))
                patched = True

            if not article.get("category"):
                text_blob = " ".join([
                    article.get("title", ""),
                    article.get("summary", ""),
                    " ".join(s.get("body", "") for s in article.get("sections", [])),
                ])
                article["category"] = normalize_category(
                    _classify_text_category(text_blob)
                )
                patched = True

            if patched:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(article, f, indent=2, ensure_ascii=False)

            self.save_article(article)
            saved += 1
            if saved % 200 == 0:
                print(f"  ... {saved:,}/{len(files):,} rebuilt")

        print(f"  Rebuild complete: {saved:,} articles")
        return saved
