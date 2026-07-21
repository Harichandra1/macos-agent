"""
Ask Different (apple.stackexchange.com) data dump parser for the RAG knowledge base.

OFFLINE batch job — reads the downloaded XML dump, no HTTP requests during parsing.

Setup
-----
1. Download the Apple Stack Exchange data dump from the Internet Archive:
     https://archive.org/download/stackexchange/apple.stackexchange.com.7z

2. Extract (requires 7-Zip or p7zip):
     7z x apple.stackexchange.com.7z -o./se_dump/

   You only need Posts.xml from the archive.

3. Run standalone:
     python ask_different_scraper.py --dump-dir ./se_dump/
   Or via the pipeline:
     python main.py --step 4 --dump-dir ./se_dump/

Quality filters (CLAUDE.md Tier 2 spec)
----------------------------------------
  - PostTypeId == 1  (questions only; answers are fetched by AcceptedAnswerId)
  - Tags include 'macos' or 'osx'
  - Question Score >= 3
  - AcceptedAnswerId is set (a verified answer exists)
  - Accepted answer Score >= 1  (accepted but heavily down-voted answers are excluded)

Document shape
--------------
  One document per Q&A pair: question body + accepted answer body stored as
  two sections ("Problem" / "Solution").  This keeps the problem→solution unit
  together in a single retrievable chunk, matching the chunking contract in
  CLAUDE.md.
"""

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

# --- path bootstrap: make the package root importable when run standalone ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from knowledge_base import (
    KnowledgeBase,
    normalize_category,
    normalize_embed_text,
    _classify_text_category,
    extract_versions_from_text,
)

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

MIN_QUESTION_SCORE = 3
MIN_ANSWER_SCORE   = 1   # exclude extreme negatives even if accepted

# Tags that qualify a question as macOS-related
MACOS_TAGS = {"macos", "osx", "mac-osx", "mac"}

# ---------------------------------------------------------------------------
# Tag → controlled category (shared vocabulary from knowledge_base.py)
# ---------------------------------------------------------------------------

TAG_CATEGORY: Dict[str, str] = {
    # Bluetooth
    "bluetooth":             "bluetooth",
    # Wi-Fi / networking
    "wifi":                  "wifi",
    "wi-fi":                 "wifi",
    "airport":               "wifi",
    "ethernet":              "wifi",
    "networking":            "wifi",
    "network":               "wifi",
    "vpn":                   "wifi",
    "bonjour":               "wifi",
    # Disk / storage
    "hard-drive":            "disk",
    "ssd":                   "disk",
    "disk-utility":          "disk",
    "apfs":                  "disk",
    "hfs-plus":              "disk",
    "ntfs":                  "disk",
    "fusion-drive":          "disk",
    "time-machine":          "disk",
    "storage":               "disk",
    "disk":                  "disk",
    # Battery / power
    "battery":               "battery",
    "power-management":      "battery",
    "sleep":                 "battery",
    "shutdown":              "battery",
    "energy-saver":          "battery",
    # Performance
    "performance":           "performance",
    "memory":                "performance",
    "ram":                   "performance",
    "cpu":                   "performance",
    "freeze":                "performance",
    "spinning-beachball":    "performance",
    "activity-monitor":      "performance",
    # Permissions / privacy / security
    "permissions":           "permissions",
    "privacy":               "permissions",
    "security":              "permissions",
    "tcc":                   "permissions",
    "gatekeeper":            "permissions",
    "code-signing":          "permissions",
    "keychain":              "permissions",
    "sandboxing":            "permissions",
    # Diagnostics / logs
    "console":               "diagnostics",
    "crash":                 "diagnostics",
    "log":                   "diagnostics",
    "kernel-extension":      "diagnostics",
    "kernel-panic":          "diagnostics",
    "diagnostics":           "diagnostics",
    "system-information":    "diagnostics",
    # System
    "startup":               "system",
    "boot":                  "system",
    "launchctl":             "system",
    "plist":                 "system",
    "defaults":              "system",
    "recovery":              "system",
    "spotlight":             "system",
    "finder":                "system",
    "preferences":           "system",
    "homebrew":              "system",
    "terminal":              "system",
}

# ---------------------------------------------------------------------------
# macOS version tag → semver string
# ---------------------------------------------------------------------------

VERSION_TAG_MAP: Dict[str, str] = {
    "macos-sequoia":   "15",
    "macos-sonoma":    "14",
    "macos-ventura":   "13",
    "macos-monterey":  "12",
    "macos-big-sur":   "11",
    "macos-catalina":  "10.15",
    "macos-mojave":    "10.14",
    "macos-high-sierra": "10.13",
    "macos-sierra":    "10.12",
    "el-capitan":      "10.11",
    "osx-yosemite":    "10.10",
    "osx-mavericks":   "10.9",
    "osx-mountain-lion": "10.8",
    "osx-lion":        "10.7",
}

# ---------------------------------------------------------------------------
# Difficulty tier inference — keyword scan on the accepted answer HTML
# ---------------------------------------------------------------------------

_TIER3_SIGNALS = (
    "log show --predicate", "kernel extension", "kext",
    "recovery mode", "internet recovery", "smc reset",
    "system integrity protection", "csrutil",
    "codesign --verify", "entitlements",
    "/dev/", "iokit", "dtrace", "dtruss",
    "kernel panic", "kernel debug",
)

_TIER2_SIGNALS = (
    "terminal", "sudo ", "command line",
    "defaults write", "defaults delete",
    "tccutil", "launchctl", "nvram ",
    "diskutil ", "pmset ", "chmod ", "chown ",
    "/etc/", "/var/", "/usr/", "/Library/",
    "system preferences", "log show",
)


def _infer_difficulty(answer_html: str) -> int:
    t = answer_html.lower()
    if any(s in t for s in _TIER3_SIGNALS):
        return 3
    if any(s in t for s in _TIER2_SIGNALS):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

def _parse_tags(tags_str: str) -> List[str]:
    """'|macos|bluetooth|' or '<macos><bluetooth>' → ['macos', 'bluetooth']"""
    s = tags_str or ""
    if "|" in s:
        return [t for t in s.strip("|").split("|") if t]
    return re.findall(r"<([^>]+)>", s)


def _infer_category(tags: List[str], text: str = "") -> str:
    """Tags first (most reliable); fall back to scoring the body text."""
    for tag in tags:
        cat = TAG_CATEGORY.get(tag)
        if cat:
            return cat
    return _classify_text_category(text)


def _extract_versions(tags: List[str], text: str = "") -> List[str]:
    """Union of version tags and version mentions found in the body text."""
    versions = {VERSION_TAG_MAP[t] for t in tags if t in VERSION_TAG_MAP}
    for v in extract_versions_from_text(text):
        if v != "all":
            versions.add(v)
    return sorted(versions) if versions else ["all"]


# ---------------------------------------------------------------------------
# HTML → plain text (preserves code blocks for command readability)
# ---------------------------------------------------------------------------

def _html_to_text(body_html: str) -> str:
    soup = BeautifulSoup(body_html, "html.parser")
    # Wrap code / pre content so commands survive as readable text
    for tag in soup.find_all(["code", "pre"]):
        tag.replace_with(f" `{tag.get_text()}` ")
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Two-pass XML parser (memory-efficient iterparse)
# ---------------------------------------------------------------------------

class AskDifferentDumpParser:
    """
    Parses Posts.xml from the apple.stackexchange.com data dump.

    Two-pass strategy to avoid loading the full XML into memory:
      Pass 1 — collect qualifying questions + their AcceptedAnswerId.
      Pass 2 — collect answer bodies for those AcceptedAnswerIds.

    Expected Posts.xml size: ~150-500 MB uncompressed.
    iterparse keeps memory usage flat regardless of file size.
    """

    def __init__(
        self,
        dump_dir: str,
        min_question_score: int = MIN_QUESTION_SCORE,
        min_answer_score:   int = MIN_ANSWER_SCORE,
    ):
        self.posts_file     = Path(dump_dir) / "Posts.xml"
        self.min_q_score    = min_question_score
        self.min_a_score    = min_answer_score

    def _iter_rows(self) -> Iterator[Dict[str, str]]:
        for _event, elem in ET.iterparse(str(self.posts_file), events=("end",)):
            if elem.tag == "row":
                yield elem.attrib
                elem.clear()

    def _collect_questions(self) -> Dict[str, Dict]:
        """Pass 1: {accepted_answer_id → question metadata}."""
        questions: Dict[str, Dict] = {}
        for row in self._iter_rows():
            if row.get("PostTypeId") != "1":
                continue
            accepted_id = row.get("AcceptedAnswerId")
            if not accepted_id:
                continue
            try:
                score = int(row.get("Score", "0"))
            except ValueError:
                continue
            if score < self.min_q_score:
                continue
            tags = _parse_tags(row.get("Tags", ""))
            if not (MACOS_TAGS & set(tags)):
                continue
            questions[accepted_id] = {
                "post_id":       row["Id"],
                "title":         html.unescape(row.get("Title", "")),
                "body":          row.get("Body", ""),
                "tags":          tags,
                "score":         score,
                "creation_date": row.get("CreationDate", ""),
            }
        return questions

    def _collect_answers(self, target_ids: set) -> Dict[str, Dict]:
        """Pass 2: {answer_id → answer metadata} for the target IDs only."""
        answers: Dict[str, Dict] = {}
        for row in self._iter_rows():
            if row.get("PostTypeId") != "2":
                continue
            if row["Id"] not in target_ids:
                continue
            try:
                score = int(row.get("Score", "0"))
            except ValueError:
                score = 0
            if score < self.min_a_score:
                continue
            answers[row["Id"]] = {
                "body":  row.get("Body", ""),
                "score": score,
            }
        return answers

    def parse(self) -> List[Dict]:
        """
        Returns a list of paired Q&A records ready for KB document construction.
        Raises FileNotFoundError if Posts.xml is missing.
        """
        if not self.posts_file.exists():
            raise FileNotFoundError(
                f"Posts.xml not found at {self.posts_file}\n\n"
                "Download the Ask Different data dump:\n"
                "  https://archive.org/download/stackexchange/apple.stackexchange.com.7z\n"
                "Extract it and pass --dump-dir pointing to the directory with Posts.xml."
            )

        print(f"  [Pass 1] Scanning {self.posts_file} for qualifying questions...")
        questions = self._collect_questions()
        print(f"  → {len(questions):,} qualifying questions")

        if not questions:
            return []

        print(f"  [Pass 2] Collecting accepted answer bodies...")
        answers = self._collect_answers(set(questions.keys()))
        print(f"  → {len(answers):,} accepted answers found")

        pairs: List[Dict] = []
        for accepted_id, q_meta in questions.items():
            ans = answers.get(accepted_id)
            if ans:
                pairs.append({**q_meta, "answer_body": ans["body"], "answer_score": ans["score"]})

        print(f"  → {len(pairs):,} paired Q&A documents ready for ingestion")
        return pairs


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def build_kb_document(pair: Dict) -> Optional[Dict]:
    """Convert a Q&A pair dict into a KB-ready article (document contract)."""
    post_id     = pair["post_id"]
    title       = pair["title"]
    tags        = pair["tags"]
    q_text      = _html_to_text(pair["body"])
    a_text      = _html_to_text(pair["answer_body"])

    if not q_text or not a_text:
        return None

    body_blob  = f"{title} {q_text} {a_text}"
    category   = normalize_category(_infer_category(tags, body_blob))
    versions   = _extract_versions(tags, body_blob)
    difficulty = _infer_difficulty(pair["answer_body"])

    # embed_text: title + question + answer, truncated at 2000 chars
    embed_text = normalize_embed_text(
        f"{title}. {q_text} {a_text}",
        title=title,
        category=category,
    )
    if len(embed_text) < 100:
        return None

    return {
        "id":               f"askdiff_{post_id}",
        "article_id":       f"ASKDIFF_{post_id}",
        "locale":           "en-us",
        "title":            title,
        "url":              f"https://apple.stackexchange.com/questions/{post_id}",
        "scraped_at":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_modified":    pair.get("creation_date"),
        "affected_devices": ["Mac"],
        "macos_versions":   versions,
        "difficulty_tier":  difficulty,
        "category":         category,
        "categories":       ["Ask Different", "Community Q&A"],
        "summary":          q_text[:300],
        "sections": [
            {"heading": "Problem",  "body": q_text},
            {"heading": "Solution", "body": a_text},
        ],
        "steps":      [],
        "embed_text": embed_text,
        "source":     "ask_different",
    }


# ---------------------------------------------------------------------------
# Scraper class — pipeline integration
# ---------------------------------------------------------------------------

class AskDifferentScraper:
    """
    Processes the offline Ask Different data dump and writes to the KnowledgeBase.
    Interface mirrors all other scrapers: scrape_all(kb) → int (saved count).
    """

    def __init__(self, dump_dir: str):
        self.dump_dir = dump_dir

    def scrape_all(self, kb: KnowledgeBase) -> int:
        try:
            parser = AskDifferentDumpParser(self.dump_dir)
            pairs  = parser.parse()
        except FileNotFoundError as e:
            print(f"\n  [Ask Different] {e}")
            return 0

        saved = skipped = failed = 0
        for pair in pairs:
            article_id = f"ASKDIFF_{pair['post_id']}"
            if kb.already_scraped(article_id):
                skipped += 1
                continue

            doc = build_kb_document(pair)
            if not doc:
                failed += 1
                continue

            kb.save_article(doc)
            saved += 1
            if saved % 1000 == 0:
                print(f"  ... {saved:,} saved")

        print(f"\n  Ask Different: {saved:,} saved, {skipped:,} skipped, {failed:,} failed")
        return saved


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Ask Different SE dump into the RAG knowledge base"
    )
    parser.add_argument(
        "--dump-dir", required=True,
        help="Directory containing Posts.xml (extracted from apple.stackexchange.com.7z)",
    )
    args = parser.parse_args()

    kb      = KnowledgeBase()
    scraper = AskDifferentScraper(args.dump_dir)
    scraper.scrape_all(kb)

    stats = kb.stats()
    print(f"\n  KB: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")


if __name__ == "__main__":
    main()
