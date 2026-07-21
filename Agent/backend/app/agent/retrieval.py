"""
retrieval.py — Hybrid search + depth triage for the macOS RAG system.

Usage (standalone query):
    python retrieval.py "wifi keeps dropping on MacBook Pro M2 Ventura"
    python retrieval.py "disk not mounting" --version 14 --chip apple_silicon --n 5

As a module:
    from retrieval import Retriever
    r = Retriever()
    results = r.retrieve("wifi drops after sleep", macos_version="14")
    for hit in results:
        print(hit["score"], hit["title"])

Architecture (from CLAUDE.md):
    - Vector search against Qdrant Cloud (text-embedding-3-small, cosine)
    - Metadata pre-filters: macos_versions, category (optional)
    - Confidence threshold: if top score is below CONFIDENCE_THRESHOLD,
      fall back to web search rather than returning a weak match
    - Depth triage: suppress Tier 1 results when the user signal implies
      they've already tried basic steps
"""

import os
import re
import sys
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

try:
    from .taxonomy import normalize_category          # imported as app.agent.retrieval
except ImportError:                                    # standalone: python retrieval.py
    from taxonomy import normalize_category

# ---------------------------------------------------------------------------
# Retrieval config
# ---------------------------------------------------------------------------

EMBED_MODEL = "text-embedding-3-small"
COLLECTION  = "macos_kb"

# Cosine similarity threshold below which we consider the KB a miss.
# 1.0 = identical vectors; text-embedding-3-small real matches typically land 0.50–0.70.
# Empirically calibrated: wifi queries against wifi docs score ~0.58–0.59.
# Anything below 0.52 is likely off-topic noise. Tune further against a test set.
CONFIDENCE_THRESHOLD = 0.52

# Secondary fallback gate: even if top-1 squeaks above the threshold, an
# off-domain query produces a uniformly mediocre cluster (no single strong hit).
# Fall back when top-1 is only marginal AND the whole top-5 is weak on average.
FALLBACK_MARGINAL_TOP1 = 0.58
FALLBACK_MEAN_TOP5     = 0.50

# Over-retrieve this multiple of n, then rerank + dedup down to n. A larger
# candidate pool lets dedup replace duplicate chunks with genuinely new docs
# instead of scraping the bottom of the list.
OVERFETCH_FACTOR = 3

# Source-credibility weights. Apple's own documentation should win a TIE against a
# community post, but must not override a clearly more relevant semantic match —
# the spread is deliberately narrow so cosine relevance dominates and authority only
# breaks near-ties. (Wide weights were measured to hurt relevance: they promote
# tangential-but-authoritative docs over on-point community answers.)
SOURCE_WEIGHT = {
    "apple_support":          1.05,
    "apple_technotes":        1.04,
    "macos_man_pages":        1.04,
    "apple_developer_docs":   1.03,
    "apple_developer_forums": 1.02,
    "ask_different":          1.00,   # baseline
    "stackoverflow":          1.00,
    "wwdc_transcripts":       0.99,
    "github_issues":          0.98,
    "reddit":                 0.97,
}
DEFAULT_SOURCE_WEIGHT = 1.00

# Additive boost when a hit's macos_versions contains the exact queried version.
# Small — a tiebreaker, not a hammer. Large boosts surface version-matched-but-weak
# docs ahead of stronger general matches.
VERSION_MATCH_BOOST = 0.02

# Command-aware retrieval. When a query names a CLI tool or asks for exact syntax,
# the authoritative man page is what wins over a frontier model — but it gets
# out-competed semantically by chatty community posts and never enters the
# candidate pool. So we run a second, source-filtered retrieval for man pages and
# give the matching command's man page a strong additive boost. This directly
# targets the benchmark's biggest loss mode: exact-syntax questions.
MANPAGE_COMMAND_BOOST = 0.25
_MANPAGE_SOURCE = "macos_man_pages"
# CLI tools whose man page carries verbatim exact-syntax truth.
_MANPAGE_COMMANDS = {
    "log", "pmset", "tccutil", "defaults", "launchctl", "diskutil", "dscl",
    "spctl", "xattr", "codesign", "kmutil", "csrutil", "nvram", "mdutil",
    "mdfind", "scutil", "networksetup", "systemsetup", "ioreg", "system_profiler",
    "fsck", "hdiutil", "plutil", "bputil", "profiles", "softwareupdate", "wdutil",
    "syslog", "powermetrics", "kextstat", "sysctl", "dscacheutil", "dsconfigad",
    "tmutil", "kmutil",
}


def _detect_commands(query: str) -> set[str]:
    """CLI tools explicitly named in the query (word-boundary match)."""
    words = set(re.findall(r"[a-z_][a-z0-9_]+", query.lower()))
    return words & _MANPAGE_COMMANDS

# Additive category-match boost. MEASURED to hurt MRR at 0.03 (it promotes
# same-category-but-less-relevant docs above cross-category-but-more-relevant ones;
# the embedding already captures topicality), so it is disabled by default.
# Category is still available as an opt-in HARD filter (category_hard=True), which
# maximises NDCG when the caller is confident about the category. Kept as a tunable
# knob rather than deleted so the experiment is reproducible.
CATEGORY_MATCH_BOOST = 0.0

# Signals in the query that imply the user has already tried Tier 1 steps.
# When any of these appear, suppress Tier 1 results (difficulty_tier == 1).
_TIER1_ALREADY_TRIED_SIGNALS = (
    "already tried",
    "doesn't work",
    "still not",
    "tried restarting",
    "tried turning off",
    "rebooted",
    "reinstalled",
    "still happening",
    "still broken",
    "even after restart",
    "toggled",
    "reset smc",
    "reset nvram",
    "log show",
    "terminal",
    "checked the logs",
    "sudo",
)


# ---------------------------------------------------------------------------
# Depth triage
# ---------------------------------------------------------------------------

def _user_has_tried_basics(query: str) -> bool:
    """Return True if the query implies the user has already done Tier 1 steps."""
    q = query.lower()
    return any(sig in q for sig in _TIER1_ALREADY_TRIED_SIGNALS)


def _rerank_for_depth(hits: list[dict], query: str) -> list[dict]:
    """
    Reorder results so that if the user has tried Tier 1 steps, Tier 2/3 docs
    sort before Tier 1 docs — without ever hiding them entirely.
    Within each tier group, preserve original score order.
    """
    if not _user_has_tried_basics(query):
        return hits

    tier1 = [h for h in hits if h.get("difficulty_tier", 1) == 1]
    tier2_plus = [h for h in hits if h.get("difficulty_tier", 1) > 1]
    return tier2_plus + tier1


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Retrieves relevant KB chunks for a macOS troubleshooting query.

    Parameters
    ----------
    confidence_threshold : float
        Minimum cosine similarity score to consider a hit relevant.
        Below this, `retrieve()` returns an empty list and signals fallback.
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        load_dotenv()
        self._threshold = confidence_threshold

        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_key = os.environ.get("QDRANT_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        collection  = os.environ.get("QDRANT_COLLECTION", COLLECTION)

        if not all([qdrant_url, qdrant_key, openai_key]):
            print(
                "\n  ✗ Missing env vars. Copy .env.example → .env and fill in:\n"
                "    OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY\n"
            )
            sys.exit(1)

        self._collection = collection
        self._openai     = OpenAI(api_key=openai_key)
        self._qdrant     = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:          str,
        macos_version:  Optional[str] = None,
        mac_chip:       Optional[str] = None,
        category:       Optional[str] = None,
        n:              int           = 5,
        min_tier:       Optional[int] = None,
        category_hard:  bool          = False,
    ) -> dict:
        """
        Run a similarity search and return a result dict.

        Returns
        -------
        {
          "hits":      list of chunk dicts, best first (after depth triage)
          "fallback":  True if no hits exceeded the confidence threshold
          "query":     the original query string
        }

        Each hit dict contains all payload fields plus:
          "score":     reranked score (cosine × source weight + version boost)
          "raw_score": the original cosine similarity from Qdrant
        """
        vector = self._embed(query)
        # Category is a SOFT boost by default (see CATEGORY_MATCH_BOOST). Only
        # hard-filter on category when explicitly requested.
        filter_category = category if category_hard else None
        qdrant_filter = self._build_filter(macos_version, mac_chip, filter_category, min_tier)

        # Over-retrieve: pull n*OVERFETCH_FACTOR candidates so dedup can replace
        # duplicate chunks with genuinely new documents instead of running short.
        candidate_n = max(n * OVERFETCH_FACTOR, n)
        raw = self._query_with_retry(vector, qdrant_filter, candidate_n)

        # Command-aware injection FIRST: if the query names a CLI tool, pull the
        # man pages via a source-filtered search and merge them into the pool.
        # This MUST happen before the fallback decision — otherwise a depth-tier
        # filter can make the main query look weak and we'd bail out before ever
        # surfacing the authoritative exact-syntax doc (the #1 hard-slice loss).
        commands = _detect_commands(query)
        points = list(raw)
        manpage_hits = []
        if commands:
            manpage_hits = self._query_manpages(vector, candidate_n=6)
            seen_ids = {p.id for p in points}
            for mp in manpage_hits:
                if mp.id not in seen_ids:
                    points.append(mp)
                    seen_ids.add(mp.id)

        # Fallback decision on RAW cosine scores across the WHOLE pool:
        #  - top-1 below the hard threshold, OR
        #  - top-1 only marginal AND the whole cluster is weak (off-domain signature)
        # A named command's man page IS authoritative, so its presence at a
        # reasonable score suppresses fallback.
        cosines = sorted((p.score for p in points), reverse=True)
        top1    = cosines[0] if cosines else 0.0
        mean5   = sum(cosines[:5]) / len(cosines[:5]) if cosines else 0.0
        strong_manpage = any(mp.score >= self._threshold for mp in manpage_hits)
        if (not points
                or (top1 < self._threshold and not strong_manpage)
                or (top1 < FALLBACK_MARGINAL_TOP1 and mean5 < FALLBACK_MEAN_TOP5
                    and not strong_manpage)):
            return {"hits": [], "fallback": True, "query": query}

        # Rerank: source-credibility weight × cosine, + version/command boosts
        scored = []
        for point in points:
            payload   = dict(point.payload)
            cosine    = point.score
            src       = payload.get("source", "")
            weight    = SOURCE_WEIGHT.get(src, DEFAULT_SOURCE_WEIGHT)
            score     = cosine * weight
            if macos_version and macos_version in payload.get("macos_versions", []):
                score += VERSION_MATCH_BOOST
            if category and payload.get("category") == normalize_category(category):
                score += CATEGORY_MATCH_BOOST
            # Strong boost for a named command's own man page.
            if commands and src == _MANPAGE_SOURCE:
                title = (payload.get("title", "") or "").lower()
                if any(title.startswith(c) or re.search(rf"\b{re.escape(c)}\b", title)
                       for c in commands):
                    score += MANPAGE_COMMAND_BOOST
            payload["raw_score"] = round(cosine, 4)
            payload["score"]     = round(score, 4)
            scored.append(payload)

        scored.sort(key=lambda h: h["score"], reverse=True)

        # Dedup by article_id — keep only the best chunk per source article
        deduped = []
        seen_articles = set()
        for h in scored:
            aid = h.get("article_id") or h.get("chunk_id")
            if aid in seen_articles:
                continue
            seen_articles.add(aid)
            deduped.append(h)
            if len(deduped) >= n:
                break

        hits = _rerank_for_depth(deduped, query)
        return {"hits": hits, "fallback": False, "query": query}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def _query_with_retry(self, vector, qdrant_filter, limit, attempts: int = 4):
        """Run query_points, retrying transient Qdrant Cloud timeouts with backoff."""
        import time as _time
        for attempt in range(attempts):
            try:
                return self._qdrant.query_points(
                    collection_name=self._collection,
                    query=vector,
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                ).points
            except Exception:
                if attempt == attempts - 1:
                    raise
                _time.sleep(2 ** attempt)

    def _query_manpages(self, vector, candidate_n: int = 6):
        """Source-filtered retrieval over just the man-page docs."""
        mp_filter = Filter(must=[FieldCondition(
            key="source", match=MatchValue(value=_MANPAGE_SOURCE))])
        try:
            return self._query_with_retry(vector, mp_filter, candidate_n)
        except Exception:
            return []

    def _build_filter(
        self,
        macos_version: Optional[str],
        mac_chip:      Optional[str],
        category:      Optional[str],
        min_tier:      Optional[int],
    ) -> Optional[Filter]:
        conditions = []

        if macos_version and macos_version.lower() not in ("all", "any", ""):
            conditions.append(
                FieldCondition(
                    key="macos_versions",
                    match=MatchAny(any=[macos_version, "all"]),
                )
            )

        if category:
            normalized = normalize_category(category)
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=normalized),
                )
            )

        if min_tier is not None:
            from qdrant_client.models import Range
            conditions.append(
                FieldCondition(
                    key="difficulty_tier",
                    range=Range(gte=min_tier),
                )
            )

        # mac_chip: we only filter when the KB has chip-specific docs.
        # Currently most docs are chip-agnostic, so over-filtering here
        # would starve retrieval. Skip chip filter for now.

        if not conditions:
            return None
        if len(conditions) == 1:
            return Filter(must=conditions)
        return Filter(must=conditions)


# ---------------------------------------------------------------------------
# Standalone CLI — quick smoke test
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Query the macOS troubleshooting knowledge base"
    )
    parser.add_argument("query", help="The troubleshooting question")
    parser.add_argument("--version",  default=None, help="macOS version, e.g. 14")
    parser.add_argument("--chip",     default=None, help="apple_silicon or intel")
    parser.add_argument("--category", default=None, help="Category filter (wifi, disk, …)")
    parser.add_argument("--n",        type=int, default=5, help="Number of results")
    parser.add_argument("--min-tier", type=int, default=None, help="Min difficulty tier")
    parser.add_argument("--json",     action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    r = Retriever()
    result = r.retrieve(
        query=args.query,
        macos_version=args.version,
        mac_chip=args.chip,
        category=args.category,
        n=args.n,
        min_tier=args.min_tier,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"\nQuery    : {result['query']}")
    print(f"Fallback : {result['fallback']}")
    print(f"Hits     : {len(result['hits'])}\n")

    if result["fallback"]:
        print("  No KB results above confidence threshold.")
        print("  → Would fall back to web search (apple.com/support, discussions.apple.com)\n")
        return

    for i, hit in enumerate(result["hits"], 1):
        tier    = hit.get("difficulty_tier", "?")
        score   = hit.get("score", 0)
        source  = hit.get("source", "")
        cat     = hit.get("category", "")
        vers    = ", ".join(hit.get("macos_versions", []))
        title   = hit.get("title", "(no title)")
        url     = hit.get("url", "")
        text    = hit.get("text", "")[:300]

        print(f"[{i}] score={score:.4f}  tier={tier}  {source}/{cat}  versions={vers}")
        print(f"    {title}")
        if url:
            print(f"    {url}")
        print(f"    {text}...")
        print()


if __name__ == "__main__":
    main()
