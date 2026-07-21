"""
migrate_metadata.py — One-time, improve-only metadata backfill over existing
article JSONs, then rebuild chunks.jsonl + index.json.

What it fixes (no network, no dump re-parse):
  1. category == "general"  → re-derive from title+summary+section text via the
     scored keyword classifier. Improve-only: a non-general category is never
     overwritten, so this can only help.
  2. macos_versions == ["all"]  → scan body text for version mentions; replace
     only if something is found. Otherwise left as ["all"].

After patching the article JSONs in place, it calls
KnowledgeBase.rebuild_from_articles() to regenerate the chunk store. Chunk text
is unchanged, so Qdrant point IDs stay stable and a payload-only re-sync
(`python ingest.py --sync-payload`) is all that's needed afterwards.

Usage:
    python migrate_metadata.py            # patch + rebuild
    python migrate_metadata.py --dry-run  # report what would change, write nothing
"""

import argparse
import glob
import json
import os
from collections import Counter

from knowledge_base import (
    KnowledgeBase,
    KB_ARTICLES_DIR,
    normalize_category,
    _classify_text_category,
    extract_versions_from_text,
)

# Single source of truth: reuse the KB's own (now data/-anchored) articles dir.
ARTICLES_GLOB = os.path.join(KB_ARTICLES_DIR, "*.json")


def _article_text(article: dict) -> str:
    """Concatenate the human-readable text fields for classification."""
    return " ".join([
        article.get("title", ""),
        article.get("summary", ""),
        " ".join(s.get("body", "") for s in article.get("sections", [])),
    ])


def _pct(n: int, total: int) -> str:
    return f"{n * 100 / total:.1f}%" if total else "0.0%"


def main():
    parser = argparse.ArgumentParser(description="Improve-only metadata backfill + rebuild")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report changes without writing or rebuilding.")
    args = parser.parse_args()

    files = sorted(glob.glob(ARTICLES_GLOB))
    if not files:
        print(f"  No article files found at {ARTICLES_GLOB}")
        return

    print(f"\n{'='*60}")
    print("METADATA MIGRATION — improve-only category + version backfill")
    print(f"{'='*60}\n")
    print(f"  Article files: {len(files):,}\n")

    # --- Single patch loop: tally before + after in-memory, write if not dry-run ---
    cat_before = Counter()
    cat_after  = Counter()
    ver_specific_before = 0
    ver_specific_after  = 0
    cat_changed = 0
    ver_changed = 0

    for p in files:
        a = json.load(open(p, encoding="utf-8"))

        # before tallies
        cat_before[normalize_category(a.get("category"))] += 1
        if a.get("macos_versions", ["all"]) != ["all"]:
            ver_specific_before += 1

        patched = False

        # 1. Category — improve only where currently general/missing
        if normalize_category(a.get("category")) == "general":
            new_cat = _classify_text_category(_article_text(a))
            if new_cat != "general":
                a["category"] = new_cat
                cat_changed += 1
                patched = True

        # 2. Versions — improve only where currently ["all"]
        if a.get("macos_versions", ["all"]) == ["all"]:
            found = extract_versions_from_text(_article_text(a))
            if found != ["all"]:
                a["macos_versions"] = found
                ver_changed += 1
                patched = True

        # after tallies (reflect in-memory patch even in dry-run)
        cat_after[normalize_category(a.get("category"))] += 1
        if a.get("macos_versions", ["all"]) != ["all"]:
            ver_specific_after += 1

        if patched and not args.dry_run:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(a, f, indent=2, ensure_ascii=False)

    total = len(files)
    print("  --- Category distribution (article level) ---")
    print(f"  {'category':<14} {'before':>10} {'after':>10}")
    for cat in sorted(set(cat_before) | set(cat_after)):
        print(f"  {cat:<14} {cat_before.get(cat,0):>10,} {cat_after.get(cat,0):>10,}")
    print(f"\n  'general' : {_pct(cat_before['general'], total)} → {_pct(cat_after['general'], total)}")
    print(f"  category re-labelled : {cat_changed:,}")
    print(f"  still general        : {cat_after['general']:,}")

    print("\n  --- Version coverage (article level) ---")
    print(f"  version-specific : {ver_specific_before:,} ({_pct(ver_specific_before,total)})"
          f" → {ver_specific_after:,} ({_pct(ver_specific_after,total)})")
    print(f"  versions added   : {ver_changed:,}")

    if args.dry_run:
        print("\n  [DRY RUN] No files written, no rebuild. Re-run without --dry-run to apply.\n")
        return

    # --- Rebuild chunk store from patched articles ---
    print(f"\n{'='*60}")
    print("REBUILD — regenerating chunks.jsonl + index.json from patched articles")
    print(f"{'='*60}\n")
    kb = KnowledgeBase()
    kb.rebuild_from_articles()

    stats = kb.stats()
    print(f"\n  KB now: {stats['total_articles']:,} articles, {stats['total_chunks']:,} chunks")
    print("\n  Next: python ingest.py --sync-payload   (push corrected payloads, $0)\n")


if __name__ == "__main__":
    main()
