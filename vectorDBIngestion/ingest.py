"""
ingest.py — Embed chunks.jsonl and upsert to Qdrant Cloud.

Pipeline:
    chunks.jsonl  →  OpenAI text-embedding-3-small  →  Qdrant Cloud (macos_kb)

Usage:
    python ingest.py                  # embed + upsert all chunks
    python ingest.py --dry-run        # cost estimate only, no API calls
    python ingest.py --limit 100      # test with first 100 chunks
    python ingest.py --incremental    # skip already-embedded chunks
    python ingest.py --reset          # delete collection and re-embed from scratch

Prerequisites:
    1. Copy .env.example to .env and fill in all four values.
    2. Create a free Qdrant Cloud cluster at https://cloud.qdrant.io
       (1 GB free tier is enough for this project — ~250 MB of vectors).
    3. uv sync   (installs openai, qdrant-client, python-dotenv)
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    KeywordIndexParams,
    IntegerIndexParams,
    MatchValue,
    OverwritePayloadOperation,
    PayloadSchemaType,
    PointStruct,
    SetPayload,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).parent
KB_DATA_DIR    = PROJECT_ROOT / "data" / "knowledge_base"
CHUNKS_FILE    = KB_DATA_DIR / "chunks.jsonl"
PROGRESS_FILE  = KB_DATA_DIR / "ingest_progress.json"

# ---------------------------------------------------------------------------
# Embedding config
# ---------------------------------------------------------------------------

EMBED_MODEL    = "text-embedding-3-small"
VECTOR_DIM     = 1536
EMBED_COST_PER_1M_TOKENS = 0.02   # USD, as of 2024
BATCH_SIZE     = 100               # chunks per OpenAI call + Qdrant upsert

# ---------------------------------------------------------------------------
# Helpers — stable point IDs
# ---------------------------------------------------------------------------

def chunk_to_point_id(chunk_id: str) -> str:
    """
    Derive a stable UUID from chunk_id.
    Same chunk always maps to the same Qdrant point ID.
    Qdrant upsert with the same ID overwrites in place — safe to re-run.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"macos_kb.{chunk_id}"))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load and validate environment variables from .env."""
    load_dotenv()
    required = {
        "OPENAI_API_KEY":    os.environ.get("OPENAI_API_KEY"),
        "QDRANT_URL":        os.environ.get("QDRANT_URL"),
        "QDRANT_API_KEY":    os.environ.get("QDRANT_API_KEY"),
        "QDRANT_COLLECTION": os.environ.get("QDRANT_COLLECTION", "macos_kb"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print(
            f"\n  ✗ Missing environment variables: {', '.join(missing)}\n"
            "  Copy .env.example to .env and fill in all values.\n"
        )
        sys.exit(1)
    return required


# ---------------------------------------------------------------------------
# Progress tracking — checkpoint to disk every batch
# ---------------------------------------------------------------------------

def load_progress() -> set:
    """Return set of chunk_ids already successfully embedded and upserted."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_progress(embedded_ids: set):
    """Persist the set of completed chunk_ids."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(embedded_ids), f)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

def load_chunks(limit: Optional[int] = None) -> list[dict]:
    """Read all chunks from chunks.jsonl. Apply --limit if given."""
    if not CHUNKS_FILE.exists():
        print(f"  ✗ chunks.jsonl not found at {CHUNKS_FILE}")
        sys.exit(1)
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                break
    return chunks


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------

def estimate_cost(chunks: list[dict]) -> float:
    """Print a cost estimate and return total estimated tokens."""
    total_chars  = sum(len(c.get("text", "")) for c in chunks)
    total_tokens = total_chars / 4           # ~4 chars per token (rough)
    cost_usd     = (total_tokens / 1_000_000) * EMBED_COST_PER_1M_TOKENS
    print(f"  Chunks to embed   : {len(chunks):,}")
    print(f"  Estimated tokens  : {total_tokens:,.0f}")
    print(f"  Estimated cost    : ${cost_usd:.4f} USD")
    return total_tokens


# ---------------------------------------------------------------------------
# Qdrant collection setup
# ---------------------------------------------------------------------------

def setup_collection(client: QdrantClient, collection: str, reset: bool = False):
    """
    Create the Qdrant collection and payload indexes if they don't exist.
    With --reset: delete first so you get a clean slate.
    """
    existing = {c.name for c in client.get_collections().collections}

    if reset and collection in existing:
        print(f"  --reset: deleting collection '{collection}'...")
        client.delete_collection(collection)
        existing.discard(collection)
        # Also clear progress so we re-embed everything
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("  Progress file cleared.")

    if collection not in existing:
        print(f"  Creating collection '{collection}' (dim={VECTOR_DIM}, cosine)...")
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,
                on_disk=False,   # keep in RAM — free tier has enough headroom
            ),
        )
        # Payload indexes enable fast metadata pre-filtering at query time.
        # These correspond to the filters described in CLAUDE.md retrieval spec.
        for field, schema in [
            ("category",        PayloadSchemaType.KEYWORD),
            ("source",          PayloadSchemaType.KEYWORD),
            ("difficulty_tier", PayloadSchemaType.INTEGER),
            ("macos_versions",  PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
            )
        print(f"  Collection created with 4 payload indexes.")
    else:
        count = client.count(collection_name=collection).count
        print(f"  Collection '{collection}' already exists ({count:,} points).")


# ---------------------------------------------------------------------------
# Embedding with exponential backoff
# ---------------------------------------------------------------------------

def embed_batch(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI embeddings API for a batch of texts.
    Retries up to 3 times on rate limit or transient errors.
    """
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            response = openai_client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except RateLimitError:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"\n  [OpenAI] Rate limited — retrying in {wait}s...")
            time.sleep(wait)
        except APIError as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"\n  [OpenAI] API error ({e}) — retrying in {wait}s...")
            time.sleep(wait)
    return []  # unreachable


# ---------------------------------------------------------------------------
# Build Qdrant points
# ---------------------------------------------------------------------------

def build_payload(chunk: dict) -> dict:
    """
    Build the full Qdrant payload for a chunk. Every chunk field is stored so
    retrieved points are self-contained — no join back to the article files.
    """
    return {
        # Identity
        "chunk_id":        chunk["chunk_id"],
        "article_id":      chunk.get("article_id", ""),
        # Content
        "text":            chunk["text"],
        "section_heading": chunk.get("section_heading", ""),
        "title":           chunk.get("title", ""),
        "url":             chunk.get("url", ""),
        # Metadata used for filtering at retrieval time
        "source":          chunk.get("source", ""),
        "category":        chunk.get("category", "general"),
        "difficulty_tier": chunk.get("difficulty_tier", 1),
        "macos_versions":  chunk.get("macos_versions", ["all"]),
        "affected_devices": chunk.get("affected_devices", []),
        # Provenance
        "locale":          chunk.get("locale", "en-us"),
        "scraped_at":      chunk.get("scraped_at", ""),
    }


def build_point(chunk: dict, embedding: list[float]) -> PointStruct:
    """Convert a chunk dict + its embedding into a Qdrant PointStruct."""
    return PointStruct(
        id=chunk_to_point_id(chunk["chunk_id"]),
        vector=embedding,
        payload=build_payload(chunk),
    )


# ---------------------------------------------------------------------------
# Payload-only re-sync (no re-embedding unless a chunk's text actually changed)
# ---------------------------------------------------------------------------

def _qdrant_retry(fn, *args, _attempts: int = 4, **kwargs):
    """Call a Qdrant client method, retrying transient cloud timeouts with backoff."""
    for attempt in range(_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == _attempts - 1:
                raise
            wait = 2 ** attempt
            print(f"\n  [Qdrant] {type(e).__name__} — retrying in {wait}s "
                  f"(attempt {attempt + 1}/{_attempts})...")
            time.sleep(wait)


def sync_payloads(
    chunks:        list[dict],
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    collection:    str,
    embedded_ids:  set,
):
    """
    Refresh payloads after a metadata migration without re-embedding.

    Strategy (no per-batch reads — those are the main cloud-timeout risk):
      - NEW chunks (chunk_id not in the ingest progress set) → embed + upsert.
      - ALL chunks → overwrite payload with current metadata. Idempotent: safe to
        re-run if interrupted. Uses wait=False so the cloud applies writes async,
        which keeps each round-trip short.
    `embed_text` is unchanged for migrated chunks, so existing vectors stay valid;
    only the handful of brand-new chunks cost an embedding call.
    """
    total = len(chunks)
    overwritten = 0
    new = 0

    # 1. Embed + upsert any genuinely new chunks (chunk_id absent from progress)
    new_chunks = [c for c in chunks if c["chunk_id"] not in embedded_ids]
    if new_chunks:
        print(f"  {len(new_chunks):,} new chunk(s) need embedding...")
        for s in range(0, len(new_chunks), BATCH_SIZE):
            grp = new_chunks[s : s + BATCH_SIZE]
            embs = embed_batch(openai_client, [c["text"] for c in grp])
            pts  = [build_point(c, e) for c, e in zip(grp, embs)]
            _qdrant_retry(qdrant_client.upsert, collection_name=collection, points=pts)
            for c in grp:
                embedded_ids.add(c["chunk_id"])
            new += len(grp)
        save_progress(embedded_ids)

    # 2. Overwrite payloads for every chunk (metadata refresh, no vectors sent)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        ops = [
            OverwritePayloadOperation(
                overwrite_payload=SetPayload(
                    payload=build_payload(c),
                    points=[chunk_to_point_id(c["chunk_id"])],
                )
            )
            for c in batch
        ]
        _qdrant_retry(
            qdrant_client.batch_update_points,
            collection_name=collection, update_operations=ops, wait=False,
        )
        overwritten += len(batch)

        done = batch_start + len(batch)
        if done % 2000 < BATCH_SIZE or done >= total:
            pct = done * 100 // total
            print(f"  [{pct:>3}%] {done:>6,}/{total:,} payloads overwritten "
                  f"| new embedded {new:,}")

    return overwritten, 0, new


# ---------------------------------------------------------------------------
# Main embed + upsert loop
# ---------------------------------------------------------------------------

def embed_and_upsert(
    chunks:         list[dict],
    openai_client:  OpenAI,
    qdrant_client:  QdrantClient,
    collection:     str,
    embedded_ids:   set,
):
    """
    Embed chunks in batches of BATCH_SIZE and upsert to Qdrant.
    Checkpoints progress after every batch so an interrupted run can resume.
    """
    total     = len(chunks)
    saved     = 0
    skipped   = 0
    failed    = 0
    start_ts  = time.time()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]

        # Skip chunks already embedded in a previous run
        pending = [c for c in batch if c["chunk_id"] not in embedded_ids]
        if not pending:
            skipped += len(batch)
            continue

        texts = [c["text"] for c in pending]

        # Embed
        try:
            embeddings = embed_batch(openai_client, texts)
        except Exception as e:
            print(f"\n  ✗ Embed failed for batch at {batch_start}: {e}")
            failed += len(pending)
            continue

        # Build Qdrant points
        points = [build_point(c, emb) for c, emb in zip(pending, embeddings)]

        # Upsert
        try:
            qdrant_client.upsert(collection_name=collection, points=points)
        except Exception as e:
            print(f"\n  ✗ Upsert failed for batch at {batch_start}: {e}")
            failed += len(pending)
            continue

        # Checkpoint
        for c in pending:
            embedded_ids.add(c["chunk_id"])
        save_progress(embedded_ids)
        saved += len(pending)
        skipped += len(batch) - len(pending)

        # Progress log every 1000 chunks
        done = batch_start + len(batch)
        if done % 1000 < BATCH_SIZE or done >= total:
            elapsed  = time.time() - start_ts
            rate     = saved / elapsed if elapsed > 0 else 0
            pct      = done * 100 // total
            eta_secs = int((total - done) / rate) if rate > 0 else 0
            print(
                f"  [{pct:>3}%] {done:>6,}/{total:,} chunks "
                f"| saved {saved:,} | skipped {skipped:,} | failed {failed} "
                f"| {rate:.0f} chunks/s | ETA {eta_secs}s"
            )

    return saved, skipped, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Embed chunks.jsonl and upsert to Qdrant Cloud."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show cost estimate only. No API calls, no data written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N chunks (useful for testing).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Skip chunks that were already embedded in a previous run.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the Qdrant collection and re-embed everything from scratch.",
    )
    parser.add_argument(
        "--sync-payload",
        action="store_true",
        help="Refresh payloads from chunks.jsonl after a metadata migration. "
             "Reuses existing vectors; only re-embeds new/changed chunks ($0 for "
             "pure metadata changes).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("INGEST — embed chunks.jsonl → Qdrant Cloud")
    print("=" * 60)

    # 1. Load and validate config
    config = load_config()
    collection = config["QDRANT_COLLECTION"]

    # 2. Load chunks
    print(f"\n[1] Loading chunks from {CHUNKS_FILE.relative_to(PROJECT_ROOT)}...")
    chunks = load_chunks(limit=args.limit)
    print(f"  Loaded {len(chunks):,} chunks")

    # 3. Dry-run: show cost estimate and exit
    if args.dry_run:
        print("\n[DRY RUN] Cost estimate:")
        estimate_cost(chunks)
        print("\n  No API calls made. Remove --dry-run to proceed.")
        return

    # 3b. Payload-only re-sync mode (metadata migration follow-up)
    if args.sync_payload:
        print(f"\n[2] Connecting to Qdrant at {config['QDRANT_URL']}...")
        qdrant = QdrantClient(url=config["QDRANT_URL"],
                              api_key=config["QDRANT_API_KEY"], timeout=60)
        before = qdrant.count(collection_name=collection).count
        print(f"  Collection '{collection}' currently holds {before:,} points.")
        openai = OpenAI(api_key=config["OPENAI_API_KEY"])
        embedded_ids = load_progress()
        print(f"  Progress file: {len(embedded_ids):,} chunk_ids already embedded.")
        print(f"\n[3] Re-syncing payloads (overwrite metadata, embed only new chunks)...\n")
        start = time.time()
        overwritten, _, new = sync_payloads(chunks, openai, qdrant, collection, embedded_ids)
        after = qdrant.count(collection_name=collection).count
        print("\n" + "=" * 60)
        print("SYNC COMPLETE")
        print("=" * 60)
        print(f"  Payloads overwritten : {overwritten:,}")
        print(f"  New chunks embedded  : {new:,}")
        print(f"  Points: {before:,} → {after:,}")
        print(f"  Time  : {time.time() - start:.1f}s")
        print()
        return

    # 4. Load progress (for incremental mode)
    embedded_ids = load_progress() if args.incremental else set()
    if args.incremental and embedded_ids:
        print(f"  Incremental mode: {len(embedded_ids):,} chunks already done, will skip them")

    # 5. Show cost estimate and confirm
    print("\n[2] Cost estimate:")
    estimate_cost(chunks)
    if not args.limit and sys.stdin.isatty():
        confirm = input("\n  Proceed with embedding? [y/N]: ").strip().lower()
        if confirm != "y":
            print("  Aborted.")
            return

    # 6. Connect to Qdrant and set up collection
    print(f"\n[3] Connecting to Qdrant at {config['QDRANT_URL']}...")
    qdrant = QdrantClient(
        url=config["QDRANT_URL"],
        api_key=config["QDRANT_API_KEY"],
        timeout=30,
    )
    setup_collection(qdrant, collection, reset=args.reset)

    # 7. Connect to OpenAI
    openai = OpenAI(api_key=config["OPENAI_API_KEY"])

    # 8. Embed + upsert
    print(f"\n[4] Embedding and upserting to '{collection}'...")
    print(f"  Batch size: {BATCH_SIZE} | Model: {EMBED_MODEL}")
    print()
    start = time.time()
    saved, skipped, failed = embed_and_upsert(
        chunks, openai, qdrant, collection, embedded_ids
    )
    elapsed = time.time() - start

    # 9. Final report
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Saved    : {saved:,} new points upserted")
    print(f"  Skipped  : {skipped:,} already in Qdrant")
    print(f"  Failed   : {failed}")
    print(f"  Time     : {elapsed:.1f}s")
    final_count = qdrant.count(collection_name=collection).count
    print(f"  Total points in '{collection}': {final_count:,}")
    print()
    print(f"  Progress saved to: {PROGRESS_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  Query your collection via:")
    print(f"    URL : {config['QDRANT_URL']}/collections/{collection}")
    print()


if __name__ == "__main__":
    main()
