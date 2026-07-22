# Vector ingestion

This directory contains the offline knowledge-base pipeline. It scrapes and
normalizes macOS troubleshooting sources, creates `chunks.jsonl`, embeds the
chunks, and upserts the resulting vectors and metadata into the Qdrant
collection configured by `QDRANT_COLLECTION`.

The deployed FastAPI application does not need the raw source dump or the
embedding artifacts. It queries Qdrant directly at runtime. The generated
dataset under `data/` is intentionally ignored by Git because it contains
large source archives and derived files; keep it locally or in object storage
if the ingestion job must be reproducible elsewhere.

Typical workflow:

1. Run the scrapers locally to refresh the source material.
2. Build or update `chunks.jsonl`.
3. Run `ingest.py` with the local provider, Qdrant, and OpenAI environment.
4. Deploy the application; it reads the already-populated Qdrant collection.

Do not run ingestion as part of the web container startup. Re-run it only when
the knowledge base changes.
