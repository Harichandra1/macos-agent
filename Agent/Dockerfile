# macOS Troubleshooting Agent — serving container (backend API + static frontend).
#
# Build context is the REPO ROOT (pyproject.toml + uv.lock live there):
#   docker build -f Agent/Dockerfile -t macos-agent .
#   docker run --rm -p 8000:8000 --env-file .env macos-agent
#
# The KB lives in Qdrant Cloud and app state in managed Postgres, so this image
# contains NO knowledge-base data and NO database — only the query/serve code.
# Credentials + DATABASE_URL are injected as env vars at run time, never baked in.
#
# On startup the app runs Alembic migrations (app.main lifespan) before serving,
# so a fresh database reaches the current schema with no manual step. Weekly
# credits reset lazily at request time; no worker or hosted cron is required.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    PORT=8000

WORKDIR /app

# uv for fast, locked dependency installs.
RUN pip install --no-cache-dir uv

# Install dependencies from the lockfile first (cached layer). --no-install-project
# skips building the repo itself (it's a collection of scripts, not a package).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy only the serving code (no vectorDBIngestion, no KB data).
COPY Agent/backend  ./Agent/backend
COPY Agent/frontend ./Agent/frontend

WORKDIR /app/Agent/backend

# Non-root runtime user.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Liveness for local Docker and the Oracle Compose deployment. Pure-stdlib
# check — slim has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

# WEB_CONCURRENCY workers (default 1). The production checkpointer uses the
# shared Postgres DATABASE_URL, so sessions can move between workers. Keep the
# default at 1 on free tiers for resource limits, not for correctness.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
