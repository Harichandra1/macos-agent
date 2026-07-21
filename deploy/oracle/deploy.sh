#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/macos-agent}"
cd "$DEPLOY_ROOT"

test -f .env.production || {
  echo "Missing $DEPLOY_ROOT/.env.production" >&2
  exit 1
}

git pull --ff-only origin main
docker compose --env-file .env.production -f deploy/oracle/docker-compose.yml config >/dev/null
docker compose --env-file .env.production -f deploy/oracle/docker-compose.yml up -d --build
docker compose --env-file .env.production -f deploy/oracle/docker-compose.yml ps
