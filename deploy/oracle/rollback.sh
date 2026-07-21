#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/macos-agent}"
TARGET_COMMIT="${1:?usage: rollback.sh <known-good-commit>}"
cd "$DEPLOY_ROOT"

git fetch origin main
git show --quiet "$TARGET_COMMIT"
git switch --detach "$TARGET_COMMIT"
docker compose --env-file .env.production -f deploy/oracle/docker-compose.yml up -d --build
docker compose --env-file .env.production -f deploy/oracle/docker-compose.yml ps
