#!/usr/bin/env bash
# Local wrapper around qdrant_keepalive.sh for the scheduled job on this Mac.
#
# WHY A WRAPPER: the keepalive itself takes its credentials from the
# environment, which is right for CI but useless to launchd — a LaunchAgent
# starts with an almost empty environment and inherits nothing from a login
# shell. This wrapper is the piece that loads .env, so the credentials live in
# exactly one place (the repo's .env, already gitignored) and are never copied
# into a plist, a crontab line, or a scheduler's secret store.
#
# It also appends to a log, because a scheduled job that fails silently is the
# exact failure this whole mechanism exists to prevent. Check it with:
#   tail -20 ~/Library/Logs/qdrant-keepalive.log
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${QDRANT_KEEPALIVE_LOG:-$HOME/Library/Logs/qdrant-keepalive.log}"
ENV_FILE="${QDRANT_KEEPALIVE_ENV:-$REPO_ROOT/.env}"

mkdir -p "$(dirname "$LOG_FILE")"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

  if [[ ! -r "$ENV_FILE" ]]; then
    echo "FAIL: cannot read $ENV_FILE"
    exit 2
  fi

  # Only export the three keys this needs. Sourcing the whole file would pull
  # every other secret in it into the process environment for no reason.
  while IFS= read -r line; do
    case "$line" in
      QDRANT_URL=*|QDRANT_API_KEY=*|QDRANT_COLLECTION=*) export "${line?}" ;;
    esac
  done < "$ENV_FILE"

  bash "$REPO_ROOT/scripts/qdrant_keepalive.sh"
  status=$?

  if [[ $status -eq 0 ]]; then
    echo "OK"
  else
    echo "FAILED (exit $status)"
  fi
  exit $status
} >> "$LOG_FILE" 2>&1
