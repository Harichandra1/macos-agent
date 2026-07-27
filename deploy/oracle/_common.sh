#!/usr/bin/env bash
# Shared helpers for the Oracle Always Free deployment scripts.
#
# The VM never builds the serving image — GitHub Actions publishes it to GHCR
# tagged with the commit SHA, and these scripts pin APP_IMAGE_TAG to the commit
# that is checked out. That keeps the container and the compose/Caddy config on
# the same revision, and makes rollback a tag change instead of a rebuild.

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/macos-agent}"
COMPOSE_FILE="${COMPOSE_FILE:-deploy/oracle/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-.env.production}"
# CI builds two native architectures in parallel; 15 minutes is generous.
IMAGE_WAIT_SECONDS="${IMAGE_WAIT_SECONDS:-900}"

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

require_env_file() {
  test -f "$ENV_FILE" || {
    echo "Missing $DEPLOY_ROOT/$ENV_FILE — copy deploy/oracle/.env.production.example and fill it." >&2
    exit 1
  }
}

# Ask Compose itself for the resolved image reference rather than duplicating
# the APP_IMAGE/APP_IMAGE_TAG default here — one source of truth.
resolve_app_image() {
  compose config --format json \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["app"]["image"])'
}

# Block until the tag exists in the registry. Without this, deploying a commit
# faster than CI can build it would fail on an image-not-found and leave the
# previous container running with the new config already checked out.
wait_for_image() {
  local ref="$1" deadline
  deadline=$(( SECONDS + IMAGE_WAIT_SECONDS ))
  echo "Waiting for $ref to be published (up to ${IMAGE_WAIT_SECONDS}s)..."
  until docker buildx imagetools inspect "$ref" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $ref." >&2
      echo "Check the 'Build and publish serving image' workflow run, and confirm" >&2
      echo "the GHCR package is public or that this host has run 'docker login ghcr.io'." >&2
      exit 1
    fi
    sleep 15
  done
  echo "Image available: $ref"
}

# Bring the stack up on a pinned tag and report what is running.
launch_pinned() {
  local tag="$1" ref
  export APP_IMAGE_TAG="$tag"
  compose config >/dev/null
  ref="$(resolve_app_image)"
  wait_for_image "$ref"
  compose pull app
  compose up -d --remove-orphans
  compose ps
}
