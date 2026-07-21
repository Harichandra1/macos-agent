#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/macos-agent}"
REPO_URL="${REPO_URL:?set REPO_URL to the GitHub HTTPS or SSH repository URL}"

sudo apt-get update
sudo apt-get install -y ca-certificates curl git
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo "Log out and back in once so the docker group is active, then rerun this script."
  exit 0
fi

sudo mkdir -p "$DEPLOY_ROOT"
sudo chown -R "$USER":"$USER" "$DEPLOY_ROOT"
if [ ! -d "$DEPLOY_ROOT/.git" ]; then
  git clone "$REPO_URL" "$DEPLOY_ROOT"
fi
cd "$DEPLOY_ROOT"
test -f .env.production || {
  echo "Copy deploy/oracle/.env.production.example to $DEPLOY_ROOT/.env.production and fill it first." >&2
  exit 1
}
docker compose --env-file .env.production -f deploy/oracle/docker-compose.yml up -d --build
