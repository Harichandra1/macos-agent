#!/usr/bin/env bash
# Rebuild the host from scratch after Oracle reclaims a VM. Application state
# (users, credits, feedback, LangGraph checkpoints) lives in Neon and the KB
# lives in Qdrant Cloud, so nothing here restores data — this only recreates
# the runtime.
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/macos-agent}"
REPO_URL="${REPO_URL:?set REPO_URL to the GitHub HTTPS or SSH repository URL}"

sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3
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
# shellcheck source=deploy/oracle/_common.sh
source deploy/oracle/_common.sh

require_env_file
launch_pinned "$(git rev-parse HEAD)"
