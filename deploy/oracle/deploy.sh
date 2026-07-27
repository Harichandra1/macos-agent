#!/usr/bin/env bash
# Normal update: fast-forward to origin/main and run the image CI built for it.
set -euo pipefail

cd "${DEPLOY_ROOT:-/opt/macos-agent}"
# shellcheck source=deploy/oracle/_common.sh
source deploy/oracle/_common.sh

require_env_file

git pull --ff-only origin main
launch_pinned "$(git rev-parse HEAD)"
