#!/usr/bin/env bash
# Roll back to a known-good commit: check out its config and run the image CI
# published for that same commit. No rebuild happens on the VM.
set -euo pipefail

TARGET_COMMIT="${1:?usage: rollback.sh <known-good-commit>}"

cd "${DEPLOY_ROOT:-/opt/macos-agent}"
# shellcheck source=deploy/oracle/_common.sh
source deploy/oracle/_common.sh

require_env_file

git fetch origin main
# Expand short SHAs — image tags are the full 40-character SHA.
RESOLVED="$(git rev-parse --verify "${TARGET_COMMIT}^{commit}")"
git switch --detach "$RESOLVED"
launch_pinned "$RESOLVED"

echo
echo "Rolled back to $RESOLVED. Verify before walking away:"
echo "  scripts/smoke_test.sh \"https://\${APP_DOMAIN}\""
