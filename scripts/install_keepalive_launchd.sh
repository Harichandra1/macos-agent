#!/usr/bin/env bash
# Install / remove the daily Qdrant keepalive as a macOS LaunchAgent.
#
# WHY LAUNCHD AND NOT ONLY GITHUB ACTIONS: the Actions workflow in
# .github/workflows/qdrant-keepalive.yml is the primary heartbeat, but it has
# two conditions this does not — it only runs from the repository's DEFAULT
# branch, and GitHub disables scheduled workflows in a public repository after
# 60 days without repository activity. Both failure modes are silent, which is
# precisely how the cluster was lost before.
#
# This LaunchAgent has neither condition, and needs no credentials uploaded
# anywhere: it reads the repo's own .env. Its one weakness is the obvious one
# — it cannot run while the Mac is off. That matters less than it sounds,
# because launchd runs a StartCalendarInterval job that was missed during
# sleep at the next wake. Qdrant suspends at one week and deletes at four, so
# any Mac that gets opened even weekly keeps the cluster alive.
#
# Run both. They fail independently, which is the point.
#
# Usage:
#   scripts/install_keepalive_launchd.sh install
#   scripts/install_keepalive_launchd.sh status
#   scripts/install_keepalive_launchd.sh uninstall
set -euo pipefail

LABEL="com.macosagent.qdrant-keepalive"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$REPO_ROOT/scripts/qdrant_keepalive_local.sh"
LOG_FILE="$HOME/Library/Logs/qdrant-keepalive.log"

# launchctl's modern subcommands are addressed by GUI domain, not by label
# alone; getting this wrong is the usual reason an agent appears installed but
# never fires.
DOMAIN="gui/$(id -u)"

usage() { echo "usage: $(basename "$0") {install|uninstall|status}" >&2; exit 64; }

cmd_install() {
  [[ -x "$RUNNER" ]] || chmod +x "$RUNNER"
  mkdir -p "$(dirname "$PLIST")"

  # 03:17 rather than a round hour: nothing else on the machine wants that
  # minute, and it keeps this out of the way of any on-the-hour jobs.
  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${RUNNER}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>17</integer>
    </dict>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>${LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_FILE}</string>
</dict>
</plist>
PLIST_EOF

  # bootout first so "install" is idempotent and picks up an edited plist.
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$PLIST"

  echo "Installed $LABEL (daily at 03:17)."
  echo "Plist: $PLIST"
  echo "Log:   $LOG_FILE"
  echo
  echo "Run it once now to prove it works:"
  echo "  launchctl kickstart -p $DOMAIN/$LABEL"
}

cmd_uninstall() {
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL."
}

cmd_status() {
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "$LABEL is loaded."
    launchctl print "$DOMAIN/$LABEL" | grep -E 'state|last exit code|runs' || true
  else
    echo "$LABEL is NOT loaded."
  fi
  echo
  echo "--- last 10 log lines ---"
  tail -10 "$LOG_FILE" 2>/dev/null || echo "(no log yet)"
}

case "${1-}" in
  install)   cmd_install ;;
  uninstall) cmd_uninstall ;;
  status)    cmd_status ;;
  *)         usage ;;
esac
