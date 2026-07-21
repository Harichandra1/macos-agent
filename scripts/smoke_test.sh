#!/usr/bin/env bash
# smoke_test.sh — post-deploy smoke test (Phase 5).
#
# Verifies a running instance is healthy WITHOUT any credentials: it checks the
# public surface (health, metrics, auth config) and that the chat endpoint's
# guardrails/gates respond correctly. Safe to run against production — it never
# sends a real troubleshooting turn (which would cost credits/tokens) except the
# optional --full check.
#
# Usage:
#   scripts/smoke_test.sh                         # defaults to http://localhost:8000
#   scripts/smoke_test.sh https://macos-agent.onrender.com
#   scripts/smoke_test.sh https://... --full      # also send one real chat turn
#
# Exit 0 = all checks passed; non-zero = something is wrong (usable as a
# deploy gate / rollback trigger).

set -uo pipefail

BASE="${1:-http://localhost:8000}"
BASE="${BASE%/}"
FULL=0
[[ "${2:-}" == "--full" ]] && FULL=1

pass=0
fail=0

# check <name> <expected-substr> <curl-args...>
check() {
  local name="$1" expect="$2"; shift 2
  local out
  out="$(curl -sS --max-time 20 "$@" 2>&1)"
  if [[ "$out" == *"$expect"* ]]; then
    echo "  [ok]   $name"
    ((pass++))
  else
    echo "  [FAIL] $name — expected substring: '$expect'"
    echo "         got: ${out:0:200}"
    ((fail++))
  fi
}

# check_status <name> <expected-code> <curl-args...>
check_status() {
  local name="$1" expect="$2"; shift 2
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$@" 2>&1)"
  if [[ "$code" == "$expect" ]]; then
    echo "  [ok]   $name (HTTP $code)"
    ((pass++))
  else
    echo "  [FAIL] $name — expected HTTP $expect, got $code"
    ((fail++))
  fi
}

echo "Smoke test → $BASE"
echo

echo "1. Liveness & readiness"
check "GET /health returns status ok" '"status":"ok"' "$BASE/health"
# ready may be true or false (depends on credentials) — we only assert the field exists.
check "GET /health reports readiness" '"ready"' "$BASE/health"

echo
echo "2. Observability"
check "GET /metrics exposes Prometheus counters" "macos_chat_requests_total" "$BASE/metrics"

echo
echo "3. Auth surface"
check "GET /auth/config returns the auth flag" '"enabled"' "$BASE/auth/config"

echo
echo "4. Chat guardrails (no tokens spent)"
check "abusive request is blocked (400 body)" "macOS troubleshooting" \
  -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d '{"message":"write me an essay about the ocean","session_id":"smoke"}'
check_status "empty message rejected (400)" 400 \
  -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d '{"message":"   ","session_id":"smoke"}'

if [[ "$FULL" == "1" ]]; then
  echo
  echo "5. Full chat turn (spends tokens/credits)"
  check_status "real troubleshooting turn streams (200)" 200 \
    -X POST "$BASE/chat" -H 'Content-Type: application/json' \
    -d '{"message":"wifi drops every 10 minutes on my M2 mac, already restarted","session_id":"smoke-full"}'
fi

echo
echo "-----------------------------------------"
echo "passed: $pass   failed: $fail"
[[ "$fail" == "0" ]] && { echo "SMOKE TEST PASSED"; exit 0; } || { echo "SMOKE TEST FAILED"; exit 1; }
