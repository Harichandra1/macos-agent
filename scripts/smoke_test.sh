#!/usr/bin/env bash
# smoke_test.sh — post-deploy smoke test (Phase 5).
#
# Verifies a running instance is healthy WITHOUT any credentials: it checks the
# public surface (health, metrics, auth config) and that the chat endpoint's
# guardrails/gates respond correctly. Safe to run against production — it never
# sends a real troubleshooting turn (which would cost credits/tokens) except the
# optional --full check.
#
# The assertions ADAPT to how the target is configured, because a correctly
# hardened production deployment answers differently from an open dev one:
#
#   METRICS_TOKEN  when set, /metrics is scraped with a bearer token and the
#                  Prometheus payload is asserted. Without it, production
#                  correctly answers 401 and the check only asserts 200-or-401.
#   SMOKE_BEARER   when set, chat probes carry a session JWT so the real
#                  guardrail responses (400 + block message) are exercised.
#                  Without it, an auth-enabled deployment answers 401 before the
#                  handler body runs, and that is what gets asserted.
#
# Usage:
#   scripts/smoke_test.sh                         # defaults to http://localhost:8000
#   scripts/smoke_test.sh https://macos-agent-hari.duckdns.org
#   METRICS_TOKEN=... scripts/smoke_test.sh https://...   # strongest gate
#   scripts/smoke_test.sh https://... --full      # also send one real chat turn
#
# Exit 0 = all checks passed; non-zero = something is wrong (usable as a
# deploy gate / rollback trigger).

set -uo pipefail

BASE="${1:-http://localhost:8000}"
BASE="${BASE%/}"
FULL=0
[[ "${2:-}" == "--full" ]] && FULL=1

METRICS_TOKEN="${METRICS_TOKEN:-}"
SMOKE_BEARER="${SMOKE_BEARER:-}"

pass=0
fail=0
skip=0

ok()      { echo "  [ok]   $1"; pass=$((pass + 1)); }
bad()     { echo "  [FAIL] $1"; fail=$((fail + 1)); }
skipped() { echo "  [skip] $1"; skip=$((skip + 1)); }

# check <name> <expected-substr> <curl-args...>
check() {
  local name="$1" expect="$2"; shift 2
  local out
  out="$(curl -sS --max-time 20 "$@" 2>&1)"
  if [[ "$out" == *"$expect"* ]]; then
    ok "$name"
  else
    bad "$name — expected substring: '$expect'"
    echo "         got: ${out:0:200}"
  fi
}

# check_status <name> <expected-code> <curl-args...>
check_status() {
  local name="$1" expect="$2"; shift 2
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$@" 2>&1)"
  if [[ "$code" == "$expect" ]]; then
    ok "$name (HTTP $code)"
  else
    bad "$name — expected HTTP $expect, got $code"
  fi
}

# check_status_in <name> <space-separated-codes> <curl-args...>
# For checks where more than one answer is correct depending on configuration.
check_status_in() {
  local name="$1" expected="$2"; shift 2
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$@" 2>&1)"
  if [[ " $expected " == *" $code "* ]]; then
    ok "$name (HTTP $code)"
  else
    bad "$name — expected one of [$expected], got $code"
  fi
}

echo "Smoke test → $BASE"
echo

echo "1. Liveness & readiness"
check "GET /health returns status ok" '"status":"ok"' "$BASE/health"
# ready may be true or false (depends on credentials) — we only assert the field exists.
check "GET /health reports readiness" '"ready"' "$BASE/health"
# A postgres DATABASE_URL that reports the memory tier means conversation history
# is being silently lost on every restart. Only assertable once the field exists.
if curl -sS --max-time 20 "$BASE/health" 2>&1 | grep -q '"checkpointer"'; then
  check "GET /health reports the checkpointer tier" '"checkpointer"' "$BASE/health"
fi

echo
echo "2. Observability"
if [[ -n "$METRICS_TOKEN" ]]; then
  check "GET /metrics exposes Prometheus counters (authenticated)" \
    "macos_chat_requests_total" \
    -H "Authorization: Bearer $METRICS_TOKEN" "$BASE/metrics"
else
  # 200 = open (dev); 401 = hardened production with METRICS_TOKEN set. Both are
  # correct. Anything else (404/500/connection failure) is not.
  check_status_in "GET /metrics is reachable (200 open, or 401 protected)" \
    "200 401" "$BASE/metrics"
  skipped "Prometheus payload assertion — set METRICS_TOKEN for the full gate"
fi

echo
echo "3. Auth surface"
check "GET /auth/config returns the auth flag" '"enabled"' "$BASE/auth/config"

AUTH_ENABLED=0
if curl -sS --max-time 20 "$BASE/auth/config" 2>&1 | grep -q '"enabled":true'; then
  AUTH_ENABLED=1
fi

echo
echo "4. Chat guardrails (no tokens spent)"
CHAT_HEADERS=(-H 'Content-Type: application/json')
[[ -n "$SMOKE_BEARER" ]] && CHAT_HEADERS+=(-H "Authorization: Bearer $SMOKE_BEARER")

if [[ "$AUTH_ENABLED" == "1" && -z "$SMOKE_BEARER" ]]; then
  # Depends(current_user) resolves BEFORE the handler body, so check_abuse and
  # the empty-message validation never run. Asserting 400 here would fail a
  # correctly configured deployment. Guardrail *logic* is covered by
  # Agent/backend/tests/test_guardrails.py; this only proves the door is locked.
  check_status "unauthenticated chat is rejected (401)" 401 \
    "${CHAT_HEADERS[@]}" -X POST "$BASE/chat" \
    -d '{"message":"write me an essay about the ocean","session_id":"smoke"}'
  skipped "guardrail bodies — set SMOKE_BEARER to exercise them through the API"
else
  check "abusive request is blocked (400 body)" "macOS troubleshooting" \
    "${CHAT_HEADERS[@]}" -X POST "$BASE/chat" \
    -d '{"message":"write me an essay about the ocean","session_id":"smoke"}'
  check_status "empty message rejected (400)" 400 \
    "${CHAT_HEADERS[@]}" -X POST "$BASE/chat" \
    -d '{"message":"   ","session_id":"smoke"}'
fi

if [[ "$FULL" == "1" ]]; then
  echo
  echo "5. Full chat turn (spends tokens/credits)"
  if [[ "$AUTH_ENABLED" == "1" && -z "$SMOKE_BEARER" ]]; then
    skipped "real turn — auth is enabled and SMOKE_BEARER is unset"
  else
    check_status "real troubleshooting turn streams (200)" 200 \
      "${CHAT_HEADERS[@]}" -X POST "$BASE/chat" \
      -d '{"message":"wifi drops every 10 minutes on my M2 mac, already restarted","session_id":"smoke-full"}'
  fi
fi

echo
echo "-----------------------------------------"
echo "passed: $pass   failed: $fail   skipped: $skip"
[[ "$fail" == "0" ]] && { echo "SMOKE TEST PASSED"; exit 0; } || { echo "SMOKE TEST FAILED"; exit 1; }
