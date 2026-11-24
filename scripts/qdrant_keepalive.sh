#!/usr/bin/env bash
# WHY THIS EXISTS: Qdrant Cloud suspends a free-tier cluster after one week
# "without use" and deletes it after four weeks of inactivity — and once
# deleted, "a cluster can't be recovered". This project has already lost the
# macos_kb collection twice that way, each loss costing a full re-ingest of
# ~66k embedded chunks. This script is the heartbeat that stops it happening a
# third time.
#
# Qdrant's docs state the one-week and four-week thresholds but never define
# what "use" means, so we deliberately do NOT rely on a cheap metadata GET
# resetting the timer. Instead each run performs real data-plane operations
# against the collection — a point read and a vector query — so the traffic is
# indistinguishable from an actual application query. Doing more than the
# minimum costs nothing here and removes the guess.
#
# Exits non-zero with a specific message on any failure, so the scheduled
# caller goes red and notifies rather than failing silently. Silent failure is
# the whole problem this guards against.
#
# Usage:  QDRANT_URL=... QDRANT_API_KEY=... [QDRANT_COLLECTION=macos_kb] \
#           scripts/qdrant_keepalive.sh
set -uo pipefail

COLLECTION="${QDRANT_COLLECTION:-macos_kb}"

if [[ -z "${QDRANT_URL:-}" || -z "${QDRANT_API_KEY:-}" ]]; then
  echo "FAIL: QDRANT_URL and QDRANT_API_KEY must both be set." >&2
  exit 2
fi

# Strip a trailing slash so the URLs below never double up.
BASE="${QDRANT_URL%/}"

# Never let the key reach stdout/stderr: it would land in scheduler logs, and
# on a public repository those logs are world-readable.
AUTH_HEADER="api-key: ${QDRANT_API_KEY}"

# A suspended cluster resets the TLS connection rather than returning an HTTP
# status, so a plain curl failure is the signature we most expect to see.
# --max-time is generous: a cold free-tier node can be slow to first byte.
#
# Request bodies arrive on stdin via --data-binary @-, never as a -d argument,
# and are built by python3 rather than string-interpolated. A literal JSON
# object written inline in shell source is a brace-expansion hazard:
# {"query":[...],"limit":1,"with_payload":false} expands to three separate
# words, stripping the outer braces and splitting at the top-level commas,
# which silently turns one request into three malformed ones. Stdin also keeps
# a 1536-float vector out of argv.
call() {
  local method="$1" path="$2" body="${3-}"
  local args=(-sS --max-time 90 -X "$method"
              -H "$AUTH_HEADER" -H 'Content-Type: application/json'
              -w '\n%{http_code}')
  if [[ -n "$body" ]]; then
    printf '%s' "$body" | curl "${args[@]}" --data-binary @- "${BASE}${path}" 2>&1
  else
    curl "${args[@]}" "${BASE}${path}" 2>&1
  fi
}

check() {
  local label="$1" response="$2"
  local code body
  code="$(printf '%s' "$response" | tail -n1)"
  body="$(printf '%s' "$response" | sed '$d')"

  if [[ ! "$code" =~ ^2[0-9][0-9]$ ]]; then
    echo "FAIL: ${label} -> HTTP ${code:-<no response>}" >&2
    # A reset connection yields no status code at all; surface curl's own text.
    printf '%s\n' "$body" | head -c 600 >&2
    echo >&2
    if [[ ! "$code" =~ ^[0-9]{3}$ ]] || [[ "$code" == "000" ]]; then
      echo "Hint: no HTTP status at all usually means the cluster is suspended" >&2
      echo "or deleted. Check the Qdrant Cloud console — a suspended cluster" >&2
      echo "can be reactivated with its data intact." >&2
    fi
    return 1
  fi
  printf '%s' "$body"
}

echo "keepalive: ${BASE} collection=${COLLECTION}"

# 1 — Collection info. Also tells us the vector size, so the query below does
#     not hardcode a dimension that could drift from the ingest config.
info="$(check "GET /collections/${COLLECTION}" "$(call GET "/collections/${COLLECTION}")")" || exit 1

read -r status points dim <<<"$(
  printf '%s' "$info" | python3 -c '
import json, sys
r = json.load(sys.stdin)["result"]
vectors = r["config"]["params"]["vectors"]
# Unnamed vectors give {"size": N}; named give {"name": {"size": N}}.
size = vectors.get("size") if isinstance(vectors, dict) and "size" in vectors \
    else next(iter(vectors.values()))["size"]
print(r.get("status"), r.get("points_count"), size)
'
)" || { echo "FAIL: could not parse collection info." >&2; exit 1; }

echo "  status=${status} points=${points} dim=${dim}"

if [[ "$points" == "0" || "$points" == "None" ]]; then
  echo "FAIL: collection '${COLLECTION}' has no points — the cluster is up but" >&2
  echo "the data is gone. Re-ingest with: python ingest.py --reset" >&2
  exit 1
fi

# 2 — Real point read. Touches the stored data, not just cluster metadata.
scroll_body="$(python3 -c 'import json; print(json.dumps({"limit": 1, "with_payload": False, "with_vector": False}))')"
check "POST /points/scroll" \
  "$(call POST "/collections/${COLLECTION}/points/scroll" "$scroll_body")" \
  >/dev/null || exit 1
echo "  scroll ok"

# 3 — Real vector query. The vector content is irrelevant (we discard the
#     result); what matters is exercising the search path the app itself uses.
query_body="$(python3 -c '
import json, sys
print(json.dumps({"query": [0.01] * int(sys.argv[1]), "limit": 1, "with_payload": False}))
' "$dim")"
check "POST /points/query" \
  "$(call POST "/collections/${COLLECTION}/points/query" "$query_body")" \
  >/dev/null || exit 1
echo "  query ok"

echo "keepalive OK: ${points} points, cluster responsive."
