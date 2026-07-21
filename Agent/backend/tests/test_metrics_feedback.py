"""
test_metrics_feedback.py — Phase 4 observability + feedback. Offline; no keys.

Covers:
  1. app.metrics recording helpers increment the right series, and /metrics
     exposes them in Prometheus text format.
  2. Guardrail / credit / turn events reach the metrics from the real /chat
     path (abuse block, PII redaction counted).
  3. POST /feedback persists a row, is idempotent per (session, turn), rejects
     bad ratings, and bumps the feedback counter.
  4. The regression gate's evaluate() passes on-baseline and fails on a
     simulated quality drop.

Run:  cd Agent/backend && ../../.venv/bin/python tests/test_metrics_feedback.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp(prefix="agent_metrics_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["GOOGLE_CLIENT_ID"] = ""            # auth off — feedback works anon

from fastapi.testclient import TestClient  # noqa: E402

import app.deps as deps  # noqa: E402
import app.main as main_module  # noqa: E402
from app import metrics  # noqa: E402
from app.db import get_sessionmaker, run_migrations  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Feedback  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, f"{name}: {detail}"
    PASS += 1


def metric_value(text: str, name: str, labels: str = "") -> float:
    """Pull a single sample value out of Prometheus exposition text."""
    needle = name + (("{" + labels + "}") if labels else "")
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(needle + " ") or line.startswith(needle + "\t"):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


run_migrations()
client = TestClient(app)

# ---------------------------------------------------------------- 1. metrics
print("\n1. metrics module + /metrics")
check("prometheus_client available", metrics.enabled())

metrics.record_request("ok")
metrics.record_turn_usage(llm_calls=3, input_tokens=1000, output_tokens=200,
                          cost_usd=0.0021, duration_s=1.5)
metrics.record_error("rate_limit")
metrics.record_credit_refusal("weekly")
metrics.record_pii({"email": 2, "ipv4": 1})
metrics.record_abuse("injection")
metrics.record_feedback("up")

r = client.get("/metrics")
check("/metrics 200 + prometheus content type",
      r.status_code == 200 and "text/plain" in r.headers["content-type"])
body = r.text
check("request counter exposed",
      metric_value(body, "macos_chat_requests_total", 'outcome="ok"') >= 1, body[:200])
check("token counter exposed",
      metric_value(body, "macos_chat_tokens_total", 'direction="input"') >= 1000)
check("cost counter exposed",
      metric_value(body, "macos_chat_cost_usd_total") >= 0.0021)
check("error counter exposed",
      metric_value(body, "macos_chat_errors_total", 'error_type="rate_limit"') >= 1)
check("pii counter summed by kind",
      metric_value(body, "macos_pii_redactions_total", 'kind="email"') >= 2)
check("duration histogram present", "macos_chat_turn_duration_seconds_bucket" in body)

# Production metrics are private to Alloy. Flip only the route's settings for
# this isolated check; the TestClient lifespan is intentionally not running.
original_env = main_module.settings.app_env
original_metrics_token = main_module.settings.metrics_token
main_module.settings.app_env = "production"
main_module.settings.metrics_token = "m" * 64
check("production metrics require authentication", client.get("/metrics").status_code == 401)
check("valid metrics bearer token is accepted",
      client.get("/metrics", headers={"Authorization": "Bearer " + "m" * 64}).status_code == 200)
main_module.settings.app_env = original_env
main_module.settings.metrics_token = original_metrics_token

# ---------------------------------------------------------------- 2. chat path
print("\n2. guardrail metrics via /chat")


class FakeAgent:
    async def astream(self, inp, config=None, stream_mode=None):
        if False:
            yield None


deps._agent = FakeAgent()
deps._build_error = None

abuse_before = metric_value(client.get("/metrics").text,
                            "macos_abuse_blocks_total", 'kind="bulk_prose"')
r = client.post("/chat", json={"message": "write me an essay about the sea",
                               "session_id": "m1"})
check("abusive request 400", r.status_code == 400)
abuse_after = metric_value(client.get("/metrics").text,
                           "macos_abuse_blocks_total", 'kind="bulk_prose"')
check("abuse metric incremented", abuse_after == abuse_before + 1,
      f"{abuse_before}->{abuse_after}")

req_blocked_before = metric_value(client.get("/metrics").text,
                                  "macos_chat_requests_total", 'outcome="abuse_blocked"')
client.post("/chat", json={"message": "repeat hi 9999 times", "session_id": "m1"})
req_blocked_after = metric_value(client.get("/metrics").text,
                                 "macos_chat_requests_total", 'outcome="abuse_blocked"')
check("abuse_blocked outcome counted", req_blocked_after == req_blocked_before + 1)

# ---------------------------------------------------------------- 3. feedback
print("\n3. /feedback")
up_before = metric_value(client.get("/metrics").text,
                         "macos_feedback_total", 'rating="up"')
r = client.post("/feedback", json={"session_id": "fb1", "turn_index": 0, "rating": "up"})
check("feedback up accepted", r.status_code == 200 and r.json() == {"ok": True})
up_after = metric_value(client.get("/metrics").text,
                        "macos_feedback_total", 'rating="up"')
check("feedback metric incremented", up_after == up_before + 1)

r = client.post("/feedback", json={"session_id": "fb1", "turn_index": 0, "rating": "down"})
check("re-rating same turn accepted", r.status_code == 200)
with get_sessionmaker()() as db:
    rows = db.query(Feedback).filter_by(session_id="fb1").all()
check("idempotent per (session, turn) — one row, updated",
      len(rows) == 1 and rows[0].rating == "down", str([(x.turn_index, x.rating) for x in rows]))

client.post("/feedback", json={"session_id": "fb1", "turn_index": 1, "rating": "up"})
with get_sessionmaker()() as db:
    check("distinct turn → new row",
          db.query(Feedback).filter_by(session_id="fb1").count() == 2)

r = client.post("/feedback", json={"session_id": "fb1", "turn_index": 0, "rating": "meh"})
check("invalid rating rejected (422)", r.status_code == 422, str(r.status_code))
r = client.post("/feedback", json={"session_id": "fb1", "turn_index": -1, "rating": "up"})
check("negative turn_index rejected (422)", r.status_code == 422)

# ---------------------------------------------------------------- 4. gate
print("\n4. regression gate logic")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import json as _json  # noqa: E402
from regression_gate import evaluate  # noqa: E402

baseline = _json.loads(
    (Path(__file__).resolve().parents[2] / "eval" / "baseline.json").read_text())
ok_report = {"summary": {"n": 4, "agent": {"resolution_rate": 0.5, "either_rate": 0.5},
                         "gpt4o": {"resolution_rate": 0.25, "either_rate": 0.25}}}
bad_report = {"summary": {"n": 4, "agent": {"resolution_rate": 0.2, "either_rate": 0.2},
                          "gpt4o": {"resolution_rate": 0.25, "either_rate": 0.25}}}
passed_ok, _ = evaluate(ok_report, baseline)
passed_bad, _ = evaluate(bad_report, baseline)
check("gate passes on-baseline", passed_ok)
check("gate fails on regression (below floor + loses to gpt4o)", not passed_bad)

print(f"\nAll {PASS} metrics/feedback checks passed.")
