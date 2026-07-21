"""
test_credits.py — Phase 3 hybrid credit system + provider-layer cost tracking.
Offline; no keys, no network.

Covers:
  1. ISO week / month keys and the LAZY reset (new week refreshes credits,
     new month zeroes the cost aggregate — no cron required).
  2. check_and_consume: up-front decrement, weekly exhaustion, monthly cap
     (cap wins even with credits left), refusals don't consume.
  3. record_usage: ledger row + monthly aggregate roll-up.
  4. UsageTracker: exact usage (llm_output), streamed usage (usage_metadata),
     chars/4 estimate fallback (flagged), failed-call input accounting,
     pricing table substring match.
  5. /chat integration: credits event streamed with the answer, 429 with the
     right message on weekly/monthly exhaustion, the agent never runs on a
     refused turn, ledger rows carry the user id.

Run:  cd Agent/backend && ../../.venv/bin/python tests/test_credits.py
"""

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp(prefix="agent_credits_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["GOOGLE_CLIENT_ID"] = "test-client-id.apps.googleusercontent.com"
os.environ["AUTH_JWT_SECRET"] = "test-secret-not-for-production-0000"
os.environ["WEEKLY_CREDIT_LIMIT"] = "2"     # small limit → short test

from fastapi.testclient import TestClient  # noqa: E402

import app.auth as auth  # noqa: E402
import app.deps as deps  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.credits import (  # noqa: E402
    MONTHLY_CAP_MESSAGE, WEEKLY_EXHAUSTED_MESSAGE, check_and_consume,
    get_account, month_key, record_usage, week_key,
)
from app.db import get_sessionmaker, run_migrations  # noqa: E402
from app.main import app  # noqa: E402
from app.models import CreditAccount, UsageLog, User  # noqa: E402
from app.usage import DEFAULT_PRICE, UsageTracker, price_for  # noqa: E402

get_settings.cache_clear()
run_migrations()

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, f"{name}: {detail}"
    PASS += 1


def make_user(email: str) -> str:
    with get_sessionmaker()() as db:
        u = User(email=email)
        db.add(u)
        db.commit()
        return u.user_id


W1 = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)   # week 2026-W30, month 2026-07
W2 = datetime(2026, 7, 27, 12, tzinfo=timezone.utc)   # next ISO week, same month
M2 = datetime(2026, 8, 3, 12, tzinfo=timezone.utc)    # next month

# ---------------------------------------------------------------- 1. windows
print("\n1. window keys + lazy reset")
check("week keys", week_key(W1) == "2026-W30" and week_key(W2) == "2026-W31")
check("month keys", month_key(W1) == "2026-07" and month_key(M2) == "2026-08")

uid = make_user("credits1@example.com")
with get_sessionmaker()() as db:
    acct = get_account(db, uid, now=W1)
    check("account created with full credits",
          acct.weekly_left == 2 and acct.month_cost_usd == 0.0)
    acct.weekly_left = 0
    acct.month_cost_usd = 0.20
    db.commit()
with get_sessionmaker()() as db:
    acct = get_account(db, uid, now=W2)
    check("new week refreshes credits, cost untouched",
          acct.weekly_left == 2 and abs(acct.month_cost_usd - 0.20) < 1e-9)
    db.commit()
with get_sessionmaker()() as db:
    acct = get_account(db, uid, now=M2)
    check("new month zeroes the cost aggregate", acct.month_cost_usd == 0.0)
    db.commit()

# ---------------------------------------------------------------- 2. consume
print("\n2. check_and_consume")
uid2 = make_user("credits2@example.com")
with get_sessionmaker()() as db:
    s1 = check_and_consume(db, uid2, now=W1)
    s2 = check_and_consume(db, uid2, now=W1)
    s3 = check_and_consume(db, uid2, now=W1)
check("turns decrement 2→1→0", s1.allowed and s1.weekly_left == 1
      and s2.allowed and s2.weekly_left == 0)
check("exhausted week refuses", not s3.allowed and s3.reason == "weekly")
with get_sessionmaker()() as db:
    check("refusal did not go negative",
          get_account(db, uid2, now=W1).weekly_left == 0)
    s4 = check_and_consume(db, uid2, now=W2)
check("next week allows again", s4.allowed and s4.weekly_left == 1)

with get_sessionmaker()() as db:
    acct = get_account(db, uid2, now=W2)
    acct.month_cost_usd = get_settings().monthly_cost_cap_usd   # at the cap
    db.commit()
    s5 = check_and_consume(db, uid2, now=W2)
check("monthly cap wins even with credits left",
      not s5.allowed and s5.reason == "monthly")

# ---------------------------------------------------------------- 3. ledger
print("\n3. record_usage ledger")
uid3 = make_user("credits3@example.com")
with get_sessionmaker()() as db:
    record_usage(db, user_id=uid3, session_id="s-led", llm_calls=4,
                 input_tokens=3000, output_tokens=800, cost_usd=0.0031,
                 estimated=False, now=W1)
    record_usage(db, user_id=uid3, session_id="s-led", llm_calls=2,
                 input_tokens=1000, output_tokens=200, cost_usd=0.0009,
                 estimated=True, error_type="rate_limit", now=W1)
with get_sessionmaker()() as db:
    rows = db.query(UsageLog).filter_by(user_id=uid3).all()
    acct = db.get(CreditAccount, uid3)
check("two ledger rows", len(rows) == 2)
check("error_type recorded", rows[1].error_type == "rate_limit" and rows[1].estimated)
check("monthly aggregate rolled up", abs(acct.month_cost_usd - 0.004) < 1e-9,
      str(acct.month_cost_usd))

# ---------------------------------------------------------------- 4. tracker
print("\n4. UsageTracker")
check("pricing substring match",
      price_for("meta/llama-3.3-70b-instruct") == (0.59, 0.79)
      and price_for("llama-3.1-8b-instant") == (0.05, 0.08)
      and price_for("mystery-model-x") == DEFAULT_PRICE)

t = UsageTracker()
t.on_llm_end(SimpleNamespace(
    llm_output={"token_usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                "model_name": "llama-3.3-70b-versatile"},
    generations=[]), run_id="r1")
expect = (1000 * 0.59 + 500 * 0.79) / 1e6
check("exact usage path", t.input_tokens == 1000 and t.output_tokens == 500
      and abs(t.cost_usd - expect) < 1e-12 and not t.estimated)

msg = SimpleNamespace(content="answer text", usage_metadata={
    "input_tokens": 200, "output_tokens": 40},
    response_metadata={"model_name": "llama-3.1-8b-instant"})
t.on_llm_end(SimpleNamespace(llm_output={}, generations=[[
    SimpleNamespace(text="answer text", message=msg)]]), run_id="r2")
check("streamed usage_metadata path", t.input_tokens == 1200
      and t.output_tokens == 540 and not t.estimated)

t2 = UsageTracker()
t2.on_chat_model_start({}, [[SimpleNamespace(content="x" * 400)]], run_id="r3")
t2.on_llm_end(SimpleNamespace(llm_output={}, generations=[[
    SimpleNamespace(text="y" * 200, message=None)]]), run_id="r3")
check("estimate fallback (chars/4, flagged)",
      t2.input_tokens == 100 and t2.output_tokens == 50 and t2.estimated)

t3 = UsageTracker()
t3.on_chat_model_start({}, [[SimpleNamespace(content="z" * 400)]], run_id="r4")
t3.on_llm_error(RuntimeError("429"), run_id="r4")
check("failed call still bills its input estimate",
      t3.input_tokens == 100 and t3.output_tokens == 0 and t3.estimated)

# ---------------------------------------------------------------- 5. endpoint
print("\n5. /chat integration")


class FakeAgent:
    def __init__(self):
        self.runs = 0

    async def astream(self, inp, config=None, stream_mode=None):
        self.runs += 1
        if False:
            yield None


fake = FakeAgent()
deps._agent = fake
deps._build_error = None
auth._google_verify = lambda cred, cid: {
    "email": "enduser@example.com", "email_verified": True, "sub": "g-9"}
client = TestClient(app)

tok = client.post("/auth/google",
                  json={"credential": "good-token-good-token"}).json()["token"]
hdrs = {"Authorization": f"Bearer {tok}"}

r = client.get("/auth/me", headers=hdrs)
check("me exposes credits", r.json()["credits"] == {"weekly_left": 2, "weekly_limit": 2})

r1 = client.post("/chat", json={"message": "wifi drops on my mac", "session_id": "c1"},
                 headers=hdrs)
check("turn 1 streams a credits event",
      r1.status_code == 200 and '"weekly_left":1' in r1.text.replace(" ", ""), r1.text)
r2 = client.post("/chat", json={"message": "still dropping on my mac", "session_id": "c1"},
                 headers=hdrs)
check("turn 2 → 0 left", '"weekly_left":0' in r2.text.replace(" ", ""))

runs_before = fake.runs
r3 = client.post("/chat", json={"message": "wifi still broken on my mac", "session_id": "c1"},
                 headers=hdrs)
check("turn 3 refused with weekly message",
      r3.status_code == 429 and r3.json()["detail"] == WEEKLY_EXHAUSTED_MESSAGE, r3.text)
check("refused turn never ran the agent", fake.runs == runs_before)

with get_sessionmaker()() as db:
    me_row = db.query(User).filter_by(email="enduser@example.com").one()
    n_ledger = db.query(UsageLog).filter_by(user_id=me_row.user_id).count()
    acct = db.get(CreditAccount, me_row.user_id)
    acct.weekly_left = 2                                  # restore credits…
    acct.month_cost_usd = get_settings().monthly_cost_cap_usd   # …but cap the month
    db.commit()
check("ledger rows written for both turns", n_ledger == 2, str(n_ledger))

r4 = client.post("/chat", json={"message": "wifi acting up on my mac", "session_id": "c1"},
                 headers=hdrs)
check("monthly cap → 429 with cap message",
      r4.status_code == 429 and r4.json()["detail"] == MONTHLY_CAP_MESSAGE, r4.text)

print(f"\nAll {PASS} credit checks passed.")
