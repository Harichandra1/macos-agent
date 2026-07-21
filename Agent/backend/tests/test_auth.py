"""
test_auth.py — DB + Google-auth gate tests. No network, no real Google.

Covers:
  1. Auth disabled (no GOOGLE_CLIENT_ID): /auth/config reports off, /chat is
     open (passes the auth gate; 503 because no agent is built in tests).
  2. /auth/google verifies the Google credential (mocked seam), creates the
     user row once, and re-login maps to the SAME user (upsert by email).
  3. Session JWT round-trip; tampered and expired tokens are rejected.
  4. Auth enabled: /chat without/with-bad token → 401; with a valid session
     token it passes the gate; /auth/me reflects the session.
  5. Unverified Google email → 401, no user row created.

Run:  cd Agent/backend && ../../.venv/bin/python tests/test_auth.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Isolate the DB and auth config BEFORE importing the app modules.
_TMP = tempfile.mkdtemp(prefix="agent_auth_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["GOOGLE_CLIENT_ID"] = ""          # start with auth disabled
os.environ["AUTH_JWT_SECRET"] = "test-secret-not-for-production-0000"

from fastapi.testclient import TestClient  # noqa: E402

import app.auth as auth  # noqa: E402
import app.deps as deps  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import get_sessionmaker, run_migrations  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User  # noqa: E402

# Tests never build the real agent (needs network + keys) — /chat with a
# passing auth gate answers 503, which is exactly the assertion we want.
deps._agent = None
deps._build_error = "disabled in tests"

run_migrations()
client = TestClient(app)  # no `with` → lifespan (migrations + agent build) skipped

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"
    PASS += 1


def set_auth_enabled(enabled: bool):
    os.environ["GOOGLE_CLIENT_ID"] = "test-client-id.apps.googleusercontent.com" if enabled else ""
    get_settings.cache_clear()


def fake_verify_factory(email="user@example.com", verified=True):
    def _fake(credential, client_id):
        if credential == "bad-token-bad-token-bad":
            raise ValueError("invalid token")
        return {"email": email, "email_verified": verified, "sub": "g-123"}
    return _fake


# ---------------------------------------------------------------- 1. disabled
print("\n1. auth disabled (dev mode)")
set_auth_enabled(False)
r = client.get("/auth/config")
check("config reports disabled", r.json() == {"enabled": False, "google_client_id": ""})
r = client.post("/chat", json={"message": "wifi drops", "session_id": "s1"})
check("chat open without token (503 = past the auth gate)", r.status_code == 503, str(r.status_code))
r = client.get("/auth/me")
check("me: unauthenticated", r.json() == {"authenticated": False})
r = client.post("/auth/google", json={"credential": "x" * 24})
check("google login 404 while disabled", r.status_code == 404)

# ---------------------------------------------------------------- 2. login/upsert
print("\n2. google login + upsert")
set_auth_enabled(True)
auth._google_verify = fake_verify_factory()

r = client.post("/auth/google", json={"credential": "good-token-good-token"})
check("login ok", r.status_code == 200, r.text)
tok1 = r.json()["token"]
uid1 = r.json()["user_id"]
check("email returned", r.json()["email"] == "user@example.com")

r = client.post("/auth/google", json={"credential": "good-token-good-token"})
check("re-login same user (upsert by email)", r.json()["user_id"] == uid1)

s = get_sessionmaker()()
check("exactly one user row", s.query(User).count() == 1)
row = s.query(User).one()
check("row fields", row.email == "user@example.com" and row.created_at is not None)
s.close()

# ---------------------------------------------------------------- 3. JWT
print("\n3. session JWT")
claims = auth.decode_session_token(tok1)
check("claims round-trip", claims["sub"] == uid1 and claims["email"] == "user@example.com")

tampered = tok1[:-4] + ("AAAA" if tok1[-4:] != "AAAA" else "BBBB")
try:
    auth.decode_session_token(tampered)
    check("tampered token rejected", False)
except Exception:
    check("tampered token rejected", True)

import jwt as pyjwt  # noqa: E402
expired = pyjwt.encode(
    {"iss": "macos-agent", "sub": uid1, "email": row.email,
     "iat": int(time.time()) - 7200, "exp": int(time.time()) - 3600},
    "test-secret-not-for-production-0000", algorithm="HS256")
try:
    auth.decode_session_token(expired)
    check("expired token rejected", False)
except Exception:
    check("expired token rejected", True)

# ---------------------------------------------------------------- 4. the gate
print("\n4. /chat auth gate")
r = client.post("/chat", json={"message": "wifi drops", "session_id": "s1"})
check("no token → 401", r.status_code == 401, str(r.status_code))
r = client.post("/chat", json={"message": "wifi drops", "session_id": "s1"},
                headers={"Authorization": f"Bearer {tampered}"})
check("bad token → 401", r.status_code == 401)
r = client.post("/chat", json={"message": "wifi drops", "session_id": "s1"},
                headers={"Authorization": f"Bearer {tok1}"})
check("valid token → past gate (503: no agent in tests)", r.status_code == 503, str(r.status_code))
r = client.get("/auth/me", headers={"Authorization": f"Bearer {tok1}"})
check("me: authenticated", r.json()["email"] == "user@example.com")
r = client.get("/auth/me")
check("me without token → 401", r.status_code == 401)

# ---------------------------------------------------------------- 5. rejections
print("\n5. rejected logins")
r = client.post("/auth/google", json={"credential": "bad-token-bad-token-bad"})
check("invalid google token → 401", r.status_code == 401)

auth._google_verify = fake_verify_factory(email="shady@example.com", verified=False)
r = client.post("/auth/google", json={"credential": "good-token-good-token"})
check("unverified email → 401", r.status_code == 401)
s = get_sessionmaker()()
check("no row created for rejected login",
      s.query(User).filter_by(email="shady@example.com").count() == 0)
s.close()

print(f"\nAll {PASS} auth checks passed.")
