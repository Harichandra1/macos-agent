"""
auth.py — Google Sign-In → our session JWT, and the /chat auth gate.

Flow (frontend drives Google Identity Services):
  1. The browser gets a Google ID token via the GIS button.
  2. POST /auth/google {credential} — we verify the token's signature and
     audience against GOOGLE_CLIENT_ID (google-auth fetches Google's JWKS),
     upsert the user row by email, and mint OUR OWN short-lived HS256 JWT.
  3. Every /chat call carries `Authorization: Bearer <our jwt>`; the
     `current_user` dependency resolves it to a User row or raises 401.

Auth is feature-flagged on GOOGLE_CLIENT_ID: unset (local dev, tests) means
the API runs open, `current_user` yields None, and /auth/config tells the
frontend to hide the sign-in UI. The Google ID token itself is never stored —
only our derived session JWT lives in the browser.
"""

import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .logging import get_logger
from .models import User

logger = get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])

_JWT_ALG = "HS256"
_JWT_ISS = "macos-agent"

# Per-process fallback secret when AUTH_JWT_SECRET is unset: dev sessions just
# don't survive a restart. Generated once at import so all workers of one
# process agree.
_EPHEMERAL_SECRET = secrets.token_hex(32)
_warned_ephemeral = False


def _jwt_secret() -> str:
    global _warned_ephemeral
    configured = get_settings().auth_jwt_secret.strip()
    if configured:
        return configured
    if not _warned_ephemeral:
        logger.warning("AUTH_JWT_SECRET not set — using an ephemeral secret; "
                       "sessions reset on restart",
                       extra={"extra_fields": {"component": "auth"}})
        _warned_ephemeral = True
    return _EPHEMERAL_SECRET


# ---------------------------------------------------------------------------
# Google ID-token verification (seam: tests monkeypatch _google_verify)
# ---------------------------------------------------------------------------

def _google_verify(credential: str, client_id: str) -> dict:
    """Verify a Google ID token; returns its claims. Raises on any failure."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(
        credential, google_requests.Request(), audience=client_id)


# ---------------------------------------------------------------------------
# Our session JWT
# ---------------------------------------------------------------------------

def mint_session_token(user: User) -> str:
    import jwt

    settings = get_settings()
    now = int(time.time())
    return jwt.encode(
        {
            "iss": _JWT_ISS,
            "sub": user.user_id,
            "email": user.email,
            "iat": now,
            "exp": now + settings.auth_token_ttl_hours * 3600,
        },
        _jwt_secret(),
        algorithm=_JWT_ALG,
    )


def decode_session_token(token: str) -> dict:
    """Decode + validate our JWT. Raises jwt exceptions on any problem."""
    import jwt

    return jwt.decode(token, _jwt_secret(), algorithms=[_JWT_ALG],
                      issuer=_JWT_ISS, options={"require": ["exp", "sub", "iss"]})


# ---------------------------------------------------------------------------
# FastAPI dependency — the auth gate
# ---------------------------------------------------------------------------

def current_user(request: Request,
                 db: Session = Depends(get_db)) -> Optional[User]:
    """
    Resolve the request's bearer token to a User row.

    Auth disabled (no GOOGLE_CLIENT_ID) → None, request proceeds open.
    Auth enabled → missing/invalid/expired token or unknown user → 401.
    """
    if not get_settings().auth_enabled:
        return None

    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="sign in to continue")
    token = header[7:].strip()

    try:
        claims = decode_session_token(token)
    except Exception:  # noqa: BLE001 — expired/forged/malformed all read the same
        raise HTTPException(status_code=401, detail="session expired — sign in again")

    user = db.get(User, claims.get("sub", ""))
    if user is None:
        raise HTTPException(status_code=401, detail="unknown user — sign in again")
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class GoogleLoginRequest(BaseModel):
    credential: str = Field(..., min_length=20, description="Google ID token from GIS")


class SessionResponse(BaseModel):
    token: str
    email: str
    user_id: str


@router.get("/config")
def auth_config():
    """Frontend bootstrap: is auth on, and which Google client id to use."""
    settings = get_settings()
    return {"enabled": settings.auth_enabled,
            "google_client_id": settings.google_client_id}


@router.post("/google", response_model=SessionResponse)
def google_login(body: GoogleLoginRequest, db: Session = Depends(get_db)):
    settings = get_settings()
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="authentication is not enabled")

    try:
        claims = _google_verify(body.credential, settings.google_client_id)
    except Exception as e:  # noqa: BLE001 — bad audience/signature/expiry
        logger.warning("google token rejected",
                       extra={"extra_fields": {"error": repr(e)}})
        raise HTTPException(status_code=401, detail="Google sign-in was rejected")

    email = (claims.get("email") or "").strip().lower()
    if not email or not claims.get("email_verified", False):
        raise HTTPException(status_code=401, detail="a verified Google email is required")

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is None:
        user = User(email=email)
        db.add(user)
        db.commit()
        logger.info("user created", extra={"extra_fields": {"user_id": user.user_id}})

    return SessionResponse(token=mint_session_token(user),
                           email=user.email, user_id=user.user_id)


@router.get("/me")
def me(user: Optional[User] = Depends(current_user),
       db: Session = Depends(get_db)):
    if user is None:   # auth disabled
        return {"authenticated": False}
    from .credits import get_account
    acct = get_account(db, user.user_id)
    db.commit()   # persist a lazy weekly/monthly reset if one just applied
    return {
        "authenticated": True, "email": user.email, "user_id": user.user_id,
        "credits": {"weekly_left": acct.weekly_left,
                    "weekly_limit": get_settings().weekly_credit_limit},
    }
