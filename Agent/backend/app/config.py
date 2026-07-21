"""
config.py — typed application settings (pydantic-settings).

Reads the shared repo-root .env (same file the ingestion side uses), so the
serving layer needs no separate credentials. In a container, real environment
variables take precedence over the .env file automatically.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Agent/backend/app/config.py → repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- credentials (upper-cased env names map case-insensitively) ---
    openai_api_key: str = ""
    groq_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "macos_kb"
    tavily_api_key: str = ""

    # --- database (Phase 1: users; later phases add credits/costs/feedback) ---
    # SQLite file by default so dev needs zero infra; deployment overrides with
    # a Postgres URL — the schema and Alembic migrations are dialect-neutral.
    database_url: str = f"sqlite:///{_REPO_ROOT / 'Agent' / 'backend' / 'data' / 'app.db'}"
    # Postgres pool sizing (ignored for SQLite). Small by default to fit
    # managed free-tier connection caps.
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # --- authentication (Google Sign-In → our session JWT) ---
    # Auth is ON when a Google client id is configured. Without one (local dev,
    # tests) the API runs open and /auth/config tells the frontend to hide the
    # sign-in UI — a deliberate feature flag, not a fallback.
    google_client_id: str = ""
    # HS256 secret for our session JWTs. Empty → a random per-process secret is
    # generated at startup (sessions won't survive a restart; fine for dev, set
    # it in deployment).
    auth_jwt_secret: str = ""
    auth_token_ttl_hours: int = 24 * 7

    # --- credits (Phase 3; only enforced for authenticated users) ---
    weekly_credit_limit: int = 5        # chat turns per ISO week
    monthly_cost_cap_usd: float = 0.25  # cumulative provider cost per calendar month

    # --- serving config ---
    app_env: str = "development"
    app_origin: str = "http://localhost:8000"
    agent_model: str = "llama-3.3-70b-versatile"
    agent_checkpoint_db: str | None = None      # sqlite path; None → in-memory
    cors_origins: str = "*"                      # comma-separated, or "*"
    trusted_hosts: str = "*"                     # comma-separated, or "*"
    metrics_token: str = ""                      # required for production /metrics
    # Input guardrail. Generous because the agent ASKS users to paste diagnostic
    # command output (`log show`, `pmset -g assertions`) which easily runs long;
    # retrieval/planning cap the paste's contribution internally.
    max_message_chars: int = 16000
    # Redact PII (emails, MACs, IPs, /Users/<name>, serials) from user messages
    # before any external provider sees them. Off-switch for local debugging only.
    pii_redaction: bool = True
    rate_limit: str = "20/minute"                # per-client /chat limit

    # --- production observability transport ---
    grafana_prometheus_remote_write_url: str = ""
    grafana_prometheus_username: str = ""
    grafana_loki_write_url: str = ""
    grafana_loki_username: str = ""
    grafana_cloud_api_key: str = ""

    @field_validator("database_url")
    @classmethod
    def _normalize_db_url(cls, v: str) -> str:
        """Managed Postgres providers hand out a bare `postgres://` /
        `postgresql://` URL, but SQLAlchemy 2 + psycopg3 needs the driver
        suffix. Rewrite it here, once, so the engine, Alembic, and the cron
        job all get a working URL without each re-parsing it."""
        v = v.strip()
        if v.startswith("postgres://"):
            v = "postgresql://" + v[len("postgres://"):]
        if v.startswith("postgresql://"):
            v = "postgresql+psycopg://" + v[len("postgresql://"):]
        return v

    @property
    def auth_enabled(self) -> bool:
        return bool(self.google_client_id.strip())

    @property
    def production_mode(self) -> bool:
        return self.app_env.strip().lower() in {"production", "prod"}

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def trusted_host_list(self) -> list[str]:
        if self.trusted_hosts.strip() == "*":
            return ["*"]
        return [h.strip() for h in self.trusted_hosts.split(",") if h.strip()]

    def production_errors(self) -> list[str]:
        """Return actionable configuration errors for the live deployment.

        Development and test environments deliberately keep the historical
        open defaults. Production is different: a missing identity, database,
        origin, or observability credential should stop startup instead of
        silently running in an unsafe degraded mode.
        """
        if not self.production_mode:
            return []

        from dotenv import load_dotenv
        load_dotenv(self.model_config.get("env_file"))
        errors: list[str] = []
        if not self.google_client_id.strip():
            errors.append("GOOGLE_CLIENT_ID")
        if len(self.auth_jwt_secret.strip()) < 32:
            errors.append("AUTH_JWT_SECRET (at least 32 characters)")
        if not self.database_url.startswith("postgresql+psycopg://"):
            errors.append("DATABASE_URL (direct PostgreSQL URL)")
        if not self.cors_origin_list or "*" in self.cors_origin_list:
            errors.append("CORS_ORIGINS (an explicit HTTPS origin)")
        if not self.trusted_host_list or "*" in self.trusted_host_list:
            errors.append("TRUSTED_HOSTS (explicit production host)")
        if not self.app_origin.startswith("https://"):
            errors.append("APP_ORIGIN (HTTPS production origin)")
        if len(self.metrics_token.strip()) < 32:
            errors.append("METRICS_TOKEN (at least 32 characters)")

        required_observability = {
            "GRAFANA_PROMETHEUS_REMOTE_WRITE_URL": self.grafana_prometheus_remote_write_url,
            "GRAFANA_PROMETHEUS_USERNAME": self.grafana_prometheus_username,
            "GRAFANA_LOKI_WRITE_URL": self.grafana_loki_write_url,
            "GRAFANA_LOKI_USERNAME": self.grafana_loki_username,
            "GRAFANA_CLOUD_API_KEY": self.grafana_cloud_api_key,
        }
        errors.extend(name for name, value in required_observability.items()
                      if not value.strip())

        required_infra = self.missing_agent_keys()
        errors.extend(name for name in required_infra if name not in errors)

        # The normal provider resolver is the source of truth for role-specific
        # keys and fallback order; reuse it so custom provider selection is
        # validated exactly as it is during agent construction.
        try:
            from app.agent.providers import key_present, required_key_envs
            errors.extend(
                name for name in required_key_envs(["intake", "synth"])
                if not key_present(name) and name not in errors
            )
        except Exception as exc:  # keep startup error actionable if config is malformed
            errors.append(f"provider configuration ({exc})")
        return errors

    def missing_agent_keys(self) -> list[str]:
        """
        Infrastructure credentials the agent ALWAYS needs, regardless of which LLM
        provider is selected: OpenAI (embeddings) + Qdrant (vector DB). The
        per-role LLM provider keys (Groq/NVIDIA/OpenRouter) are checked separately
        and provider-awarely in deps.build() via providers.required_key_envs().
        """
        need = {
            "OPENAI_API_KEY": self.openai_api_key,   # text-embedding-3-small
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_API_KEY": self.qdrant_api_key,
        }
        return [k for k, v in need.items() if not v or v in ("...", "changeme")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
