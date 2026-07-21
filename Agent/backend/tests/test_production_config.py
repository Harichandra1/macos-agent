"""Production configuration guard tests.

Run from Agent/backend:
  ../../.venv/bin/python tests/test_production_config.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402


def check(name: str, condition: bool):
    print(f"  [{'ok' if condition else 'FAIL'}] {name}")
    assert condition, name


def production(**overrides) -> Settings:
    values = {
        "app_env": "production",
        "app_origin": "https://macosagent-hari.duckdns.org",
        "cors_origins": "https://macosagent-hari.duckdns.org",
        "trusted_hosts": "macosagent-hari.duckdns.org",
        "google_client_id": "client.apps.googleusercontent.com",
        "auth_jwt_secret": "a" * 64,
        "database_url": "postgresql://db.example/app?sslmode=require",
        "metrics_token": "b" * 64,
        "openai_api_key": "openai-test",
        "groq_api_key": "groq-test",
        "qdrant_url": "https://qdrant.example:6333",
        "qdrant_api_key": "qdrant-test",
        "grafana_prometheus_remote_write_url": "https://grafana.example/api/prom/push",
        "grafana_prometheus_username": "12345",
        "grafana_loki_write_url": "https://grafana.example/loki/api/v1/push",
        "grafana_loki_username": "12345",
        "grafana_cloud_api_key": "grafana-test",
    }
    values.update(overrides)
    return Settings(**values)


print("\nproduction configuration guards")
os.environ["GROQ_API_KEY"] = "groq-test"
valid = production()
check("complete production settings pass", valid.production_errors() == [])
check("postgres URL is normalized", valid.database_url.startswith("postgresql+psycopg://"))

missing_auth = production(google_client_id="")
check("missing Google client id is rejected", "GOOGLE_CLIENT_ID" in missing_auth.production_errors())

wildcard_cors = production(cors_origins="*")
check("wildcard CORS is rejected", "CORS_ORIGINS (an explicit HTTPS origin)" in wildcard_cors.production_errors())

sqlite_db = production(database_url="sqlite:///tmp/app.db")
check("SQLite is rejected in production", any("DATABASE_URL" in e for e in sqlite_db.production_errors()))

missing_grafana = production(grafana_cloud_api_key="")
check("missing Grafana credential is rejected", "GRAFANA_CLOUD_API_KEY" in missing_grafana.production_errors())

print("\nAll production configuration checks passed.")
