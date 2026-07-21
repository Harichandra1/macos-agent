"""
taxonomy.py — controlled category vocabulary for the serving/read path.

Vendored from vectorDBIngestion/knowledge_base.py so the serving layer is
self-contained and does not import the ingestion package. The vocabulary is
FROZEN by contract (CLAUDE.md: "do not extend ad hoc") — the ingestion side is
the source of truth for what values land in Qdrant payloads, and this copy must
match it. If the controlled vocabulary ever changes, update BOTH this file and
knowledge_base.py together, per CLAUDE.md.
"""

from typing import Optional

# Controlled category vocabulary. Qdrant metadata pre-filtering depends on
# `category` being exactly one of these values.
ALLOWED_CATEGORIES = {
    "bluetooth", "wifi", "disk", "battery", "performance",
    "permissions", "system", "diagnostics", "general",
}

# Off-vocabulary values seen in the wild → the controlled term they map onto.
_CATEGORY_ALIASES = {
    "network": "wifi",        "networking": "wifi",   "wi-fi": "wifi",
    "wireless": "wifi",       "ethernet": "wifi",
    "security": "permissions", "privacy": "permissions",
    "preferences": "system",  "prefs": "system",      "config": "system",
    "spotlight": "diagnostics", "logs": "diagnostics", "logging": "diagnostics",
    "processes": "performance", "process": "performance",
    "cpu": "performance",     "memory": "performance",
    "storage": "disk",        "power": "battery",
}


def normalize_category(value: Optional[str]) -> str:
    """Coerce an arbitrary category string into the controlled vocabulary."""
    if not value:
        return "general"
    v = value.strip().lower()
    if v in ALLOWED_CATEGORIES:
        return v
    return _CATEGORY_ALIASES.get(v, "general")
