"""
test_checkpointer_postgres.py — V3: Postgres-backed conversation memory.

Closes a real gap: `make_memory_checkpointer()` previously only persisted
when a SEPARATE `AGENT_CHECKPOINT_DB` sqlite path was set (never wired in
render.yaml), so production ran on an in-process MemorySaver — conversation
memory vanished on every restart/redeploy and wasn't shared across workers.

This test proves the fix two ways:
  1. Routing: DATABASE_URL (postgres-shaped, either bare `postgresql://` or
     the settings-normalized `postgresql+psycopg://`) selects PostgresSaver;
     a sqlite/absent DATABASE_URL still falls through to the sqlite/memory
     tiers exactly as before (no regression to the dev path).
  2. Real persistence: build a full compiled agent + PostgresSaver against a
     REAL local Postgres, run one turn, close the connection ("process 1"),
     then build a COMPLETELY FRESH checkpointer + agent against the same
     Postgres ("process 2" — simulates a redeploy) and confirm the case file
     and message history survive.

Requires a real Postgres reachable via TEST_DATABASE_URL (skips gracefully
if unset/unreachable — this is the one integration point in the test suite
that can't be faked, since the whole point is proving cross-process disk
persistence, not just calling the right constructor).

Run:
  # start any throwaway Postgres, e.g. via Homebrew:
  #   brew install postgresql@16
  #   LC_ALL=en_US.UTF-8 pg_ctl -D /opt/homebrew/var/postgresql@16 \\
  #       -l /tmp/pg16.log -o "-p 5433" start
  #   createdb -p 5433 macosagent_test
  TEST_DATABASE_URL=postgresql://localhost:5433/macosagent_test \\
      ../../.venv/bin/python tests/test_checkpointer_postgres.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_SEARCH_MODE"] = "off"

from langchain_core.messages import HumanMessage  # noqa: E402

from app.agent.graph import _build_graph, _last_answer, make_memory_checkpointer  # noqa: E402
from tests.test_agent_flow import FakeRetriever, ScriptedLLM, intake_json  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, f"{name}: {detail}"
    PASS += 1


# ---------------------------------------------------------------- 1. routing
print("\n1. checkpointer routing (no real DB needed)")

for key in ("DATABASE_URL", "AGENT_CHECKPOINT_DB"):
    os.environ.pop(key, None)

cp = make_memory_checkpointer()
check("no DATABASE_URL, no AGENT_CHECKPOINT_DB -> in-memory",
      type(cp).__name__ == "InMemorySaver", type(cp).__name__)

os.environ["DATABASE_URL"] = "sqlite:///unused.db"
cp = make_memory_checkpointer()
check("sqlite DATABASE_URL does NOT trigger the postgres branch",
      type(cp).__name__ != "PostgresSaver", type(cp).__name__)
os.environ.pop("DATABASE_URL", None)

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")
if not TEST_DB:
    print("\n(TEST_DATABASE_URL not set — skipping the real-Postgres "
          "persistence checks below. See this file's docstring to run them.)")
    print(f"\n{PASS} routing checks passed (persistence checks skipped).")
    sys.exit(0)

try:
    import psycopg
    psycopg.connect(TEST_DB.replace("postgresql+psycopg://", "postgresql://", 1),
                    connect_timeout=3).close()
except Exception as e:  # noqa: BLE001
    print(f"\n(TEST_DATABASE_URL set but unreachable ({e!r}) — skipping "
          f"persistence checks.)")
    print(f"\n{PASS} routing checks passed (persistence checks skipped).")
    sys.exit(0)

for url_form in (TEST_DB, TEST_DB.replace("postgresql://", "postgresql+psycopg://", 1)):
    os.environ["DATABASE_URL"] = url_form
    cp = make_memory_checkpointer()
    check(f"DATABASE_URL={'normalized' if '+psycopg' in url_form else 'bare'} -> PostgresSaver",
          type(cp).__name__ == "PostgresSaver", type(cp).__name__)
    cp.conn.close()

# ---------------------------------------------------------- 2. real persistence
print("\n2. real cross-process persistence (the actual gap being closed)")

os.environ["DATABASE_URL"] = TEST_DB
CFG = {"configurable": {"thread_id": "test-checkpointer-restart"}}
ANSWER = json.dumps({"action": "answer", "reason": "clear"})

# "process 1": save one turn, then close — nothing kept in memory afterward.
cp1 = make_memory_checkpointer()
agent1 = _build_graph(
    llm=ScriptedLLM(["Try `sudo wdutil info` [Source 1]."]),
    intake_llm=ScriptedLLM([intake_json()]),
    filter_llm=ScriptedLLM([]),
    planner_llm=ScriptedLLM([ANSWER]),
    retriever=FakeRetriever(),
    checkpointer=cp1,
)
r1 = agent1.invoke(
    {"messages": [HumanMessage(content="wifi drops every 10 minutes on my macbook")]},
    config=CFG)
check("process 1: turn answered", "wdutil" in _last_answer(r1))
cp1.conn.close()
del cp1, agent1   # nothing lingers in this process's memory

# "process 2": FRESH checkpointer + FRESH compiled graph, same thread_id —
# this is exactly what a redeploy does (deps.build() runs make_memory_
# checkpointer() again with an empty process).
cp2 = make_memory_checkpointer()
agent2 = _build_graph(
    llm=ScriptedLLM(["Second answer [Source 1]."]),
    intake_llm=ScriptedLLM([intake_json(clean_query="")]),
    filter_llm=ScriptedLLM([]),
    planner_llm=ScriptedLLM([ANSWER]),
    retriever=FakeRetriever(),
    checkpointer=cp2,
)
r2 = agent2.invoke({"messages": [HumanMessage(content="still dropping, what else")]},
                   config=CFG)
check("process 2 (simulated restart): case_file topic survived",
      r2["case_file"]["last_query"] == "wifi drops every 10 minutes",
      r2["case_file"]["last_query"])
check("process 2: full message history survived (both turns present)",
      len(r2["messages"]) >= 4, len(r2["messages"]))
cp2.conn.close()

print(f"\nAll {PASS} checkpointer checks passed (including real Postgres persistence).")
