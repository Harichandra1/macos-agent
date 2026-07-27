"""
test_checkpointer_async.py — the checkpointer must work under astream().

Closes a total production outage. `make_memory_checkpointer()` returned the
SYNCHRONOUS `PostgresSaver` whenever DATABASE_URL was postgres-shaped, but
main.py drives the graph with `astream()`. `PostgresSaver` implements only
get_tuple/put/put_writes — it inherits `BaseCheckpointSaver.aget_tuple`, whose
body is `raise NotImplementedError`. So every /chat turn in production died
before a single node ran, surfacing as the generic
"internal error while generating the answer".

Why the existing suite missed it: test_checkpointer_postgres.py exercises the
SYNC `.invoke()` path exclusively — the one code path the server never uses. A
checkpointer test that never calls astream() cannot catch an astream-only bug.

The load-bearing assertion here is `natively_async()`: EVERY checkpointer the
factories can return — including the degraded fallback — must implement the
async interface itself rather than inheriting the NotImplementedError stubs.

Run (no database required):
  cd Agent/backend && ../../.venv/bin/python tests/test_checkpointer_async.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_SEARCH_MODE"] = "off"

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402

from app.agent.graph import (  # noqa: E402
    _build_graph, make_async_checkpointer, make_memory_checkpointer,
)
from tests.test_agent_flow import FakeRetriever, ScriptedLLM, intake_json  # noqa: E402

PASS = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, f"{name}: {detail}"
    PASS += 1


# The async graph loop awaits these three. A saver that inherits any of them
# from BaseCheckpointSaver raises NotImplementedError the moment astream runs.
ASYNC_METHODS = ("aget_tuple", "aput", "aput_writes")


def natively_async(cls) -> bool:
    return all(getattr(cls, m) is not getattr(BaseCheckpointSaver, m)
               for m in ASYNC_METHODS)


ANSWER = json.dumps({"action": "answer", "reason": "clear"})


def build(checkpointer):
    return _build_graph(
        llm=ScriptedLLM(["Try `sudo wdutil info` [Source 1]."]),
        intake_llm=ScriptedLLM([intake_json()]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([ANSWER]),
        retriever=FakeRetriever(),
        checkpointer=checkpointer,
    )


async def astream_turn(checkpointer, thread: str):
    """Drive one turn exactly the way main.py does."""
    agent = build(checkpointer)
    async for _ in agent.astream(
        {"messages": [HumanMessage(content="wifi drops every 10 minutes")]},
        config={"configurable": {"thread_id": thread}},
        stream_mode=["updates", "messages"],
    ):
        pass


# --------------------------------------------------- 1. the interface contract
print("\n1. async interface contract (pure introspection, no I/O)")

check("sync PostgresSaver is NOT natively async — this is why it broke prod",
      not natively_async(PostgresSaver))
check("AsyncPostgresSaver is natively async", natively_async(AsyncPostgresSaver))
check("MemorySaver is natively async (why dev never saw the bug)",
      natively_async(MemorySaver))

# ------------------------------------------------------- 2. behavioural pinning
print("\n2. astream() behaviour, so a langgraph change cannot silently regress")


class SyncOnlySaver(PostgresSaver):
    """Stands in for the pre-fix production checkpointer without needing a DB."""

    def __init__(self):  # skip the real connection
        pass


try:
    asyncio.run(astream_turn(SyncOnlySaver(), "sync-saver"))
    check("sync saver under astream raises", False, "it unexpectedly succeeded")
except NotImplementedError:
    check("sync saver under astream raises NotImplementedError (the prod bug)", True)

asyncio.run(astream_turn(MemorySaver(), "memory-saver"))
check("natively-async saver streams a full turn under astream", True)

# ------------------------------------------------- 3. the factory's guarantee
print("\n3. make_async_checkpointer() always returns an astream-safe saver")

os.environ.pop("DATABASE_URL", None)
cp, pool = asyncio.run(make_async_checkpointer())
check("no DATABASE_URL -> in-memory tier", type(cp).__name__ == "InMemorySaver",
      type(cp).__name__)
check("in-memory tier reports no pool to close", pool is None)
check("in-memory tier is astream-safe", natively_async(type(cp)))

# Unreachable Postgres: must degrade rather than raise, AND the thing it
# degrades to must still be astream-safe. The old code failed this even when
# Postgres was REACHABLE, because the success path returned a sync saver.
os.environ["DATABASE_URL"] = "postgresql://nobody:nobody@127.0.0.1:1/none"

records = []


class _Capture(logging.Handler):
    def emit(self, record):
        records.append(record)


app_logger = logging.getLogger("macos_agent")
handler = _Capture()
app_logger.addHandler(handler)
try:
    cp, pool = asyncio.run(make_async_checkpointer())
finally:
    app_logger.removeHandler(handler)

check("unreachable Postgres degrades instead of raising", cp is not None)
check("degraded fallback is STILL astream-safe", natively_async(type(cp)),
      type(cp).__name__)
check("degrade is logged at ERROR (not a silent fallback)",
      any(r.levelno >= logging.ERROR for r in records))
check("degrade log carries a traceback (exc_info set)",
      any(r.exc_info is not None for r in records if r.levelno >= logging.ERROR))

# ----------------------------------------------- 4. the sync-path guard rails
print("\n4. make_memory_checkpointer() guard for async-capable callers")

os.environ["DATABASE_URL"] = "postgresql://nobody:nobody@127.0.0.1:1/none"
cp = make_memory_checkpointer(allow_sync_postgres=False)
check("allow_sync_postgres=False never returns a sync Postgres saver",
      natively_async(type(cp)), type(cp).__name__)

os.environ.pop("DATABASE_URL", None)
cp = make_memory_checkpointer()
check("default signature still works for sync callers (CLI, evals)",
      type(cp).__name__ == "InMemorySaver", type(cp).__name__)

print(f"\nall {PASS} checks passed\n")
