"""
test_history_summary.py — rolling-summary conversation memory (Phase 1).

Offline, scripted-LLM test of the cost-control compression in graph.py:

  1. Short sessions (≤4 turns) are untouched — no summary, full transcript.
  2. Turn 5 crosses the keep-window: turns older than the last 6 messages are
     folded into `history_summary` and DELETED from checkpointed state.
  3. The summary is injected into both the planner (decide) and synthesize
     prompts; later folds feed the existing summary back in (rolling).
  4. Summarizer outage → degrade: keep the full transcript this turn, no
     message loss, the turn still answers.

Run:  cd Agent/backend && ../../.venv/bin/python tests/test_history_summary.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_SEARCH_MODE"] = "off"          # keep tests offline

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from test_agent_flow import FakeRetriever, ScriptedLLM, intake_json  # noqa: E402
from app.agent.graph import _HISTORY_KEEP, _build_graph, _last_answer  # noqa: E402

ANSWER = json.dumps({"action": "answer", "reason": "clear"})
GROUNDED = "Try `sudo wdutil info` and check the channel [Source 1]."
SUMMARY_1 = ("User's M2 MacBook on Sonoma drops Wi-Fi every 10 minutes; "
             "restarting and toggling Wi-Fi did not help.")
SUMMARY_2 = (SUMMARY_1 + " wdutil output showed a DFS channel; a non-DFS "
             "channel change is being tested.")

CFG = {"configurable": {"thread_id": "t1"}}


def turn(agent, text):
    return agent.invoke({"messages": [HumanMessage(content=text)]}, config=CFG)


def run():
    # intake_llm serves BOTH the summarizer and intake extraction, in call
    # order within node_intake: summary first (when folding), then intake.
    intake_llm = ScriptedLLM([
        intake_json(),   # turn 1
        intake_json(),   # turn 2
        intake_json(),   # turn 3
        intake_json(),   # turn 4      (7 msgs at intake → below threshold)
        SUMMARY_1,       # turn 5 fold (9 msgs at intake → fold 3, keep 6)
        intake_json(),   # turn 5
        RuntimeError("groq 429 daily cap"),   # turn 6 fold attempt → outage
        intake_json(),   # turn 6      (degrades: transcript kept)
        SUMMARY_2,       # turn 7 fold (rolls SUMMARY_1 + outage backlog in)
        intake_json(),   # turn 7
    ])
    planner = ScriptedLLM([ANSWER] * 7)
    synth = ScriptedLLM([GROUNDED] * 7)
    agent = _build_graph(
        llm=synth, intake_llm=intake_llm, filter_llm=ScriptedLLM([]),
        planner_llm=planner, retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )

    texts = [
        "My M2 MacBook on Sonoma drops wifi every 10 minutes",
        "still drops, wifi already toggled off and on",
        "no luck, same problem with wifi after that",
        "wifi still dropping, what else",
        "still the same wifi issue, tried that too",
        "wifi drop again, no change at all",
        "wifi still broken, same thing",
    ]

    # ---- turns 1-4: below the fold threshold — untouched transcript --------
    for t in texts[:4]:
        r = turn(agent, t)
        assert _last_answer(r) == GROUNDED
    assert (r.get("history_summary") or "") == "", r.get("history_summary")
    assert len(r["messages"]) == 8, len(r["messages"])
    print("  ✓ short session untouched (no summary, 8 messages after turn 4)")

    # ---- turn 5: fold fires — old turns summarized + removed ---------------
    r5 = turn(agent, texts[4])
    assert r5["history_summary"] == SUMMARY_1, r5["history_summary"]
    # 9 pre-turn messages − 3 folded + 1 new answer = 7
    assert len(r5["messages"]) == 7, len(r5["messages"])
    surviving = " | ".join(str(m.content) for m in r5["messages"])
    assert texts[0] not in surviving, "folded turn still in transcript"
    assert texts[2] in surviving, "kept turn was wrongly removed"
    # summarizer saw an empty existing summary + the folded turn's text
    summ_call = intake_llm.calls[4][1]
    assert "EXISTING SUMMARY:\n(none)" in summ_call, summ_call[:200]
    assert texts[0] in summ_call
    # the summary reached the synthesis prompt. (decide doesn't get it: past
    # 3 user turns it short-circuits to "answer" without calling the planner,
    # and compression can only fire from turn 5 — the planner path never
    # coexists with a summary.)
    assert "## Earlier conversation (compressed)\n" + SUMMARY_1 in synth.calls[4][1]
    print("  ✓ turn 5 folds history: summary stored, messages trimmed, prompts fed")

    # ---- turn 6: summarizer outage → keep transcript, turn still answers ---
    r6 = turn(agent, texts[5])
    assert _last_answer(r6) == GROUNDED
    assert r6["history_summary"] == SUMMARY_1, "summary must survive an outage"
    # nothing was removed: 7 + user + answer = 9
    assert len(r6["messages"]) == 9, len(r6["messages"])
    print("  ✓ summarizer outage degrades safely (no message loss)")

    # ---- turn 7: rolling — existing summary folded together with backlog ---
    r7 = turn(agent, texts[6])
    assert r7["history_summary"] == SUMMARY_2
    summ_call = intake_llm.calls[8][1]
    assert SUMMARY_1 in summ_call, "existing summary not fed back to the summarizer"
    assert len(r7["messages"]) == _HISTORY_KEEP + 1, len(r7["messages"])
    print("  ✓ rolling fold: prior summary + backlog compressed together")


if __name__ == "__main__":
    print("\nrolling-summary memory")
    run()
    print("\nAll history-summary checks passed.")
