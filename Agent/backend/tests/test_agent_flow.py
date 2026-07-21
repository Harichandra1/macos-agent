"""
test_agent_flow.py — scripted multi-turn agent-flow test with stubbed LLMs.

No API keys, no network: fake LLMs + a fake retriever are injected through the
`_build_graph` seam. Verifies the agentic algorithm's contract:

  1. Turn 1 (machine-state symptom) → agent requests a GROUNDED diagnostic,
     `pending_diagnostic` is recorded, budgets tick.
  2. Turn 2 (pasted output)         → post-diagnostic flow: synthesize gets the
     POST_DIAGNOSTIC prompt, answer opens with "Root cause:", pending cleared.
  3. Self-correction (refine) works EVERY turn, not once per session
     (refine_count resets at intake).
  4. Turn 3 (different problem)     → topic + per-problem budgets reset, machine
     facts survive.
  5. Guardrails: an intake-LLM outage or a retriever outage degrades — the turn
     still completes with an answer.

Run:  cd Agent/backend && ../../.venv/bin/python tests/test_agent_flow.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_SEARCH_MODE"] = "off"          # keep tests offline

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.agent.graph import DECLINE_MESSAGE, _build_graph, _last_answer  # noqa: E402
from app.agent.prompts import POST_CLARIFY_PROMPT, POST_DIAGNOSTIC_PROMPT  # noqa: E402


# ---------------------------------------------------------------- fakes

class ScriptedLLM:
    """Returns canned responses in order; records every (system, human) prompt."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def invoke(self, messages):
        system = str(messages[0].content) if messages else ""
        human = str(messages[-1].content) if messages else ""
        self.calls.append((system, human))
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        # A real AIMessage — synthesize puts the LLM's response object straight
        # into graph state (id-preserving, so token streams dedupe correctly).
        return AIMessage(content=resp)


class FakeRetriever:
    """Returns one canned KB doc whose text grounds `sudo wdutil info`."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.queries: list[str] = []

    def retrieve(self, query, **kwargs):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("qdrant down")
        return {
            "hits": [{
                "title": "Diagnosing Wi-Fi drops (wdutil)",
                "url": "kb://diag_wifi",
                "text": ("When Wi-Fi drops repeatedly, run `sudo wdutil info` "
                         "in Terminal and check the channel: DFS channels "
                         "(52-144, 149) drop on radar detection. Fix: set the "
                         "router to a non-DFS channel (36-48) or 2.4 GHz. "
                         "Also check RSSI. networksetup can list interfaces."),
                "source": "macos_man_pages",
                "category": "wifi",
                "difficulty_tier": 2,
                "score": 0.72,
                "article_id": "diag_wifi",
            }],
            "fallback": False,
            "query": query,
        }


def intake_json(**over):
    base = {
        "macos_version": "14", "mac_chip": "apple_silicon", "category": "wifi",
        "already_tried": [], "clean_query": "wifi drops every 10 minutes",
        "new_problem": False,
    }
    base.update(over)
    return json.dumps(base)


def build(intake_resps, planner_resps, synth_resps, retriever=None):
    return _build_graph(
        llm=ScriptedLLM(synth_resps),
        intake_llm=ScriptedLLM(intake_resps),
        filter_llm=ScriptedLLM([]),          # single-source merge → never called
        planner_llm=ScriptedLLM(planner_resps),
        retriever=retriever or FakeRetriever(),
        checkpointer=MemorySaver(),
    ), None


CFG = {"configurable": {"thread_id": "t1"}}


def turn(agent, text):
    return agent.invoke({"messages": [HumanMessage(content=text)]}, config=CFG)


# ---------------------------------------------------------------- tests

def test_diagnostic_loop_and_topic_reset():
    diagnose = json.dumps({
        "action": "diagnose", "reason": "state-dependent",
        "diagnostic": {"command": "sudo wdutil info",
                       "rationale": "reveals channel + RSSI",
                       "look_for": "DFS channel or weak RSSI"},
    })
    answer = json.dumps({"action": "answer", "reason": "output received"})

    synth = ScriptedLLM([
        # turn 2: post-diagnostic answer (grounded command)
        "**Root cause:** DFS radar hits — your output shows channel 149 (DFS), "
        "RSSI -52 [Source 1]\n\n**Fix:** set the router to a non-DFS channel "
        "(36-48).\n\n**Verify:** run `sudo wdutil info` again after the change.",
        # turn 3: new-problem answer
        "**Fix:** check Bluetooth [Source 1]",
    ])
    agent = _build_graph(
        llm=synth,
        intake_llm=ScriptedLLM([
            intake_json(),                                            # turn 1
            intake_json(clean_query="", already_tried=[]),            # turn 2 paste
            intake_json(category="bluetooth", new_problem=True,       # turn 3
                        clean_query="bluetooth mouse disconnects randomly"),
        ]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([diagnose, answer, answer]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )

    # ---- turn 1: symptom → grounded diagnostic requested
    r1 = turn(agent, "My M2 MacBook on Sonoma drops wifi every 10 minutes")
    a1 = _last_answer(r1)
    assert "sudo wdutil info" in a1 and "paste the output" in a1.lower(), a1
    cf = r1["case_file"]
    assert cf["pending_diagnostic"]["command"] == "sudo wdutil info", cf
    assert cf["diagnostic_count"] == 1
    assert cf["last_query"] == "wifi drops every 10 minutes"

    # ---- turn 2: pasted output → post-diagnostic crisp answer
    paste = ("WDUTIL INFO\n" + "boring line\n" * 60 +
             "Channel : 149 (DFS)\nRSSI : -52 dBm\nlast disconnect: radar detected\n")
    r2 = turn(agent, paste)
    a2 = _last_answer(r2)
    assert a2.startswith("**Root cause:**"), a2
    # synthesize must have used the POST_DIAGNOSTIC prompt
    assert synth.calls[0][0] == POST_DIAGNOSTIC_PROMPT
    # …and its prompt must carry the ORIGINAL problem + the diagnostic we asked for
    assert "wifi drops every 10 minutes" in synth.calls[0][1]
    assert "sudo wdutil info" in synth.calls[0][1]
    assert r2["case_file"]["pending_diagnostic"] is None, "pending not cleared"
    # verification ran and the wdutil command is grounded
    assert r2["verification"]["total"] >= 1
    assert r2["verification"]["ungrounded"] == [], r2["verification"]

    # ---- turn 3: DIFFERENT problem → topic + budgets reset, facts survive
    r3 = turn(agent, "Now my bluetooth mouse keeps disconnecting randomly")
    cf3 = r3["case_file"]
    assert cf3["last_query"] == "bluetooth mouse disconnects randomly", cf3
    assert cf3["diagnostic_count"] == 0 and cf3["asked_clarify"] is False
    assert cf3["macos_version"] == "14" and cf3["mac_chip"] == "apple_silicon"
    print("  ✓ diagnostic loop + post-diagnostic prompt + topic reset")


def test_refine_resets_every_turn():
    answer = json.dumps({"action": "answer", "reason": "clear"})
    # Both turns synthesize an UNGROUNDED command first (hdiutil is not in the
    # fake KB text), forcing verify → refine → re-synthesize. If refine_count
    # leaked across turns, turn 2 would emit the bad draft with no refine.
    bad = "Run this:\n```\nhdiutil attach /Volumes/Foo.dmg\n```"
    good = "Run `sudo wdutil info` and check the channel [Source 1]."
    synth = ScriptedLLM([bad, good, bad, good])
    agent = _build_graph(
        llm=synth,
        intake_llm=ScriptedLLM([intake_json(), intake_json()]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer, answer]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r1 = turn(agent, "wifi drops on my macbook constantly, already restarted")
    assert r1["refine_count"] == 1, r1["refine_count"]
    assert _last_answer(r1) == good
    r2 = turn(agent, "it still drops, what else can I try here")
    assert r2["refine_count"] == 1, (
        f"refine_count={r2['refine_count']} — self-correction did not reset per turn")
    assert _last_answer(r2) == good
    print("  ✓ refine/self-correction resets every turn")


def test_outage_guardrails():
    answer = json.dumps({"action": "answer", "reason": "clear"})

    # (a) intake LLM down → regex fallback carries the turn
    agent = _build_graph(
        llm=ScriptedLLM(["Try `sudo wdutil info` [Source 1]."]),
        intake_llm=ScriptedLLM([RuntimeError("groq 429 daily cap")]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "My M2 MacBook on Sonoma keeps dropping wifi")
    assert "wdutil" in _last_answer(r)
    assert r["intake"]["macos_version"] == "14"       # regex fallback extracted it
    assert r["intake"]["category"] == "wifi"

    # (b) retriever down → KB skipped, turn still answers (no sources)
    agent = _build_graph(
        llm=ScriptedLLM(["I could not find grounded sources for this."] * 2),
        intake_llm=ScriptedLLM([intake_json()]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer]),
        retriever=FakeRetriever(fail=True),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "wifi drops all the time on my macbook")
    assert r["kb_low_conf"] is True and r["kb_results"] == []
    assert _last_answer(r)                             # an answer was produced

    # (c) synth LLM down entirely → honest apology, never an exception
    agent = _build_graph(
        llm=ScriptedLLM([RuntimeError("all providers down")] * 2),
        intake_llm=ScriptedLLM([intake_json()]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "wifi drops all the time on my macbook")
    assert "temporary problem" in _last_answer(r), _last_answer(r)
    print("  ✓ outage guardrails (intake / retriever / synth) degrade, never die")


def test_planner_outage_answers():
    agent = _build_graph(
        llm=ScriptedLLM(["Answer text `sudo wdutil info` [Source 1]."]),
        intake_llm=ScriptedLLM([intake_json()]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([RuntimeError("planner down")]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "wifi drops constantly on my macbook air")
    assert r["action"] == "answer" and _last_answer(r)
    print("  ✓ planner outage falls back to answer")


class EmptyRetriever(FakeRetriever):
    """Simulates a KB miss: no hits, fallback signalled."""

    def retrieve(self, query, **kwargs):
        self.queries.append(query)
        return {"hits": [], "fallback": True, "query": query}


def test_clarify_batched_questions_and_reply():
    clarify = json.dumps({
        "action": "clarify", "reason": "cause is ambiguous",
        "questions": [
            {"text": "Does it drop only on this network, or on every network?",
             "options": ["Only this network", "Every network"]},
            {"text": "Did it start after a macOS update?",
             "options": ["Yes", "No"]},
        ],
    })
    # Turn 2: planner tries to clarify AGAIN → budget must downgrade to answer.
    clarify_again = json.dumps({
        "action": "clarify", "reason": "want more",
        "questions": [{"text": "Anything else?", "options": []}],
    })
    planner = ScriptedLLM([clarify, clarify_again])
    synth = ScriptedLLM(["Since it only drops at home, the router is the cause "
                         "[Source 1]. Run `sudo wdutil info` to confirm."])
    agent = _build_graph(
        llm=synth,
        intake_llm=ScriptedLLM([intake_json(), intake_json(clean_query="")]),
        filter_llm=ScriptedLLM([]),
        planner_llm=planner,
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )

    # ---- turn 1: clarify with TWO batched questions
    r1 = turn(agent, "my macbook wifi keeps dropping, its so annoying")
    assert r1["action"] == "clarify"
    a1 = _last_answer(r1)
    assert "1. Does it drop only on this network" in a1 and "2. Did it start" in a1, a1
    assert len(r1["questions"]) == 2
    assert r1["questions"][0]["options"] == ["Only this network", "Every network"]
    cf = r1["case_file"]
    assert cf["asked_clarify"] is True
    assert len(cf["pending_questions"]) == 2

    # ---- turn 2: the ANSWERS — long, zero token-overlap with the topic; would
    # trip the new-problem backstop if clarify_reply didn't suppress it.
    r2 = turn(agent, "It happens only at home actually, and yes it started after the last update")
    assert r2["case_file"]["last_query"] == "wifi drops every 10 minutes", (
        "clarify answers were mistaken for a new problem")
    # decide saw the answered-questions stage and the budget forced answer
    assert "ANSWERED our clarifying questions" in planner.calls[1][1]
    assert r2["action"] == "answer"
    assert r2["case_file"]["pending_questions"] is None, "pending_questions not cleared"
    assert _last_answer(r2).startswith("Since it only drops at home")
    # synthesize reasoned over THEIR ANSWERS: post-clarify prompt + Q&A pairing
    assert synth.calls[0][0] == POST_CLARIFY_PROMPT
    assert "Does it drop only on this network" in synth.calls[0][1]
    print("  ✓ batched clarify questions + reply safety + one-clarify budget")


def test_new_problem_overrides_pending_diagnostic():
    # Turn 1: wifi symptom → diagnostic requested (pending_diagnostic set).
    # Turn 2: the user IGNORES it and asks a DIFFERENT problem, which the
    # intake LLM flags new_problem — it must NOT be treated as command output
    # (no post-diagnostic prompt), and the topic/budgets must reset.
    diagnose = json.dumps({
        "action": "diagnose", "reason": "state",
        "diagnostic": {"command": "sudo wdutil info", "rationale": "r",
                       "look_for": "DFS"},
    })
    answer = json.dumps({"action": "answer", "reason": "clear"})
    synth = ScriptedLLM(["Bluetooth answer `sudo wdutil info` [Source 1]."])
    agent = _build_graph(
        llm=synth,
        intake_llm=ScriptedLLM([
            intake_json(),
            intake_json(category="bluetooth", new_problem=True,
                        clean_query="bluetooth mouse disconnects randomly"),
        ]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([diagnose, answer]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r1 = turn(agent, "My M2 MacBook on Sonoma drops wifi every 10 minutes")
    assert r1["case_file"]["pending_diagnostic"], "diagnostic not pending"
    r2 = turn(agent, "Actually forget that — my bluetooth mouse keeps disconnecting randomly")
    assert r2["case_file"]["last_query"] == "bluetooth mouse disconnects randomly"
    assert r2["case_file"]["pending_diagnostic"] is None, "stale pending survived reset"
    assert r2["case_file"]["diagnostic_count"] == 0
    assert synth.calls[0][0] != POST_DIAGNOSTIC_PROMPT, (
        "new problem was misread as diagnostic output")
    print("  ✓ explicit new problem overrides a stale pending diagnostic")


def test_no_context_forces_clarify():
    # Retrieval finds NOTHING; the planner (wrongly) wants to answer anyway.
    # The anti-hallucination gate must override to clarify with template
    # questions instead of letting it guess.
    answer = json.dumps({"action": "answer", "reason": "confident"})
    agent = _build_graph(
        llm=ScriptedLLM([]),                      # synthesize must NOT run
        intake_llm=ScriptedLLM([intake_json(macos_version=None, mac_chip=None)]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer]),
        retriever=EmptyRetriever(),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "my macbook wifi keeps dropping all the time at my house")
    assert r["action"] == "clarify", (
        f"action={r['action']} — empty context should force questions, not a guess")
    qs = r["questions"]
    assert 1 <= len(qs) <= 3
    assert any("network" in q["text"].lower() for q in qs)        # wifi discriminator
    assert any(q["options"] for q in qs)                           # chips present
    assert r["case_file"]["asked_clarify"] is True
    print("  ✓ anti-hallucination gate: empty context → clarify, not a guess")


def test_off_topic_declines():
    # Off-topic message ("give me html code") — decide must short-circuit to
    # decline WITHOUT ever calling the planner or synth LLM (empty response
    # lists would raise if invoked), and without running retrieval.
    retriever = FakeRetriever()
    agent = _build_graph(
        llm=ScriptedLLM([]),
        intake_llm=ScriptedLLM([intake_json(
            category="general", clean_query="", on_topic=False)]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([]),
        retriever=retriever,
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "give me html code for a login page")
    assert r["action"] == "decline"
    assert _last_answer(r) == DECLINE_MESSAGE
    assert retriever.queries == [], "retrieval ran for an off-topic message"
    print("  ✓ off-topic message declines without retrieval or synthesis")


def test_on_topic_keyword_backstop():
    # Intake LLM wrongly says on_topic=False, but the message clearly names a
    # Mac symptom — the deterministic keyword backstop must force it back on.
    answer = json.dumps({"action": "answer", "reason": "clear"})
    agent = _build_graph(
        llm=ScriptedLLM(["Try `sudo wdutil info` [Source 1]."]),
        intake_llm=ScriptedLLM([intake_json(on_topic=False)]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "wifi keeps dropping on my macbook")
    assert r["action"] != "decline", "keyword backstop failed to override a false negative"
    assert r["intake"]["on_topic"] is True
    print("  ✓ on-topic keyword backstop overrides a wrong LLM verdict")


def test_decline_preserves_pending_diagnostic():
    # Turn 1: normal flow → a diagnostic gets requested (pending_diagnostic set).
    # Turn 2: an off-topic aside — must decline WITHOUT touching the pending
    # diagnostic/budget, so a real reply next turn still resumes correctly.
    diagnose = json.dumps({
        "action": "diagnose", "reason": "state-dependent",
        "diagnostic": {"command": "sudo wdutil info", "rationale": "r",
                       "look_for": "DFS channel"},
    })
    agent = _build_graph(
        llm=ScriptedLLM([]),
        intake_llm=ScriptedLLM([
            intake_json(),
            intake_json(clean_query="", on_topic=False),
        ]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([diagnose]),
        retriever=FakeRetriever(),
        checkpointer=MemorySaver(),
    )
    r1 = turn(agent, "My M2 MacBook on Sonoma drops wifi every 10 minutes")
    assert r1["case_file"]["pending_diagnostic"]["command"] == "sudo wdutil info"
    r2 = turn(agent, "write me a poem about the ocean")
    assert r2["action"] == "decline"
    assert _last_answer(r2) == DECLINE_MESSAGE
    cf = r2["case_file"]
    assert cf["pending_diagnostic"]["command"] == "sudo wdutil info", (
        "off-topic aside clobbered the pending diagnostic")
    assert cf["diagnostic_count"] == 1
    assert cf["last_query"] == "wifi drops every 10 minutes", (
        "off-topic aside was mistaken for a new problem")
    print("  ✓ decline leaves pending diagnostic/topic state untouched")


class RescueRetriever(FakeRetriever):
    """Symptom queries return a generic doc; a query naming pmset returns its doc."""

    def retrieve(self, query, **kwargs):
        self.queries.append(query)
        if "pmset" in query:
            return {"hits": [{
                "title": "Diagnosing sleep/battery drain (pmset)",
                "url": "kb://diag_battery",
                "text": "Run `pmset -g assertions` and look for "
                        "PreventUserIdleSystemSleep held by a named process.",
                "source": "macos_man_pages", "category": "battery",
                "difficulty_tier": 2, "score": 0.7, "article_id": "diag_battery",
            }], "fallback": False, "query": query}
        return {"hits": [{
            "title": "Generic battery tips", "url": "kb://tips",
            "text": "Batteries drain. Check settings and reduce brightness.",
            "source": "ask_different", "category": "battery",
            "difficulty_tier": 1, "score": 0.6, "article_id": "tips",
        }], "fallback": False, "query": query}


def test_decide_rescue_retrieval():
    # Planner proposes the decisive command, but the symptom retrieval didn't
    # surface its doc — rescue retrieval must ground it instead of downgrading
    # to a blind answer.
    diagnose = json.dumps({
        "action": "diagnose", "reason": "state-dependent",
        "diagnostic": {"command": "pmset -g assertions",
                       "rationale": "find the process blocking sleep",
                       "look_for": "PreventUserIdleSystemSleep"},
    })
    agent = _build_graph(
        llm=ScriptedLLM([]),
        intake_llm=ScriptedLLM([intake_json(category="battery",
                                            clean_query="battery drains overnight")]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([diagnose]),
        retriever=RescueRetriever(),
        checkpointer=MemorySaver(),
    )
    r = turn(agent, "my macbook battery drains overnight even when closed")
    assert r["action"] == "diagnose", (
        f"action={r['action']} — rescue retrieval failed to ground the diagnostic")
    assert "pmset -g assertions" in _last_answer(r)
    # the rescued doc joined the context
    assert any((m.get("url") == "kb://diag_battery") for m in r["merged"]), r["merged"]
    print("  ✓ decide-time rescue retrieval grounds the decisive diagnostic")


def test_paste_query_stays_salient():
    answer = json.dumps({"action": "answer", "reason": "clear"})
    retriever = FakeRetriever()
    agent = _build_graph(
        llm=ScriptedLLM(["ok `sudo wdutil info` [Source 1]"]),
        intake_llm=ScriptedLLM([intake_json(), intake_json(clean_query="")]),
        filter_llm=ScriptedLLM([]),
        planner_llm=ScriptedLLM([answer, answer]),
        retriever=retriever,
        checkpointer=MemorySaver(),
    )
    turn(agent, "wifi drops every 10 minutes on my macbook")
    big_paste = "boring log line\n" * 400 + "error: radar detected on channel 149\n"
    agent2_unused = None  # readability
    # second turn on same thread — capture what retrieval embedded
    agentless = agent.invoke({"messages": [HumanMessage(content=big_paste)]}, config=CFG)
    q = retriever.queries[-1]
    assert len(q) < 700, f"retrieval query not capped: {len(q)} chars"
    assert "wifi drops every 10 minutes" in q            # topic anchor kept
    assert "radar detected" in q                          # salient line kept
    assert "boring log line" not in q                     # noise dropped
    print("  ✓ pasted output reduced to salient excerpt in retrieval query")


if __name__ == "__main__":
    test_diagnostic_loop_and_topic_reset()
    test_refine_resets_every_turn()
    test_outage_guardrails()
    test_planner_outage_answers()
    test_decide_rescue_retrieval()
    test_clarify_batched_questions_and_reply()
    test_new_problem_overrides_pending_diagnostic()
    test_off_topic_declines()
    test_on_topic_keyword_backstop()
    test_decline_preserves_pending_diagnostic()
    test_no_context_forces_clarify()
    test_paste_query_stays_salient()
    print("\nAll agent-flow tests passed.")
