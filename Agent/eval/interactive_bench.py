"""
interactive_bench.py — the multi-turn "beat the frontier" proof.

The one-shot benchmark showed GPT-4o wins when it already knows the answer. But a
real troubleshooter is INTERACTIVE and can see the user's machine. This eval
measures exactly that edge.

Each scenario has a HIDDEN ground truth: the real `root_cause`, a `system_state`
(what specific diagnostics would reveal), and the `correct_fix`. A **simulated
user** (LLM) states only the surface symptom and, when asked a question or given a
command, reveals ONLY the matching detail from the hidden state — never the cause.

Two arms play the same scenarios ↔ the simulated user, up to N turns:
  - our AGENT (multi-turn, grounded, requests exact diagnostics)
  - raw GPT-4o (may also ask questions across turns, but has no KB/tools)

A judge panel (Groq gpt-oss + qwen, reused from benchmark.py) decides whether the
final diagnosis matches the hidden root_cause + correct_fix. We report resolution
rate and average turns-to-resolution, agent vs GPT-4o.

Usage:
  python interactive_bench.py --quick        # 3 scenarios
  python interactive_bench.py                # all
  python interactive_bench.py --max-turns 6
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "backend"))

from app.agent.providers import make_chat_model, key_present, PROVIDERS  # noqa: E402
from benchmark import JUDGE_REGISTRY, _judge_spec, _parse_judgment, _load, _save  # noqa: E402

load_dotenv()

RESULTS_DIR = _HERE / "results"
TRANSCRIPTS = RESULTS_DIR / "interactive_transcripts.md"
REPORT      = RESULTS_DIR / "interactive_report.json"

MAX_TURNS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Scenarios — hidden ground truth the agent must uncover interactively.
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    symptom: str                 # what the user says first
    macos_version: str
    chip: str
    root_cause: str              # HIDDEN
    system_state: dict           # HIDDEN: command/topic -> what it reveals
    correct_fix: str             # HIDDEN
    tags: list[str] = field(default_factory=list)


SCENARIOS: list[Scenario] = [
    Scenario(
        id="wifi_dfs_channel",
        symptom="My MacBook keeps dropping Wi-Fi every few minutes at home. It's really annoying.",
        macos_version="14", chip="apple_silicon",
        root_cause="The router is on a DFS channel (149) and the Mac disconnects on radar detection (CSA events).",
        system_state={
            "wdutil info / airport": "Wi-Fi: channel 149 (DFS), RSSI -52 (strong signal), PHY mode 802.11ax",
            "log show wifi / disconnect reason": "repeated 'CSA' / 'channel switch announcement' and 'radar detected' events",
            "facts": "macOS Sonoma 14.5, M2 MacBook Air. Signal is strong. Only this Mac drops; a Windows laptop is fine.",
        },
        correct_fix="Change the router's 5GHz channel OFF the DFS range (use 36-48) so radar events stop forcing disconnects.",
    ),
    Scenario(
        id="battery_assertion",
        symptom="My MacBook battery drains overnight even when it's closed and asleep in my bag.",
        macos_version="14", chip="apple_silicon",
        root_cause="A process is holding a power assertion preventing sleep (PreventUserIdleSystemSleep by 'bird', the iCloud daemon).",
        system_state={
            "pmset -g assertions": "PreventUserIdleSystemSleep 1 held by 'bird' (com.apple.bird)",
            "pmset -g log / sleep wake": "system repeatedly wakes with 'Wake reason: bird' and fails to enter standby",
            "facts": "Sonoma 14.5, M3. Battery Settings look normal. Closing the lid does not stop the drain.",
        },
        correct_fix="The iCloud sync daemon (bird) is blocking sleep — sign out/in of iCloud (or restart bird) to clear the stuck assertion.",
    ),
    Scenario(
        id="kext_panic",
        symptom="My Mac randomly restarts with a message that it panicked. It happens a few times a day.",
        macos_version="14", chip="intel",
        root_cause="A third-party kernel extension (com.thirdparty.vpnkext) appears in the panic backtrace and is causing the crashes.",
        system_state={
            "panic report / DiagnosticReports": "panic .panic file lists 'com.thirdparty.vpnkext' in the Kernel Extensions in backtrace",
            "kextstat / kmutil showloaded": "com.thirdparty.vpnkext (a VPN app's network extension) is loaded",
            "facts": "Ventura 14.4, Intel Mac. Installed a VPN app recently. Panics started around then.",
        },
        correct_fix="Uninstall/disable the third-party VPN kext (com.thirdparty.vpnkext) named in the panic backtrace.",
    ),
    Scenario(
        id="disk_snapshots",
        symptom="My disk says it's almost full but I can't find what's using the space. Finder shows way less than the disk usage.",
        macos_version="15", chip="apple_silicon",
        root_cause="Local Time Machine APFS snapshots are consuming the 'missing' space.",
        system_state={
            "tmutil listlocalsnapshots /": "lists ~12 local snapshots com.apple.TimeMachine.* dates over the last week",
            "diskutil apfs / storage": "APFS container shows large 'purgeable' space; Finder undercounts snapshot space",
            "facts": "Sequoia 15.5, M2. Time Machine is on. About 80GB unaccounted for.",
        },
        correct_fix="Thin/delete local APFS Time Machine snapshots (tmutil deletelocalsnapshots or thinlocalsnapshots) to reclaim the space.",
    ),
    Scenario(
        id="mic_tcc_stuck",
        symptom="My microphone doesn't work in one specific app, but it works everywhere else.",
        macos_version="14", chip="apple_silicon",
        root_cause="The app's TCC microphone entry is in a stuck/denied state even though it looks granted.",
        system_state={
            "privacy settings / tccutil": "the app is missing from or greyed out in Settings > Privacy > Microphone; toggling does nothing",
            "reset": "resetting the app's TCC microphone entry makes the prompt reappear and then it works",
            "facts": "Sonoma 14.5, M1. Mic works in FaceTime and Zoom, only fails in this one app.",
        },
        correct_fix="Reset the app's Microphone TCC entry with `tccutil reset Microphone <bundle-id>`, then relaunch and re-grant.",
    ),
    Scenario(
        id="spotlight_cpu",
        symptom="My Mac is hot and the fans are loud, and it feels sluggish, but I'm not running anything heavy.",
        macos_version="14", chip="intel",
        root_cause="Spotlight (mds/mds_stores) is stuck reindexing and pinning the CPU.",
        system_state={
            "activity monitor / top": "mds_stores and mdworker using very high CPU sustained",
            "mdutil status": "indexing is stuck / a volume shows 'Indexing enabled' but never completes",
            "facts": "Ventura 14.4, Intel. Started after copying a large external drive. No heavy apps open.",
        },
        correct_fix="Spotlight is stuck reindexing — turn indexing off/on with `sudo mdutil -i off /` then `-i on /` (or add+remove the volume in Privacy) to rebuild the index.",
    ),
    Scenario(
        id="external_display_flicker",
        symptom="My external monitor keeps flickering and sometimes goes black for a second. It's driving me crazy.",
        macos_version="14", chip="apple_silicon",
        root_cause="A third-party USB-C hub between the Mac and the display can't sustain the bandwidth; connecting the display directly (or via the original cable) is stable.",
        system_state={
            "is it one display or all / how is it connected": "one external monitor, connected through a cheap USB-C hub; the built-in screen never flickers",
            "does it happen with a different cable/port": "when the user plugs the display straight into the Mac with its own cable, the flicker stops completely",
            "when did it start / what changed": "started when they bought the hub; the display was fine for a year before that",
            "facts": "Sonoma 14.6, M2 MacBook Pro. Built-in display is fine. Only the external screen flickers.",
        },
        correct_fix="The USB-C hub is the culprit — connect the display directly to the Mac (or use a certified/original cable or a better hub); the flicker stops without any software change.",
        tags=["ask_to_resolve"],   # no shell command reveals this — the agent must ASK
    ),
]

QUICK_IDS = ["wifi_dfs_channel", "battery_assertion", "disk_snapshots",
             "external_display_flicker"]


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------

SIM_USER_SYSTEM = """\
You are role-playing a NON-EXPERT macOS user in a troubleshooting chat. You know
only the surface symptom and basic facts about your Mac. You have a HIDDEN system
state (what commands would reveal). Rules:
- Reply briefly and casually, like a real user.
- If the assistant ASKS A QUESTION you can answer from the basic facts, answer it.
- If the assistant asks you to RUN A COMMAND, "run" it and paste ONLY the output
  that matches the hidden state (find the closest matching entry). If no entry
  matches, say the command returned nothing useful / you're not sure.
- NEVER volunteer the root cause or the fix. Only reveal what you'd actually see.
- If the assistant gives you a final fix, just acknowledge briefly.
Output ONLY the user's message.
"""


def sim_user_reply(sim_llm, scenario: Scenario, assistant_msg: str) -> str:
    state = "\n".join(f"- {k}: {v}" for k, v in scenario.system_state.items())
    prompt = (
        f"YOUR SYMPTOM (what you first said): {scenario.symptom}\n"
        f"BASIC FACTS: macOS {scenario.macos_version}, chip {scenario.chip}.\n"
        f"HIDDEN SYSTEM STATE (reveal only what matches what the assistant asks):\n{state}\n\n"
        f"ASSISTANT JUST SAID:\n{assistant_msg}\n\n"
        f"Your reply as the user:"
    )
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = sim_llm.invoke([SystemMessage(content=SIM_USER_SYSTEM), HumanMessage(content=prompt)])
    return str(resp.content).strip()


# ---------------------------------------------------------------------------
# Arm 1 — our agent (multi-turn, grounded)
# ---------------------------------------------------------------------------

def run_agent_arm(scenario: Scenario, sim_llm, max_turns: int) -> dict:
    from app.agent.graph import build_agent, make_memory_checkpointer, chat
    agent = build_agent(checkpointer=make_memory_checkpointer())
    thread = f"iact-{scenario.id}"
    transcript = [("user", scenario.symptom)]
    msg = scenario.symptom
    final = ""
    turns = 0
    for turns in range(1, max_turns + 1):
        out = chat(agent, msg, thread_id=thread)
        reply = out["answer"]
        transcript.append(("agent", f"[{out['action']}] {reply}"))
        if out["action"] == "answer":
            final = reply
            break
        # clarify / diagnose → the simulated user responds
        msg = sim_user_reply(sim_llm, scenario, reply)
        transcript.append(("user", msg))
    if not final:
        final = transcript[-1][1]  # ran out of turns; take last agent message
    return {"final": final, "turns": turns, "transcript": transcript}


# ---------------------------------------------------------------------------
# Arm 2 — raw GPT-4o (interactive, but no KB/tools)
# ---------------------------------------------------------------------------

GPT4O_SYSTEM = """\
You are an expert macOS troubleshooting assistant in a multi-turn chat. You may
ask ONE clarifying question or request ONE diagnostic command's output per turn if
it helps. When you are confident of the root cause and fix, give your FINAL answer
and prefix it with 'FIX:'. Be concise and specific.
"""


def run_gpt4o_arm(scenario: Scenario, gpt4o_llm, sim_llm, max_turns: int) -> dict:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    history = [SystemMessage(content=GPT4O_SYSTEM), HumanMessage(content=scenario.symptom)]
    transcript = [("user", scenario.symptom)]
    final = ""
    turns = 0
    for turns in range(1, max_turns + 1):
        reply = str(gpt4o_llm.invoke(history).content).strip()
        history.append(AIMessage(content=reply))
        transcript.append(("gpt4o", reply))
        if "FIX:" in reply.upper() or turns == max_turns:
            final = reply
            break
        user = sim_user_reply(sim_llm, scenario, reply)
        history.append(HumanMessage(content=user))
        transcript.append(("user", user))
    if not final:
        final = transcript[-1][1]
    return {"final": final, "turns": turns, "transcript": transcript}


# ---------------------------------------------------------------------------
# Judge — did the final diagnosis match the hidden ground truth?
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are grading a macOS troubleshooting conversation. You are given the HIDDEN true
root cause + correct fix, and an assistant's FINAL answer. Decide if the assistant
correctly identified the real cause AND gave a correct, actionable fix for it.

Reply with ONLY JSON: {"resolved": true|false, "reason": "<short>"}
Be strict: generic advice that doesn't reach the specific root cause is NOT resolved.
"""


def judge_resolution(judge_name: str, scenario: Scenario, final_answer: str) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    provider, model = _judge_spec(judge_name)
    llm = make_chat_model(provider=provider, model=model, temperature=0, timeout=60, max_retries=0)
    prompt = (
        f"TRUE ROOT CAUSE: {scenario.root_cause}\n"
        f"CORRECT FIX: {scenario.correct_fix}\n\n"
        f"ASSISTANT FINAL ANSWER:\n{final_answer}\n\n"
        f"Resolved? JSON only."
    )
    try:
        resp = llm.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)])
        raw = str(resp.content)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        return {"resolved": bool(data.get("resolved")), "reason": str(data.get("reason", ""))[:160]}
    except Exception as e:  # noqa: BLE001
        return {"resolved": False, "reason": f"judge error: {type(e).__name__}"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(quick: bool, max_turns: int, judges: list[str]):
    scenarios = [s for s in SCENARIOS if s.id in QUICK_IDS] if quick else SCENARIOS

    sim_llm   = make_chat_model(provider="groq", model="llama-3.1-8b-instant", temperature=0.3)
    gpt4o_llm = make_chat_model(provider="openai", model="gpt-4o", temperature=0)

    rows = []
    for i, sc in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {sc.id}")
        print("   agent arm …")
        agent_run = run_agent_arm(sc, sim_llm, max_turns)
        print(f"     agent finished in {agent_run['turns']} turns")
        print("   gpt4o arm …")
        gpt_run = run_gpt4o_arm(sc, gpt4o_llm, sim_llm, max_turns)
        print(f"     gpt4o finished in {gpt_run['turns']} turns")

        verdicts = {"agent": {}, "gpt4o": {}}
        for j in judges:
            verdicts["agent"][j] = judge_resolution(j, sc, agent_run["final"])
            verdicts["gpt4o"][j] = judge_resolution(j, sc, gpt_run["final"])

        rows.append({"scenario": asdict(sc), "agent": agent_run, "gpt4o": gpt_run,
                     "verdicts": verdicts})

    report = aggregate(rows, judges)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _save(REPORT, {"summary": report, "detail": rows})
    write_transcripts(rows)
    render(report, judges)
    return report


def _votes(verdict_map: dict) -> list[bool]:
    return [bool(v.get("resolved")) for v in verdict_map.values()]


def _resolved_strict(verdict_map: dict) -> bool:
    """ALL judges agree it's resolved (the punishing bar)."""
    votes = _votes(verdict_map)
    return bool(votes) and all(votes)


def _resolved_either(verdict_map: dict) -> bool:
    """At least ONE judge calls it resolved (the lenient bar)."""
    return any(_votes(verdict_map))


def aggregate(rows: list[dict], judges: list[str]) -> dict:
    out = {"n": len(rows), "agent": {}, "gpt4o": {}, "per_judge": {}}
    for arm in ("agent", "gpt4o"):
        strict = sum(_resolved_strict(r["verdicts"][arm]) for r in rows)
        either = sum(_resolved_either(r["verdicts"][arm]) for r in rows)
        turns = [r[arm]["turns"] for r in rows]
        out[arm] = {
            # strict (unanimous) is the headline; either-agree shows how much of
            # the gap is judge disagreement rather than a real quality gap.
            "resolution_rate": round(strict / len(rows), 3) if rows else 0.0,
            "resolved": strict,
            "either_rate": round(either / len(rows), 3) if rows else 0.0,
            "either_resolved": either,
            "avg_turns": round(sum(turns) / len(turns), 2) if turns else 0.0,
        }
    for j in judges:
        out["per_judge"][j] = {
            arm: sum(bool(r["verdicts"][arm][j].get("resolved")) for r in rows)
            for arm in ("agent", "gpt4o")
        }
    return out


def write_transcripts(rows: list[dict]):
    lines = ["# Interactive benchmark transcripts\n"]
    for r in rows:
        sc = r["scenario"]
        lines.append(f"\n## {sc['id']}\n\n**Symptom:** {sc['symptom']}\n\n"
                     f"**Hidden root cause:** {sc['root_cause']}\n")
        for arm in ("agent", "gpt4o"):
            lines.append(f"\n### {arm} ({r[arm]['turns']} turns)\n")
            for who, msg in r[arm]["transcript"]:
                lines.append(f"- **{who}:** {msg}\n")
    TRANSCRIPTS.write_text("\n".join(lines))


def render(report: dict, judges: list[str]):
    print("\n" + "=" * 66)
    print("  INTERACTIVE BENCHMARK — agent vs GPT-4o (multi-turn, hidden state)")
    print("=" * 66)
    print(f"\n  Scenarios: {report['n']}   Judges: {judges}\n")
    print(f"  {'arm':<8} {'strict (all agree)':>19} {'either judge':>13} {'avg turns':>11}")
    print("  " + "-" * 56)
    for arm in ("agent", "gpt4o"):
        a = report[arm]
        print(f"  {arm:<8} {a['resolution_rate']*100:>15.0f}% ({a['resolved']}) "
              f"{a.get('either_rate', 0)*100:>9.0f}% ({a.get('either_resolved', 0)}) "
              f"{a['avg_turns']:>10}")
    print("\n  per-judge resolved counts:")
    for j, d in report["per_judge"].items():
        print(f"    {j:<10} agent={d['agent']}  gpt4o={d['gpt4o']}")
    print("\n  Transcripts → results/interactive_transcripts.md\n")


def main():
    p = argparse.ArgumentParser(description="Multi-turn interactive agent-vs-frontier eval")
    p.add_argument("--quick", action="store_true", help="3 scenarios")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS_DEFAULT)
    p.add_argument("--judges", default="gptoss,qwen")
    args = p.parse_args()

    if not key_present("GROQ_API_KEY") or not key_present("OPENAI_API_KEY"):
        print("\n  ✗ Needs GROQ_API_KEY (agent + sim-user) and OPENAI_API_KEY (GPT-4o arm).\n")
        sys.exit(1)

    req_j = [j.strip() for j in args.judges.split(",") if j.strip()]
    judges = [j for j in req_j if j in JUDGE_REGISTRY and key_present(PROVIDERS[JUDGE_REGISTRY[j][0]][1])]
    if not judges:
        print("\n  ✗ No judges available (need Groq for gptoss/qwen).\n")
        sys.exit(1)

    run(quick=args.quick, max_turns=args.max_turns, judges=judges)


if __name__ == "__main__":
    main()
