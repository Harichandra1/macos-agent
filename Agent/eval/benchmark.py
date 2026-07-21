"""
benchmark.py — head-to-head "beat the frontier model" evaluation (HEADLINE).

Thesis under test: our RAG agent (Groq Llama-3.3-70B + macOS KB) beats a raw
frontier model on this narrow domain — because of retrieval grounding and depth
triage, not model size.

Contestants (answers generated per question):
  - rag_agent : our full pipeline (app.agent.graph) — retrieval + depth triage.
  - gpt4o     : raw GPT-4o, no RAG. The frontier baseline we claim to beat.
  - llama70b  : raw Llama-3.3-70B via Groq, no RAG. Ablation — same model our
                agent uses, minus the KB. Isolates RAG's contribution from
                model choice (the honest, interview-defensible comparison).

  All contestants get the SAME strong system prompt (including the depth-triage
  instruction), so any win is attributable to KB grounding, not prompt asymmetry.

Judges (independent — none is a contestant; all run on FREE tiers via the same
provider factory as the agent):
  - nvidia     : Qwen-2.5-72B on NVIDIA NIM      needs NVIDIA_API_KEY
  - openrouter : DeepSeek-V3 (free) on OpenRouter needs OPENROUTER_API_KEY
  - openai / xai / groq also selectable (see JUDGE_REGISTRY).

  Each question's answers are shown BLIND and in RANDOMIZED order (labels A/B/C).
  Judges score each answer on 4 rubric dimensions (0-5) and pick the best.
  We report each judge's win-rate for rag_agent vs each baseline, per-dimension
  means, and inter-judge agreement.

Everything is cached (answers + judgments) so reruns are ~free. Missing keys are
skipped gracefully; the report needs rag_agent + >=1 baseline + >=1 judge.

Usage:
  python benchmark.py --quick            # 8-question subset
  python benchmark.py                    # full set
  python benchmark.py --no-cache         # regenerate answers + re-judge
  python benchmark.py --contestants rag_agent,gpt4o --judges nvidia,openrouter
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

# --- make the serving package importable: Agent/backend is the package root ---
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "backend"))

from questions import BenchmarkQuestion, get_questions  # noqa: E402

load_dotenv()

RESULTS_DIR      = _HERE / "results"
ANSWERS_CACHE    = RESULTS_DIR / "answers_cache.json"
JUDGMENTS_CACHE  = RESULTS_DIR / "judgments_cache.json"
REPORT_JSON      = RESULTS_DIR / "benchmark_report.json"
TRANSCRIPTS_MD   = RESULTS_DIR / "transcripts.md"

# Same strong prompt for the raw baselines — fair fight. Only difference vs our
# agent is that the baselines have no retrieved KB context to ground in.
BASELINE_SYSTEM = """\
You are an expert macOS troubleshooting assistant with the depth of a senior
Apple support engineer. Diagnose the user's problem and give precise, correct,
actionable steps.

Rules:
- Depth triage: if the user says they already tried basic steps (restart, toggle
  off/on, update, reinstall), DO NOT repeat those — lead with intermediate or
  advanced fixes they have not tried.
- Prefer exact commands, file paths, and Settings locations over vague advice.
- Be concise: lead with the most likely fix, offer 2-3 ranked options max.
- Do not invent commands or macOS behaviors you are unsure about.
"""

CONTESTANTS = ("rag_agent", "gpt4o", "llama70b")
# Default judge panel: two independent FREE Groq models, neither a contestant.
JUDGES      = ("gptoss", "qwen")

DIMENSIONS = ("correctness", "actionability", "depth_triage", "grounding")

JUDGE_SYSTEM = """\
You are an impartial expert judge evaluating macOS troubleshooting answers.
You will see a user's problem, a grading rubric, and several candidate answers
labeled A, B, C. You do NOT know which system produced which answer.

Score EACH answer on these four dimensions, each 0-5:
  - correctness    : technically accurate; no wrong or dangerous advice.
  - actionability  : concrete commands / exact paths the user can follow.
  - depth_triage   : does NOT re-suggest steps the user already tried; pitched at
                     the right difficulty for this user.
  - grounding      : specific and well-supported; not vague or hallucinated.

Then pick the single best answer overall.

Respond with ONLY a JSON object, no prose:
{
  "scores": {
    "A": {"correctness": int, "actionability": int, "depth_triage": int, "grounding": int, "note": "short"},
    "B": {...},
    "C": {...}
  },
  "best": "A" | "B" | "C"
}
Only include labels that were actually shown.
"""


# ---------------------------------------------------------------------------
# Key / client detection
# ---------------------------------------------------------------------------

def _have(key: str) -> bool:
    v = os.environ.get(key, "")
    return bool(v) and v not in ("...", "changeme")


def available_contestants(requested: list[str]) -> list[str]:
    from app.agent.providers import required_key_envs
    out = []
    for c in requested:
        if c == "rag_agent":
            # embeddings (OpenAI) + Qdrant + whatever provider(s) the agent's
            # intake/synth roles resolve to (Groq by default, or NVIDIA/OpenRouter).
            need = ["OPENAI_API_KEY", "QDRANT_URL"] + required_key_envs(["intake", "synth"])
            if all(_have(k) for k in need):
                out.append(c)
        elif c == "gpt4o" and _have("OPENAI_API_KEY"):
            out.append(c)
        elif c == "llama70b" and _have("GROQ_API_KEY"):   # this contestant is Groq-specific
            out.append(c)
    return out


# Judge registry: name -> (provider, model). Judges go through the same
# provider factory as the agent, so they run on FREE tiers and are independent of
# the contestants (GPT-4o / Llama-70B). Override a judge's model with
# {NAME}_JUDGE_MODEL env (e.g. NVIDIA_JUDGE_MODEL=meta/llama-3.1-405b-instruct).
JUDGE_REGISTRY: dict[str, tuple[str, str]] = {
    # Default panel: two independent families on Groq (reliable free tier, no
    # rate-limit flakiness), neither a contestant (contestants = GPT-4o / Llama-70B):
    "gptoss": ("groq", "openai/gpt-oss-120b"),   # open-weight GPT-OSS
    "qwen":   ("groq", "qwen/qwen3-32b"),         # Qwen
    # Others available if their keys/models are set:
    "nvidia":     ("nvidia",     "qwen/qwen3-next-80b-a3b-instruct"),
    "openrouter": ("openrouter", "google/gemma-4-31b-it:free"),
    "openai":     ("openai",     "gpt-4o-mini"),
    "xai":        ("xai",        "grok-2-latest"),
}


def _judge_spec(name: str) -> tuple[str, str]:
    provider, model = JUDGE_REGISTRY[name]
    return provider, os.environ.get(f"{name.upper()}_JUDGE_MODEL", model)


def available_judges(requested: list[str]) -> list[str]:
    """Keep judges whose provider key is present."""
    from app.agent.providers import PROVIDERS
    out = []
    for j in requested:
        if j not in JUDGE_REGISTRY:
            continue
        provider = JUDGE_REGISTRY[j][0]
        if _have(PROVIDERS[provider][1]):
            out.append(j)
    return out


# ---------------------------------------------------------------------------
# Answer generation (contestants)
# ---------------------------------------------------------------------------

class Contestants:
    """Lazily-built clients so we only touch what's configured."""

    def __init__(self):
        self._openai = None
        self._groq = None
        self._rag_agent = None

    def _openai_client(self):
        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._openai

    def _groq_client(self):
        if self._groq is None:
            from groq import Groq
            self._groq = Groq(api_key=os.environ["GROQ_API_KEY"])
        return self._groq

    def _agent(self):
        if self._rag_agent is None:
            from app.agent.graph import build_agent
            self._rag_agent = build_agent()
        return self._rag_agent

    def answer(self, contestant: str, q: BenchmarkQuestion) -> str:
        if contestant == "rag_agent":
            return self._rag_answer(q)
        if contestant == "gpt4o":
            return self._chat_openai("gpt-4o", q.prompt)
        if contestant == "llama70b":
            return self._chat_groq("llama-3.3-70b-versatile", q.prompt)
        raise ValueError(contestant)

    def _rag_answer(self, q: BenchmarkQuestion) -> str:
        # Stateless single-shot (no checkpointer) — each benchmark question is
        # independent, so we reuse the agent's own fresh-state + answer helpers.
        from app.agent.graph import _fresh_state, _last_answer
        agent = self._agent()
        result = agent.invoke(_fresh_state(q.prompt))
        return _last_answer(result)

    def _chat_openai(self, model: str, prompt: str) -> str:
        resp = self._openai_client().chat.completions.create(
            model=model, temperature=0,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or "(no answer)"

    def _chat_groq(self, model: str, prompt: str) -> str:
        resp = self._groq_client().chat.completions.create(
            model=model, temperature=0,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or "(no answer)"


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

def _judge_prompt(q: BenchmarkQuestion, labeled: dict[str, str]) -> str:
    rubric = "\n".join(f"  - {r}" for r in q.rubric)
    answers = "\n\n".join(
        f"### Answer {label}\n{text}" for label, text in labeled.items()
    )
    return (
        f"USER PROBLEM:\n{q.prompt}\n\n"
        f"CONTEXT: macOS version = {q.macos_version or 'unspecified'}, "
        f"chip = {q.chip or 'unspecified'}.\n\n"
        f"GRADING RUBRIC (what a great answer contains):\n{rubric}\n\n"
        f"CANDIDATE ANSWERS:\n{answers}\n\n"
        f"Score each shown answer and pick the best. JSON only."
    )


def judge_with(judge: str, q: BenchmarkQuestion, labeled: dict[str, str]) -> dict:
    """Run one judge (via the provider factory) and parse its verdict, with a
    couple of retries for transient rate-limit / timeout errors."""
    import time as _time
    from app.agent.providers import make_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    provider, model = _judge_spec(judge)
    llm = make_chat_model(provider=provider, model=model, temperature=0,
                          timeout=60, max_retries=0)
    msgs = [SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=_judge_prompt(q, labeled))]
    last_err = None
    for attempt in range(3):
        try:
            resp = llm.invoke(msgs)
            return _parse_judgment(str(resp.content), set(labeled.keys()))
        except Exception as e:  # noqa: BLE001  (429 / timeout / 5xx)
            last_err = e
            _time.sleep(2 ** attempt * 3)   # 3s, 6s, 12s backoff
    return {"scores": {}, "best": None, "error": f"{type(last_err).__name__}: {last_err}"[:160]}


def _parse_judgment(raw: str, labels: set[str]) -> dict:
    """Best-effort JSON extraction; clamp scores; validate 'best'."""
    text = raw.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip() if text.count("```") >= 2 else text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"scores": {}, "best": None, "parse_error": True}

    scores = {}
    for label in labels:
        s = data.get("scores", {}).get(label, {}) or {}
        scores[label] = {d: max(0, min(5, int(s.get(d, 0) or 0))) for d in DIMENSIONS}
        scores[label]["note"] = str(s.get("note", ""))[:200]
    best = data.get("best")
    if best not in labels:
        # fall back to highest dimension-sum
        best = max(labels, key=lambda l: sum(scores[l][d] for d in DIMENSIONS)) if labels else None
    return {"scores": scores, "best": best, "parse_error": False}


def _overall(dim_scores: dict) -> int:
    return sum(dim_scores.get(d, 0) for d in DIMENSIONS)  # 0-20


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(quick: bool, use_cache: bool, contestants: list[str], judges: list[str],
        hard: bool = False):
    questions = get_questions(quick=quick, hard=hard)
    ans_cache = _load(ANSWERS_CACHE) if use_cache else {}
    jud_cache = _load(JUDGMENTS_CACHE) if use_cache else {}
    gen = Contestants()

    print(f"\nBenchmark: {len(questions)} questions | contestants={contestants} | judges={judges}\n")

    per_question = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:>2}/{len(questions)}] {q.id}")

        # 1. Generate / load answers
        answers = {}
        for c in contestants:
            key = f"{q.id}::{c}"
            if key in ans_cache:
                answers[c] = ans_cache[key]
            else:
                print(f"       generating {c} …")
                try:
                    answers[c] = gen.answer(c, q)
                except Exception as e:  # noqa: BLE001
                    answers[c] = f"(generation error: {e})"
                ans_cache[key] = answers[c]
                _save(ANSWERS_CACHE, ans_cache)

        # 2. Blind + randomized labeling (deterministic per question id)
        rng = random.Random(q.id)
        ordered = list(answers.items())
        rng.shuffle(ordered)
        letters = ["A", "B", "C", "D"][:len(ordered)]
        label_to_contestant = {letters[k]: ordered[k][0] for k in range(len(ordered))}
        labeled_text = {letters[k]: ordered[k][1] for k in range(len(ordered))}

        # 3. Judge
        q_judgments = {}
        for j in judges:
            jkey = f"{q.id}::{j}"
            if jkey in jud_cache:
                q_judgments[j] = jud_cache[jkey]
            else:
                print(f"       judging with {j} …")
                try:
                    verdict = judge_with(j, q, labeled_text)
                except Exception as e:  # noqa: BLE001
                    verdict = {"scores": {}, "best": None, "error": str(e)}
                # map letters back to contestant names for storage
                verdict["label_to_contestant"] = label_to_contestant
                q_judgments[j] = verdict
                jud_cache[jkey] = verdict
                _save(JUDGMENTS_CACHE, jud_cache)

        per_question.append({
            "question": asdict(q),
            "answers": answers,
            "label_to_contestant": label_to_contestant,
            "judgments": q_judgments,
        })

    report = aggregate(per_question, contestants, judges)
    _save(REPORT_JSON, {"summary": report, "detail": per_question})
    write_transcripts(per_question)
    render(report, contestants, judges)
    return report


def aggregate(per_question: list[dict], contestants: list[str], judges: list[str]) -> dict:
    baselines = [c for c in contestants if c != "rag_agent"]
    summary = {"judges": {}, "inter_judge_agreement": None}

    for j in judges:
        winrate = {b: {"win": 0, "tie": 0, "loss": 0, "n": 0} for b in baselines}
        dim_means = {c: {d: [] for d in DIMENSIONS} for c in contestants}
        best_counts = {c: 0 for c in contestants}

        for pq in per_question:
            verdict = pq["judgments"].get(j, {})
            scores = verdict.get("scores", {})
            l2c = verdict.get("label_to_contestant", {})
            if not scores:
                continue
            # per-contestant overall
            c_overall = {}
            for label, dims in scores.items():
                c = l2c.get(label)
                if not c:
                    continue
                c_overall[c] = _overall(dims)
                for d in DIMENSIONS:
                    dim_means[c][d].append(dims.get(d, 0))
            best_c = l2c.get(verdict.get("best"))
            if best_c in best_counts:
                best_counts[best_c] += 1
            if "rag_agent" in c_overall:
                for b in baselines:
                    if b not in c_overall:
                        continue
                    winrate[b]["n"] += 1
                    if c_overall["rag_agent"] > c_overall[b]:
                        winrate[b]["win"] += 1
                    elif c_overall["rag_agent"] == c_overall[b]:
                        winrate[b]["tie"] += 1
                    else:
                        winrate[b]["loss"] += 1

        summary["judges"][j] = {
            "winrate_vs": winrate,
            "dim_means": {c: {d: round(sum(v) / len(v), 2) if v else None
                              for d, v in dims.items()}
                          for c, dims in dim_means.items()},
            "best_counts": best_counts,
        }

    # inter-judge agreement: fraction of questions where both judges agree on best
    if len(judges) == 2:
        j1, j2 = judges
        agree = total = 0
        for pq in per_question:
            v1, v2 = pq["judgments"].get(j1, {}), pq["judgments"].get(j2, {})
            b1 = v1.get("label_to_contestant", {}).get(v1.get("best"))
            b2 = v2.get("label_to_contestant", {}).get(v2.get("best"))
            if b1 and b2:
                total += 1
                agree += int(b1 == b2)
        summary["inter_judge_agreement"] = round(agree / total, 3) if total else None
        summary["inter_judge_n"] = total

    return summary


def write_transcripts(per_question: list[dict]):
    lines = ["# Benchmark transcripts\n"]
    for pq in per_question:
        q = pq["question"]
        lines.append(f"\n## {q['id']}\n\n**Prompt:** {q['prompt']}\n")
        for c, text in pq["answers"].items():
            lines.append(f"\n<details><summary>{c}</summary>\n\n{text}\n\n</details>\n")
    TRANSCRIPTS_MD.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_MD.write_text("\n".join(lines))


def render(report: dict, contestants: list[str], judges: list[str]):
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD BENCHMARK — RAG agent vs frontier / ablation")
    print("=" * 70)
    for j in judges:
        jr = report["judges"].get(j)
        if not jr:
            continue
        print(f"\n  Judge: {j}")
        print("  " + "-" * 50)
        print("  rag_agent win-rate (overall score, 0-20):")
        for b, w in jr["winrate_vs"].items():
            n = w["n"] or 1
            print(f"    vs {b:<10} : {w['win']}/{w['n']} win "
                  f"({w['win']*100//n}%)  tie={w['tie']} loss={w['loss']}")
        print("  mean dimension scores (0-5):")
        print(f"    {'contestant':<12} " + " ".join(f"{d[:5]:>6}" for d in DIMENSIONS))
        for c in contestants:
            dm = jr["dim_means"].get(c, {})
            print(f"    {c:<12} " + " ".join(
                f"{(dm.get(d) if dm.get(d) is not None else 0):>6.2f}" for d in DIMENSIONS))
        print(f"  best-answer counts: {jr['best_counts']}")

    if report.get("inter_judge_agreement") is not None:
        print(f"\n  Inter-judge agreement on best answer: "
              f"{report['inter_judge_agreement']*100:.0f}% "
              f"(n={report.get('inter_judge_n')})")
    print("\n  Full detail → results/benchmark_report.json")
    print("  Transcripts → results/transcripts.md\n")


def main():
    p = argparse.ArgumentParser(description="Head-to-head RAG vs frontier benchmark")
    p.add_argument("--quick", action="store_true", help="smaller subset (6-8 questions)")
    p.add_argument("--hard", action="store_true",
                   help="run the HARD 'frontier-fails' set (exact-syntax + recent-version)")
    p.add_argument("--no-cache", action="store_true", help="regenerate answers + re-judge")
    p.add_argument("--contestants", default=",".join(CONTESTANTS),
                   help="comma list from: rag_agent,gpt4o,llama70b")
    p.add_argument("--judges", default=",".join(JUDGES),
                   help="comma list from: claude,grok")
    args = p.parse_args()

    req_c = [c.strip() for c in args.contestants.split(",") if c.strip()]
    req_j = [j.strip() for j in args.judges.split(",") if j.strip()]
    contestants = available_contestants(req_c)
    judges = available_judges(req_j)

    missing_c = [c for c in req_c if c not in contestants]
    missing_j = [j for j in req_j if j not in judges]
    if missing_c:
        print(f"  ⚠ skipping contestants (missing keys): {missing_c}")
    if missing_j:
        print(f"  ⚠ skipping judges (missing keys): {missing_j}")

    if "rag_agent" not in contestants:
        print("\n  ✗ rag_agent unavailable — needs GROQ_API_KEY + OPENAI_API_KEY + QDRANT_URL.")
        print("    Nothing to compare against our system. Add keys to .env and retry.\n")
        sys.exit(1)
    if len(contestants) < 2:
        print("\n  ✗ Need at least one baseline (gpt4o or llama70b) to compare against.\n")
        sys.exit(1)
    if not judges:
        print("\n  ✗ No judges available — set NVIDIA_API_KEY and/or OPENROUTER_API_KEY "
              "(or pick others via --judges).\n")
        sys.exit(1)

    run(quick=args.quick, use_cache=not args.no_cache,
        contestants=contestants, judges=judges, hard=args.hard)


if __name__ == "__main__":
    main()
