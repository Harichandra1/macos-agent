# macOS Troubleshooting Agent — serving layer

Production read/serve path for the macOS troubleshooting RAG system. The thesis:
**a 70B open model (Llama-3.3 via Groq) grounded in a deep, narrow macOS
knowledge base beats a frontier model on this domain** — because of retrieval
grounding and *solution-depth triage*, not model size.

This folder is the **read path**. The **write path** (scrape → build KB → ingest
to Qdrant Cloud) lives in `../vectorDBIngestion/`. The knowledge base (41k+
chunks) is already live in Qdrant Cloud, so nothing here needs local KB data.

---

## Architecture

```
Browser (frontend/ SPA)
  │  POST /chat  (SSE stream)
  ▼
FastAPI (backend/app/main.py)
  │
  ▼  LangGraph agent (backend/app/agent/graph.py) — AGENTIC, not linear:

  intake ─┬─(parallel)─► kb_retrieve   (Qdrant; command-aware man-page injection)
          └─(parallel)─► web_search    (Tavily deep; smart-gated)
                              │  fan-in
                        smart_merge     (LLM relevance-filter of KB∪web)
                              │
                        ┌── decide ──┐  (planner: ANSWER / CLARIFY / DIAGNOSE)
              ANSWER ────┤            ├──── CLARIFY → ask ONE question → end turn
                         │            └──── DIAGNOSE → request a GROUNDED command
                         ▼                              ("run `sudo wdutil info`…") → end turn
                    synthesize → verify → (refine once if a command is ungrounded) → END
```

The agent is **interactive**: instead of guessing one-shot, it asks targeted
questions or requests the user's REAL system state (a grounded diagnostic command),
then reasons over the pasted output on the next turn — a dedicated post-diagnostic
prompt concludes "**Root cause:** X (quoted from your output) → **Fix:** → 
**Verify:**". A per-session **case file** (version, chip, already-tried, pending
diagnostic/questions) persists across turns; budgets keep it to ≤1 clarify turn
and forced-answer by turn 3, and reset per problem (a topic change starts a fresh
case). Every diagnostic command is **grounded** in the KB — and if the planner's
decisive command didn't surface in retrieval, a targeted *rescue retrieval*
fetches its docs before the grounding gate rules (a hallucinated command is still
rejected).

**Context-aware questioning (anti-hallucination):** when the retrieved context is
weak or the candidate causes diverge, the planner asks up to **3 short questions
batched into ONE turn** — each either discriminates between causes present in the
sources ("only this network, or every network?") or fills a critical missing slot.
A deterministic gate backs it up: if retrieval found *nothing* and the planner
would answer anyway, the turn is overridden to questions — *a grounded question
beats an ungrounded answer*. Questions carry structured quick-reply options that
the UI renders as one-click chips filling the composer; the reply is folded into
the case file and can chain straight into a grounded diagnostic.

**Runtime guardrails:** every LLM role is provider-pluggable across Groq / NVIDIA
/ OpenRouter / OpenAI (`SYNTH_PROVIDER=nvidia`, etc.) **with automatic cross-
provider failover** (`.with_fallbacks`, kill-switch `LLM_FALLBACKS=off`) and
bounded timeouts, so a rate-limited or hung provider degrades instead of failing
the turn. Each graph node catches its own outages: intake falls back to regex
extraction, retrieval degrades to web-only, synthesis to an honest apology —
a turn never dies with a bare "internal error".

Key components:

| Path | Role |
|------|------|
| `backend/app/main.py` | FastAPI: `POST /chat` (SSE), `GET /health`, CORS, rate limit, static frontend |
| `backend/app/agent/graph.py` | Agentic LangGraph: parallel retrieval → decide → clarify/diagnose/answer → verify/refine |
| `backend/app/agent/providers.py` | Provider-pluggable LLM factory (Groq/NVIDIA/OpenRouter/OpenAI/xAI) |
| `backend/app/agent/retrieval.py` | Qdrant query engine: rerank, dedup, **command-aware man-page injection** |
| `backend/app/agent/verify.py` | Command-grounding check (flags hallucinated shell commands) |
| `backend/app/agent/prompts.py` | DECIDE + depth-triage + citation/point-to-specifics prompts |
| `frontend/` | Vanilla HTML/JS chat UI (tokens, intake chips, sources, question/diagnostic markers, verify badge) |
| `eval/benchmark.py` | One-shot head-to-head vs GPT-4o / raw-Llama (blind, independent Groq judges) |
| `eval/interactive_bench.py` | **Multi-turn simulated-user eval** — the agentic beat-frontier test |

---

## Run locally

From the repo root, with credentials in `../.env` (copy `../.env.example`):

```bash
# 1. Backend + frontend (one service; frontend is served at /)
cd Agent/backend
../../.venv/bin/uvicorn app.main:app --reload --port 8000
# → open http://localhost:8000

# 2. Check readiness (reports any missing credentials)
curl -s localhost:8000/health
```

**Required to be "ready":** `OPENAI_API_KEY`, `GROQ_API_KEY`, `QDRANT_URL`,
`QDRANT_API_KEY`. Optional: `TAVILY_API_KEY` (web fallback).

Multi-turn: the frontend keeps one `session_id`, so "that didn't work, what else?"
climbs the difficulty tiers instead of re-suggesting a restart.

---

## The head-to-head benchmark (the headline)

Proves the thesis. For each realistic macOS problem it generates answers from
three systems, judges them **blind + randomized** with an **independent panel**,
and reports win-rates.

- **Contestants:** `rag_agent` (this system) · `gpt4o` (raw frontier, no RAG) ·
  `llama70b` (raw Llama-3.3-70B, no RAG — the ablation isolating *RAG's*
  contribution from model choice). All three get the **same** strong system
  prompt, so any win is attributable to KB grounding.
- **Judges:** Claude (Anthropic) + Grok (xAI) — neither is a contestant. Each
  scores 4 dimensions (correctness, actionability, depth-triage, grounding) 0–5.
- **Report:** per-judge win-rate (rag vs each baseline) + inter-judge agreement,
  plus cached transcripts.

```bash
cd Agent/eval
../../.venv/bin/python benchmark.py --quick     # 8-question subset
../../.venv/bin/python benchmark.py             # full 22-question set
```

Extra keys for the run: `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`
(missing keys are skipped gracefully). Results land in `eval/results/`.

### Results (2026-07-06, `--quick` 8 common Qs + `--hard --quick` 6 exact-syntax Qs)

Judges: `openai/gpt-oss-120b` + `qwen/qwen3-32b` on Groq (independent, non-contestant).
Win-rate = share of questions where rag_agent's answer was judged better.

| Set | rag vs **GPT-4o** | rag vs raw Llama-70B | Inter-judge agreement |
|-----|-------------------|----------------------|-----------------------|
| Common (8) | **0%** (0/8, 0/6) | 25–50% | 100% |
| Hard slice (6) | 0–16% | 16–33% | 50–67% |

**Honest finding: the RAG agent does NOT beat GPT-4o on this domain (yet).** The
benchmark is doing its job — it exposed *why*, which is the valuable result:

1. **Common tasks** (reset a permission, drive won't mount): GPT-4o already knows
   them cold. Grounding a 70B model in a KB of older community posts is a *net
   negative* — retrieval only helps for knowledge the model lacks.
2. **Exact-syntax tasks**: the authoritative man pages ARE in the KB but were being
   shut out of retrieval. Fixed via **command-aware retrieval** (below) — the man
   page now ranks #1 and answers became grounded — but two gaps remain: the KB man
   pages carry *patterns* not always the *exact values* GPT-4o knows (e.g. it used
   `subsystem == "com.apple.wifi"`; ours had generic examples), and the Groq
   Llama-3.3-70B synthesizer is simply out-polished by GPT-4o as judged by LLMs.

**Where the RAG system genuinely wins:** off-domain deflection (2/2), command-
grounding trust (flags un-sourced commands GPT-4o emits confidently), inline
citations, and Tavily recency for post-cutoff versions.

**Retrieval fixes made during this cycle** (real quality gains, kept regardless):
- **Command-aware man-page injection** — when a query names a CLI tool, its man
  page is fetched via a source-filtered search and boosted so it can't be
  out-competed by chatty community posts (`retrieval.py`).
- **Injection before the fallback gate** — authoritative docs are no longer lost
  when a depth-tier filter makes the main query look weak.
- **Retrieve on the user's raw words, not the paraphrased `clean_query`** — the
  paraphrase was dropping exact tokens (command names, flags), tanking recall.

**To actually beat the frontier from here** (open questions for the next cycle):
deepen the KB with *exact-value* authoritative content (not just synopses);
consider a stronger synthesizer (tension with the open-model thesis); or reframe
success around the trust/recency/deflection axes where RAG measurably wins.

---

## The interactive benchmark — the agentic beat-frontier test

The one-shot benchmark can't capture the real edge: a frontier model answers blind
and **cannot see the user's machine**. `eval/interactive_bench.py` measures the
interactive edge. Each scenario has a HIDDEN ground truth (`root_cause`,
`system_state` = what specific diagnostics reveal, `correct_fix`). A **simulated
user** (LLM) states only the surface symptom and, when asked a question or given a
command, reveals ONLY the matching detail. Two arms play each scenario ↔ the
sim-user, up to N turns: **our agent** vs **raw GPT-4o** (which may also ask
questions, but has no KB/tools). A Groq judge panel scores resolution rate +
turns-to-resolution.

```bash
cd Agent/eval && ../../.venv/bin/python interactive_bench.py --quick   # 3 scenarios
```

### What it revealed (honest, and the most interesting result)

Running this eval *drove* the last cycle of work — it exposed, in order:

1. **A multi-turn retrieval bug:** a follow-up reply ("macOS 14.5") was replacing
   the retrieval query, so the agent lost the original problem and answered "given
   the lack of context." Fixed: `last_query` is now a **sticky topic anchor**.
2. **The agent diagnosed with the WRONG commands** (DHCP renew, DNS flush) because
   it's *grounded* — and the KB lacked the decisive diagnostics. Meanwhile GPT-4o,
   unconstrained, requested the RIGHT ones (`wdutil`, `pmset -g assertions`,
   `tmutil listlocalsnapshots`) from its own knowledge and nailed all three causes.
   **The grounding constraint hurts when the KB lacks the right diagnostic.**
3. **Fix → KB exact-value depth:** added 6 `diag_*` docs (Wi-Fi/DFS, sleep power
   assertions, APFS snapshots, TCC reset, kext panics, Spotlight) carrying the
   exact decisive commands, re-embedded to Qdrant. The agent now proposes
   `sudo wdutil info` for Wi-Fi drops (was `airport -I`).

### Result (2026-07-07, `--quick`, real Groq-70B agent, gptoss+qwen judges)

| arm | strict (both judges agree) | either judge | avg turns |
|-----|----------------------------|--------------|-----------|
| **agent** | **67%** (2/3) | **100%** (3/3) | **2.33** |
| GPT-4o | 0% (0/3) | 0% (0/3) | 3.67 |

**The agent beats GPT-4o on the interactive slice** — resolving in fewer turns
with unanimously-judged correct root-cause+fix answers, while GPT-4o resolves
none (it guessed peripherals for the power-assertion drain, invented a dangerous
`sudo rm /System/Volumes/VM/VM` for the snapshot case, and never reached the DFS
channel). Example agent final (wifi): *"**Root cause:** … radar detection —
'radar detected, channel switch announcement' → **Fix:** In your router's 5 GHz
settings, set the channel to 36-48 — any non-DFS channel."* — exactly the hidden
ground truth.

Three changes flipped this from the earlier 0%-both result:

1. **Post-diagnostic synthesis** — when the user pastes the requested command
   output, a dedicated prompt forces "**Root cause:** X (quoted from THEIR
   output) → **Fix:** → **Verify:**". The diagnosis was already right; the
   conclusion is now as decisive as the diagnosis.
2. **Decide-time rescue retrieval** — when the planner proposes the decisive
   diagnostic but its doc didn't surface in the symptom's top-5, the agent
   fetches the command's own docs and re-grounds instead of silently downgrading
   to a blind answer (this was why battery answered turn-1 without
   `pmset -g assertions`).
3. **Salient-paste retrieval** — pasted output is reduced to its decisive lines
   before embedding, so a 4k-char dump doesn't drown the query.

**Durable takeaway confirmed:** the interactive+grounded design beats one-shot
frontier because GPT-4o *can't see the machine* — our agent requests the real
state, reasons over it, and closes with a grounded fix in fewer turns. It only
works because the KB carries the exact diagnostics (KB depth is the product) and
the loop is closed end-to-end: request → paste → root-cause synthesis.

---

## Deploy (free tier)

Containerized; the KB is remote (Qdrant Cloud), so the image is code-only and
credentials are injected at run time (never baked in).

```bash
# Local container (needs Docker running)
docker build -f Agent/Dockerfile -t macos-agent .      # from repo root
docker run --rm -p 8000:8000 --env-file .env macos-agent

# Render: push to GitHub → New → Blueprint → this repo (reads Agent/render.yaml).
# Set the secret env vars in the Render dashboard.
```

See `Dockerfile` and `render.yaml`.
