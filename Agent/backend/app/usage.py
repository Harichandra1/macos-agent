"""
usage.py — per-turn token & cost accounting at the provider boundary (Phase 3).

One UsageTracker is created per /chat turn and passed as a LangChain callback
in the graph config — LangGraph propagates it to EVERY LLM call the turn makes
(intake, filter, planner, synth, summary, fallbacks), so the accounting lives
where the provider calls happen without threading state through the graph.

Token counts come from, in order of preference:
  1. response.llm_output["token_usage"]  (non-streamed OpenAI-compatible calls)
  2. the AIMessage's usage_metadata      (streamed calls with stream_usage on)
  3. a chars/4 ESTIMATE from prompt+completion text, flagged `estimated` —
     free providers that omit usage on streams still get counted, honestly.

Pricing is per-model USD/1M tokens from the providers' published paid tiers.
Free-tier calls cost $0 in cash, but we bill them at list price on purpose:
the $0.25/month cap (v2.0 plan) is a consumption budget, and pricing at $0
would make it a no-op. Unknown models are priced as 70B-class — conservative,
never under-counted.
"""

from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

# model substring -> (input, output) USD per 1M tokens. Substring match so
# provider prefixes ("meta/llama-3.3-70b-instruct", ":free" suffixes) all hit.
PRICING_PER_MTOK: tuple[tuple[str, tuple[float, float]], ...] = (
    ("llama-3.3-70b", (0.59, 0.79)),        # Groq / NVIDIA / OpenRouter 70B
    ("llama-3.1-8b",  (0.05, 0.08)),        # Groq 8B-instant class
    ("gpt-4o-mini",   (0.15, 0.60)),        # OpenAI last-resort fallback
)
DEFAULT_PRICE = (0.59, 0.79)                # unknown model → 70B-class price

# Retrieval embeds the query with text-embedding-3-small OUTSIDE LangChain
# (raw OpenAI client in retrieval.py), invisible to callbacks. At $0.02/1M a
# query is ~$0.000002 — accounted as a flat per-turn estimate, not plumbing.
EMBEDDING_COST_ESTIMATE_USD = 0.00001

_CHARS_PER_TOKEN = 4   # coarse estimate for providers that report no usage


def price_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for needle, price in PRICING_PER_MTOK:
        if needle in m:
            return price
    return DEFAULT_PRICE


class UsageTracker(BaseCallbackHandler):
    """Accumulates tokens + cost across every LLM call of one chat turn."""

    # LangChain runs sync handlers inline on async runs; nothing here blocks.
    run_inline = True

    def __init__(self) -> None:
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.estimated = False              # any call fell back to chars/4?
        self._prompt_chars: dict[str, int] = {}   # run_id -> prompt size

    # -- callbacks ---------------------------------------------------------

    def on_llm_start(self, serialized: dict, prompts: list[str], *,
                     run_id: Any = None, **kwargs: Any) -> None:
        self._prompt_chars[str(run_id)] = sum(len(p) for p in prompts)

    def on_chat_model_start(self, serialized: dict, messages: list, *,
                            run_id: Any = None, **kwargs: Any) -> None:
        chars = sum(len(str(getattr(m, "content", m)))
                    for batch in messages for m in batch)
        self._prompt_chars[str(run_id)] = chars

    def on_llm_end(self, response: Any, *, run_id: Any = None,
                   **kwargs: Any) -> None:
        self.llm_calls += 1
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        model = llm_output.get("model_name") or ""

        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")

        completion_chars = 0
        for gens in getattr(response, "generations", []) or []:
            for gen in gens:
                msg = getattr(gen, "message", None)
                completion_chars += len(str(getattr(gen, "text", "") or
                                            getattr(msg, "content", "")))
                if in_tok is None and msg is not None:
                    meta = getattr(msg, "usage_metadata", None)
                    if meta:                     # streamed path
                        in_tok = meta.get("input_tokens")
                        out_tok = meta.get("output_tokens")
                if not model and msg is not None:
                    model = (getattr(msg, "response_metadata", None)
                             or {}).get("model_name", "")

        if in_tok is None or out_tok is None:    # estimate path
            in_tok = self._prompt_chars.get(str(run_id), 0) // _CHARS_PER_TOKEN
            out_tok = completion_chars // _CHARS_PER_TOKEN
            self.estimated = True
        self._prompt_chars.pop(str(run_id), None)

        in_price, out_price = price_for(model)
        self.input_tokens += int(in_tok)
        self.output_tokens += int(out_tok)
        self.cost_usd += (int(in_tok) * in_price + int(out_tok) * out_price) / 1e6

    def on_llm_error(self, error: BaseException, *, run_id: Any = None,
                     **kwargs: Any) -> None:
        # Failed calls still consumed input tokens upstream — estimate them.
        chars = self._prompt_chars.pop(str(run_id), 0)
        if chars:
            self.llm_calls += 1
            in_tok = chars // _CHARS_PER_TOKEN
            self.input_tokens += in_tok
            self.cost_usd += in_tok * DEFAULT_PRICE[0] / 1e6
            self.estimated = True

    # -- totals ------------------------------------------------------------

    def turn_cost_usd(self, retrieval_ran: bool = True) -> float:
        """Total turn cost including the flat embedding estimate."""
        return self.cost_usd + (EMBEDDING_COST_ESTIMATE_USD if retrieval_ran else 0.0)
