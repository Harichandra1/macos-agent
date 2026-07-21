"""
evaluate.py — RAG retrieval evaluation for the macOS KB.

Metrics computed:
  - MRR@K       (Mean Reciprocal Rank)
  - MAP@K       (Mean Average Precision)
  - DCG@K       (Discounted Cumulative Gain)
  - NDCG@K      (Normalized DCG)
  - Precision@K
  - Hit Rate@K
  - Fallback Rate
  - Mean Top-1 Score
  - Per-category breakdown

Relevance judgement method: LLM-as-judge (gpt-4o-mini).
  Each (query, chunk) pair is rated 0-3:
    0 = not relevant
    1 = tangentially relevant (mentions the domain but not the problem)
    2 = relevant (addresses the problem)
    3 = highly relevant (directly answers the question with specifics)

Usage:
  python evaluate.py             # full 25-query eval
  python evaluate.py --quick     # 10 queries, faster
  python evaluate.py --no-cache  # re-judge all (ignore cached grades)
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# --- make the serving package importable: Agent/backend is the package root ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.agent.retrieval import Retriever

load_dotenv()

# The relevance-judgement cache lives with the KB (ingestion side).
_KB_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "vectorDBIngestion" / "data" / "knowledge_base"
)

# ---------------------------------------------------------------------------
# Test query set — 25 queries across all categories, tiers, and edge cases
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # --- WiFi (5 queries) ---
    {
        "id": "wifi_01",
        "query": "wifi drops every few minutes MacBook",
        "category": "wifi",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "wifi_02",
        "query": "cannot connect to 5GHz network but 2.4GHz works fine Mac",
        "category": "wifi",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "wifi_03",
        "query": "wifi slow after upgrade to Ventura MacBook Pro",
        "category": "wifi",
        "expected_tier": 1,
        "macos_version": "13",
    },
    {
        "id": "wifi_04",
        "query": "preferred network order not saving macOS wifi settings",
        "category": "wifi",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "wifi_05",
        "query": "log show predicate wifi kernel Wi-Fi disconnect reason",
        "category": "wifi",
        "expected_tier": 3,
        "macos_version": None,
    },
    # --- Bluetooth (3 queries) ---
    {
        "id": "bt_01",
        "query": "AirPods keep disconnecting from Mac while iPhone nearby",
        "category": "bluetooth",
        "expected_tier": 1,
        "macos_version": None,
    },
    {
        "id": "bt_02",
        "query": "bluetooth audio stutter Mac M1 coreaudio",
        "category": "bluetooth",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "bt_03",
        "query": "delete bluetooth plist file reset module Mac terminal",
        "category": "bluetooth",
        "expected_tier": 2,
        "macos_version": None,
    },
    # --- Disk (3 queries) ---
    {
        "id": "disk_01",
        "query": "external drive not mounting Mac disk utility greyed out",
        "category": "disk",
        "expected_tier": 1,
        "macos_version": None,
    },
    {
        "id": "disk_02",
        "query": "diskutil verifyVolume APFS errors how to repair",
        "category": "disk",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "disk_03",
        "query": "Time Machine backup stuck preparing first backup forever",
        "category": "disk",
        "expected_tier": 1,
        "macos_version": None,
    },
    # --- Battery / Power (2 queries) ---
    {
        "id": "bat_01",
        "query": "MacBook battery draining fast on Sonoma M2",
        "category": "battery",
        "expected_tier": 1,
        "macos_version": "14",
    },
    {
        "id": "bat_02",
        "query": "pmset settings hibernatemode standby MacBook sleep power",
        "category": "battery",
        "expected_tier": 2,
        "macos_version": None,
    },
    # --- Performance (2 queries) ---
    {
        "id": "perf_01",
        "query": "Mac spinning beachball all apps freezing kernel_task high CPU",
        "category": "performance",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "perf_02",
        "query": "memory pressure red mac slow activity monitor swap",
        "category": "performance",
        "expected_tier": 2,
        "macos_version": None,
    },
    # --- Permissions (3 queries) ---
    {
        "id": "perm_01",
        "query": "microphone permission not showing in privacy settings Mac app",
        "category": "permissions",
        "expected_tier": 1,
        "macos_version": None,
    },
    {
        "id": "perm_02",
        "query": "tccutil reset Accessibility app permission mac terminal",
        "category": "permissions",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "perm_03",
        "query": "Gatekeeper codesign signature invalid macOS blocked app developer",
        "category": "permissions",
        "expected_tier": 3,
        "macos_version": None,
    },
    # --- System (2 queries) ---
    {
        "id": "sys_01",
        "query": "launchctl list service not starting on boot Mac",
        "category": "system",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "sys_02",
        "query": "defaults write NSGlobalDomain key value Mac terminal preferences",
        "category": "system",
        "expected_tier": 2,
        "macos_version": None,
    },
    # --- Diagnostics (2 queries) ---
    {
        "id": "diag_01",
        "query": "how to read kernel panic log file macOS crash report",
        "category": "diagnostics",
        "expected_tier": 2,
        "macos_version": None,
    },
    {
        "id": "diag_02",
        "query": "log show --predicate processImagePath contains kernel panic subsystem",
        "category": "diagnostics",
        "expected_tier": 3,
        "macos_version": None,
    },
    # --- Edge cases ---
    {
        "id": "edge_01",
        "query": "Mac running slow",  # very vague
        "category": "performance",
        "expected_tier": 1,
        "macos_version": None,
    },
    {
        "id": "edge_02",
        "query": "how do I install Python on Windows using pip",  # off-domain
        "category": None,
        "expected_tier": None,
        "macos_version": None,
    },
    {
        "id": "edge_03",
        "query": "Sequoia 15 bluetooth stability issue apple silicon MacBook Air M3",
        "category": "bluetooth",
        "expected_tier": 2,
        "macos_version": "15",
    },
]

QUICK_SUBSET = [
    "wifi_01", "wifi_03", "bt_01", "bt_03",
    "disk_01", "disk_02", "perf_01", "perm_02",
    "diag_02", "edge_02",
]

# ---------------------------------------------------------------------------
# LLM-as-judge: grade each (query, chunk) pair 0-3
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are a relevance judge for a macOS troubleshooting retrieval system.

Given a USER QUERY and a RETRIEVED CHUNK, rate relevance on this scale:
  0 = Not relevant. The chunk does not address the query's macOS problem.
  1 = Tangential. Mentions related technology but doesn't address the specific problem.
  2 = Relevant. Addresses the problem or closely related symptom with useful info.
  3 = Highly relevant. Directly answers the question with specific steps or explanation.

Reply with ONLY a single integer: 0, 1, 2, or 3.
No explanation. No punctuation. Just the digit.
"""

def judge_relevance(
    openai_client: OpenAI,
    query: str,
    chunk_title: str,
    chunk_text: str,
    cache: dict,
) -> int:
    cache_key = f"{query}|||{chunk_title[:80]}"
    if cache_key in cache:
        return cache[cache_key]

    prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"RETRIEVED CHUNK TITLE:\n{chunk_title}\n\n"
        f"RETRIEVED CHUNK TEXT:\n{chunk_text[:600]}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1,
            temperature=0,
        )
        grade = int(resp.choices[0].message.content.strip())
        grade = max(0, min(3, grade))
    except Exception:
        grade = 0

    cache[cache_key] = grade
    return grade


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def reciprocal_rank(grades: list[int], threshold: int = 2) -> float:
    """1/rank of the first result with grade >= threshold. 0 if none."""
    for rank, g in enumerate(grades, 1):
        if g >= threshold:
            return 1.0 / rank
    return 0.0


def average_precision(grades: list[int], threshold: int = 2) -> float:
    """AP@K: mean of precision-at-k for each position where a relevant doc appears."""
    hits = 0
    precision_sum = 0.0
    for rank, g in enumerate(grades, 1):
        if g >= threshold:
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return precision_sum / hits


def dcg(grades: list[int]) -> float:
    """DCG using graded relevance (0-3 scale)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades, 1))


def ideal_dcg(grades: list[int]) -> float:
    """IDCG: DCG of the ideal (sorted descending) ranking."""
    return dcg(sorted(grades, reverse=True))


def ndcg(grades: list[int]) -> float:
    """NDCG = DCG / IDCG. Returns 1.0 if IDCG == 0 (nothing to retrieve)."""
    idcg = ideal_dcg(grades)
    if idcg == 0:
        return 1.0  # no relevant docs exist — retrieval is vacuously perfect
    return dcg(grades) / idcg


def precision_at_k(grades: list[int], k: int, threshold: int = 2) -> float:
    return sum(1 for g in grades[:k] if g >= threshold) / k


def hit_rate_at_k(grades: list[int], k: int, threshold: int = 2) -> float:
    return 1.0 if any(g >= threshold for g in grades[:k]) else 0.0


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_evaluation(quick: bool = False, use_cache: bool = True,
                   use_category_filter: bool = False):
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retriever     = Retriever()

    queries = TEST_QUERIES
    if quick:
        queries = [q for q in TEST_QUERIES if q["id"] in QUICK_SUBSET]

    # Load / initialise relevance cache
    cache_file = _KB_DATA_DIR / "eval_cache.json"
    cache: dict = {}
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

    K_VALUES = [1, 3, 5]
    results  = []

    print(f"\nRunning evaluation on {len(queries)} queries (n=5 per query)…\n")

    for i, tq in enumerate(queries, 1):
        qid      = tq["id"]
        query    = tq["query"]
        version  = tq.get("macos_version")
        category = tq.get("category")

        print(f"[{i:>2}/{len(queries)}] {qid}: {query[:60]}…")

        result = retriever.retrieve(
            query=query,
            macos_version=version,
            category=(category if use_category_filter else None),
            n=5,
        )

        fallback = result["fallback"]
        hits     = result["hits"]

        # Judge each hit
        grades = []
        for hit in hits:
            g = judge_relevance(
                openai_client,
                query,
                hit.get("title", ""),
                hit.get("text", ""),
                cache,
            )
            grades.append(g)
            time.sleep(0.05)  # light rate-limit buffer

        # Pad to 5 with 0s if fewer hits returned
        grades = (grades + [0] * 5)[:5]
        scores = [h.get("score", 0) for h in hits] + [0.0] * (5 - len(hits))

        row = {
            "id":           qid,
            "query":        query,
            "category":     category,
            "expected_tier": tq.get("expected_tier"),
            "fallback":     fallback,
            "grades":       grades,
            "scores":       scores,
            "top1_score":   scores[0] if scores else 0.0,
            "top1_grade":   grades[0] if grades else 0,
            "actual_tiers": [h.get("difficulty_tier") for h in hits],
        }

        for k in K_VALUES:
            row[f"mrr@{k}"]   = reciprocal_rank(grades[:k])
            row[f"ap@{k}"]    = average_precision(grades[:k])
            row[f"dcg@{k}"]   = dcg(grades[:k])
            row[f"ndcg@{k}"]  = ndcg(grades[:k])
            row[f"p@{k}"]     = precision_at_k(grades, k)
            row[f"hit@{k}"]   = hit_rate_at_k(grades, k)

        results.append(row)
        print(
            f"         grades={grades}  "
            f"ndcg@5={row['ndcg@5']:.3f}  "
            f"hit@3={row['hit@3']:.0f}  "
            f"fallback={fallback}"
        )

    # Persist cache
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(results: list[dict]):
    K_VALUES   = [1, 3, 5]
    n          = len(results)
    off_domain = [r for r in results if r["category"] is None]
    on_domain  = [r for r in results if r["category"] is not None]
    fallback_r = [r for r in results if r["fallback"]]

    def mean(vals): return sum(vals) / len(vals) if vals else 0.0

    print("\n" + "=" * 70)
    print("  RAG RETRIEVAL EVALUATION REPORT")
    print("  macOS Troubleshooting Knowledge Base — Qdrant / text-embedding-3-small")
    print("=" * 70)
    print(f"\n  Queries evaluated : {n}")
    print(f"  On-domain         : {len(on_domain)}")
    print(f"  Off-domain        : {len(off_domain)}")
    print(f"  Fallback triggered: {len(fallback_r)} ({len(fallback_r)*100//n}%)")
    print()

    # --- Global metrics table ---
    print("─" * 70)
    print(f"  {'Metric':<18} {'@1':>8} {'@3':>8} {'@5':>8}")
    print("─" * 70)

    all_r = on_domain  # exclude off-domain from meaningful metrics

    for metric in ("mrr", "ap", "ndcg", "p", "hit"):
        label = {
            "mrr":  "MRR",
            "ap":   "MAP",
            "ndcg": "NDCG",
            "p":    "Precision",
            "hit":  "Hit Rate",
        }[metric]
        vals = {k: mean([r[f"{metric}@{k}"] for r in all_r]) for k in K_VALUES}
        print(
            f"  {label:<18} "
            f"{vals[1]:>8.3f} "
            f"{vals[3]:>8.3f} "
            f"{vals[5]:>8.3f}"
        )

    # Mean DCG (not normalized — raw scale for context)
    dcg_vals = {k: mean([r[f"dcg@{k}"] for r in all_r]) for k in K_VALUES}
    print(
        f"  {'DCG (mean)':<18} "
        f"{dcg_vals[1]:>8.3f} "
        f"{dcg_vals[3]:>8.3f} "
        f"{dcg_vals[5]:>8.3f}"
    )

    print("─" * 70)
    print(f"\n  Mean top-1 similarity score  : {mean([r['top1_score'] for r in all_r]):.4f}")
    print(f"  Fallback rate (on-domain)    : {len([r for r in on_domain if r['fallback']]) * 100 // max(len(on_domain),1)}%")
    off_fb = len([r for r in off_domain if r["fallback"]])
    print(f"  Fallback rate (off-domain)   : {off_fb}/{len(off_domain)} queries correctly deflected")

    # --- Grade distribution ---
    print("\n  Grade distribution (0–3 across all retrieved chunks):")
    all_grades = [g for r in all_r for g in r["grades"]]
    for grade in range(4):
        count = all_grades.count(grade)
        bar   = "█" * int(count * 30 / len(all_grades)) if all_grades else ""
        label = {0: "Not relevant", 1: "Tangential", 2: "Relevant", 3: "Highly relevant"}[grade]
        print(f"    {grade} — {label:<18} {count:>4} ({count*100//len(all_grades):>2}%)  {bar}")

    # --- Per-category breakdown (NDCG@5) ---
    print("\n  Per-category NDCG@5:")
    cats = sorted({r["category"] for r in all_r if r["category"]})
    for cat in cats:
        cat_results = [r for r in all_r if r["category"] == cat]
        score = mean([r["ndcg@5"] for r in cat_results])
        bar   = "█" * int(score * 20)
        print(f"    {cat:<16} {score:.3f}  {bar}  (n={len(cat_results)})")

    # --- Per-query detail ---
    print("\n  Per-query results:")
    print(f"  {'ID':<12} {'NDCG@5':>7} {'Hit@3':>6} {'MRR@5':>6} {'Top-1 score':>12} {'Grades':<20} {'FB'}")
    print("  " + "-" * 72)
    for r in results:
        print(
            f"  {r['id']:<12} "
            f"{r['ndcg@5']:>7.3f} "
            f"{r['hit@3']:>6.0f} "
            f"{r['mrr@5']:>6.3f} "
            f"{r['top1_score']:>12.4f} "
            f"{str(r['grades']):<20} "
            f"{'✓' if r['fallback'] else ''}"
        )

    # --- Tier distribution of returned hits ---
    print("\n  Difficulty tier distribution of retrieved hits (on-domain queries):")
    tier_counts = {1: 0, 2: 0, 3: 0}
    for r in all_r:
        for t in r["actual_tiers"]:
            if t in tier_counts:
                tier_counts[t] += 1
    total_hits = sum(tier_counts.values())
    for t, c in sorted(tier_counts.items()):
        bar = "█" * int(c * 30 / max(total_hits, 1))
        print(f"    Tier {t}: {c:>4} ({c*100//max(total_hits,1):>2}%)  {bar}")

    print("\n" + "=" * 70)

    # --- Interpretation ---
    ndcg5 = mean([r["ndcg@5"] for r in all_r])
    mrr5  = mean([r["mrr@5"]  for r in all_r])
    hit3  = mean([r["hit@3"]  for r in all_r])

    print("\n  INTERPRETATION")
    print("  " + "-" * 50)

    def rating(score, thresholds):
        # thresholds: list of (threshold, label)
        for t, label in sorted(thresholds, reverse=True):
            if score >= t:
                return label
        return thresholds[-1][1]

    ndcg_label = rating(ndcg5, [(0.85,"Excellent"),(0.70,"Good"),(0.55,"Moderate"),(0,"Needs work")])
    mrr_label  = rating(mrr5,  [(0.80,"Excellent"),(0.65,"Good"),(0.50,"Moderate"),(0,"Needs work")])
    hit_label  = rating(hit3,  [(0.90,"Excellent"),(0.75,"Good"),(0.60,"Moderate"),(0,"Needs work")])

    print(f"  NDCG@5  {ndcg5:.3f} — {ndcg_label}")
    print(f"  MRR@5   {mrr5:.3f} — {mrr_label}")
    print(f"  Hit@3   {hit3:.3f} — {hit_label}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--quick",    action="store_true", help="Run 10 queries instead of 25")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached LLM judgements")
    parser.add_argument("--filtered", action="store_true",
                        help="Apply the query's category as a metadata pre-filter "
                             "(exercises the agent path that the category fix improves).")
    parser.add_argument("--save", default=None,
                        help="Write aggregate metrics JSON to this path for before/after diffing.")
    args = parser.parse_args()

    mode = "CATEGORY-FILTERED" if args.filtered else "UNFILTERED"
    print(f"\n>>> Retrieval mode: {mode}")
    results = run_evaluation(
        quick=args.quick,
        use_cache=not args.no_cache,
        use_category_filter=args.filtered,
    )
    render_report(results)

    if args.save:
        on = [r for r in results if r["category"] is not None]
        def mean(vs): return sum(vs) / len(vs) if vs else 0.0
        summary = {
            "mode": mode,
            "n_on_domain": len(on),
            "ndcg@5": round(mean([r["ndcg@5"] for r in on]), 4),
            "mrr@5":  round(mean([r["mrr@5"]  for r in on]), 4),
            "map@5":  round(mean([r["ap@5"]   for r in on]), 4),
            "p@5":    round(mean([r["p@5"]    for r in on]), 4),
            "hit@3":  round(mean([r["hit@3"]  for r in on]), 4),
            "hit@5":  round(mean([r["hit@5"]  for r in on]), 4),
            "fallbacks": sum(1 for r in results if r["fallback"]),
        }
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved aggregate metrics → {args.save}")


if __name__ == "__main__":
    main()
