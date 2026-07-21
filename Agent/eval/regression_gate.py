"""
regression_gate.py — quality gate over the interactive benchmark (Phase 4).

The KB + depth-triage edge is this project's whole thesis, and prompt/graph
edits can silently erode it. This gate turns the existing interactive
benchmark into a pass/fail check against a committed baseline (baseline.json):

  1. agent resolution rate must not fall more than `tolerance` below baseline,
  2. agent must still beat (or tie) raw GPT-4o when `require_beats_gpt4o` is set.

Exit code is 0 on pass, 1 on regression — so it slots into CI or a pre-merge
hook as the gate the v2.0 plan calls for.

Usage:
  python regression_gate.py            # judge the CURRENT cached interactive_report.json
  python regression_gate.py --run      # regenerate the report first (costs API calls)
  python regression_gate.py --run --quick
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
BASELINE = _HERE / "baseline.json"
REPORT = _HERE / "results" / "interactive_report.json"


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"  ✗ missing {path.name} — run with --run to generate it.")
        sys.exit(2)
    return json.loads(path.read_text())


def evaluate(report: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Return (passed, lines) comparing a report summary to the baseline."""
    s = report["summary"]
    agent = s["agent"]["resolution_rate"]
    agent_either = s["agent"]["either_rate"]
    gpt4o = s["gpt4o"]["resolution_rate"]

    floor = baseline["agent_resolution_rate"] - baseline["tolerance"]
    lines = [
        f"  n scenarios            : {s['n']}  (baseline recorded n={baseline['n']})",
        f"  agent resolution rate  : {agent:.2f}  (baseline {baseline['agent_resolution_rate']:.2f}, floor {floor:.2f})",
        f"  agent either-judge rate: {agent_either:.2f}  (baseline {baseline['agent_either_rate']:.2f})",
        f"  gpt4o resolution rate  : {gpt4o:.2f}",
    ]

    passed = True
    if agent + 1e-9 < floor:
        passed = False
        lines.append(f"  ✗ agent resolution {agent:.2f} fell below floor {floor:.2f}")
    else:
        lines.append(f"  ✓ agent resolution within tolerance of baseline")

    if baseline.get("require_beats_gpt4o", True):
        if agent + 1e-9 < gpt4o:
            passed = False
            lines.append(f"  ✗ agent ({agent:.2f}) lost to raw GPT-4o ({gpt4o:.2f}) — core thesis broken")
        else:
            lines.append(f"  ✓ agent ≥ GPT-4o ({agent:.2f} ≥ {gpt4o:.2f})")

    return passed, lines


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive-benchmark regression gate")
    p.add_argument("--run", action="store_true",
                   help="regenerate interactive_report.json first (costs API calls)")
    p.add_argument("--quick", action="store_true", help="with --run: 3-scenario subset")
    p.add_argument("--judges", default="gptoss,qwen")
    args = p.parse_args()

    baseline = _load(BASELINE)

    if args.run:
        # Import lazily: the module pulls in the whole agent stack + provider keys.
        sys.path.insert(0, str(_HERE))
        from interactive_bench import MAX_TURNS_DEFAULT, run
        print("Regenerating interactive report …\n")
        run(quick=args.quick, max_turns=MAX_TURNS_DEFAULT,
            judges=[j.strip() for j in args.judges.split(",") if j.strip()])

    report = _load(REPORT)
    passed, lines = evaluate(report, baseline)

    print("\nRegression gate — interactive benchmark")
    print("\n".join(lines))
    print(f"\n{'PASS ✓' if passed else 'FAIL ✗'} — quality "
          f"{'held' if passed else 'REGRESSED'} vs baseline.\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
