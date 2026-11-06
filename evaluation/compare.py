"""Compare two experiment reports: metric deltas, regressions, and per-item flips.

    python -m evaluation.compare BASELINE.json CANDIDATE.json [--threshold 0.02] [--fail-on-regression]

Direction-aware: for lower-is-better metrics (noise_sensitivity, hallucination_rate,
unsupported_claim_rate) an increase counts as a regression.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def compare(base: dict, cand: dict, threshold: float = 0.02) -> dict:
    rows = []
    regressions = []
    for name in sorted(set(base["aggregates"]) | set(cand["aggregates"])):
        a, b = base["aggregates"].get(name), cand["aggregates"].get(name)
        am = a["mean"] if a else None
        bm = b["mean"] if b else None
        lower = bool((a or b)["lower_is_better"])
        delta: Optional[float] = None if am is None or bm is None else round(bm - am, 4)
        verdict = "n/a"
        if delta is not None:
            signed = -delta if lower else delta
            verdict = "improved" if signed > threshold else "REGRESSED" if signed < -threshold else "same"
            if verdict == "REGRESSED":
                regressions.append(name)
        rows.append({"metric": name, "baseline": am, "candidate": bm, "delta": delta, "verdict": verdict})

    base_items = {i["id"]: i for i in base["items"]}
    flips = []
    for it in cand["items"]:
        old = base_items.get(it["id"])
        if not old:
            continue
        for metric in ("faithfulness", "answer_correctness", "ref_context_hit_rate", "abstention_correctness"):
            o, n = old["scores"].get(metric), it["scores"].get(metric)
            if o is not None and n is not None and n < o - 0.2:
                flips.append({"id": it["id"], "question": it["question"], "metric": metric, "from": o, "to": n,
                              "trace_url": it.get("trace_url")})
    return {"rows": rows, "regressions": regressions, "flips": flips,
            "infra": {"baseline": base["infra"], "candidate": cand["infra"]},
            "config_diff": {k: (base["config"].get(k), cand["config"].get(k))
                            for k in sorted(set(base["config"]) | set(cand["config"]))
                            if base["config"].get(k) != cand["config"].get(k)}}


def render(base: dict, cand: dict, result: dict) -> str:
    out = [f"baseline : {base['config_name']}  ({base['run_name']})",
           f"candidate: {cand['config_name']}  ({cand['run_name']})",
           f"config changes: {result['config_diff'] or 'none'}"]
    if base["dataset"]["version"] != cand["dataset"]["version"]:
        out.append("WARNING: golden dataset versions differ - results are not directly comparable")
    if base["evaluator_version"] != cand["evaluator_version"]:
        out.append(f"WARNING: evaluator versions differ ({base['evaluator_version']} vs {cand['evaluator_version']})")
    out.append("")
    out.append(f"{'metric':28s} {'baseline':>9s} {'candidate':>9s} {'delta':>8s}  verdict")
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    for r in result["rows"]:
        d = "" if r["delta"] is None else f"{r['delta']:+.3f}"
        out.append(f"{r['metric']:28s} {fmt(r['baseline']):>9s} {fmt(r['candidate']):>9s} {d:>8s}  {r['verdict']}")
    ib, ic = result["infra"]["baseline"], result["infra"]["candidate"]
    out.append(f"\nlatency p50/p95 ms: {ib['latency_ms_p50']}/{ib['latency_ms_p95']} -> {ic['latency_ms_p50']}/{ic['latency_ms_p95']}")
    out.append(f"total cost usd    : {ib['total_cost_usd']} -> {ic['total_cost_usd']}")
    if result["flips"]:
        out.append(f"\nitems that got worse ({len(result['flips'])}):")
        for f in result["flips"][:15]:
            out.append(f"  {f['id']} {f['metric']}: {f['from']:.2f} -> {f['to']:.2f}  {f['question'][:60]}")
    out.append(f"\nregressions: {result['regressions'] or 'none'}")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--fail-on-regression", action="store_true")
    args = p.parse_args()
    base, cand = load(args.baseline), load(args.candidate)
    result = compare(base, cand, args.threshold)
    print(render(base, cand, result))
    if args.fail_on_regression and result["regressions"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
