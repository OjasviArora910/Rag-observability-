"""Offline evaluation: run the golden dataset through a RAG configuration,
score every answer, and (when Langfuse is configured) record the run as a
Langfuse experiment so configurations can be compared in the UI.

    # baseline
    python -m evaluation.experiment --config evaluation/configs/baseline.json
    # sweep chunk size x top-k (each combination is its own run)
    python -m evaluation.experiment --config evaluation/configs/baseline.json \\
        --sweep chunk_size=400,800 --sweep top_k=3,5
    # compare two runs
    python -m evaluation.compare evaluation/results/A.json evaluation/results/B.json

The same code path runs with and without Langfuse; results are always written
to evaluation/results/<run>.json so runs are comparable offline and in CI.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import Settings, get_settings
from app.embeddings import get_embedder
from app.ingestion import ingest_bytes
from app.llm import get_chat
from app.observability import flush, get_langfuse, init_langfuse, shutdown_langfuse, tracing_enabled
from app.rag import RAGPipeline
from app.retriever import Retriever
from app.vectorstore import VectorStore, make_client
from evaluation.metrics import ALL_METRICS, EVALUATED, REGISTRY, Sample
from evaluation.scoring import Scorer

logger = logging.getLogger("rag.eval.experiment")

ROOT = Path(__file__).resolve().parent
GOLDEN_PATH = ROOT / "golden" / "golden_dataset.jsonl"
CORPUS_DIR = ROOT / "golden" / "corpus"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baseline"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    hybrid: bool = False
    reranker: str = "none"
    prompt_version: str = "v2"
    temperature: float = 0.0
    candidate_multiplier: int = 3

    def knobs(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("name")
        return d


def load_config(path: Path) -> ExperimentConfig:
    data = json.loads(Path(path).read_text())
    allowed = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
    return ExperimentConfig(**data)


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def expand_sweep(base: ExperimentConfig, sweeps: list[str]) -> list[ExperimentConfig]:
    """['chunk_size=400,800', 'top_k=3,5'] -> 4 configs (cartesian product)."""
    if not sweeps:
        return [base]
    axes: list[tuple[str, list[Any]]] = []
    for s in sweeps:
        key, _, values = s.partition("=")
        if key not in ExperimentConfig.__dataclass_fields__ or key == "name":
            raise ValueError(f"Cannot sweep '{key}'")
        axes.append((key, [_coerce(v) for v in values.split(",") if v != ""]))
    out = []
    for combo in itertools.product(*[vals for _, vals in axes]):
        overrides = {k: v for (k, _), v in zip(axes, combo)}
        label = "+".join(f"{k}={v}" for k, v in overrides.items())
        out.append(replace(base, name=f"{base.name}+{label}", **overrides))
    return out


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_version(path: Path = GOLDEN_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def build_pipeline(settings: Settings, cfg: ExperimentConfig, corpus_dir: Path = CORPUS_DIR) -> RAGPipeline:
    """Fresh in-memory index per configuration (chunking/embeddings change the index)."""
    embedder = get_embedder(settings, cfg.embedding_model)
    store = VectorStore(make_client(settings, in_memory=True), f"exp_{cfg.name[:40]}", embedder.dim)
    for f in sorted(Path(corpus_dir).glob("*")):
        if f.suffix.lower() in {".md", ".txt", ".pdf"}:
            ingest_bytes(store=store, embedder=embedder, settings=settings, filename=f.name, data=f.read_bytes(),
                         chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, traced=False)
    retriever = Retriever(store, embedder, cost_fn=settings.estimate_cost)
    return RAGPipeline(settings, retriever, get_chat(settings), eval_queue=None, overrides=cfg.knobs())


# ---------------------------------------------------------------------------------
def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _sample(output: dict, row: dict) -> Sample:
    answerable = bool(row.get("answerable", True))
    return Sample(
        question=row["question"], answer=output["answer"], contexts=output["contexts"],
        reference=row.get("reference_answer") if answerable else None,
        reference_contexts=row.get("reference_contexts") or None,
        answerable=answerable, expected_behavior=row.get("expected_behavior"),
        abstained=output["abstained"], cited_numbers=output.get("cited_numbers", []),
    )


def _failure_reasons(scores: dict[str, Optional[float]]) -> list[str]:
    checks = [
        ("abstention_correctness", lambda v: v < 1, "wrong abstain/answer decision"),
        ("faithfulness", lambda v: v < 0.7, "low faithfulness"),
        ("answer_correctness", lambda v: v < 0.5, "low answer correctness"),
        ("ref_context_hit_rate", lambda v: v < 1, "gold passage not retrieved"),
        ("citation_correctness", lambda v: v < 0.5, "citations not supported"),
    ]
    return [msg for name, bad, msg in checks if scores.get(name) is not None and bad(scores[name])]


def _mean(vals: list[float]) -> Optional[float]:
    return round(statistics.fmean(vals), 4) if vals else None


def _pct(vals: list[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 1)


def run_experiment(
    cfg: ExperimentConfig,
    items: list[dict],
    settings: Settings,
    scorer: Scorer,
    *,
    metrics: Optional[list[str]] = None,
    max_concurrency: int = 4,
    dataset_name: Optional[str] = None,
    corpus_dir: Path = CORPUS_DIR,
    golden_path: Path = GOLDEN_PATH,
) -> dict:
    metric_names = metrics or ALL_METRICS
    pipeline = build_pipeline(settings, cfg, corpus_dir)
    run_name = f"{cfg.name}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    outputs: dict[str, dict] = {}
    evals: dict[str, list] = {}

    def task(*, item: Any, **_: Any) -> dict:
        row = _get(item, "metadata") or {}
        row = {**row, "question": _get(item, "input") if isinstance(_get(item, "input"), str) else row.get("question")}
        res = pipeline.answer(
            row["question"], tags=["experiment", f"experiment:{cfg.name}"],
            extra_metadata={"experiment": cfg.name, "golden_id": row.get("id", ""), "question_type": row.get("type", "")},
            enqueue_eval=False,
        )
        out = {
            "answer": res.answer, "contexts": [c.text for c in res.contexts],
            "chunk_ids": [c.chunk_id for c in res.contexts], "abstained": res.abstained,
            "cited_numbers": res.cited_numbers, "trace_id": res.trace_id, "trace_url": res.trace_url,
            "latency_ms": round(res.latency_ms, 1), "usage": res.usage,
            "cost": None if res.cost is None else res.cost["total"],
        }
        outputs[row["id"]] = out
        return out

    async def evaluator(*, input: Any, output: dict, expected_output: Any, metadata: Optional[dict], **_: Any):
        from langfuse import Evaluation

        row = {**(metadata or {}), "question": input if isinstance(input, str) else (metadata or {}).get("question")}
        results = await scorer.score(_sample(output, row), metric_names)
        evals[row["id"]] = results
        return [
            Evaluation(name=r.name, value=r.value, metadata={"evaluator_version": scorer.version, "config": cfg.name})
            for r in results if r.status == EVALUATED and r.value is not None
        ]

    started = time.perf_counter()
    experiment_url = None
    lf = get_langfuse() if tracing_enabled() else None
    if lf is not None:
        def run_level(*, item_results: list, **_: Any):
            from langfuse import Evaluation

            agg: dict[str, list[float]] = {}
            for ir in item_results:
                for ev in ir.evaluations:
                    if isinstance(ev.value, (int, float)):
                        agg.setdefault(ev.name, []).append(float(ev.value))
            return [Evaluation(name=f"mean_{k}", value=statistics.fmean(v)) for k, v in agg.items() if v]

        kwargs = dict(
            name="rag-golden-eval", run_name=run_name,
            description=f"RAG config {cfg.name}", task=task, evaluators=[evaluator], run_evaluators=[run_level],
            max_concurrency=max_concurrency, metadata={k: str(v) for k, v in cfg.knobs().items()},
        )
        if dataset_name:
            result = lf.get_dataset(dataset_name).run_experiment(**kwargs)
        else:
            data = [{"input": r["question"], "expected_output": r.get("reference_answer"), "metadata": r} for r in items]
            result = lf.run_experiment(data=data, **kwargs)
        experiment_url = getattr(result, "experiment_url", None)
        trace_by_item = {}
        for ir in result.item_results:
            rid = (_get(ir.item, "metadata") or {}).get("id")
            if rid:
                trace_by_item[rid] = ir.trace_id
        for rid, tid in trace_by_item.items():
            if rid in outputs and tid:
                outputs[rid]["trace_id"] = tid
        flush()
    else:
        async def evaluate_all():
            sem = asyncio.Semaphore(max_concurrency)

            async def one(row: dict):
                async with sem:
                    out = outputs[row["id"]]
                    await evaluator(input=row["question"], output=out, expected_output=row.get("reference_answer"), metadata=row)

            await asyncio.gather(*[one(r) for r in items if r["id"] in outputs])

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = [pool.submit(task, item={"input": r["question"], "metadata": r}) for r in items]
            for f in futures:
                f.result()
        asyncio.run(evaluate_all())
    duration = time.perf_counter() - started

    return _report(cfg, run_name, items, outputs, evals, scorer, settings, duration, experiment_url, golden_path)


def _report(cfg, run_name, items, outputs, evals, scorer, settings, duration, experiment_url, golden_path) -> dict:
    rows = []
    per_metric: dict[str, dict[str, Any]] = {}
    by_type: dict[str, dict[str, list[float]]] = {}
    for it in items:
        rid = it["id"]
        if rid not in outputs:
            continue
        results = evals.get(rid, [])
        scores = {r.name: r.value for r in results if r.status == EVALUATED}
        for r in results:
            m = per_metric.setdefault(r.name, {"values": [], "not_evaluated": 0, "errors": 0, "reasons": {}})
            if r.status == EVALUATED and r.value is not None:
                m["values"].append(r.value)
                by_type.setdefault(it["type"], {}).setdefault(r.name, []).append(r.value)
            elif r.status == "error":
                m["errors"] += 1
                m["reasons"][r.reason or "error"] = m["reasons"].get(r.reason or "error", 0) + 1
            else:
                m["not_evaluated"] += 1
        out = outputs[rid]
        rows.append({
            "id": rid, "type": it["type"], "question": it["question"], "reference_answer": it.get("reference_answer"),
            "answer": out["answer"], "abstained": out["abstained"], "retrieved_chunk_ids": out["chunk_ids"],
            "trace_id": out.get("trace_id"), "trace_url": out.get("trace_url"), "latency_ms": out["latency_ms"],
            "cost_usd": out.get("cost"),
            "scores": scores,
            "not_evaluated": {r.name: r.reason for r in results if r.status == "not_evaluated"},
            "errors": {r.name: r.reason for r in results if r.status == "error"},
            "failure_reasons": _failure_reasons(scores),
        })
    aggregates = {
        name: {"mean": _mean(m["values"]), "n_evaluated": len(m["values"]), "n_not_evaluated": m["not_evaluated"],
               "n_errors": m["errors"], "error_reasons": m["reasons"],
               "lower_is_better": REGISTRY[name].lower_is_better}
        for name, m in per_metric.items()
    }
    lat = [r["latency_ms"] for r in rows]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    return {
        "run_name": run_name, "config_name": cfg.name, "config": cfg.knobs(),
        "provider_mode": settings.provider_mode, "evaluator_version": scorer.version,
        "dataset": {"path": str(golden_path.name), "version": dataset_version(golden_path), "n_items": len(rows)},
        "created_at": datetime.now(timezone.utc).isoformat(), "duration_s": round(duration, 1),
        "experiment_url": experiment_url,
        "aggregates": aggregates,
        "by_type": {t: {k: _mean(v) for k, v in ms.items()} for t, ms in by_type.items()},
        "infra": {"latency_ms_p50": _pct(lat, 0.5), "latency_ms_p95": _pct(lat, 0.95),
                  "total_cost_usd": round(sum(costs), 6) if costs else None},
        "failures": [r for r in rows if r["failure_reasons"]],
        "items": rows,
    }


def save_report(report: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe = report["run_name"].replace("/", "_").replace("=", "-").replace("+", "_")
    path = results_dir / f"{safe}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def print_summary(report: dict) -> None:
    print(f"\n== {report['config_name']}  ({report['dataset']['n_items']} items, provider={report['provider_mode']}) ==")
    print(f"{'metric':28s} {'mean':>7s}  evaluated / not-evaluated / errors")
    for name, a in report["aggregates"].items():
        mean = "n/a" if a["mean"] is None else f"{a['mean']:.3f}"
        print(f"{name:28s} {mean:>7s}  {a['n_evaluated']:>3d} / {a['n_not_evaluated']:>3d} / {a['n_errors']:>3d}")
    i = report["infra"]
    print(f"latency p50/p95 ms: {i['latency_ms_p50']}/{i['latency_ms_p95']}  cost: {i['total_cost_usd']}  failures: {len(report['failures'])}")
    if report.get("experiment_url"):
        print("Langfuse experiment:", report["experiment_url"])


def main() -> None:
    p = argparse.ArgumentParser(description="Run the golden dataset against a RAG configuration")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "baseline.json")
    p.add_argument("--sweep", action="append", default=[], help="key=v1,v2 (repeatable; cartesian product)")
    p.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    p.add_argument("--types", help="comma-separated question types to include")
    p.add_argument("--limit", type=int)
    p.add_argument("--metrics", help="comma-separated subset of metrics")
    p.add_argument("--provider", choices=["openai", "groq", "fake"], help="override LLM_PROVIDER/PROVIDER_MODE")
    p.add_argument("--dataset-name", help="run against a Langfuse dataset (see scripts/sync_golden_dataset.py)")
    p.add_argument("--no-langfuse", action="store_true")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    s = get_settings()
    if args.provider:
        if args.provider == "fake":
            s = s.model_copy(update={"provider_mode": "fake"})
        else:
            s = s.model_copy(update={"provider_mode": "openai", "llm_provider": args.provider})
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not args.no_langfuse:
        init_langfuse(s)

    items = load_golden(args.golden)
    if args.types:
        keep = set(args.types.split(","))
        items = [i for i in items if i["type"] in keep]
    if args.limit:
        items = items[: args.limit]
    metrics = args.metrics.split(",") if args.metrics else None

    scorer = Scorer(s)
    try:
        for cfg in expand_sweep(load_config(args.config), args.sweep):
            report = run_experiment(cfg, items, s, scorer, metrics=metrics, max_concurrency=args.concurrency,
                                    dataset_name=args.dataset_name, golden_path=args.golden)
            path = save_report(report)
            print_summary(report)
            print("saved:", path)
    finally:
        asyncio.run(scorer.backend.aclose())
        shutdown_langfuse()


if __name__ == "__main__":
    main()
