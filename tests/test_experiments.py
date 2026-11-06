"""Offline experiment runner, sweeps and run comparison."""
from __future__ import annotations

import copy
import json

import pytest

from evaluation.compare import compare, render
from evaluation.experiment import (
    ExperimentConfig, expand_sweep, load_config, load_golden, run_experiment, save_report,
)
from evaluation.metrics import ALL_METRICS, REGISTRY
from evaluation.ragas_backend import RagasBackend
from evaluation.scoring import Scorer


def const(v):
    async def f(sample):
        return v
    return f


@pytest.fixture
def scorer(settings):
    ragas = {n: const(0.8) for n in ALL_METRICS if REGISTRY[n].kind == "ragas"} | {"answer_completeness": const(0.7)}
    return Scorer(settings, RagasBackend(settings, scorers=ragas))


def test_sweep_expands_cartesian_product_with_typed_values():
    base = ExperimentConfig(name="b")
    cfgs = expand_sweep(base, ["chunk_size=400,800", "top_k=3,5", "hybrid=false,true"])
    assert len(cfgs) == 8
    assert {c.chunk_size for c in cfgs} == {400, 800} and {c.hybrid for c in cfgs} == {True, False}
    assert len({c.name for c in cfgs}) == 8 and all(c.name.startswith("b+") for c in cfgs)
    with pytest.raises(ValueError):
        expand_sweep(base, ["nonsense=1"])
    assert expand_sweep(base, []) == [base]


def test_config_files_load_and_reject_unknown_keys(tmp_path):
    from evaluation.experiment import ROOT

    assert load_config(ROOT / "configs" / "baseline.json").name == "baseline"
    assert load_config(ROOT / "configs" / "hybrid_rerank.json").reranker == "lexical"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "chunksize": 5}))
    with pytest.raises(ValueError):
        load_config(bad)


def test_experiment_runs_end_to_end_and_reports_not_evaluated_separately(settings, scorer, tmp_path):
    items = load_golden()
    report = run_experiment(ExperimentConfig(name="t", hybrid=True), items, settings, scorer, max_concurrency=2)
    assert report["dataset"]["n_items"] == len(items) and report["dataset"]["version"]
    assert report["evaluator_version"].startswith("rag-eval/1|ragas=")

    agg = report["aggregates"]
    # unanswerable + vague items have no gold answer => reference metrics are NOT evaluated (not zero)
    n_unanswerable = sum(1 for i in items if i["type"] == "unanswerable")
    assert agg["answer_correctness"]["n_not_evaluated"] >= n_unanswerable
    assert agg["answer_correctness"]["n_evaluated"] + agg["answer_correctness"]["n_not_evaluated"] == len(items)
    assert agg["abstention_correctness"]["n_evaluated"] == len(items) - sum(i["type"] == "ambiguous" for i in items)
    assert 0 <= agg["ref_context_hit_rate"]["mean"] <= 1

    by_type = report["by_type"]
    assert {"factual", "exact_match", "multi_hop"} <= set(by_type)
    assert report["infra"]["latency_ms_p50"] is not None
    first = report["items"][0]
    assert {"id", "answer", "retrieved_chunk_ids", "scores", "not_evaluated", "failure_reasons"} <= set(first)

    path = save_report(report, tmp_path)
    assert json.loads(path.read_text())["run_name"] == report["run_name"]


def test_failures_list_flags_bad_items(settings, scorer):
    items = load_golden()[:20]
    report = run_experiment(ExperimentConfig(name="t", top_k=1), items, settings, scorer, max_concurrency=2)
    for f in report["failures"]:
        assert f["failure_reasons"]


def test_changing_top_k_changes_retrieval_not_the_dataset(settings, scorer):
    items = [i for i in load_golden() if i["type"] in {"factual", "exact_match"}]
    a = run_experiment(ExperimentConfig(name="k1", top_k=1), items, settings, scorer, metrics=["ref_context_hit_rate"])
    b = run_experiment(ExperimentConfig(name="k6", top_k=6), items, settings, scorer, metrics=["ref_context_hit_rate"])
    assert a["dataset"]["version"] == b["dataset"]["version"]
    assert b["aggregates"]["ref_context_hit_rate"]["mean"] >= a["aggregates"]["ref_context_hit_rate"]["mean"]


def _fake_report(**means):
    return {
        "run_name": "r", "config_name": "c", "config": {"top_k": 4}, "evaluator_version": "ev1",
        "dataset": {"version": "d1", "n_items": 1},
        "aggregates": {k: {"mean": v, "lower_is_better": k in {"hallucination_rate"}} for k, v in means.items()},
        "infra": {"latency_ms_p50": 1, "latency_ms_p95": 2, "total_cost_usd": 0.1},
        "items": [{"id": "q1", "question": "q", "scores": {"faithfulness": means.get("faithfulness", 1)}}],
    }


def test_compare_is_direction_aware_and_flags_regressions():
    base = _fake_report(faithfulness=0.90, hallucination_rate=0.10, context_recall=0.80)
    cand = copy.deepcopy(base)
    cand["aggregates"]["faithfulness"]["mean"] = 0.70      # worse
    cand["aggregates"]["hallucination_rate"]["mean"] = 0.30  # higher is worse for this metric
    cand["aggregates"]["context_recall"]["mean"] = 0.85    # better
    cand["config"] = {"top_k": 8}
    cand["items"][0]["scores"]["faithfulness"] = 0.2
    res = compare(base, cand, threshold=0.02)
    verdict = {r["metric"]: r["verdict"] for r in res["rows"]}
    assert verdict == {"faithfulness": "REGRESSED", "hallucination_rate": "REGRESSED", "context_recall": "improved"}
    assert set(res["regressions"]) == {"faithfulness", "hallucination_rate"}
    assert res["config_diff"] == {"top_k": (4, 8)}
    assert res["flips"] and res["flips"][0]["id"] == "q1"
    text = render(base, cand, res)
    assert "REGRESSED" in text and "regressions:" in text


def test_compare_warns_when_evaluator_or_dataset_differ():
    a, b = _fake_report(faithfulness=0.9), _fake_report(faithfulness=0.9)
    b["evaluator_version"], b["dataset"]["version"] = "ev2", "d2"
    text = render(a, b, compare(a, b))
    assert "evaluator versions differ" in text and "dataset versions differ" in text


def test_langfuse_experiment_path_links_traces(traced_settings, exporter):
    """With Langfuse configured the runner uses langfuse.run_experiment: every item gets
    its own rag-chat trace under the experiment's item span, and the trace id is reported."""
    ragas = {n: const(0.8) for n in ALL_METRICS if REGISTRY[n].kind == "ragas"} | {"answer_completeness": const(0.7)}
    scorer = Scorer(traced_settings, RagasBackend(traced_settings, scorers=ragas))
    items = load_golden()[:4]
    report = run_experiment(ExperimentConfig(name="lf"), items, traced_settings, scorer, max_concurrency=2)
    spans = exporter.get_finished_spans()
    assert sum(s.name == "rag-chat" for s in spans) == 4
    assert {"experiment-item-run", "experiment-item-evaluation"} <= {s.name for s in spans}
    assert all(i["trace_id"] and len(i["trace_id"]) == 32 for i in report["items"])
    assert report["aggregates"]["faithfulness"]["mean"] == 0.8
