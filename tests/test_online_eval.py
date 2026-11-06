"""Online evaluation: durable queue, async worker, retries, score attachment, evaluator versioning."""
from __future__ import annotations

import time

import pytest

from evaluation.eval_queue import DONE, FAILED, PENDING, EvalQueue
from evaluation.online_worker import OnlineEvaluator
from evaluation.ragas_backend import RagasBackend
from evaluation.scoring import Scorer

PAYLOAD = {
    "question": "What is the minimum password length?",
    "answer": "Passwords must be at least 14 characters long. [1]",
    "contexts": ["Passwords must be at least 14 characters long."],
    "abstained": False, "cited_numbers": [1],
}


def const(v):
    async def f(sample):
        return v
    return f


def make_worker(settings, queue, **scorers):
    scorers = {"faithfulness": const(0.9), "answer_relevancy": const(0.8), **scorers}
    return OnlineEvaluator(settings, queue, Scorer(settings, RagasBackend(settings, scorers=scorers)))


@pytest.fixture
def queue(tmp_path):
    return EvalQueue(str(tmp_path / "q.sqlite3"), max_attempts=3, retry_base_seconds=0)


# ---- queue -----------------------------------------------------------------------------------------
def test_enqueue_is_idempotent_per_trace(queue):
    assert queue.enqueue("t1", PAYLOAD)
    assert queue.enqueue("t1", PAYLOAD) is None
    assert queue.stats() == {PENDING: 1}


def test_claim_complete_cycle(queue):
    queue.enqueue("t1", PAYLOAD)
    job = queue.claim()
    assert job.trace_id == "t1" and job.attempts == 1 and queue.claim() is None
    queue.complete(job, evaluator_version="v-test", result={"faithfulness": 0.9})
    row = queue.get("t1")
    assert row["status"] == DONE and row["evaluator_version"] == "v-test"


def test_failure_backs_off_then_fails_permanently(tmp_path):
    q = EvalQueue(str(tmp_path / "b.sqlite3"), max_attempts=2, retry_base_seconds=3600)
    q.enqueue("t1", PAYLOAD)
    job = q.claim()
    assert q.fail(job, "boom") == PENDING
    assert q.claim() is None  # backoff: not due for an hour
    q2 = EvalQueue(str(tmp_path / "c.sqlite3"), max_attempts=2, retry_base_seconds=0)
    q2.enqueue("t2", PAYLOAD)
    q2.fail(q2.claim(), "e1")
    assert q2.fail(q2.claim(), "e2") == FAILED
    assert q2.get("t2")["last_error"] == "e2"


def test_stale_running_jobs_are_recovered(queue):
    queue.enqueue("t1", PAYLOAD)
    queue.claim()
    time.sleep(0.01)
    assert queue.requeue_stale(older_than_seconds=0) == 1
    assert queue.claim() is not None


# ---- worker ---------------------------------------------------------------------------------------------
def test_worker_attaches_scores_to_the_original_trace(settings, queue, exporter):
    queue.enqueue("trace-abc", PAYLOAD)
    worker = make_worker(settings, queue)
    assert worker.run(drain=True) == 1

    by_name = {s["name"]: s for s in exporter.scores}
    assert all(s["trace_id"] == "trace-abc" for s in exporter.scores)
    assert by_name["faithfulness"]["value"] == 0.9 and by_name["answer_relevancy"]["value"] == 0.8
    assert by_name["hallucination_rate"]["value"] == pytest.approx(0.1)
    assert by_name["citation_coverage"]["value"] == 1.0
    version = worker.scorer.version
    assert by_name["faithfulness"]["metadata"]["evaluator_version"] == version  # evaluator version tracked
    assert queue.get("trace-abc")["status"] == DONE and queue.get("trace-abc")["evaluator_version"] == version


def test_reference_dependent_metrics_are_reported_not_evaluated_never_scored(settings, queue, exporter):
    queue.enqueue("t1", PAYLOAD)
    make_worker(settings, queue).run(drain=True)
    names = {s["name"] for s in exporter.scores}
    assert not names & {"context_precision", "context_recall", "answer_correctness", "noise_sensitivity"}
    status = next(s for s in exporter.scores if s["name"] == "online_eval_status")
    assert status["value"] == "COMPLETED" and status["data_type"] == "CATEGORICAL"
    assert "context_precision" in status["metadata"]["not_evaluated"]
    assert "requires gold labels" in status["metadata"]["not_evaluated"]["context_precision"]


def test_transient_failure_is_retried_and_scores_are_idempotent(settings, queue, exporter):
    calls = {"n": 0}

    async def flaky(sample):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("judge timeout")
        return 0.7

    queue.enqueue("t1", PAYLOAD)
    worker = make_worker(settings, queue, faithfulness=flaky)
    assert worker.run(drain=True) == 2  # same job claimed twice (attempt 1 failed, attempt 2 ok)

    assert queue.get("t1")["status"] == DONE and queue.get("t1")["attempts"] == 2
    faith = [s for s in exporter.scores if s["name"] == "faithfulness"]
    assert len(faith) == 1 and faith[0]["value"] == 0.7
    statuses = [s for s in exporter.scores if s["name"] == "online_eval_status"]
    assert [s["value"] for s in statuses] == ["COMPLETED"]  # no terminal status published for the failed attempt
    # deterministic score ids: attempt 1 and 2 would upsert the same score, never duplicate
    answer_rel = [s for s in exporter.scores if s["name"] == "answer_relevancy"]
    assert len({s["score_id"] for s in answer_rel}) == 1 and len(answer_rel) == 2


def test_permanent_failure_publishes_partial_status(settings, tmp_path, exporter):
    async def always(sample):
        raise RuntimeError("judge down")

    q = EvalQueue(str(tmp_path / "f.sqlite3"), max_attempts=2, retry_base_seconds=0)
    q.enqueue("t1", PAYLOAD)
    make_worker(settings, q, faithfulness=always).run(drain=True)
    assert q.get("t1")["status"] == FAILED and "judge down" in q.get("t1")["last_error"]
    status = [s for s in exporter.scores if s["name"] == "online_eval_status"]
    assert len(status) == 1 and status[0]["value"] == "PARTIAL"  # other metrics succeeded
    assert not any(s["name"] == "faithfulness" for s in exporter.scores)  # failed metric is absent, not 0


def test_chat_request_only_enqueues_and_never_runs_the_evaluator(traced_pipeline, traced_settings, tmp_path, exporter):
    queue = EvalQueue(str(tmp_path / "chat.sqlite3"))
    traced_pipeline.eval_queue = queue
    traced_pipeline.settings = traced_settings.model_copy(update={"online_eval_enabled": True, "online_eval_sample_rate": 1.0})
    result = traced_pipeline.answer("What is the minimum password length?")
    job = queue.get(result.trace_id)
    assert job and job["status"] == PENDING  # waiting for the async worker
    assert not any(s["name"] in {"faithfulness", "answer_relevancy"} for s in exporter.scores)


def test_sampling_rate_zero_enqueues_nothing(traced_pipeline, traced_settings, tmp_path, exporter):
    queue = EvalQueue(str(tmp_path / "s.sqlite3"))
    traced_pipeline.eval_queue = queue
    traced_pipeline.settings = traced_settings.model_copy(update={"online_eval_sample_rate": 0.0})
    traced_pipeline.answer("What is the minimum password length?")
    assert queue.stats() == {}


def test_queued_payload_is_redacted(traced_pipeline, traced_settings, tmp_path, exporter):
    queue = EvalQueue(str(tmp_path / "r.sqlite3"))
    traced_pipeline.eval_queue = queue
    result = traced_pipeline.answer("My email is jane.doe@example.com - what is the minimum password length?")
    payload = queue.get(result.trace_id)["payload"]
    assert "jane.doe@example.com" not in payload and "[EMAIL]" in payload
