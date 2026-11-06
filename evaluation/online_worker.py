"""Asynchronous online evaluation worker (separate process from the API).

    python -m evaluation.online_worker            # run forever
    python -m evaluation.online_worker --drain    # process everything due, then exit

For every queued production trace it:
  1. builds a Sample from the stored (question, answer, retrieved contexts),
  2. runs the reference-free metrics (Ragas faithfulness / answer_relevancy +
     deterministic citation and unsupported-claim checks),
  3. attaches one Langfuse score per evaluated metric to the ORIGINAL trace,
  4. attaches `online_eval_status` (COMPLETED | PARTIAL | FAILED) describing what
     was evaluated, not evaluated (and why), and which evaluator version ran.

Failures are retried with exponential backoff (EVAL_MAX_ATTEMPTS, EVAL_RETRY_BASE_SECONDS).
Score ids are deterministic, so retries upsert instead of duplicating.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import signal
import time
from typing import Any, Optional

from app.config import Settings, get_settings
from app.observability import create_score, flush, init_langfuse, shutdown_langfuse
from evaluation.eval_queue import EvalQueue, Job
from evaluation.metrics import ALL_METRICS, ERROR, EVALUATED, ONLINE_METRICS, ScoreResult, Sample
from evaluation.scoring import Scorer

logger = logging.getLogger("rag.eval.online")


def _score_id(trace_id: str, name: str, version: str) -> str:
    return hashlib.sha1(f"{trace_id}|{name}|{version}".encode()).hexdigest()[:32]


def sample_from_payload(p: dict[str, Any]) -> Sample:
    return Sample(
        question=p["question"], answer=p["answer"], contexts=p.get("contexts", []),
        abstained=p.get("abstained"), cited_numbers=p.get("cited_numbers", []),
    )


class OnlineEvaluator:
    def __init__(self, settings: Settings, queue: EvalQueue, scorer: Optional[Scorer] = None):
        self.settings = settings
        self.queue = queue
        self.scorer = scorer or Scorer(settings)
        self._stop = False

    # -- scoring + publishing -----------------------------------------------------
    async def evaluate(self, job: Job) -> list[ScoreResult]:
        sample = sample_from_payload(job.payload)
        return await self.scorer.score(sample, online=True)

    def publish(self, job: Job, results: list[ScoreResult], *, final: bool) -> str:
        version = self.scorer.version
        meta = {"evaluator_version": version, "attempt": job.attempts}
        for r in results:
            if r.status == EVALUATED and r.value is not None:
                create_score(
                    trace_id=job.trace_id, name=r.name, value=r.value, data_type="NUMERIC",
                    metadata={**meta, "evaluation_status": "evaluated"},
                    score_id=_score_id(job.trace_id, r.name, version),
                )
        evaluated = [r for r in results if r.status == EVALUATED]
        errors = [r for r in results if r.status == ERROR]
        not_eval = {r.name: r.reason for r in results if r.status == "not_evaluated"}
        not_eval.update({n: "requires gold labels (offline experiments only)" for n in ALL_METRICS if n not in ONLINE_METRICS})
        if errors and not evaluated:
            status = "FAILED"
        elif errors:
            status = "PARTIAL"
        else:
            status = "COMPLETED"
        if errors and not final:
            return status  # retry first; only publish a terminal status once
        comment = (
            f"evaluator={version}; evaluated={[r.name for r in evaluated]}; "
            f"errors={ {r.name: r.reason for r in errors} }; not_evaluated={not_eval}"
        )
        create_score(
            trace_id=job.trace_id, name="online_eval_status", value=status, data_type="CATEGORICAL",
            comment=comment, metadata={**meta, "not_evaluated": not_eval},
            score_id=_score_id(job.trace_id, "online_eval_status", version),
        )
        return status

    async def process(self, job: Job) -> str:
        results = await self.evaluate(job)
        errors = [r for r in results if r.status == ERROR]
        final = (not errors) or job.attempts >= job.max_attempts
        status = self.publish(job, results, final=final)
        flush()
        summary = {r.name: (r.value if r.status == EVALUATED else r.status) for r in results}
        if errors:
            msg = "; ".join(f"{r.name}: {r.reason}" for r in errors)
            outcome = self.queue.fail(job, msg)
            logger.warning("trace %s: %d metric error(s), job -> %s (%s)", job.trace_id, len(errors), outcome, msg)
        else:
            self.queue.complete(job, evaluator_version=self.scorer.version, result=summary)
            logger.info("trace %s evaluated: %s", job.trace_id, summary)
        return status

    # -- loop ----------------------------------------------------------------------------
    def run(self, *, drain: bool = False) -> int:
        """Process jobs until stopped. With drain=True, exit when nothing is due."""
        loop = asyncio.new_event_loop()  # one loop for the worker's lifetime (async HTTP clients are loop-bound)
        asyncio.set_event_loop(loop)
        processed = 0
        self.queue.requeue_stale()
        try:
            while not self._stop:
                job = self.queue.claim()
                if job is None:
                    if drain:
                        break
                    time.sleep(self.settings.eval_poll_interval_seconds)
                    continue
                try:
                    loop.run_until_complete(self.process(job))
                except Exception as exc:  # unexpected failure outside per-metric handling
                    logger.exception("job %s crashed", job.id)
                    self.queue.fail(job, f"{type(exc).__name__}: {exc}")
                processed += 1
        finally:
            loop.run_until_complete(self.scorer.backend.aclose())
            loop.close()
        return processed

    def stop(self, *_: Any) -> None:
        self._stop = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Online RAG evaluation worker")
    parser.add_argument("--drain", action="store_true", help="process due jobs then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="score jobs without Langfuse (scores are NOT attached anywhere); for local debugging only")
    args = parser.parse_args()
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not init_langfuse(s):
        if not args.dry_run:
            raise SystemExit(
                "Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST). "
                "Refusing to start: jobs would be consumed and their scores discarded. Use --dry-run to score locally."
            )
        logger.warning("--dry-run: Langfuse not configured; scores will NOT be attached to traces.")
    worker = OnlineEvaluator(s, EvalQueue(s.resolved_eval_db, s.eval_max_attempts, s.eval_retry_base_seconds))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    try:
        n = worker.run(drain=args.drain)
        logger.info("worker finished, processed=%d queue=%s", n, worker.queue.stats())
    finally:
        shutdown_langfuse()


if __name__ == "__main__":
    main()
