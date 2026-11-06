"""Scores one Sample against the metric registry.

Every metric ends in exactly one of three states:
  evaluated      - numeric value
  not_evaluated  - inputs unavailable / not applicable (reason recorded, NEVER 0)
  error          - the metric was attempted and failed (reason recorded)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Optional

from app.config import Settings
from evaluation.deterministic import DETERMINISTIC
from evaluation.metrics import ALL_METRICS, ONLINE_METRICS, REGISTRY, ScoreResult, Sample, applicability
from evaluation.ragas_backend import RagasBackend, evaluator_version

logger = logging.getLogger("rag.eval.scoring")


class Scorer:
    def __init__(self, settings: Settings, backend: Optional[RagasBackend] = None, concurrency: int = 4):
        self.settings = settings
        self.backend = backend or RagasBackend(settings)
        self._sem = asyncio.Semaphore(concurrency)

    @property
    def version(self) -> str:
        return evaluator_version(self.settings)

    async def _one(self, name: str, sample: Sample, faith: dict[str, ScoreResult]) -> ScoreResult:
        spec = REGISTRY[name]
        reason = applicability(spec, sample)
        if reason:
            return ScoreResult.skip(name, reason)
        try:
            if name in DETERMINISTIC:
                return DETERMINISTIC[name](sample)
            if name == "hallucination_rate":
                f = faith.get("faithfulness")
                if f is None or f.value is None:
                    return ScoreResult.skip(name, "faithfulness not available")
                return ScoreResult.ok(name, 1.0 - f.value)
            if not self.backend.judge_available:
                return ScoreResult.skip(name, "no judge LLM configured")
            async with self._sem:
                if name == "answer_completeness":
                    return ScoreResult.ok(name, await self.backend.completeness(sample))
                if not self.backend.has(name):
                    return ScoreResult.skip(name, "metric backend not available")
                return ScoreResult.ok(name, await self.backend.score(name, sample))
        except Exception as exc:  # a failed metric is an error, not a zero
            logger.warning("metric %s failed: %s: %s", name, type(exc).__name__, exc)
            return ScoreResult.error(name, f"{type(exc).__name__}: {exc}")

    async def score(self, sample: Sample, metrics: Optional[Iterable[str]] = None, *, online: bool = False) -> list[ScoreResult]:
        names = list(metrics) if metrics else list(ONLINE_METRICS if online else ALL_METRICS)
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}")
        # hallucination_rate derives from faithfulness: make sure it is scored first.
        base = [n for n in names if n != "hallucination_rate"]
        derived = [n for n in names if n == "hallucination_rate"]
        results = await asyncio.gather(*[self._one(n, sample, {}) for n in base])
        faith = {r.name: r for r in results}
        results += await asyncio.gather(*[self._one(n, sample, faith) for n in derived])
        order = {n: i for i, n in enumerate(names)}
        return sorted(results, key=lambda r: order[r.name])
