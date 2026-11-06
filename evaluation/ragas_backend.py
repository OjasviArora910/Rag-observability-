"""Ragas 0.4 (`ragas.metrics.collections`) wiring.

Metrics are built once from an OpenAI-compatible async client. Each metric is
invoked with exactly the inputs it requires (see evaluation/metrics.py).
For tests, `scorers=` injects deterministic async callables instead.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

from app.config import Settings
from evaluation.metrics import EVALUATOR_SCHEMA, Sample

logger = logging.getLogger("rag.eval.ragas")

Scorer = Callable[[Sample], Awaitable[float]]

_CALL_KWARGS: dict[str, Callable[[Sample], dict]] = {
    "faithfulness": lambda s: dict(user_input=s.question, response=s.answer, retrieved_contexts=s.contexts),
    "answer_relevancy": lambda s: dict(user_input=s.question, response=s.answer),
    "answer_correctness": lambda s: dict(user_input=s.question, response=s.answer, reference=s.reference),
    "semantic_similarity": lambda s: dict(reference=s.reference, response=s.answer),
    "context_precision": lambda s: dict(user_input=s.question, reference=s.reference, retrieved_contexts=s.contexts),
    "context_recall": lambda s: dict(user_input=s.question, retrieved_contexts=s.contexts, reference=s.reference),
    "context_entity_recall": lambda s: dict(reference=s.reference, retrieved_contexts=s.contexts),
    "noise_sensitivity": lambda s: dict(
        user_input=s.question, response=s.answer, reference=s.reference, retrieved_contexts=s.contexts
    ),
}


def ragas_version() -> str:
    try:
        import ragas

        return ragas.__version__
    except Exception:  # pragma: no cover
        return "unavailable"


def evaluator_version(settings: Settings) -> str:
    return f"{EVALUATOR_SCHEMA}|ragas={ragas_version()}|judge={settings.judge_model}|emb={settings.judge_embedding_model}"


class RagasBackend:
    def __init__(self, settings: Settings, *, scorers: Optional[dict[str, Scorer]] = None):
        self._settings = settings
        self._override = scorers or {}
        self._metrics: dict[str, object] = {}
        self._client = None
        self._embedding_client = None
        if (
            not self._override
            and settings.provider_mode == "openai"
            and settings.llm_api_key
            and settings.embedding_api_key
        ):
            self._build(settings)

    def _build(self, settings: Settings) -> None:
        from openai import AsyncOpenAI
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness, AnswerRelevancy, ContextEntityRecall, ContextPrecision, ContextRecall,
            Faithfulness, NoiseSensitivity, SemanticSimilarity,
        )

        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.resolved_llm_base_url,
            max_retries=2,
            timeout=settings.llm_timeout_seconds,
        )
        self._embedding_client = AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            max_retries=2,
        )
        llm = llm_factory(settings.judge_model, client=self._client)
        emb = embedding_factory(
            "openai",
            model=settings.judge_embedding_model,
            client=self._embedding_client,
        )
        self._metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
            "answer_correctness": AnswerCorrectness(llm=llm, embeddings=emb),
            "semantic_similarity": SemanticSimilarity(embeddings=emb),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
            "context_entity_recall": ContextEntityRecall(llm=llm),
            "noise_sensitivity": NoiseSensitivity(llm=llm),
        }

    @property
    def judge_available(self) -> bool:
        return bool(self._override or self._metrics)

    def has(self, name: str) -> bool:
        return name in self._override or name in self._metrics

    async def score(self, name: str, sample: Sample) -> float:
        if name in self._override:
            return float(await self._override[name](sample))
        metric = self._metrics[name]
        result = await metric.ascore(**_CALL_KWARGS[name](sample))  # type: ignore[attr-defined]
        return float(result.value)

    async def completeness(self, sample: Sample) -> float:
        """LLM-judged: fraction of the gold answer's key points covered by the answer."""
        if "answer_completeness" in self._override:
            return float(await self._override["answer_completeness"](sample))
        if self._client is None:
            raise RuntimeError("no judge client")
        prompt = (
            "You compare a candidate answer with a reference answer.\n"
            "1. List the distinct key facts in the REFERENCE (numbers, conditions, names).\n"
            "2. Count how many of those facts are present and correct in the CANDIDATE.\n"
            'Reply as JSON: {"total": <int>, "covered": <int>}.\n\n'
            f"Question: {sample.question}\nReference: {sample.reference}\nCandidate: {sample.answer}"
        )
        resp = await self._client.chat.completions.create(
            model=self._settings.judge_model, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        total, covered = int(data["total"]), int(data["covered"])
        if total <= 0:
            raise ValueError("judge returned total=0")
        return max(0.0, min(1.0, covered / total))

    async def aclose(self) -> None:
        for client in (self._client, self._embedding_client):
            if client is None:
                continue
            try:
                await client.close()
            except Exception:  # pragma: no cover
                pass
