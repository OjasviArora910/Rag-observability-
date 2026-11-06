"""The RAG request path:
query -> retrieval (embed, vector, bm25, fusion, rerank) -> context assembly
-> prompt -> LLM generation -> citation parsing -> response.

Every request is one Langfuse trace:

rag-chat (chain)
 |- retrieval (retriever)
 |   |- embed-query (embedding) | vector-search | bm25-search | rank-fusion | rerank
 |- context-assembly (span)
 |- generation (generation)     <- model, usage, cost, prompt version
 |- answer-postprocess (span)   <- citations, abstention

Evaluation is NOT executed here. After the answer is built, a small job is
written to the eval queue and picked up by a separate worker process.
"""
from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import Settings
from app.llm import ChatLLM
from app.observability import (
    create_score,
    current_trace_id,
    observation,
    trace_attributes,
    trace_url,
)
from app.privacy import redact
from app.prompts import (
    ABSTAIN_PHRASE,
    build_user_message,
    get_system_prompt,
    is_abstention,
    is_document_overview,
)
from app.retriever import Retriever
from app.vectorstore import RetrievedChunk

logger = logging.getLogger("rag.pipeline")

_CITATION = re.compile(r"[\[【](\d+(?:\s*,\s*\d+)*)(?:\]|】)")


def parse_citations(answer: str) -> list[int]:
    nums: list[int] = []
    for m in _CITATION.finditer(answer):
        nums.extend(int(x) for x in re.split(r"\s*,\s*", m.group(1)))
    return nums


def _select_overview_chunks(chunks: list[RetrievedChunk], budget: int) -> list[RetrievedChunk]:
    """Choose passages spread across each document while staying near the context budget."""
    grouped: dict[str, list[RetrievedChunk]] = {}
    for chunk in sorted(chunks, key=lambda c: (c.document_id, c.chunk_index)):
        grouped.setdefault(chunk.document_id, []).append(chunk)
    if not grouped:
        return []

    documents = list(grouped.values())
    average_block_size = sum(len(c.text) + len(c.filename) + 20 for c in chunks) / len(chunks)
    slots = min(len(chunks), max(1, int(budget / max(average_block_size, 1))))

    while True:
        quotas = [0] * len(documents)
        for _ in range(slots):
            eligible = [i for i, doc in enumerate(documents) if quotas[i] < len(doc)]
            if not eligible:
                break
            selected = min(eligible, key=lambda i: quotas[i] / len(documents[i]))
            quotas[selected] += 1

        chosen: list[RetrievedChunk] = []
        for document, quota in zip(documents, quotas):
            if quota == len(document):
                chosen.extend(document)
            elif quota == 1:
                chosen.append(document[0])
            elif quota > 1:
                indexes = {round(i * (len(document) - 1) / (quota - 1)) for i in range(quota)}
                chosen.extend(document[i] for i in sorted(indexes))

        size = sum(len(c.text) + len(c.filename) + 20 for c in chosen)
        if size <= budget or slots <= 1:
            return chosen
        slots -= 1


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[RetrievedChunk]
    cited_numbers: list[int]
    citations: list[dict]
    invalid_citations: list[int]
    abstained: bool
    empty_context: bool
    trace_id: Optional[str]
    trace_url: Optional[str]
    config: dict[str, Any]
    latency_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    cost: Optional[dict] = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    def to_api(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "abstained": self.abstained,
            "trace_id": self.trace_id,
            "trace_url": self.trace_url,
            "latency_ms": round(self.latency_ms, 1),
            "config": self.config,
        }


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        chat: ChatLLM,
        *,
        eval_queue: Any = None,
        overrides: Optional[dict[str, Any]] = None,
    ):
        self.settings = settings
        self.retriever = retriever
        self.chat = chat
        self.eval_queue = eval_queue
        self.cfg = {**settings.rag_config(), "candidate_multiplier": settings.candidate_multiplier,
                    "max_context_chars": settings.max_context_chars, **(overrides or {})}

    # ------------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        enqueue_eval: bool = True,
    ) -> RAGResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        trace_tags = ["rag", f"prompt:{cfg['prompt_version']}", f"model:{cfg['llm_model']}", *(tags or [])]
        meta = {**{k: v for k, v in cfg.items()}, **(extra_metadata or {})}

        with observation("rag-chat", as_type="chain", input=question, metadata=meta) as root:
            with trace_attributes(
                session_id=session_id, user_id=user_id, tags=trace_tags,
                version=cfg["prompt_version"], metadata=meta, trace_name="rag-chat",
            ):
                trace_id = current_trace_id()
                timings: dict[str, float] = {}

                # 1) retrieval --------------------------------------------------
                t0 = time.perf_counter()
                overview_request = is_document_overview(question)
                retrieval_top_k = max(cfg["top_k"], self.retriever.store.count()) if overview_request else cfg["top_k"]
                with observation(
                    "retrieval", as_type="retriever", input=question,
                    metadata={
                        "top_k": retrieval_top_k,
                        "hybrid": cfg["hybrid"],
                        "reranker": cfg["reranker"],
                        "overview_request": overview_request,
                    },
                ) as rspan:
                    out = self.retriever.retrieve(
                        question, top_k=retrieval_top_k, hybrid=cfg["hybrid"], reranker=cfg["reranker"],
                        candidate_multiplier=cfg["candidate_multiplier"], document_ids=document_ids,
                    )
                    chunks = out.chunks
                    rspan.update(output=[c.trace_view() for c in chunks], metadata={
                        "chunk_ids": [c.chunk_id for c in chunks],
                        "document_ids": sorted({c.document_id for c in chunks}),
                        "document_versions": sorted({c.document_version for c in chunks}),
                        "retrieved": len(chunks),
                    })
                timings["retrieval"] = (time.perf_counter() - t0) * 1000
                timings.update({f"retrieval.{k}": v for k, v in out.timings_ms.items()})
                empty_context = len(chunks) == 0

                # 2) context assembly ---------------------------------------------
                with observation("context-assembly", as_type="span", input={"n_chunks": len(chunks)}) as cspan:
                    blocks: list[str] = []
                    used: list[RetrievedChunk] = []
                    budget = cfg["max_context_chars"]
                    context_chunks = _select_overview_chunks(chunks, budget) if overview_request else chunks
                    for c in context_chunks:
                        block = f"[{len(used) + 1}] ({c.filename}) {c.text}"
                        if blocks and sum(len(b) for b in blocks) + len(block) > budget:
                            break
                        blocks.append(block)
                        used.append(c)
                    cspan.update(output={
                        "used_chunk_ids": [c.chunk_id for c in used],
                        "dropped_for_budget": len(chunks) - len(used),
                        "context_chars": sum(len(b) for b in blocks),
                        "empty_context": empty_context,
                    })

                # 3) generation ---------------------------------------------------
                usage: dict[str, int] = {}
                cost = None
                if empty_context:
                    answer = ABSTAIN_PHRASE  # nothing to ground on: skip the LLM call entirely
                else:
                    messages = [
                        {"role": "system", "content": get_system_prompt(cfg["prompt_version"])},
                        {"role": "user", "content": build_user_message(question, blocks)},
                    ]
                    t0 = time.perf_counter()
                    with observation(
                        "generation", as_type="generation", model=cfg["llm_model"], input=messages,
                        model_parameters={"temperature": cfg["temperature"]},
                        metadata={"prompt_version": cfg["prompt_version"]},
                    ) as gen:
                        resp = self.chat.complete(messages, model=cfg["llm_model"], temperature=cfg["temperature"])
                        answer = resp.text.strip()
                        usage = {"input": resp.input_tokens, "output": resp.output_tokens}
                        cost = self.settings.estimate_cost(cfg["llm_model"], resp.input_tokens, resp.output_tokens)
                        gen.update(
                            output=answer, usage_details=usage,
                            metadata={"prompt_version": cfg["prompt_version"], "llm_retries": resp.retries,
                                      "resolved_model": resp.model},
                            **({"cost_details": cost} if cost else {}),
                        )
                    timings["generation"] = (time.perf_counter() - t0) * 1000

                # 4) post-process: citations & abstention ----------------------------
                with observation("answer-postprocess", as_type="span", input=answer) as pspan:
                    numbers = parse_citations(answer)
                    invalid = sorted({n for n in numbers if not 1 <= n <= len(used)})
                    cited = sorted({n for n in numbers if 1 <= n <= len(used)})
                    citations = [
                        {"number": n, "chunk_id": used[n - 1].chunk_id, "document_id": used[n - 1].document_id,
                         "document_version": used[n - 1].document_version, "filename": used[n - 1].filename,
                         "page": used[n - 1].page, "score": round(used[n - 1].score, 5)}
                        for n in cited
                    ]
                    abstained = empty_context or is_abstention(answer)
                    pspan.update(output={"citations": citations, "invalid_citations": invalid, "abstained": abstained})

                latency_ms = (time.perf_counter() - t_start) * 1000
                root.update(
                    output={"answer": answer, "citations": [c["chunk_id"] for c in citations]},
                    metadata={**meta, "abstained": abstained, "empty_context": empty_context,
                              "latency_ms": round(latency_ms, 1), "usage": usage,
                              "estimated_cost_usd": None if cost is None else round(cost["total"], 8)},
                )

                # Cheap, deterministic request-time signals (RAG health, not LLM-judged).
                if trace_id:
                    create_score(trace_id=trace_id, name="empty_context", value=1.0 if empty_context else 0.0, data_type="BOOLEAN")
                    create_score(trace_id=trace_id, name="abstained", value=1.0 if abstained else 0.0, data_type="BOOLEAN")
                    create_score(trace_id=trace_id, name="invalid_citation_count", value=float(len(invalid)), data_type="NUMERIC")
                    if used:
                        create_score(trace_id=trace_id, name="top_retrieval_score", value=float(used[0].score), data_type="NUMERIC")

        result = RAGResult(
            question=question, answer=answer, contexts=used, cited_numbers=cited, citations=citations,
            invalid_citations=invalid, abstained=abstained, empty_context=empty_context, trace_id=trace_id,
            trace_url=trace_url(trace_id), config=dict(cfg), latency_ms=latency_ms, usage=usage, cost=cost,
            stage_timings_ms=timings,
        )
        if enqueue_eval:
            self._enqueue_online_eval(result, session_id)
        return result

    # ------------------------------------------------------------------
    def _enqueue_online_eval(self, r: RAGResult, session_id: Optional[str]) -> None:
        """Hand the finished request to the async evaluator. Never raises."""
        s = self.settings
        if not (s.online_eval_enabled and self.eval_queue and r.trace_id):
            return
        if random.random() > s.online_eval_sample_rate:
            return
        try:
            payload = {
                "question": r.question,
                "answer": r.answer,
                "contexts": [c.text for c in r.contexts],
                "context_chunk_ids": [c.chunk_id for c in r.contexts],
                "cited_numbers": r.cited_numbers,
                "invalid_citations": r.invalid_citations,
                "abstained": r.abstained,
                "empty_context": r.empty_context,
                "config": r.config,
                "session_id": session_id,
            }
            if s.redact_pii:
                payload = redact(payload)
            self.eval_queue.enqueue(r.trace_id, payload)
        except Exception:
            logger.exception("Could not enqueue online evaluation for trace %s", r.trace_id)
