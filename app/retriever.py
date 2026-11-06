"""Retrieval: dense search, optional BM25 hybrid (reciprocal-rank fusion),
optional lightweight lexical rerank. Every stage is its own Langfuse span."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from app.embeddings import Embedder
from app.llm import content_tokens
from app.observability import observation
from app.vectorstore import RetrievedChunk, VectorStore

RRF_K = 60


@dataclass
class RetrievalOutput:
    chunks: list[RetrievedChunk]
    embed_tokens: int
    timings_ms: dict[str, float]


def _rrf(dense: list[RetrievedChunk], sparse: list[RetrievedChunk]) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    fused: dict[str, float] = {}
    for lst in (dense, sparse):
        for c in lst:
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + 1.0 / (RRF_K + c.rank)
            if c.chunk_id in merged:
                m = merged[c.chunk_id]
                m.dense_score = m.dense_score if m.dense_score is not None else c.dense_score
                m.bm25_score = m.bm25_score if m.bm25_score is not None else c.bm25_score
                m.sources = sorted(set(m.sources) | set(c.sources))
            else:
                merged[c.chunk_id] = c
    out = sorted(merged.values(), key=lambda c: fused[c.chunk_id], reverse=True)
    for i, c in enumerate(out, start=1):
        c.score = fused[c.chunk_id]
        c.rank = i
    return out


def _lexical_rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    q = content_tokens(query)
    top = max((c.score for c in chunks), default=0.0) or 1.0
    for c in chunks:
        coverage = len(q & content_tokens(c.text)) / (len(q) or 1)
        c.rerank_score = 0.7 * coverage + 0.3 * (c.score / top)
    out = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
    for i, c in enumerate(out, start=1):
        c.rank = i
    return out


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        cost_fn: Optional[Callable[[str, int, int], Optional[dict]]] = None,
    ):
        self.store = store
        self.embedder = embedder
        self._cost_fn = cost_fn

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        hybrid: bool = False,
        reranker: str = "none",
        candidate_multiplier: int = 3,
        document_ids: Optional[list[str]] = None,
    ) -> RetrievalOutput:
        timings: dict[str, float] = {}
        fetch_k = max(top_k, top_k * candidate_multiplier) if (hybrid or reranker != "none") else top_k

        t0 = time.perf_counter()
        with observation(
            "embed-query", as_type="embedding", model=self.embedder.model_name, input=query
        ) as span:
            emb = self.embedder.embed([query])
            cost = self._cost_fn(self.embedder.model_name, emb.tokens, 0) if self._cost_fn else None
            span.update(usage_details={"input": emb.tokens}, **({"cost_details": cost} if cost else {}))
        timings["embed_query"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with observation(
            "vector-search", as_type="retriever",
            input={"query": query, "top_k": fetch_k, "filter_documents": document_ids},
        ) as span:
            dense = self.store.dense_search(emb.vectors[0], fetch_k, document_ids)
            span.update(output={"returned": len(dense), "top_scores": [round(c.score, 4) for c in dense[:5]]})
        timings["vector_search"] = (time.perf_counter() - t0) * 1000

        candidates = dense
        if hybrid:
            t0 = time.perf_counter()
            with observation(
                "bm25-search", as_type="retriever", input={"query": query, "top_k": fetch_k},
            ) as span:
                sparse = self.store.bm25_search(query, fetch_k, document_ids)
                span.update(output={"returned": len(sparse)})
            with observation("rank-fusion", as_type="span", input={"method": "rrf", "k": RRF_K}) as span:
                candidates = _rrf(dense, sparse)
                span.update(output={"fused": len(candidates)})
            timings["hybrid"] = (time.perf_counter() - t0) * 1000

        if reranker != "none" and candidates:
            t0 = time.perf_counter()
            with observation("rerank", as_type="span", input={"method": reranker, "candidates": len(candidates)}) as span:
                candidates = _lexical_rerank(query, candidates)
                span.update(output={"order": [c.chunk_id for c in candidates[:top_k]]})
            timings["rerank"] = (time.perf_counter() - t0) * 1000

        final = candidates[:top_k]
        for i, c in enumerate(final, start=1):
            c.rank = i
        return RetrievalOutput(final, emb.tokens, timings)
