"""Qdrant-backed chunk store with a lazily-built BM25 index for hybrid search."""
from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import portalocker
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from rank_bm25 import BM25Okapi

from app.config import Settings

_TOK = re.compile(r"[a-z0-9]+")
logger = logging.getLogger("rag.vectorstore")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


@contextmanager
def embedded_qdrant_lock(path: str) -> Iterator[None]:
    """Keep embedded Qdrant exclusive across app processes, with an actionable error."""
    storage_path = Path(path).resolve()
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = storage_path.with_name(f".{storage_path.name}.app.lock")
    owner_path = lock_path.with_name(f"{lock_path.name}.owner")
    lock = portalocker.Lock(str(lock_path), mode="a+", fail_when_locked=True)
    try:
        lock.acquire()
    except portalocker.exceptions.AlreadyLocked as exc:
        try:
            owner = owner_path.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            owner = "unknown"
        raise RuntimeError(
            f"Embedded Qdrant storage '{storage_path}' is already in use by application PID {owner}. "
            "Stop the other app process before starting another. For multiple workers, run a Qdrant "
            "server and set QDRANT_URL. Do not delete the storage or lock files."
        ) from exc

    try:
        owner_path.write_text(str(os.getpid()), encoding="utf-8")
        yield
    finally:
        try:
            if owner_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                owner_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            lock.release()


@dataclass
class StoredChunk:
    chunk_id: str
    document_id: str
    document_version: str
    filename: str
    chunk_index: int
    page: Optional[int]
    text: str


@dataclass
class RetrievedChunk(StoredChunk):
    score: float = 0.0
    rank: int = 0
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None
    sources: list[str] = field(default_factory=list)

    def trace_view(self, *, include_text: bool = True, max_chars: int = 600) -> dict:
        d = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "filename": self.filename,
            "page": self.page,
            "rank": self.rank,
            "score": round(self.score, 5),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 5),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 5),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 5),
            "sources": self.sources,
        }
        if include_text:
            d["text"] = self.text[:max_chars]
        return d


def make_client(settings: Settings, *, in_memory: bool = False) -> QdrantClient:
    if in_memory:
        return QdrantClient(":memory:")
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=settings.resolved_qdrant_path)
    except RuntimeError as exc:
        if "already accessed by another instance of Qdrant client" not in str(exc):
            raise
        raise RuntimeError(
            f"Embedded Qdrant storage '{Path(settings.resolved_qdrant_path).resolve()}' is already in use. "
            "Stop the other app process before starting another. For multiple workers, run a Qdrant "
            "server and set QDRANT_URL. Do not delete the storage or lock files."
        ) from exc


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int):
        self.client = client
        self.collection = collection
        self.dim = dim
        self._version = 0
        self._bm25_cache: tuple[int, BM25Okapi, list[StoredChunk]] | None = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors
            existing_dim = vectors.size if isinstance(vectors, qm.VectorParams) else None
            if existing_dim is not None and existing_dim != self.dim:
                original = self.collection
                self.collection = f"{original}_{self.dim}d"
                logger.warning(
                    "Qdrant collection %s has vector dimension %d, but the active embedder uses %d. "
                    "Using separate collection %s; re-upload documents to index them with this embedder.",
                    original,
                    existing_dim,
                    self.dim,
                    self.collection,
                )
                existing = {c.name for c in self.client.get_collections().collections}

        if self.collection not in existing:
            self.client.create_collection(
                self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )

    @staticmethod
    def _payload(c: StoredChunk) -> dict:
        return {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "document_version": c.document_version,
            "filename": c.filename,
            "chunk_index": c.chunk_index,
            "page": c.page,
            "text": c.text,
        }

    @staticmethod
    def _from_payload(p: dict) -> StoredChunk:
        return StoredChunk(
            chunk_id=p["chunk_id"],
            document_id=p["document_id"],
            document_version=p.get("document_version", ""),
            filename=p.get("filename", ""),
            chunk_index=p.get("chunk_index", 0),
            page=p.get("page"),
            text=p["text"],
        )

    def upsert(self, chunks: list[StoredChunk], vectors: list[list[float]]) -> None:
        points = [
            qm.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)), vector=v, payload=self._payload(c))
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128])
        self._version += 1

    def delete_document(self, document_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.Filter(
                must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
            ),
        )
        self._version += 1

    def _filter(self, document_ids: Optional[list[str]]) -> Optional[qm.Filter]:
        if not document_ids:
            return None
        return qm.Filter(must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=document_ids))])

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def dense_search(
        self, vector: list[float], k: int, document_ids: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        res = self.client.query_points(
            self.collection,
            query=vector,
            limit=k,
            query_filter=self._filter(document_ids),
            with_payload=True,
        )
        out = []
        for rank, p in enumerate(res.points, start=1):
            base = self._from_payload(p.payload)
            out.append(
                RetrievedChunk(**base.__dict__, score=float(p.score), rank=rank, dense_score=float(p.score), sources=["dense"])
            )
        return out

    def all_chunks(self) -> list[StoredChunk]:
        chunks: list[StoredChunk] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(self._from_payload(p.payload) for p in points)
            if offset is None:
                break
        return chunks

    def bm25_search(
        self, query: str, k: int, document_ids: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        if self._bm25_cache is None or self._bm25_cache[0] != self._version:
            chunks = self.all_chunks()
            if not chunks:
                return []
            index = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])
            self._bm25_cache = (self._version, index, chunks)
        _, index, chunks = self._bm25_cache
        scores = index.get_scores(tokenize(query))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:
                break
            if document_ids and chunks[i].document_id not in document_ids:
                continue
            out.append(
                RetrievedChunk(
                    **chunks[i].__dict__, score=float(scores[i]), rank=len(out) + 1,
                    bm25_score=float(scores[i]), sources=["bm25"],
                )
            )
            if len(out) >= k:
                break
        return out
