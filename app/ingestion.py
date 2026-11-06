"""Document ingestion: parse (PDF/TXT/MD) -> chunk -> embed -> upsert to Qdrant.

Ingestion is traced too (one trace per document) so embedding cost and
failures are visible next to chat traces.
"""
from __future__ import annotations

import hashlib
import io
import logging
import sqlite3
import uuid
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from app.chunking import chunk_pages
from app.config import Settings
from app.embeddings import Embedder
from app.observability import _NullObservation, observation
from app.vectorstore import StoredChunk, VectorStore

logger = logging.getLogger("rag.ingestion")

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _untraced(name: str, **_: object):
    return nullcontext(_NullObservation())


def parse_document(filename: str, data: bytes) -> list[tuple[Optional[int], str]]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if ext in {".txt", ".md"}:
        return [(None, data.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")


def content_version(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def ingest_bytes(
    *,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    filename: str,
    data: bytes,
    document_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    traced: bool = True,
) -> dict:
    """`traced=False` is used when building throw-away experiment indexes so they
    do not clutter Langfuse with ingestion traces."""
    span = observation if traced else _untraced
    document_id = document_id or str(uuid.uuid4())
    version = content_version(data)
    size = chunk_size or settings.chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    with span("ingest-document", as_type="chain", input={"filename": filename, "bytes": len(data)},
                     metadata={"document_id": document_id, "document_version": version,
                               "chunk_size": size, "chunk_overlap": overlap}) as root:
        with span("parse-and-chunk", as_type="span") as pspan:
            pages = parse_document(filename, data)
            chunks = chunk_pages(pages, size, overlap)
            pspan.update(output={"pages": len(pages), "chunks": len(chunks)})
        if not chunks:
            raise ValueError("No extractable text found in the document.")
        with span("embed-chunks", as_type="embedding", model=embedder.model_name) as espan:
            emb = embedder.embed([c.text for c in chunks])
            cost = settings.estimate_cost(embedder.model_name, emb.tokens, 0)
            espan.update(usage_details={"input": emb.tokens}, **({"cost_details": cost} if cost else {}))
        with span("upsert-vectors", as_type="span") as uspan:
            stored = [
                StoredChunk(
                    chunk_id=f"{document_id}:{c.index}", document_id=document_id, document_version=version,
                    filename=filename, chunk_index=c.index, page=c.page, text=c.text,
                )
                for c in chunks
            ]
            store.upsert(stored, emb.vectors)
            uspan.update(output={"upserted": len(stored)})
        result = {"document_id": document_id, "document_version": version, "chunks": len(stored)}
        root.update(output=result)
    return result


# ---- document registry (small SQLite table) --------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, filename TEXT NOT NULL, version TEXT, file_size INTEGER,
    status TEXT NOT NULL, error_message TEXT, chunk_count INTEGER DEFAULT 0, created_at TEXT NOT NULL
);
"""


class DocumentRegistry:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create(self, doc_id: str, filename: str, file_size: int) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO documents (id, filename, file_size, status, created_at) VALUES (?,?,?,?,?)",
                (doc_id, filename, file_size, "Processing", datetime.now(timezone.utc).isoformat()),
            )

    def mark_completed(self, doc_id: str, version: str, chunks: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET status='Completed', version=?, chunk_count=?, error_message=NULL WHERE id=?",
                      (version, chunks, doc_id))

    def mark_failed(self, doc_id: str, error: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET status='Failed', error_message=? WHERE id=?", (error[:500], doc_id))

    def list(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM documents ORDER BY created_at DESC")]

    def get(self, doc_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def delete(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
