"""FastAPI application: document upload + chat, instrumented with Langfuse."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.embeddings import get_embedder
from app.ingestion import ALLOWED_EXTENSIONS, DocumentRegistry, ingest_bytes
from app.llm import get_chat
from app.observability import create_score, init_langfuse, observation, shutdown_langfuse, tracing_enabled
from app.rag import RAGPipeline
from app.retriever import Retriever
from app.vectorstore import VectorStore, embedded_qdrant_lock, make_client
from evaluation.eval_queue import EvalQueue

logger = logging.getLogger("rag.api")
MAX_FILE_SIZE = 50 * 1024 * 1024
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    document_ids: Optional[list[str]] = None


class FeedbackRequest(BaseModel):
    trace_id: str
    thumbs_up: bool
    comment: Optional[str] = Field(default=None, max_length=1000)


def create_app(settings: Optional[Settings] = None, *, span_exporter: Any = None, store_client: Any = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        logging.basicConfig(level=s.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        s.data_dir.mkdir(parents=True, exist_ok=True)
        client = None
        storage_lock = (
            embedded_qdrant_lock(s.resolved_qdrant_path)
            if store_client is None and not s.qdrant_url
            else nullcontext()
        )
        try:
            with storage_lock:
                try:
                    init_langfuse(s, span_exporter=span_exporter, force=span_exporter is not None)
                    embedder = get_embedder(s)
                    client = store_client or make_client(s)
                    store = VectorStore(client, s.collection_name, embedder.dim)
                    queue = EvalQueue(s.resolved_eval_db, s.eval_max_attempts, s.eval_retry_base_seconds)
                    retriever = Retriever(store, embedder, cost_fn=s.estimate_cost)
                    app.state.s = s
                    app.state.store = store
                    app.state.embedder = embedder
                    app.state.registry = DocumentRegistry(s.documents_db)
                    app.state.queue = queue
                    app.state.pipeline = RAGPipeline(s, retriever, get_chat(s), eval_queue=queue)
                    yield
                finally:
                    if client is not None:
                        client.close()
                        client = None
        finally:
            shutdown_langfuse()

    app = FastAPI(title=(settings or Settings()).app_name, version="1.0.0", lifespan=lifespan)

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/health/ready")
    def ready():
        s: Settings = app.state.s
        return {
            "status": "ready",
            "chunks_indexed": app.state.store.count(),
            "langfuse_tracing": tracing_enabled(),
            "provider_mode": s.provider_mode,
            "vector_collection": app.state.store.collection,
            "embedding_dimension": app.state.embedder.dim,
            "eval_queue": app.state.queue.stats(),
            "rag_config": s.rag_config(),
        }

    # -------------------------------------------------------------- documents
    def _ingest_job(doc_id: str, filename: str, data: bytes) -> None:
        try:
            result = ingest_bytes(
                store=app.state.store, embedder=app.state.embedder, settings=app.state.s,
                filename=filename, data=data, document_id=doc_id,
            )
            app.state.registry.mark_completed(doc_id, result["document_version"], result["chunks"])
        except Exception as exc:
            logger.exception("Ingestion failed for %s", filename)
            app.state.registry.mark_failed(doc_id, f"{type(exc).__name__}: {exc}")

    @app.post("/api/v1/documents/upload", status_code=status.HTTP_201_CREATED)
    async def upload_document(background: BackgroundTasks, file: UploadFile = File(...)):
        filename = (file.filename or "").strip()
        if not filename:
            raise HTTPException(400, "Filename is required.")
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Uploaded file is empty.")
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(413, "File size exceeds the 50 MB limit.")
        doc_id = str(uuid.uuid4())
        app.state.registry.create(doc_id, filename, len(data))
        background.add_task(_ingest_job, doc_id, filename, data)
        return app.state.registry.get(doc_id)

    @app.get("/api/v1/documents")
    def list_documents():
        return app.state.registry.list()

    @app.get("/api/v1/documents/{document_id}")
    def get_document(document_id: str):
        doc = app.state.registry.get(document_id)
        if doc is None:
            raise HTTPException(404, "Document not found.")
        return doc

    @app.delete("/api/v1/documents/{document_id}")
    def delete_document(document_id: str):
        if app.state.registry.get(document_id) is None:
            raise HTTPException(404, "Document not found.")
        try:
            app.state.store.delete_document(document_id)
        except Exception as exc:
            raise HTTPException(500, f"Failed to remove document vectors: {exc}") from exc
        app.state.registry.delete(document_id)
        return {"deleted": True, "id": document_id}

    # ------------------------------------------------------------------- chat
    @app.post("/api/v1/chat")
    def chat(req: ChatRequest):
        pipeline: RAGPipeline = app.state.pipeline
        try:
            result = pipeline.answer(
                req.question.strip(), session_id=req.session_id, user_id=req.user_id,
                document_ids=req.document_ids,
            )
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(502, f"The request could not be completed: {type(exc).__name__}") from exc
        return result.to_api()

    @app.post("/api/v1/feedback")
    def feedback(req: FeedbackRequest):
        ok = create_score(
            trace_id=req.trace_id, name="user_feedback", value=1.0 if req.thumbs_up else 0.0,
            data_type="BOOLEAN", comment=req.comment,
        )
        return {"recorded": ok}

    return app


app = create_app()
