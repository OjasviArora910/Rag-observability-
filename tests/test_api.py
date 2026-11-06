"""End-to-end API: upload a document, chat over it, get a traced + cited answer."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.main import create_app
from app.llm import LLMResponse
from evaluation.experiment import CORPUS_DIR


def minimal_pdf(lines: list[str]) -> bytes:
    """Build a tiny valid one-page text PDF (no extra dependency needed)."""
    text = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(text)} >>\nstream\n{text}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


@pytest.fixture
def client(traced_settings, exporter):
    app = create_app(traced_settings, store_client=QdrantClient(":memory:"))
    with TestClient(app) as c:
        yield c


def upload(client, name: str, data: bytes, ctype="text/markdown"):
    return client.post("/api/v1/documents/upload", files={"file": (name, data, ctype)})


def test_upload_then_ask_returns_cited_answer_and_trace_id(client, exporter):
    r = upload(client, "security_policy.md", (CORPUS_DIR / "security_policy.md").read_bytes())
    assert r.status_code == 201
    doc = client.get(f"/api/v1/documents/{r.json()['id']}").json()
    assert doc["status"] == "Completed" and doc["chunk_count"] > 0 and doc["version"]

    # (the offline FakeChat is extractive: it needs word overlap with the passage to answer)
    chat = client.post("/api/v1/chat", json={"question": "How many characters long must passwords be?", "session_id": "s1"})
    assert chat.status_code == 200
    body = chat.json()
    assert "14 characters" in body["answer"] and body["citations"] and body["abstained"] is False
    assert body["citations"][0]["filename"] == "security_policy.md"
    assert body["trace_id"] and len(body["trace_id"]) == 32
    # the trace for this request exists in the exporter
    import app.observability as obs

    obs.flush()
    assert any(format(s.context.trace_id, "032x") == body["trace_id"] for s in exporter.get_finished_spans())
    # and an async evaluation job was queued, not executed
    assert client.app.state.queue.get(body["trace_id"])["status"] == "pending"


def test_document_overview_uses_context_across_the_document(client, monkeypatch):
    sections = [
        f"Section {i}: " + (f"topic{i} policy details are recorded here. " * 8)
        for i in range(36)
    ]
    response = upload(client, "long_report.md", "\n".join(sections).encode())
    assert response.status_code == 201
    document = client.app.state.registry.get(response.json()["id"])
    assert document["status"] == "Completed"
    assert document["chunk_count"] > 4

    captured = {}

    def summarize(messages, **kwargs):
        captured["prompt"] = messages[-1]["content"]
        return LLMResponse("The report covers the uploaded policy sections. 【1】", kwargs.get("model") or "test")

    monkeypatch.setattr(client.app.state.pipeline.chat, "complete", summarize)
    result = client.post(
        "/api/v1/chat", json={"question": "What do you understand by the given document?"}
    )

    assert result.status_code == 200
    body = result.json()
    assert body["abstained"] is False
    assert body["citations"][0]["filename"] == "long_report.md"
    assert "Section 0:" in captured["prompt"]
    assert "Section 35:" in captured["prompt"]
    assert captured["prompt"].count("long_report.md") > 4


def test_pdf_upload_and_retrieval(client):
    pdf = minimal_pdf(["Refunds are processed within 14 business days.", "Shipping is free above 50 euros."])
    r = upload(client, "policy.pdf", pdf, "application/pdf")
    assert r.status_code == 201
    doc = client.get(f"/api/v1/documents/{r.json()['id']}").json()
    assert doc["status"] == "Completed", doc
    body = client.post("/api/v1/chat", json={"question": "How many business days do refunds take?"}).json()
    assert "14 business days" in body["answer"] and body["citations"][0]["page"] == 1


def test_unanswerable_question_abstains(client):
    upload(client, "security_policy.md", (CORPUS_DIR / "security_policy.md").read_bytes())
    body = client.post("/api/v1/chat", json={"question": "What is the cafeteria menu this week?"}).json()
    assert body["abstained"] is True and body["citations"] == []


def test_empty_index_is_an_empty_context_abstention_without_calling_the_llm(client, exporter):
    called = []
    client.app.state.pipeline.chat.complete = lambda *a, **k: called.append(1)  # type: ignore[method-assign]
    body = client.post("/api/v1/chat", json={"question": "Anything?"}).json()
    assert body["abstained"] is True and not called
    assert {s["name"]: s["value"] for s in exporter.scores}["empty_context"] == 1.0


def test_delete_removes_vectors(client):
    doc_id = upload(client, "security_policy.md", (CORPUS_DIR / "security_policy.md").read_bytes()).json()["id"]
    assert client.app.state.store.count() > 0
    assert client.delete(f"/api/v1/documents/{doc_id}").json()["deleted"] is True
    assert client.app.state.store.count() == 0
    assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404
    assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 404


def test_document_filter_restricts_retrieval(client):
    a = upload(client, "security_policy.md", (CORPUS_DIR / "security_policy.md").read_bytes()).json()["id"]
    upload(client, "employee_handbook.md", (CORPUS_DIR / "employee_handbook.md").read_bytes())
    body = client.post("/api/v1/chat", json={"question": "How many days of paid annual leave?", "document_ids": [a]}).json()
    assert body["abstained"] is True  # leave policy lives in the handbook, which is filtered out


@pytest.mark.parametrize("name,data,code", [("notes.exe", b"x", 400), ("empty.md", b"", 400)])
def test_upload_validation(client, name, data, code):
    assert upload(client, name, data).status_code == code


def test_scanned_pdf_without_text_fails_cleanly(client):
    doc_id = upload(client, "blank.pdf", minimal_pdf([""]), "application/pdf").json()["id"]
    doc = client.get(f"/api/v1/documents/{doc_id}").json()
    assert doc["status"] == "Failed" and "No extractable text" in doc["error_message"]


def test_llm_failure_returns_502_with_error_trace(client, exporter):
    upload(client, "security_policy.md", (CORPUS_DIR / "security_policy.md").read_bytes())

    def boom(*a, **k):
        raise RuntimeError("provider outage")

    client.app.state.pipeline.chat.complete = boom  # type: ignore[method-assign]
    r = client.post("/api/v1/chat", json={"question": "What is the minimum password length?"})
    assert r.status_code == 502 and "provider outage" not in r.text  # no internals leaked
    import app.observability as obs

    obs.flush()
    assert any(s.attributes.get("langfuse.observation.level") == "ERROR" for s in exporter.get_finished_spans())


def test_feedback_is_recorded_as_a_score(client, monkeypatch):
    sent = []
    monkeypatch.setattr("app.main.create_score", lambda **k: sent.append(k) or True)
    r = client.post("/api/v1/feedback", json={"trace_id": "abc", "thumbs_up": False, "comment": "wrong"})
    assert r.json() == {"recorded": True}
    assert sent[0]["name"] == "user_feedback" and sent[0]["value"] == 0.0 and sent[0]["trace_id"] == "abc"


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "healthy"}
    ready = client.get("/health/ready").json()
    assert ready["langfuse_tracing"] is True and "rag_config" in ready


def test_embedded_qdrant_rejects_second_app_with_owner_and_releases_lock(settings):
    first_app = create_app(settings)
    second_app = create_app(settings)

    with TestClient(first_app) as first:
        assert first.get("/health/ready").status_code == 200
        with pytest.raises(
            RuntimeError, match=f"already in use by application PID {os.getpid()}"
        ):
            with TestClient(second_app):
                pass

    with TestClient(second_app) as second:
        assert second.get("/health/ready").status_code == 200


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Document Chat" in r.text
