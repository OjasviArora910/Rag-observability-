"""Demonstrates that every chatbot request produces a complete Langfuse trace.

Uses the REAL Langfuse SDK (v5) with an in-memory OTel exporter, so these tests
assert on the spans/attributes that would be sent to Langfuse."""
from __future__ import annotations

import json

import pytest

import app.observability as obs

EXPECTED_SPANS = {
    "rag-chat", "retrieval", "embed-query", "vector-search", "bm25-search", "rank-fusion", "rerank",
    "context-assembly", "generation", "answer-postprocess",
}


def _chat_spans(exporter):
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == "rag-chat")
    return root, [s for s in spans if s.context.trace_id == root.context.trace_id]


def test_request_produces_full_trace(traced_pipeline, exporter):
    result = traced_pipeline.answer("What is the minimum password length?", session_id="sess-1", user_id="user-1",
                                    enqueue_eval=False)
    obs.flush()
    root, spans = _chat_spans(exporter)

    assert result.trace_id == format(root.context.trace_id, "032x")
    assert EXPECTED_SPANS <= {s.name for s in spans}
    assert len({s.context.trace_id for s in spans}) == 1  # one trace for the whole request

    types = {s.name: s.attributes["langfuse.observation.type"] for s in spans}
    assert types["generation"] == "generation"
    assert types["retrieval"] == "retriever"
    assert types["vector-search"] == "retriever"
    assert types["bm25-search"] == "retriever"
    assert types["embed-query"] == "embedding"
    for name in ("vector-search", "bm25-search"):
        search_input = json.loads(next(s for s in spans if s.name == name).attributes["langfuse.observation.input"])
        assert search_input["query"] == "What is the minimum password length?"

    a = root.attributes
    assert a["session.id"] == "sess-1" and a["user.id"] == "user-1"
    assert "prompt:v2" in a["langfuse.trace.tags"]
    assert a["langfuse.version"] == "v2"  # prompt version recorded on the trace
    assert json.loads(a["langfuse.trace.metadata.prompt_version"]) == "v2"
    assert a["langfuse.observation.input"] == "What is the minimum password length?"


def test_generation_records_model_usage_and_cost(traced_pipeline, exporter):
    traced_pipeline.answer("How many days of paid annual leave do full-time employees get?", enqueue_eval=False)
    obs.flush()
    _, spans = _chat_spans(exporter)
    gen = next(s for s in spans if s.name == "generation").attributes
    assert gen["langfuse.observation.model.name"] == "gpt-4o-mini"
    usage = json.loads(gen["langfuse.observation.usage_details"])
    assert usage["input"] > 0 and usage["output"] > 0
    assert json.loads(gen["langfuse.observation.cost_details"])["total"] > 0
    assert json.loads(gen["langfuse.observation.metadata.prompt_version"]) == "v2"


def test_retrieval_span_captures_chunks_ids_scores_and_versions(traced_pipeline, exporter):
    result = traced_pipeline.answer("What is the minimum password length?", enqueue_eval=False)
    obs.flush()
    _, spans = _chat_spans(exporter)
    retrieval = next(s for s in spans if s.name == "retrieval").attributes
    out = json.loads(retrieval["langfuse.observation.output"])
    assert out and {"chunk_id", "document_id", "document_version", "score", "rank", "text"} <= set(out[0])
    assert json.loads(retrieval["langfuse.observation.metadata.chunk_ids"]) == [c["chunk_id"] for c in out]
    assert json.loads(retrieval["langfuse.observation.metadata.document_versions"])
    assert out[0]["chunk_id"] == result.contexts[0].chunk_id


def test_citations_and_request_scores_are_recorded(traced_pipeline, exporter):
    result = traced_pipeline.answer("How many days of paid annual leave do full-time employees get?", enqueue_eval=False)
    assert result.cited_numbers and not result.invalid_citations
    names = {s["name"]: s for s in exporter.scores}
    assert {"empty_context", "abstained", "invalid_citation_count", "top_retrieval_score"} <= set(names)
    assert all(s["trace_id"] == result.trace_id for s in exporter.scores)
    assert names["abstained"]["value"] == 0.0


def test_unanswerable_question_is_traced_as_abstention(traced_pipeline, exporter):
    result = traced_pipeline.answer("What is the name of the company's CEO?", enqueue_eval=False)
    assert result.abstained
    assert {s["name"]: s["value"] for s in exporter.scores}["abstained"] == 1.0


def test_errors_are_recorded_on_the_trace_and_reraised(traced_pipeline, exporter):
    def boom(*a, **k):
        raise RuntimeError("llm down")

    traced_pipeline.chat.complete = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        traced_pipeline.answer("What is the minimum password length?", enqueue_eval=False)
    obs.flush()
    _, spans = _chat_spans(exporter)
    errored = {s.name for s in spans if s.attributes.get("langfuse.observation.level") == "ERROR"}
    assert {"generation", "rag-chat"} <= errored
    gen = next(s for s in spans if s.name == "generation")
    assert "llm down" in gen.attributes["langfuse.observation.status_message"]


def test_pipeline_works_without_langfuse(pipeline):
    result = pipeline.answer("What is the minimum password length?", enqueue_eval=False)
    assert result.answer and result.trace_id is None


def test_pii_is_masked_before_export(traced_settings, tmp_path):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from evaluation.experiment import ExperimentConfig, build_pipeline

    exp = InMemorySpanExporter()
    obs.init_langfuse(traced_settings, span_exporter=exp, force=True)
    p = build_pipeline(traced_settings, ExperimentConfig(name="pii"))
    p.answer("Email me at jane.doe@example.com about the minimum password length", enqueue_eval=False)
    obs.flush()
    spans = exp.get_finished_spans()
    root = next(s for s in spans if s.name == "rag-chat")
    assert "jane.doe@example.com" not in root.attributes["langfuse.observation.input"]
    assert "[EMAIL]" in root.attributes["langfuse.observation.input"]
    search = next(s for s in spans if s.name == "vector-search")
    assert "jane.doe@example.com" not in search.attributes["langfuse.observation.input"]
    assert "[EMAIL]" in search.attributes["langfuse.observation.input"]
