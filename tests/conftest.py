from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

import app.observability as obs  # noqa: E402
from app.config import Settings  # noqa: E402
from evaluation.experiment import ExperimentConfig, build_pipeline  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Offline settings: fake providers, no Langfuse, isolated data dir."""
    return Settings(_env_file=None, provider_mode="fake", data_dir=tmp_path, redact_pii=True)


@pytest.fixture
def traced_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None, provider_mode="fake", data_dir=tmp_path,
        # Langfuse caches client resources per public key; a unique key per test keeps tests isolated.
        langfuse_public_key=f"pk-lf-{uuid.uuid4().hex[:12]}", langfuse_secret_key="sk-lf-test",
        langfuse_host="http://127.0.0.1:9",  # unreachable on purpose: spans are captured in memory
    )


@pytest.fixture
def exporter(traced_settings, monkeypatch):
    """Real Langfuse SDK, with spans captured by an in-memory OpenTelemetry exporter."""
    exp = InMemorySpanExporter()
    sent_scores: list[dict] = []

    def fake_create_score(**kwargs):
        sent_scores.append(kwargs)
        return True

    # Scores go over HTTP in Langfuse; capture them instead of calling the dummy host.
    monkeypatch.setattr("app.rag.create_score", fake_create_score)
    monkeypatch.setattr("evaluation.online_worker.create_score", fake_create_score)
    obs.init_langfuse(traced_settings, span_exporter=exp, force=True)
    exp.scores = sent_scores  # type: ignore[attr-defined]
    yield exp
    obs.shutdown_langfuse()


@pytest.fixture(autouse=True)
def _reset_langfuse():
    yield
    obs._client = None


@pytest.fixture
def pipeline(settings):
    return build_pipeline(settings, ExperimentConfig(name="test"))


@pytest.fixture
def traced_pipeline(traced_settings, exporter):
    return build_pipeline(traced_settings, ExperimentConfig(name="test", hybrid=True, reranker="lexical"))
