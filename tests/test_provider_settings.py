from __future__ import annotations

from pydantic import ValidationError
import pytest

from app.config import Settings
from app.embeddings import OpenAIEmbedder
from app.llm import OpenAICompatibleChat


def test_groq_provider_is_case_insensitive_and_uses_groq_endpoint():
    settings = Settings(_env_file=None, llm_provider="Groq")

    assert settings.llm_provider == "groq"
    assert settings.resolved_llm_base_url == "https://api.groq.com/openai/v1"


def test_default_provider_mode_is_keyless():
    settings = Settings(_env_file=None)

    assert settings.provider_mode == "fake"


def test_groq_provider_rejects_openai_endpoint():
    with pytest.raises(ValidationError, match="cannot use the OpenAI endpoint"):
        Settings(
            _env_file=None,
            llm_provider="Groq",
            llm_base_url="https://api.openai.com/v1",
        )


def test_chat_and_embedding_clients_use_separate_credentials(monkeypatch):
    clients = []

    def fake_openai(**kwargs):
        clients.append(kwargs)
        return object()

    monkeypatch.setattr("openai.OpenAI", fake_openai)
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        llm_api_key="groq-test-key",
        embedding_api_key="embedding-test-key",
    )

    OpenAICompatibleChat(settings)
    OpenAIEmbedder(settings)

    assert clients == [
        {
            "api_key": "groq-test-key",
            "base_url": "https://api.groq.com/openai/v1",
            "max_retries": 0,
            "timeout": 20.0,
        },
        {
            "api_key": "embedding-test-key",
            "base_url": None,
            "max_retries": 0,
        },
    ]
