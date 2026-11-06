"""Central configuration. Everything is read from environment variables / .env.

No credentials live in code. See .env.example for the full list.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    app_name: str = "RAG Observability Chatbot"
    log_level: str = "INFO"

    # --- providers -------------------------------------------------------
    # "openai" uses the OpenAI-compatible client against the configured LLM provider.
    # "fake" is for offline smoke tests, not meaningful quality numbers.
    provider_mode: Literal["openai", "fake"] = "fake"
    llm_provider: Literal["openai", "groq"] = "groq"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None
    llm_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = Field(default=20.0, gt=0)
    embedding_model: str = "text-embedding-3-small"
    # auto: fake mode -> hashing; otherwise OpenAI if a real EMBEDDING_API_KEY is set, else local (no key needed).
    embedding_provider: Literal["auto", "openai", "local", "hashing"] = "auto"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    temperature: float = 0.0
    llm_max_attempts: int = 3

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def resolved_llm_base_url(self) -> Optional[str]:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "groq":
            return "https://api.groq.com/openai/v1"
        return None

    @model_validator(mode="after")
    def validate_llm_endpoint(self):
        if self.llm_provider == "groq" and self.llm_base_url:
            host = urlparse(self.llm_base_url).hostname
            if host == "api.openai.com":
                raise ValueError(
                    "LLM_PROVIDER=groq cannot use the OpenAI endpoint; "
                    "set LLM_BASE_URL=https://api.groq.com/openai/v1"
                )
        return self

    # --- RAG configuration (all overridable per experiment) --------------
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 4
    hybrid: bool = False
    reranker: Literal["none", "lexical"] = "none"
    candidate_multiplier: int = 3  # fetch top_k * N candidates before fusion/rerank
    max_context_chars: int = 12000
    prompt_version: str = "v2"

    # --- storage ----------------------------------------------------------
    data_dir: Path = Path("data")
    qdrant_url: Optional[str] = None  # use a Qdrant server in production
    qdrant_path: Optional[str] = None  # embedded mode (single process only)
    collection_name: str = "rag_chunks"

    # --- Langfuse ---------------------------------------------------------
    langfuse_enabled: bool = True
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None  # e.g. https://cloud.langfuse.com
    langfuse_environment: str = "development"
    langfuse_release: Optional[str] = None

    # --- online evaluation -------------------------------------------------
    online_eval_enabled: bool = True
    online_eval_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    eval_db_path: Optional[str] = None
    eval_max_attempts: int = 3
    eval_retry_base_seconds: float = 30.0
    eval_poll_interval_seconds: float = 2.0
    judge_model: str = "openai/gpt-oss-20b"
    judge_embedding_model: str = "text-embedding-3-small"

    # --- privacy / cost -----------------------------------------------------
    redact_pii: bool = True
    # JSON: {"gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60}}
    # Used only to *estimate* cost; Langfuse model pricing takes precedence in the UI.
    pricing_json: Optional[str] = None

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def resolved_qdrant_path(self) -> str:
        return self.qdrant_path or str(self.data_dir / "qdrant")

    @property
    def resolved_eval_db(self) -> str:
        return self.eval_db_path or str(self.data_dir / "eval_queue.sqlite3")

    @property
    def documents_db(self) -> str:
        return str(self.data_dir / "documents.sqlite3")

    def pricing(self) -> dict:
        defaults = {
            # Estimates only - override with PRICING_JSON for your contract/model.
            "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
            "text-embedding-3-small": {"input_per_1m": 0.02, "output_per_1m": 0.0},
        }
        if self.pricing_json:
            try:
                defaults.update(json.loads(self.pricing_json))
            except json.JSONDecodeError:
                pass
        return defaults

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Optional[dict]:
        price = self.pricing().get(model)
        if not price:
            return None
        inp = input_tokens * price["input_per_1m"] / 1_000_000
        out = output_tokens * price["output_per_1m"] / 1_000_000
        return {"input": inp, "output": out, "total": inp + out}

    def rag_config(self) -> dict:
        """The tunable RAG knobs, as recorded on every trace."""
        return {
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "hybrid": self.hybrid,
            "reranker": self.reranker,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
