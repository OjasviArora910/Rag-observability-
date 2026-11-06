"""Embedding providers.

* OpenAIEmbedder  - real embeddings through the OpenAI-compatible API.
* HashingEmbedder - deterministic bag-of-words hashing vectors, offline only.
  Good enough to exercise retrieval + the evaluation plumbing in tests; its
  quality numbers say nothing about production quality.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    tokens: int


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> EmbedResult: ...


class HashingEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> EmbedResult:
        return EmbedResult([self._vec(t) for t in texts], sum(len(_TOKEN.findall(t)) for t in texts))


class OpenAIEmbedder:
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, settings: Settings, model: str | None = None, batch_size: int = 64):
        from openai import OpenAI

        if not settings.embedding_api_key:
            raise RuntimeError(
                "EMBEDDING_API_KEY is not set. Configure an embeddings API separately; "
                "Groq's chat endpoint does not provide embeddings "
                "(or use PROVIDER_MODE=fake for offline smoke tests)."
            )
        self.model_name = model or settings.embedding_model
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            max_retries=0,
        )
        self._batch = batch_size
        self._max_attempts = settings.llm_max_attempts
        self.dim = self._DIMS.get(self.model_name) or len(self.embed(["dimension probe"]).vectors[0])

    def _call(self, batch: list[str]):
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.embeddings.create(model=self.model_name, input=batch)
            except Exception:
                if attempt == self._max_attempts:
                    raise
                time.sleep(min(2 ** attempt, 20))

    def embed(self, texts: list[str]) -> EmbedResult:
        vectors: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), self._batch):
            resp = self._call(texts[i : i + self._batch])
            vectors.extend(d.embedding for d in resp.data)
            tokens += getattr(resp.usage, "total_tokens", 0) or 0
        return EmbedResult(vectors, tokens)


class LocalEmbedder:
    """On-device ONNX embeddings via fastembed. No API key; the model (~130 MB)
    is downloaded once and cached under DATA_DIR."""

    def __init__(self, settings: Settings, model: str | None = None):
        try:
            import truststore  # use the OS certificate store (corporate proxies)

            truststore.inject_into_ssl()
        except ImportError:
            pass
        from fastembed import TextEmbedding

        self.model_name = model or settings.local_embedding_model
        cache = settings.data_dir / "models"
        cache.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(self.model_name, cache_dir=str(cache))
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> EmbedResult:
        vectors = [v.tolist() for v in self._model.embed(texts)]
        return EmbedResult(vectors, sum(len(_TOKEN.findall(t)) for t in texts))


def _has_real_key(key: str | None) -> bool:
    return bool(key) and not key.strip().startswith("<")


def get_embedder(settings: Settings, model: str | None = None) -> Embedder:
    provider = settings.embedding_provider
    if provider == "auto":
        if settings.provider_mode == "fake":
            provider = "hashing"
        else:
            provider = "openai" if _has_real_key(settings.embedding_api_key) else "local"
    if provider == "hashing":
        return HashingEmbedder()
    if provider == "local":
        return LocalEmbedder(settings)
    return OpenAIEmbedder(settings, model)
