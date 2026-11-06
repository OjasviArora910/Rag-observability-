"""Chat LLM providers: OpenAI-compatible client with visible retries, and a
deterministic extractive FakeChat for offline smoke tests."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from app.config import Settings
from app.prompts import ABSTAIN_PHRASE

logger = logging.getLogger("rag.llm")

_STOP = set(
    "a an the of to in on for and or is are was were be been what which who whom how when where why do does did "
    "i me my we our you your it its this that these those with at by from as if can could would should will "
    "much many long often per".split()
)
_TOK = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def content_tokens(text: str) -> set[str]:
    return {_stem(t) for t in _TOK.findall(text.lower()) if t not in _STOP}


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0


class ChatLLM(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse: ...


class OpenAICompatibleChat:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        if not settings.llm_api_key:
            raise RuntimeError(
                f"LLM_API_KEY is not set for LLM_PROVIDER={settings.llm_provider} "
                "(or use PROVIDER_MODE=fake for offline smoke tests)."
            )
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.resolved_llm_base_url,
            max_retries=0,
            timeout=settings.llm_timeout_seconds,
        )
        self._model = settings.llm_model
        self._max_attempts = settings.llm_max_attempts

    def complete(self, messages, *, model=None, temperature=0.0, json_mode=False) -> LLMResponse:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
        model = model or self._model
        kwargs: dict = dict(model=model, messages=messages, temperature=temperature)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        retries = 0
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                usage = resp.usage
                return LLMResponse(
                    text=resp.choices[0].message.content or "",
                    model=resp.model or model,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    retries=retries,
                )
            except retryable as exc:
                if attempt == self._max_attempts:
                    raise
                retries += 1
                delay = min(2**attempt, 20)
                logger.warning("LLM call failed (%s); retry %d in %ss", type(exc).__name__, retries, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover


_BLOCK = re.compile(r"\[(\d+)\] \(([^)]*)\) (.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:)", re.S)


class FakeChat:
    """Extractive stand-in: answers with the best-matching context line + its
    citation, or abstains. Deterministic; used for tests and keyless demos."""

    def __init__(self, min_overlap: float = 0.5):
        self._min = min_overlap

    def complete(self, messages, *, model=None, temperature=0.0, json_mode=False) -> LLMResponse:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        question = user.rsplit("Question:", 1)[-1].strip()
        q = content_tokens(question)
        best: tuple[float, str, str] = (0.0, "", "")
        for num, _src, text in _BLOCK.findall(user):
            for line in text.splitlines():
                overlap = len(q & content_tokens(line)) / (len(q) or 1)
                if overlap > best[0]:
                    best = (overlap, num, line.strip())
        if best[0] >= self._min:
            answer = f"{best[2]} [{best[1]}]"
        else:
            answer = ABSTAIN_PHRASE
        return LLMResponse(
            text=answer,
            model=model or "fake-extractive",
            input_tokens=len(user.split()),
            output_tokens=len(answer.split()),
        )


def get_chat(settings: Settings) -> ChatLLM:
    return FakeChat() if settings.provider_mode == "fake" else OpenAICompatibleChat(settings)
