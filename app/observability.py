"""Thin, failure-proof wrapper around the Langfuse Python SDK (v4/v5 OTel-based API).

Design rules
------------
* Observability must never break the chatbot: every Langfuse call is guarded.
* If Langfuse is not configured, `observation()` yields a no-op object so the
  pipeline code is identical with and without tracing.
* One place owns client creation (masking, environment, release, exporter).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.config import Settings
from app.privacy import langfuse_mask

logger = logging.getLogger("rag.observability")

_client: Any = None


class _NullObservation:
    """Returned when tracing is disabled; mirrors the tiny API we use."""

    id: Optional[str] = None
    trace_id: Optional[str] = None

    def update(self, **_: Any) -> "_NullObservation":
        return self

    def score_trace(self, **_: Any) -> None:
        return None


def init_langfuse(settings: Settings, *, span_exporter: Any = None, force: bool = False) -> Any:
    """Create (or return) the process-wide Langfuse client.

    `span_exporter` lets tests capture spans in memory instead of sending them.
    """
    global _client
    if _client is not None and not force:
        return _client
    if not settings.langfuse_configured:
        logger.warning("Langfuse is not configured (missing keys) - tracing disabled.")
        _client = None
        return None
    try:
        from langfuse import Langfuse

        kwargs: dict[str, Any] = dict(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            environment=settings.langfuse_environment,
            release=settings.langfuse_release,
            mask=langfuse_mask if settings.redact_pii else None,
        )
        if settings.langfuse_host:
            kwargs["base_url"] = settings.langfuse_host
        if span_exporter is not None:
            kwargs["span_exporter"] = span_exporter
        _client = Langfuse(**{k: v for k, v in kwargs.items() if v is not None})
        logger.info("Langfuse tracing enabled (environment=%s)", settings.langfuse_environment)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to initialise Langfuse; tracing disabled")
        _client = None
    return _client


def get_langfuse() -> Any:
    return _client


def tracing_enabled() -> bool:
    return _client is not None


def shutdown_langfuse() -> None:
    global _client
    if _client is not None:
        try:
            _client.flush()
            _client.shutdown()
        except Exception:  # pragma: no cover
            logger.exception("Langfuse shutdown failed")
    _client = None


@contextmanager
def observation(name: str, *, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Open a Langfuse observation (span/generation/retriever/...) as the current span.

    Exceptions are recorded on the observation (level=ERROR) and re-raised.
    """
    client = _client
    if client is None:
        yield _NullObservation()
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        obs = cm.__enter__()
    except Exception:  # pragma: no cover - defensive
        logger.exception("Could not start observation %s", name)
        yield _NullObservation()
        return

    exc_info = (None, None, None)
    try:
        yield obs
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        try:
            obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
        except Exception:  # pragma: no cover
            pass
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # pragma: no cover
            logger.exception("Could not close observation %s", name)


@contextmanager
def trace_attributes(**kwargs: Any) -> Iterator[None]:
    """Propagate trace-level attributes (session_id, user_id, tags, version, metadata)
    to the current span and all children. No-op when tracing is disabled."""
    if _client is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        # Langfuse requires metadata values to be short strings.
        meta = kwargs.get("metadata")
        if meta:
            kwargs["metadata"] = {str(k): str(v)[:200] for k, v in meta.items()}
        cm = propagate_attributes(**{k: v for k, v in kwargs.items() if v is not None})
        cm.__enter__()
    except Exception:  # pragma: no cover
        logger.exception("propagate_attributes failed")
        yield
        return
    exc_info = (None, None, None)
    try:
        yield
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        try:
            cm.__exit__(*exc_info)
        except Exception:  # pragma: no cover
            pass


def current_trace_id() -> Optional[str]:
    if _client is None:
        return None
    try:
        return _client.get_current_trace_id()
    except Exception:  # pragma: no cover
        return None


def trace_url(trace_id: Optional[str]) -> Optional[str]:
    if _client is None or not trace_id:
        return None
    try:
        return _client.get_trace_url(trace_id=trace_id)
    except Exception:  # pragma: no cover
        return None


def create_score(
    *,
    trace_id: Optional[str],
    name: str,
    value: Any,
    data_type: Optional[str] = None,
    comment: Optional[str] = None,
    metadata: Optional[dict] = None,
    observation_id: Optional[str] = None,
    score_id: Optional[str] = None,
) -> bool:
    """Attach a score to a trace. Returns False (never raises) on failure."""
    if _client is None or not trace_id:
        return False
    try:
        kwargs: dict[str, Any] = dict(name=name, value=value, trace_id=trace_id)
        if data_type:
            kwargs["data_type"] = data_type
        if comment:
            kwargs["comment"] = comment[:1000]
        if metadata:
            kwargs["metadata"] = metadata
        if observation_id:
            kwargs["observation_id"] = observation_id
        if score_id:
            kwargs["score_id"] = score_id  # deterministic id => retries are idempotent upserts
        _client.create_score(**kwargs)
        return True
    except Exception:
        logger.exception("create_score failed for %s on trace %s", name, trace_id)
        return False


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:  # pragma: no cover
            logger.exception("Langfuse flush failed")
