"""PII / secret redaction applied before text leaves the process.

Used (1) as the Langfuse `mask` function, so traces never contain raw emails,
phone numbers or key-like strings, and (2) before payloads are written to the
local online-evaluation queue.
"""
from __future__ import annotations

import re
from typing import Any

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b(?:sk|pk|lf|rk)[-_][A-Za-z0-9_-]{16,}\b"), "[API_KEY]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD_OR_ID_NUMBER]"),
    (re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d"), "[PHONE]"),
]


def redact_text(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact(value: Any) -> Any:
    """Recursively redact strings inside lists/dicts. Safe on any input."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v) for v in value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


def langfuse_mask(*, data: Any, **_: Any) -> Any:
    """Signature expected by Langfuse(mask=...)."""
    try:
        return redact(data)
    except Exception:  # never break tracing because of masking
        return "[REDACTION_FAILED]"
