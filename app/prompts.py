"""Versioned prompts. The version string is attached to every trace so quality
can be compared per prompt version in Langfuse (and in experiments)."""
from __future__ import annotations

import re

ABSTAIN_PHRASE = "I don't have enough information in the provided documents to answer that."

_OVERVIEW_REQUEST = re.compile(
    r"\b(?:summari[sz](?:e|ing|ation)?|overview)\b"
    r"|\b(?:what|tell me|explain|describe)\b.{0,80}\b(?:document|file|report|pdf|text)\b"
    r"|\bwhat do you understand\b.{0,40}\b(?:document|file|report|pdf|text)\b",
    re.I,
)

PROMPTS: dict[str, str] = {
    "v1": (
        "You are a helpful assistant. Answer the user's question using the provided context. "
        "Cite the sources you used like [1], [2]."
    ),
    "v2": (
        "You answer questions strictly from the numbered context passages provided.\n"
        "Rules:\n"
        "1. Use only facts stated in the context. Never use outside knowledge.\n"
        "2. Cite every factual sentence with the passage number(s), e.g. [1] or [1][3].\n"
        f"3. If the context does not contain the answer, reply exactly: {ABSTAIN_PHRASE}\n"
        "4. If the question is ambiguous and the context supports several readings, answer for each "
        "reading briefly, or ask one short clarifying question.\n"
        "5. For an overview or summary request, synthesize the main subject and important points across "
        "the provided passages. A summary does not need to appear as one sentence in the context.\n"
        "6. Be concise and complete: include every relevant detail asked for (numbers, conditions, exceptions)."
    ),
}

_ABSTAIN_PATTERNS = [
    re.compile(r"don'?t have enough information", re.I),
    re.compile(r"do(es)? not (contain|include|mention|provide|specify)", re.I),
    re.compile(r"(no|not enough) (relevant )?information", re.I),
    re.compile(r"cannot (find|answer)|can'?t (find|answer)", re.I),
    re.compile(r"not (stated|mentioned|specified|covered) in the (provided )?(documents?|context)", re.I),
]


def get_system_prompt(version: str) -> str:
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(f"Unknown prompt version '{version}'. Available: {sorted(PROMPTS)}") from None


def is_document_overview(question: str) -> bool:
    return bool(_OVERVIEW_REQUEST.search(question))


def is_abstention(answer: str) -> bool:
    return any(p.search(answer) for p in _ABSTAIN_PATTERNS)


def build_user_message(question: str, context_blocks: list[str]) -> str:
    return "Context:\n" + "\n\n".join(context_blocks) + f"\n\nQuestion: {question}"
