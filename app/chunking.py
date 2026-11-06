"""Chunking: line/sentence-aware greedy packing with character overlap.

Units are lines (so list-style and PDF text both work). Lines longer than
`chunk_size` are split on sentence boundaries, then hard-split as a last resort.
Overlap is carried over as whole trailing units, capped at `overlap` chars.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    index: int
    page: Optional[int] = None


def _units(text: str, size: int) -> list[str]:
    units: list[str] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if len(line) <= size:
            units.append(line)
            continue
        buf = ""
        for sent in _SENTENCE.split(line):
            if len(sent) > size:  # pathological: no sentence breaks
                if buf:
                    units.append(buf)
                    buf = ""
                units.extend(sent[i : i + size] for i in range(0, len(sent), size))
                continue
            if buf and len(buf) + 1 + len(sent) > size:
                units.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf)
    return units


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    units = _units(text, chunk_size)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and length + added > chunk_size:
            chunks.append("\n".join(current))
            # carry overlap: trailing units up to `overlap` chars
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            carried_len = sum(len(u) for u in carry) + max(len(carry) - 1, 0)
            if carry and carried_len + 1 + len(unit) > chunk_size:
                carry, carried_len = [], 0  # overlap would push the chunk past chunk_size: drop it
            current = carry
            length = carried_len
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        length += added
    if current:
        tail = "\n".join(current)
        if not chunks or tail != chunks[-1]:
            chunks.append(tail)
    return chunks


def chunk_pages(pages: list[tuple[Optional[int], str]], chunk_size: int, overlap: int) -> list[Chunk]:
    """pages: [(page_number or None, text)] -> ordered chunks with global index."""
    out: list[Chunk] = []
    for page, text in pages:
        for piece in split_text(text, chunk_size, overlap):
            out.append(Chunk(text=piece, index=len(out), page=page))
    return out
