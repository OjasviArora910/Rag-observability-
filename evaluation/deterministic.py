"""Cheap, deterministic metrics that need no LLM: citations, unsupported
numbers/terms, abstention, and gold-passage retrieval hit-rate / MRR.

These are intentionally simple and documented as heuristics; the LLM-judged
Ragas metrics are the primary quality signal, these are guard-rails that are
free to run on every request and fully reproducible.
"""
from __future__ import annotations

import re
from typing import Optional

from app.llm import content_tokens
from evaluation.metrics import ScoreResult, Sample

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CAP_TERM = re.compile(r"\b[A-Z][A-Za-z0-9-]{2,}\b")
_WORD = re.compile(r"[a-z0-9]+")


def _sentences(answer: str) -> list[str]:
    out: list[str] = []
    for raw in _SENT_SPLIT.split(answer):
        s = _BULLET.sub("", raw).strip()
        if not s:
            continue
        # "...per year. [1]" splits the marker into its own fragment; re-attach it.
        if out and not _MARKER.sub("", s).strip():
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def _cited(sentence: str) -> list[int]:
    nums: list[int] = []
    for m in _MARKER.finditer(sentence):
        nums.extend(int(x) for x in re.findall(r"\d+", m.group(0)))
    return nums


def _strip_markers(sentence: str) -> str:
    return _MARKER.sub("", sentence).strip()


def _supported(sentence: str, context: str, threshold: float = 0.5) -> bool:
    claim = content_tokens(_strip_markers(sentence))
    if not claim:
        return False
    return len(claim & content_tokens(context)) / len(claim) >= threshold


def citation_correctness(s: Sample) -> ScoreResult:
    total = valid = 0
    for sent in _sentences(s.answer):
        for n in _cited(sent):
            total += 1
            if 1 <= n <= len(s.contexts) and _supported(sent, s.contexts[n - 1]):
                valid += 1
    if total == 0:
        return ScoreResult.skip("citation_correctness", "answer contains no citations (see citation_coverage)")
    return ScoreResult.ok("citation_correctness", valid / total)


def citation_coverage(s: Sample) -> ScoreResult:
    factual = [x for x in _sentences(s.answer) if len(content_tokens(_strip_markers(x))) >= 3]
    if not factual:
        return ScoreResult.skip("citation_coverage", "no factual sentences in answer")
    covered = 0
    for sent in factual:
        if any(1 <= n <= len(s.contexts) for n in _cited(sent)):
            covered += 1
    return ScoreResult.ok("citation_coverage", covered / len(factual))


def _norm_number(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def unsupported_claim_rate(s: Sample) -> ScoreResult:
    answer = _MARKER.sub(" ", s.answer)
    ctx = " ".join(s.contexts)
    ctx_numbers = {_norm_number(n) for n in _NUMBER.findall(ctx)}
    ctx_words = set(_WORD.findall(ctx.lower()))

    numbers = {_norm_number(n) for n in _NUMBER.findall(answer)}
    # capitalised terms that are not sentence-initial words
    terms: set[str] = set()
    for sent in _SENT_SPLIT.split(answer):
        words = sent.split()
        for w in words[1:]:
            for m in _CAP_TERM.findall(w):
                terms.add(m.lower())
    checkable = len(numbers) + len(terms)
    if checkable == 0:
        return ScoreResult.skip("unsupported_claim_rate", "no numbers or named terms to verify")
    bad = sum(1 for n in numbers if n not in ctx_numbers) + sum(1 for t in terms if t not in ctx_words)
    return ScoreResult.ok("unsupported_claim_rate", bad / checkable)


def abstention_correctness(s: Sample) -> ScoreResult:
    should_abstain = s.expected_behavior == "abstain"
    return ScoreResult.ok("abstention_correctness", 1.0 if bool(s.abstained) == should_abstain else 0.0)


def _token_recall(ref: str, chunk: str) -> float:
    r = set(_WORD.findall(ref.lower()))
    if not r:
        return 0.0
    return len(r & set(_WORD.findall(chunk.lower()))) / len(r)


def _hits(s: Sample, threshold: float = 0.8) -> tuple[float, Optional[int]]:
    refs = s.reference_contexts or []
    found = 0
    first_rank: Optional[int] = None
    for ref in refs:
        best = None
        for rank, chunk in enumerate(s.contexts, start=1):
            if _token_recall(ref, chunk) >= threshold:
                best = rank
                break
        if best is not None:
            found += 1
            first_rank = best if first_rank is None else min(first_rank, best)
    return found / len(refs), first_rank


def ref_context_hit_rate(s: Sample) -> ScoreResult:
    return ScoreResult.ok("ref_context_hit_rate", _hits(s)[0])


def ref_context_mrr(s: Sample) -> ScoreResult:
    _, first = _hits(s)
    return ScoreResult.ok("ref_context_mrr", 0.0 if first is None else 1.0 / first)


DETERMINISTIC = {
    "citation_correctness": citation_correctness,
    "citation_coverage": citation_coverage,
    "unsupported_claim_rate": unsupported_claim_rate,
    "abstention_correctness": abstention_correctness,
    "ref_context_hit_rate": ref_context_hit_rate,
    "ref_context_mrr": ref_context_mrr,
}
