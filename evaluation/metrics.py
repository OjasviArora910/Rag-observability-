"""Metric registry: what each metric is, which inputs it REQUIRES, and when it
is skipped. This is the single source of truth for 'not evaluated' decisions.

Rule: a metric whose inputs are unavailable is reported as NOT_EVALUATED with a
reason. It is never scored as 0 - zeros would silently drag averages down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Bump when scoring logic, prompts or metric definitions change in a way that
# makes new scores non-comparable with old ones.
EVALUATOR_SCHEMA = "rag-eval/1"

EVALUATED, NOT_EVALUATED, ERROR = "evaluated", "not_evaluated", "error"


@dataclass
class Sample:
    """One (question, answer, contexts) observation, online or offline."""

    question: str
    answer: str
    contexts: list[str]
    reference: Optional[str] = None  # gold answer
    reference_contexts: Optional[list[str]] = None  # gold supporting passages
    answerable: Optional[bool] = None  # gold: is the answer in the documents?
    expected_behavior: Optional[str] = None  # answer | abstain | clarify_or_cover_all
    abstained: Optional[bool] = None  # what the system actually did
    cited_numbers: list[int] = field(default_factory=list)


@dataclass
class ScoreResult:
    name: str
    status: str
    value: Optional[float] = None
    reason: Optional[str] = None

    @classmethod
    def ok(cls, name: str, value: float) -> "ScoreResult":
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return cls(name, ERROR, None, "metric returned NaN")
        return cls(name, EVALUATED, float(value))

    @classmethod
    def skip(cls, name: str, reason: str) -> "ScoreResult":
        return cls(name, NOT_EVALUATED, None, reason)

    @classmethod
    def error(cls, name: str, reason: str) -> "ScoreResult":
        return cls(name, ERROR, None, reason[:300])


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str  # "ragas" | "custom"
    group: str  # answer_quality | retrieval_quality | additional
    requires: tuple[str, ...]  # Sample fields that must be present
    judge_llm: bool = False  # needs an LLM judge
    embeddings: bool = False  # needs an embedding model
    skip_if_abstained: bool = False  # answer has no claims to assess
    needs_answerable: bool = False  # only meaningful when gold says answerable
    lower_is_better: bool = False
    description: str = ""

    @property
    def online_capable(self) -> bool:
        """Can run in production, where no gold labels exist."""
        return not ({"reference", "reference_contexts", "expected_behavior"} & set(self.requires))


_SPECS = [
    # ---- Ragas: answer quality ------------------------------------------------
    MetricSpec("faithfulness", "ragas", "answer_quality", ("question", "answer", "contexts"), judge_llm=True,
               skip_if_abstained=True, description="Share of answer claims supported by the retrieved contexts."),
    MetricSpec("answer_relevancy", "ragas", "answer_quality", ("question", "answer"), judge_llm=True,
               embeddings=True, skip_if_abstained=True,
               description="How well the answer addresses the question (generated-question similarity)."),
    MetricSpec("answer_correctness", "ragas", "answer_quality", ("question", "answer", "reference"),
               judge_llm=True, embeddings=True, needs_answerable=True,
               description="Factual overlap + semantic similarity of the answer vs the gold answer."),
    MetricSpec("semantic_similarity", "ragas", "answer_quality", ("answer", "reference"), embeddings=True,
               needs_answerable=True, description="Embedding cosine similarity of answer and gold answer."),
    # ---- Ragas: retrieval quality -------------------------------------------------
    MetricSpec("context_precision", "ragas", "retrieval_quality", ("question", "reference", "contexts"),
               judge_llm=True, needs_answerable=True, description="Are relevant chunks ranked above irrelevant ones?"),
    MetricSpec("context_recall", "ragas", "retrieval_quality", ("question", "reference", "contexts"),
               judge_llm=True, needs_answerable=True, description="Share of gold-answer statements found in the contexts."),
    MetricSpec("context_entity_recall", "ragas", "retrieval_quality", ("reference", "contexts"), judge_llm=True,
               needs_answerable=True, description="Share of gold-answer entities present in the contexts."),
    MetricSpec("noise_sensitivity", "ragas", "retrieval_quality", ("question", "answer", "reference", "contexts"),
               judge_llm=True, skip_if_abstained=True, needs_answerable=True, lower_is_better=True,
               description="How often the answer is wrong because of irrelevant retrieved context (lower is better)."),
    # ---- Additional -------------------------------------------------------------------
    MetricSpec("citation_correctness", "custom", "additional", ("answer", "contexts"), skip_if_abstained=True,
               description="Share of [n] citations that point to an existing passage that lexically supports the cited sentence."),
    MetricSpec("citation_coverage", "custom", "additional", ("answer", "contexts"), skip_if_abstained=True,
               description="Share of factual answer sentences carrying at least one valid citation."),
    MetricSpec("unsupported_claim_rate", "custom", "additional", ("answer", "contexts"), skip_if_abstained=True,
               lower_is_better=True,
               description="Share of numbers/named terms in the answer that do not appear in the contexts (lower is better)."),
    MetricSpec("hallucination_rate", "custom", "additional", ("question", "answer", "contexts"), judge_llm=True,
               skip_if_abstained=True, lower_is_better=True, description="1 - faithfulness (lower is better)."),
    MetricSpec("answer_completeness", "custom", "additional", ("question", "answer", "reference"), judge_llm=True,
               needs_answerable=True, description="Share of the gold answer's key points covered by the answer (LLM judge)."),
    MetricSpec("abstention_correctness", "custom", "additional", ("expected_behavior",),
               description="1 if the system abstained exactly when the gold label says it should."),
    MetricSpec("ref_context_hit_rate", "custom", "retrieval_quality", ("contexts", "reference_contexts"),
               needs_answerable=True, description="Share of gold passages found (>=80% token recall) in a retrieved chunk."),
    MetricSpec("ref_context_mrr", "custom", "retrieval_quality", ("contexts", "reference_contexts"),
               needs_answerable=True, description="Reciprocal rank of the first retrieved chunk matching a gold passage."),
]

REGISTRY: dict[str, MetricSpec] = {m.name: m for m in _SPECS}
ONLINE_METRICS = [m.name for m in _SPECS if m.online_capable]
ALL_METRICS = [m.name for m in _SPECS]

_FIELD_ALIASES = {"question": "question", "answer": "answer", "contexts": "contexts"}


def missing_inputs(spec: MetricSpec, s: Sample) -> list[str]:
    missing = []
    for f in spec.requires:
        v = getattr(s, f, None)
        if v is None or (isinstance(v, (str, list)) and len(v) == 0):
            missing.append(f)
    return missing


def applicability(spec: MetricSpec, s: Sample) -> Optional[str]:
    """Return a reason string if the metric must NOT be scored for this sample."""
    miss = missing_inputs(spec, s)
    if miss:
        return "missing input: " + ", ".join(miss)
    if spec.needs_answerable and s.answerable is False:
        return "unanswerable question: no gold answer to compare against"
    if spec.name == "abstention_correctness" and s.expected_behavior not in {"answer", "abstain"}:
        return f"expected_behavior '{s.expected_behavior}' has no single correct abstention outcome"
    if spec.skip_if_abstained and s.abstained:
        return "answer is an abstention: no claims to assess"
    return None


def metric_table_markdown() -> str:
    rows = ["| Metric | Kind | Required inputs | Judge LLM | Embeddings | Online? | Notes |", "|---|---|---|---|---|---|---|"]
    for m in _SPECS:
        rows.append(
            f"| `{m.name}` | {m.kind} | {', '.join(m.requires)} | {'yes' if m.judge_llm else 'no'} | "
            f"{'yes' if m.embeddings else 'no'} | {'yes' if m.online_capable else 'offline only'} | {m.description} |"
        )
    return "\n".join(rows)
