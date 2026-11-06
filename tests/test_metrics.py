"""Metric semantics: input requirements, 'not evaluated' (never zero), errors, deterministic checks."""
from __future__ import annotations

import asyncio

import pytest

from evaluation.deterministic import (
    abstention_correctness, citation_correctness, citation_coverage, ref_context_hit_rate, ref_context_mrr,
    unsupported_claim_rate,
)
from evaluation.metrics import (
    ALL_METRICS, ERROR, EVALUATED, NOT_EVALUATED, ONLINE_METRICS, REGISTRY, Sample, metric_table_markdown,
)
from evaluation.ragas_backend import RagasBackend
from evaluation.scoring import Scorer

CTX = ["Passwords must be at least 14 characters long.", "All data at rest is encrypted with AES-256."]


def const(value):
    async def f(sample):
        return value
    return f


def make_scorer(settings, **scorers):
    return Scorer(settings, RagasBackend(settings, scorers=scorers))


def score(scorer, sample, **kw):
    return {r.name: r for r in asyncio.run(scorer.score(sample, **kw))}


def sample(**kw):
    base = dict(question="What is the minimum password length?", answer="Passwords must be at least 14 characters long. [1]",
                contexts=CTX, abstained=False)
    base.update(kw)
    return Sample(**base)


ALL_RAGAS = {n: const(0.9) for n in ALL_METRICS if REGISTRY[n].kind == "ragas"} | {"answer_completeness": const(0.8)}


def test_online_metrics_need_no_gold_labels():
    assert {"faithfulness", "answer_relevancy", "citation_correctness", "citation_coverage",
            "unsupported_claim_rate", "hallucination_rate"} <= set(ONLINE_METRICS)
    for gold_only in ("context_precision", "context_recall", "answer_correctness", "noise_sensitivity",
                      "abstention_correctness", "ref_context_hit_rate"):
        assert gold_only not in ONLINE_METRICS


def test_every_required_metric_is_registered():
    required = {"faithfulness", "answer_relevancy", "answer_correctness", "semantic_similarity", "context_precision",
                "context_recall", "context_entity_recall", "noise_sensitivity", "citation_correctness",
                "citation_coverage", "answer_completeness", "abstention_correctness", "hallucination_rate"}
    assert required <= set(REGISTRY)
    table = metric_table_markdown()
    assert all(f"`{n}`" in table for n in REGISTRY)


def test_missing_reference_is_not_evaluated_and_never_zero(settings):
    out = score(make_scorer(settings, **ALL_RAGAS), sample())  # no reference / reference_contexts
    for name in ("answer_correctness", "semantic_similarity", "context_precision", "context_recall",
                 "context_entity_recall", "noise_sensitivity", "answer_completeness", "ref_context_hit_rate"):
        assert out[name].status == NOT_EVALUATED and out[name].value is None, name
        assert "missing input" in out[name].reason
    assert out["faithfulness"].status == EVALUATED and out["faithfulness"].value == 0.9


def test_unanswerable_question_skips_reference_metrics(settings):
    s = sample(answer="I don't have enough information in the provided documents to answer that.", abstained=True,
               reference="The documents do not contain this information.", answerable=False,
               expected_behavior="abstain", reference_contexts=[])
    out = score(make_scorer(settings, **ALL_RAGAS), s)
    assert out["answer_correctness"].status == NOT_EVALUATED
    assert out["abstention_correctness"].value == 1.0  # abstained when it should


def test_abstention_skips_claim_metrics_but_wrong_abstention_scores_low_on_correctness(settings):
    s = sample(answer="I don't have enough information in the provided documents to answer that.", abstained=True,
               reference="At least 14 characters.", answerable=True, expected_behavior="answer",
               reference_contexts=[CTX[0]])
    out = score(make_scorer(settings, **{**ALL_RAGAS, "answer_correctness": const(0.0)}), s)
    assert out["faithfulness"].status == NOT_EVALUATED  # no claims to check
    assert out["answer_correctness"].status == EVALUATED and out["answer_correctness"].value == 0.0  # real failure is visible
    assert out["abstention_correctness"].value == 0.0


def test_no_judge_llm_marks_llm_metrics_not_evaluated(settings):
    out = score(Scorer(settings), sample(reference="14", reference_contexts=[CTX[0]], answerable=True, expected_behavior="answer"))
    assert out["faithfulness"].status == NOT_EVALUATED and "no judge" in out["faithfulness"].reason
    assert out["citation_coverage"].status == EVALUATED  # deterministic metrics still run
    assert out["ref_context_hit_rate"].value == 1.0


def test_metric_exception_is_an_error_not_a_zero(settings):
    async def boom(s):
        raise TimeoutError("judge timed out")

    out = score(make_scorer(settings, faithfulness=boom, answer_relevancy=const(0.8)), sample(), online=True)
    assert out["faithfulness"].status == ERROR and out["faithfulness"].value is None
    assert "judge timed out" in out["faithfulness"].reason
    assert out["hallucination_rate"].status == NOT_EVALUATED  # cannot derive from a failed metric
    assert out["answer_relevancy"].value == 0.8


def test_nan_is_an_error(settings):
    out = score(make_scorer(settings, faithfulness=const(float("nan"))), sample(), metrics=["faithfulness"])
    assert out["faithfulness"].status == ERROR


def test_hallucination_rate_is_one_minus_faithfulness(settings):
    out = score(make_scorer(settings, faithfulness=const(0.75)), sample(), metrics=["faithfulness", "hallucination_rate"])
    assert out["hallucination_rate"].value == pytest.approx(0.25)


def test_unknown_metric_rejected(settings):
    with pytest.raises(ValueError):
        asyncio.run(Scorer(settings).score(sample(), metrics=["nope"]))


# ---- deterministic metrics -----------------------------------------------------------------
def test_citation_marker_after_the_period_is_attached_to_its_sentence():
    s = sample(answer="Passwords must be at least 14 characters long. [1]")
    assert citation_correctness(s).value == 1.0
    assert citation_coverage(s).value == 1.0


def test_citation_to_nonexistent_or_unsupportive_passage_is_penalised():
    assert citation_correctness(sample(answer="Passwords must be at least 14 characters long. [5]")).value == 0.0
    assert citation_correctness(sample(answer="Passwords must be at least 14 characters long. [2]")).value == 0.0
    mixed = sample(answer="Passwords must be at least 14 characters long [1]. Data at rest uses AES-256 [1].")
    assert citation_correctness(mixed).value == 0.5


def test_missing_citations_hurt_coverage_but_correctness_is_not_evaluated():
    s = sample(answer="Passwords must be at least 14 characters long.")
    assert citation_coverage(s).value == 0.0
    assert citation_correctness(s).status == NOT_EVALUATED


def test_multi_citation_markers():
    s = sample(answer="Passwords need 14 characters long, data at rest uses AES-256 [1, 2].")
    assert citation_coverage(s).value == 1.0


def test_unsupported_claim_rate_flags_invented_numbers_and_names():
    grounded = sample(answer="At least 14 characters. [1]")
    assert unsupported_claim_rate(grounded).value == 0.0
    invented = sample(answer="Passwords need 16 characters and are checked by Okta. [1]")
    assert unsupported_claim_rate(invented).value == 1.0
    assert unsupported_claim_rate(sample(answer="No numbers here at all")).status == NOT_EVALUATED


def test_number_formatting_is_normalised():
    s = sample(answer="The budget is 1,200 euros. [1]", contexts=["Each employee has a learning budget of 1200 euros."])
    assert unsupported_claim_rate(s).value == 0.0


def test_abstention_correctness_matrix():
    assert abstention_correctness(sample(abstained=True, expected_behavior="abstain")).value == 1.0
    assert abstention_correctness(sample(abstained=False, expected_behavior="abstain")).value == 0.0
    assert abstention_correctness(sample(abstained=True, expected_behavior="answer")).value == 0.0
    assert abstention_correctness(sample(abstained=False, expected_behavior="answer")).value == 1.0


def test_reference_context_hit_rate_and_mrr():
    s = sample(contexts=["unrelated text", CTX[0], CTX[1]], reference_contexts=[CTX[0], CTX[1]])
    assert ref_context_hit_rate(s).value == 1.0
    assert ref_context_mrr(s).value == pytest.approx(0.5)  # first hit at rank 2
    miss = sample(contexts=["unrelated text"], reference_contexts=[CTX[0]])
    assert ref_context_hit_rate(miss).value == 0.0 and ref_context_mrr(miss).value == 0.0
