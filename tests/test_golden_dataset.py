"""Integrity checks for the version-controlled golden dataset."""
from collections import Counter

from evaluation.experiment import CORPUS_DIR, GOLDEN_PATH, load_golden
from evaluation.golden.build_golden import build

REQUIRED_TYPES = {"factual", "exact_match", "multi_hop", "ambiguous", "unanswerable"}


def _corpus_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md")))


def test_size_and_type_coverage():
    rows = load_golden()
    assert 60 <= len(rows) <= 100
    counts = Counter(r["type"] for r in rows)
    assert REQUIRED_TYPES <= set(counts)
    assert all(counts[t] >= 5 for t in REQUIRED_TYPES)


def test_ids_unique_and_questions_unique():
    rows = load_golden()
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["question"] for r in rows}) == len(rows)


def test_reference_contexts_are_verbatim_corpus_text():
    corpus = _corpus_text()
    for r in load_golden():
        for ctx in r["reference_contexts"]:
            assert ctx in corpus, f"{r['id']}: gold passage not found verbatim in corpus: {ctx!r}"


def test_answerable_items_have_gold_passages_and_answers():
    for r in load_golden():
        assert r["reference_answer"].strip()
        if r["type"] == "unanswerable":
            assert r["answerable"] is False and r["reference_contexts"] == [] and r["expected_behavior"] == "abstain"
        elif r["question"] != "What is the limit?":  # deliberately vague: clarification expected, no gold passage
            assert r["answerable"] is True and r["reference_contexts"], r["id"]


def test_multi_hop_items_need_several_passages_or_reasoning():
    multi = [r for r in load_golden() if r["type"] == "multi_hop"]
    assert multi and sum(len(r["reference_contexts"]) >= 2 for r in multi) >= len(multi) // 2


def test_jsonl_is_in_sync_with_builder():
    """Fails if someone edits build_golden.py/corpus without regenerating the JSONL."""
    assert load_golden(GOLDEN_PATH) == build()
