# Metrics reference

Generated table below comes from `evaluation/metrics.py` (`python -c "from evaluation.metrics import metric_table_markdown as t; print(t())"`),
which is also what the code enforces, so this page cannot drift from behaviour.

## The "not evaluated" rule

A metric whose required inputs are unavailable is reported as **not evaluated** with a reason. It is **never scored as 0**,
because a zero is indistinguishable from "the answer was bad" and silently drags averages down.

Each metric ends in exactly one state:

| State | Meaning | In Langfuse | In experiment reports |
|---|---|---|---|
| `evaluated` | numeric score | one numeric score on the trace | counted in `n_evaluated` and the mean |
| `not_evaluated` | inputs missing / not applicable (reason recorded) | **no score is written**; listed in the `online_eval_status` score metadata | counted in `n_not_evaluated` |
| `error` | the metric ran and failed (timeout, malformed judge output, NaN) | no score; the job is retried | counted in `n_errors` with reasons |

Cases that produce `not_evaluated` by design:

* **No gold labels online.** Production traffic has no reference answer, so `answer_correctness`, `semantic_similarity`,
  `context_precision`, `context_recall`, `context_entity_recall`, `noise_sensitivity`, `answer_completeness`, `abstention_correctness`,
  `ref_context_hit_rate` and `ref_context_mrr` are offline-only.
* **Unanswerable questions** have no gold answer, so reference-comparison metrics are skipped; `abstention_correctness` is the
  metric that judges them.
* **Abstained answers** contain no claims, so claim-based metrics (`faithfulness`, `answer_relevancy`, `noise_sensitivity`, citation metrics,
  `unsupported_claim_rate`, `hallucination_rate`) are skipped. But if the gold label says the question *was* answerable, `answer_correctness`,
  `semantic_similarity` and `answer_completeness` still run, so a wrong abstention shows up as a real failure instead of disappearing.
* **Ambiguous questions** (`expected_behavior = clarify_or_cover_all`) have no single correct abstain/answer outcome, so `abstention_correctness` is skipped.
* **No judge LLM configured** (e.g. `PROVIDER_MODE=fake`): LLM-judged metrics are `not_evaluated`, deterministic ones still run.

## Sample fields

| Field | Meaning | Where it comes from |
|---|---|---|
| `question` | user question | trace input / golden `question` |
| `answer` | generated answer | trace output |
| `contexts` | retrieved passages actually placed in the prompt | retrieval + context-assembly |
| `reference` | gold answer | golden `reference_answer` (offline) |
| `reference_contexts` | gold supporting passages (verbatim corpus text) | golden `reference_contexts` (offline) |
| `answerable` / `expected_behavior` | gold: is the answer in the documents; what the system should do | golden (offline) |
| `abstained` | what the system actually did | pipeline (abstention phrase detection or empty context) |

## Notes on individual metrics

* **Ragas metrics** use `ragas.metrics.collections` (Ragas 0.4.x): `Faithfulness`, `AnswerRelevancy`, `AnswerCorrectness`,
  `SemanticSimilarity`, `ContextPrecision`, `ContextRecall`, `ContextEntityRecall`, `NoiseSensitivity`. They call the judge model
  (`JUDGE_MODEL`, `JUDGE_EMBEDDING_MODEL`). In Ragas 0.4 `ContextPrecision`/`ContextRecall`/`ContextEntityRecall` compare against the **reference answer**;
  `reference_contexts` is used by the deterministic `ref_context_*` metrics instead.
* **`citation_correctness`** is a lexical heuristic: a citation `[n]` is correct if passage *n* exists and shares at least half of the
  content words of the sentence that cites it. It catches phantom/mismatched citations cheaply; it does not understand paraphrase.
* **`citation_coverage`** counts factual sentences (>= 3 content words) carrying at least one valid citation.
* **`unsupported_claim_rate`** checks that every number and capitalised term in the answer appears in the retrieved text (thousand separators normalised). It is a fast
  hallucination tripwire for numeric/entity facts, complementary to Ragas `faithfulness`.
* **`hallucination_rate`** = 1 - `faithfulness` (same judge call, no extra cost).
* **`answer_completeness`** is LLM-judged: the judge counts how many key facts of the reference the answer covers.
* **`ref_context_hit_rate` / `ref_context_mrr`** are chunking-independent: a gold passage counts as retrieved when >= 80% of its tokens appear in one retrieved chunk.
  This makes them comparable across chunk-size experiments.
* **Judge bias:** use a judge model different from (or at least not weaker than) the answering model, and re-validate the judge on a handful of hand-labelled items before trusting trends.

## Evaluator versioning

Every score carries `evaluator_version = rag-eval/<schema>|ragas=<version>|judge=<model>|emb=<model>` in its metadata, and experiment reports store it.
Bump `EVALUATOR_SCHEMA` in `evaluation/metrics.py` whenever scoring logic changes in a way that makes old and new numbers non-comparable.
`python -m evaluation.compare` warns when two reports used different evaluator or dataset versions.

## Full table

| Metric | Kind | Required inputs | Judge LLM | Embeddings | Online? | Notes |
|---|---|---|---|---|---|---|
| `faithfulness` | ragas | question, answer, contexts | yes | no | yes | Share of answer claims supported by the retrieved contexts. |
| `answer_relevancy` | ragas | question, answer | yes | yes | yes | How well the answer addresses the question (generated-question similarity). |
| `answer_correctness` | ragas | question, answer, reference | yes | yes | offline only | Factual overlap + semantic similarity of the answer vs the gold answer. |
| `semantic_similarity` | ragas | answer, reference | no | yes | offline only | Embedding cosine similarity of answer and gold answer. |
| `context_precision` | ragas | question, reference, contexts | yes | no | offline only | Are relevant chunks ranked above irrelevant ones? |
| `context_recall` | ragas | question, reference, contexts | yes | no | offline only | Share of gold-answer statements found in the contexts. |
| `context_entity_recall` | ragas | reference, contexts | yes | no | offline only | Share of gold-answer entities present in the contexts. |
| `noise_sensitivity` | ragas | question, answer, reference, contexts | yes | no | offline only | How often the answer is wrong because of irrelevant retrieved context (lower is better). |
| `citation_correctness` | custom | answer, contexts | no | no | yes | Share of [n] citations that point to an existing passage that lexically supports the cited sentence. |
| `citation_coverage` | custom | answer, contexts | no | no | yes | Share of factual answer sentences carrying at least one valid citation. |
| `unsupported_claim_rate` | custom | answer, contexts | no | no | yes | Share of numbers/named terms in the answer that do not appear in the contexts (lower is better). |
| `hallucination_rate` | custom | question, answer, contexts | yes | no | yes | 1 - faithfulness (lower is better). |
| `answer_completeness` | custom | question, answer, reference | yes | no | offline only | Share of the gold answer's key points covered by the answer (LLM judge). |
| `abstention_correctness` | custom | expected_behavior | no | no | offline only | 1 if the system abstained exactly when the gold label says it should. |
| `ref_context_hit_rate` | custom | contexts, reference_contexts | no | no | offline only | Share of gold passages found (>=80% token recall) in a retrieved chunk. |
| `ref_context_mrr` | custom | contexts, reference_contexts | no | no | offline only | Reciprocal rank of the first retrieved chunk matching a gold passage. |
