# Monitoring dashboards in Langfuse

Two separate dashboards answer two different questions. Keep them separate so an outage is never confused with a quality drop.

> Langfuse's dashboard builder evolves between releases; menu labels may differ slightly. The *data* described here (names, tags, score names, metadata keys) is what this project emits and is stable.

## What the app emits

| Signal | Where it lives | Notes |
|---|---|---|
| Trace per chat request | trace name `rag-chat` | input = question, output = answer + cited chunk ids |
| Latency per stage | observations `retrieval`, `embed-query`, `vector-search`, `bm25-search`, `rank-fusion`, `rerank`, `context-assembly`, `generation`, `answer-postprocess` | Langfuse computes latency from span timestamps |
| Tokens + estimated cost | `generation` and `embed-query` observations | `usage_details` always; `cost_details` when the model has pricing (Langfuse model pricing, or `PRICING_JSON`) |
| Errors / failures | observations with level `ERROR` (+ status message); LLM retries in `generation` metadata `llm_retries` | the HTTP layer returns 502 without leaking internals |
| Tags | `rag`, `prompt:<version>`, `model:<name>`, `experiment`, `experiment:<config>` | filter/group dashboards by prompt version or model |
| Trace metadata | `llm_model`, `embedding_model`, `chunk_size`, `chunk_overlap`, `top_k`, `hybrid`, `reranker`, `prompt_version`, ... | exact config that produced the trace |
| Session / user | `session_id`, `user_id` | when the client sends them |
| Request-time scores (cheap, deterministic) | `empty_context`, `abstained`, `invalid_citation_count`, `top_retrieval_score` | written synchronously |
| Online evaluation scores (async) | `faithfulness`, `answer_relevancy`, `hallucination_rate`, `citation_correctness`, `citation_coverage`, `unsupported_claim_rate` | written by the worker; metadata has `evaluator_version` |
| Evaluation status | `online_eval_status` = `COMPLETED` / `PARTIAL` / `FAILED` | metadata lists not-evaluated metrics and reasons |
| User feedback | `user_feedback` (1 = thumbs up, 0 = thumbs down) | from `POST /api/v1/feedback` |

## Dashboard A - Infrastructure health

Question: *is the service fast, available and affordable?* None of these need an LLM judge.

| Widget | Data | Aggregation / grouping |
|---|---|---|
| Request volume | traces named `rag-chat` | count per hour |
| End-to-end latency | `rag-chat` | p50 / p95 / p99 over time |
| Stage latency | observations `retrieval`, `generation` | p95, one line each - shows which stage regressed |
| Error rate | observations with level `ERROR` | count per hour, group by observation name |
| Token consumption | `generation` observations | sum of input/output tokens, group by model |
| Estimated cost | `generation` + `embed-query` | sum of total cost per day, group by model |
| Cost per request | traces | total cost / trace count |
| Eval pipeline health | `GET /health/ready` -> `eval_queue` (`pending`, `failed`) and the `online_eval_status` score | alert if `pending` keeps growing or `FAILED` share rises |

Suggested alerts (in your alerting tool, fed from Langfuse/Prometheus/uptime checks): p95 latency, error rate > 2%, daily cost over budget, `/health/ready` failing.

## Dashboard B - RAG quality

Question: *are answers grounded, relevant and correct?* Everything here comes from scores.

| Widget | Score | Aggregation / grouping |
|---|---|---|
| Faithfulness trend | `faithfulness` | daily mean; also p10 to expose the worst answers |
| Answer relevancy trend | `answer_relevancy` | daily mean |
| Hallucination rate | `hallucination_rate`, `unsupported_claim_rate` | daily mean (lower is better) |
| Citation quality | `citation_coverage`, `citation_correctness` | daily mean |
| Retrieval failures | `empty_context` | daily mean of the boolean = share of requests with no retrieved context |
| Abstention rate | `abstained` | daily mean; a sudden jump means retrieval or ingestion broke |
| Weak retrieval | `top_retrieval_score` | histogram / p10 |
| User satisfaction | `user_feedback` | daily mean |
| Quality by prompt / model | any score above | group by tag `prompt:<version>` or metadata `llm_model` |
| Evaluation coverage | `online_eval_status` | counts by value; many `PARTIAL`/`FAILED` means the numbers above are biased |

`answer_correctness`, `context_precision`, `context_recall`, `context_entity_recall`, `noise_sensitivity`, `answer_completeness` and `abstention_correctness`
need gold labels, so they appear on **experiment runs** (below), not on production traces.

## Experiments (offline, regression and trend analysis)

1. `python -m scripts.sync_golden_dataset` uploads the golden set as a Langfuse dataset (`rag-golden-v1`).
2. `python -m evaluation.experiment --dataset-name rag-golden-v1 --config evaluation/configs/baseline.json`
3. Repeat with a changed config (or `--sweep`). Open **Datasets -> rag-golden-v1 -> Runs** to compare runs side by side: per-item scores, run-level means (`mean_<metric>`), linked traces.
4. Click through a low-scoring item to its trace and inspect retrieval output, assembled context and the generation.

Locally, `python -m evaluation.compare A.json B.json` produces the same comparison with regression flags, and `--fail-on-regression` makes it usable as a CI gate.

## Reading the two dashboards together

| Infra | Quality | Likely cause |
|---|---|---|
| latency up | quality flat | slower provider / larger `top_k` / index growth |
| errors up | `abstained` up | LLM or vector store failing; fallbacks returning abstentions |
| flat | faithfulness down | prompt or model change, noisier retrieval (lower `top_retrieval_score`, more chunks) |
| flat | `empty_context` up | ingestion failure, deleted documents, bad document filter |
| cost up | quality flat | longer contexts (`chunk_size`/`top_k`), more LLM retries |
