"""Smoke-check the real Langfuse connection end to end.

    python -m scripts.verify_langfuse

1. validates credentials (auth_check),
2. ingests the sample corpus into an in-memory index and asks one question,
3. prints the trace URL so you can open it in Langfuse,
4. attaches a score to that trace, proving the scoring path works.
"""
from __future__ import annotations

from app.config import get_settings
from app.observability import create_score, flush, get_langfuse, init_langfuse, shutdown_langfuse
from evaluation.experiment import ExperimentConfig, build_pipeline


def main() -> None:
    s = get_settings()
    if not init_langfuse(s):
        raise SystemExit("Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env")
    lf = get_langfuse()
    if not lf.auth_check():
        raise SystemExit("Langfuse auth_check failed: check keys and LANGFUSE_HOST (region/self-hosted URL).")
    print("Langfuse credentials OK")

    pipeline = build_pipeline(s, ExperimentConfig(name="verify", llm_model=s.llm_model,
                                                  embedding_model=s.embedding_model, prompt_version=s.prompt_version))
    result = pipeline.answer("What is the minimum password length?", tags=["verify"], enqueue_eval=False)
    print("answer  :", result.answer)
    print("trace id:", result.trace_id)
    create_score(trace_id=result.trace_id, name="verify_probe", value=1.0, data_type="NUMERIC", comment="verify_langfuse.py")
    flush()
    print("trace   :", result.trace_url or "(open Langfuse -> Tracing and search the trace id)")
    shutdown_langfuse()


if __name__ == "__main__":
    main()
