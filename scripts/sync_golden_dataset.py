"""Upload the golden dataset to Langfuse as a Dataset (idempotent by item id).

    python -m scripts.sync_golden_dataset [--name rag-golden-v1]

After syncing, run experiments against it so runs show up side by side in
Langfuse -> Datasets -> <name> -> Runs:

    python -m evaluation.experiment --dataset-name rag-golden-v1
"""
from __future__ import annotations

import argparse

from app.config import get_settings
from app.observability import get_langfuse, init_langfuse, shutdown_langfuse
from evaluation.experiment import GOLDEN_PATH, dataset_version, load_golden


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="rag-golden-v1")
    args = p.parse_args()

    s = get_settings()
    if not init_langfuse(s):
        raise SystemExit("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST.")
    lf = get_langfuse()
    rows = load_golden()
    lf.create_dataset(
        name=args.name,
        description=f"RAG golden dataset (file version {dataset_version()}), {len(rows)} questions",
        metadata={"file": GOLDEN_PATH.name, "version": dataset_version()},
    )
    for r in rows:
        lf.create_dataset_item(
            dataset_name=args.name,
            id=f"{args.name}:{r['id']}",  # stable id => re-running updates instead of duplicating
            input=r["question"],
            expected_output=r["reference_answer"],
            metadata=r,
        )
    lf.flush()
    shutdown_langfuse()
    print(f"synced {len(rows)} items to Langfuse dataset '{args.name}'")


if __name__ == "__main__":
    main()
