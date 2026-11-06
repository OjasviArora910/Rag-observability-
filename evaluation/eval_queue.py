"""Durable SQLite job queue for asynchronous online evaluation.

Why SQLite: zero extra infrastructure and it survives restarts. The interface
(enqueue / claim / complete / fail) is small, so it can be swapped for Redis or
SQS if you already run them. Jobs carry status, attempts, backoff and the
evaluator version that produced the result.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_jobs (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    next_attempt_at REAL NOT NULL,
    last_error TEXT,
    evaluator_version TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_jobs_claim ON eval_jobs(status, next_attempt_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_jobs_trace ON eval_jobs(trace_id);
"""


@dataclass
class Job:
    id: str
    trace_id: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int


class EvalQueue:
    def __init__(self, path: str, max_attempts: int = 3, retry_base_seconds: float = 30.0):
        self.path = path
        self.max_attempts = max_attempts
        self.retry_base = retry_base_seconds
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def enqueue(self, trace_id: str, payload: dict[str, Any]) -> Optional[str]:
        """Idempotent per trace_id. Returns the job id (None if already queued)."""
        now = time.time()
        job_id = str(uuid.uuid4())
        with self._conn() as c:
            try:
                c.execute(
                    "INSERT INTO eval_jobs (id, trace_id, payload, status, max_attempts, next_attempt_at, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (job_id, trace_id, json.dumps(payload), PENDING, self.max_attempts, now, now, now),
                )
            except sqlite3.IntegrityError:
                return None
        return job_id

    def claim(self) -> Optional[Job]:
        now = time.time()
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT * FROM eval_jobs WHERE status=? AND next_attempt_at<=? ORDER BY created_at LIMIT 1",
                (PENDING, now),
            ).fetchone()
            if row is None:
                c.execute("COMMIT")
                return None
            c.execute(
                "UPDATE eval_jobs SET status=?, attempts=attempts+1, updated_at=? WHERE id=?",
                (RUNNING, now, row["id"]),
            )
            c.execute("COMMIT")
        return Job(row["id"], row["trace_id"], json.loads(row["payload"]), row["attempts"] + 1, row["max_attempts"])

    def complete(self, job: Job, *, evaluator_version: str, result: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE eval_jobs SET status=?, evaluator_version=?, result=?, last_error=NULL, updated_at=? WHERE id=?",
                (DONE, evaluator_version, json.dumps(result), time.time(), job.id),
            )

    def fail(self, job: Job, error: str) -> str:
        """Schedule a retry with exponential backoff, or mark permanently failed.
        Returns the resulting status."""
        now = time.time()
        final = job.attempts >= job.max_attempts
        status = FAILED if final else PENDING
        delay = self.retry_base * (2 ** (job.attempts - 1))
        with self._conn() as c:
            c.execute(
                "UPDATE eval_jobs SET status=?, last_error=?, next_attempt_at=?, updated_at=? WHERE id=?",
                (status, error[:1000], now + delay, now, job.id),
            )
        return status

    def requeue_stale(self, older_than_seconds: float = 900) -> int:
        """Recover jobs stuck in 'running' after a worker crash."""
        cutoff = time.time() - older_than_seconds
        with self._conn() as c:
            cur = c.execute(
                "UPDATE eval_jobs SET status=?, updated_at=? WHERE status=? AND updated_at<?",
                (PENDING, time.time(), RUNNING, cutoff),
            )
            return cur.rowcount

    def stats(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) n FROM eval_jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    def get(self, trace_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM eval_jobs WHERE trace_id=?", (trace_id,)).fetchone()
        return dict(row) if row else None
