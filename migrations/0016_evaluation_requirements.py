"""
Migration 0016 — requirements snapshot per job evaluation. Idempotent.

  evaluation_requirements
        The exact normalized requirements (and their provenance: extraction
        method, model, requirements-cache version) that one evaluation step used.
        Advice is generated ONLY from this snapshot; it used to reload the
        requirements cache, whose rule-based rows expire after a day, so a run
        resumed after a long review pause got advice built on EMPTY requirements.
        Written fenced by the execution generation (agent_store), deleted with the
        run/step, and covered by an erasure guard: a late write for an erased run
        keeps the row (metrics/provenance) but blanks `requirements`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erasure_guards import install_snapshot_guards  # noqa: E402


def upgrade(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_requirements (
            id BIGSERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            step_id INTEGER NOT NULL REFERENCES steps(id) ON DELETE CASCADE,
            job_id INTEGER NOT NULL,
            execution_generation INTEGER NOT NULL,
            requirements JSONB,
            provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT evaluation_requirements_step_uniq UNIQUE (step_id)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS evaluation_requirements_run_job_idx "
                "ON evaluation_requirements (run_id, job_id)")
    install_snapshot_guards(cur)
