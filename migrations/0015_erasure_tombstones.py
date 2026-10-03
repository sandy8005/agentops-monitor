"""
Migration 0015 — MONOTONIC data erasure. Idempotent.

  erasure_tombstones / erased_resume_hashes + guard triggers (erasure_guards.py)
        Erasure was a one-shot scrub, so a superseded worker generation that was
        still finishing (an LLM call, a tool, a checkpoint flush) re-persisted
        prompts, responses, tool payloads, suggestions, step context or
        checkpoints AFTER the user erased them. Every payload-bearing table now
        has a BEFORE INSERT OR UPDATE trigger that blanks (or drops) writes for a
        tombstoned run; tombstones are append-only; an erased resume cannot be
        restored or rewritten.

Backfill: every run that was ALREADY erased gets a tombstone, so the guarantee
also covers erasures made before this migration —
  * runs of a deleted resume                     -> resume_erased
  * runs whose traces were purged by retention   -> retention
and the content hash of an erased resume is unknown (its text was overwritten),
so no hash tombstone can be backfilled; the cache rows were deleted at the time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from erasure_guards import install_app_guards, install_checkpoint_guards  # noqa: E402


def upgrade(cur):
    install_app_guards(cur)
    cur.execute("""
        INSERT INTO erasure_tombstones (run_id, reason)
        SELECT r.id, 'resume_erased' FROM runs r JOIN resumes s ON s.id = r.resume_id
        WHERE s.is_deleted
        ON CONFLICT (run_id) DO NOTHING
    """)
    cur.execute("""
        INSERT INTO erasure_tombstones (run_id, reason)
        SELECT id, 'retention' FROM runs WHERE trace_purged_at IS NOT NULL
        ON CONFLICT (run_id) DO NOTHING
    """)
    # Databases whose checkpoint tables already exist get their guards now; fresh
    # databases get them from checkpointing.setup_schema() right after migrations.
    install_checkpoint_guards(cur)