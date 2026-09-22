"""
Migration 0006 — job_queue.available_at (retry backoff).

A failed-but-retryable job is put back to 'queued', but with nothing gating WHEN it
may run again it gets claimed immediately — so a persistently-failing job spins
fail -> queued -> claim -> fail with no pause, hammering the DB and the LLM provider.

available_at gates the next claim: mark_failed() sets it to NOW() + an exponential
backoff on requeue, and claim_next() only picks up queued jobs whose available_at
has passed. NULL means "available now" — a fresh enqueue, or an orphan reclaimed
after a worker crash (which should retry promptly, not be penalised with backoff).

Idempotent: ADD COLUMN IF NOT EXISTS.
"""


def upgrade(cur):
    cur.execute("ALTER TABLE job_queue ADD COLUMN IF NOT EXISTS available_at TIMESTAMP")
    # Claim scans queued rows oldest-first and now also filters on availability;
    # keep the existing claim index and add availability to help the planner.
    cur.execute("""
        CREATE INDEX IF NOT EXISTS job_queue_available_idx
        ON job_queue (available_at)
        WHERE status = 'queued'
    """)