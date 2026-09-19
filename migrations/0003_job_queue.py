"""
Migration 0003 — durable job queue.

Replaces FastAPI BackgroundTasks (which run IN the web process and are LOST on a
restart/crash) with a durable Postgres-backed queue. The API enqueues a row and
returns immediately; a separate worker process (worker.py) claims and runs jobs
with SELECT ... FOR UPDATE SKIP LOCKED, so work survives API restarts and can be
retried or recovered.

Job kinds:
  - 'start_run'   : run the LangGraph agent for a run (payload: {run_id, resume_id,
                    target_role, location, work_mode, employment_type, evaluate,
                    live_only})
  - 'resume_run'  : resume a paused run with a human decision (payload: {run_id,
                    decision, comment})

Status lifecycle:
  queued -> running -> done | failed
  A job stuck in 'running' because its worker crashed is reclaimed by orphan
  recovery (claimed_at older than a threshold, and no heartbeat).
"""


def upgrade(cur):
    cur.execute("""
    CREATE TABLE IF NOT EXISTS job_queue (
        id SERIAL PRIMARY KEY,
        kind TEXT NOT NULL,                 -- 'start_run' | 'resume_run'
        payload JSONB NOT NULL,             -- kind-specific args
        run_id INTEGER,                     -- the run this job drives (for dedup/trace)
        status TEXT NOT NULL DEFAULT 'queued',   -- queued|running|done|failed
        attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL DEFAULT 3,
        last_error TEXT,
        enqueued_at TIMESTAMP NOT NULL DEFAULT NOW(),
        claimed_at TIMESTAMP,               -- when a worker took it (for orphan recovery)
        heartbeat_at TIMESTAMP,             -- worker liveness while running
        finished_at TIMESTAMP,
        worker_id TEXT                      -- which worker holds/ran it
    )
    """)
    # Fast claim: workers scan queued rows oldest-first.
    cur.execute("""
        CREATE INDEX IF NOT EXISTS job_queue_claim_idx
        ON job_queue (status, enqueued_at)
        WHERE status IN ('queued', 'running')
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS job_queue_run_idx ON job_queue (run_id)")