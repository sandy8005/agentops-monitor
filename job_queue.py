"""
Durable job queue on Postgres.

The API ENQUEUES jobs (write a row, return immediately). A separate worker process
CLAIMS and runs them. Because the queue lives in Postgres (not the web process's
memory like BackgroundTasks did), work survives API restarts/crashes and can be
retried or recovered.

Concurrency-safe claiming uses `SELECT ... FOR UPDATE SKIP LOCKED`: multiple
workers can run at once and never grab the same job — each skips rows another
worker has locked. This is the standard, race-free Postgres queue pattern.

Orphan recovery: if a worker dies mid-job, its row is left 'running' with a stale
heartbeat. reclaim_orphans() returns such rows to 'queued' so another worker picks
them up (or fails them if out of attempts).
"""
import json
import socket
import os
from datetime import datetime, timedelta

from database import get_connection

# A job is considered orphaned if it's been 'running' with no heartbeat for this
# long (worker likely died). Tune vs. your longest expected job.
ORPHAN_AFTER = timedelta(minutes=15)

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


# ---------------------------------------------------------------- enqueue -----

def enqueue_tx(cur, kind, payload, run_id=None, max_attempts=3):
    """
    Transactional enqueue: INSERT the job on the CALLER'S cursor and return its id
    WITHOUT committing. This lets the API write the run (create it, or flip its
    status) and the queue row in ONE transaction, so they commit or roll back
    together — a run is never left 'running'/'waiting_for_human' with no queue job.
    The caller owns the transaction (commit / rollback / close).
    """
    cur.execute("""
        INSERT INTO job_queue (kind, payload, run_id, status, max_attempts, enqueued_at)
        VALUES (%s, %s, %s, 'queued', %s, %s) RETURNING id
    """, (kind, json.dumps(payload), run_id, max_attempts, datetime.now()))
    return cur.fetchone()[0]


def enqueue(kind, payload, run_id=None, max_attempts=3):
    """Add a job to the queue in its OWN transaction and return its id — a thin
    wrapper over enqueue_tx() for callers that aren't already inside a transaction
    (e.g. the cancel path). API paths that must be atomic with a run write should
    call enqueue_tx() on their own connection instead, not this."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        job_id = enqueue_tx(cur, kind, payload, run_id=run_id, max_attempts=max_attempts)
        conn.commit()
        return job_id
    finally:
        conn.close()


# ------------------------------------------------------------------ claim -----

def claim_next(worker_id=WORKER_ID):
    """
    Atomically claim the oldest queued job for this worker, or return None.

    SELECT ... FOR UPDATE SKIP LOCKED locks the chosen row so no other worker can
    take it, and SKIP LOCKED means concurrent workers pass over already-locked rows
    instead of blocking. The UPDATE to 'running' happens in the SAME transaction as
    the lock, so the claim is atomic.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, kind, payload, run_id, attempts, max_attempts
            FROM job_queue
            WHERE status = 'queued'
            ORDER BY enqueued_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        """)
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        job_id, kind, payload, run_id, attempts, max_attempts = row
        now = datetime.now()
        cur.execute("""
            UPDATE job_queue
            SET status = 'running', attempts = attempts + 1,
                claimed_at = %s, heartbeat_at = %s, worker_id = %s
            WHERE id = %s
        """, (now, now, worker_id, job_id))
        conn.commit()
        return {
            "id": job_id, "kind": kind,
            "payload": payload if isinstance(payload, dict) else json.loads(payload),
            "run_id": run_id, "attempts": attempts + 1, "max_attempts": max_attempts,
        }
    finally:
        conn.close()


# -------------------------------------------------------------- heartbeat -----

def heartbeat(job_id):
    """Mark the running job alive (so orphan recovery doesn't reclaim a long but
    healthy job). The worker calls this periodically during long work."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE job_queue SET heartbeat_at = %s WHERE id = %s",
                (datetime.now(), job_id))
    conn.commit()
    conn.close()


# ------------------------------------------------------- complete / fail -----

def mark_done(job_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE job_queue SET status = 'done', finished_at = %s WHERE id = %s
    """, (datetime.now(), job_id))
    conn.commit()
    conn.close()


def mark_failed(job_id, error, attempts, max_attempts):
    """
    Fail a job. If it still has attempts left, put it BACK to 'queued' for retry;
    otherwise mark it permanently 'failed'. Records the error either way.
    """
    conn = get_connection()
    cur = conn.cursor()
    if attempts < max_attempts:
        cur.execute("""
            UPDATE job_queue
            SET status = 'queued', last_error = %s, claimed_at = NULL,
                heartbeat_at = NULL, worker_id = NULL
            WHERE id = %s
        """, (str(error)[:2000], job_id))
        requeued = True
    else:
        cur.execute("""
            UPDATE job_queue
            SET status = 'failed', last_error = %s, finished_at = %s
            WHERE id = %s
        """, (str(error)[:2000], datetime.now(), job_id))
        requeued = False
    conn.commit()
    conn.close()
    return requeued


# ------------------------------------------------------- orphan recovery -----

def reclaim_orphans():
    """
    Return jobs stuck in 'running' with a stale/absent heartbeat (their worker
    died) to 'queued' so another worker retries them — or 'failed' if they're out
    of attempts. Called by the worker on startup and periodically. Returns the
    number reclaimed.
    """
    cutoff = datetime.now() - ORPHAN_AFTER
    conn = get_connection()
    cur = conn.cursor()
    # Requeue orphans that still have attempts left.
    cur.execute("""
        UPDATE job_queue
        SET status = 'queued', claimed_at = NULL, heartbeat_at = NULL, worker_id = NULL,
            last_error = COALESCE(last_error, '') || ' [reclaimed orphan]'
        WHERE status = 'running'
          AND attempts < max_attempts
          AND (heartbeat_at IS NULL OR heartbeat_at < %s)
    """, (cutoff,))
    requeued = cur.rowcount
    # Fail orphans that are out of attempts.
    cur.execute("""
        UPDATE job_queue
        SET status = 'failed', finished_at = %s,
            last_error = COALESCE(last_error, '') || ' [orphan, out of attempts]'
        WHERE status = 'running'
          AND attempts >= max_attempts
          AND (heartbeat_at IS NULL OR heartbeat_at < %s)
    """, (datetime.now(), cutoff))
    failed = cur.rowcount
    conn.commit()
    conn.close()
    return requeued + failed