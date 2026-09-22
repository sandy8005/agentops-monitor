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

Fencing token: each claim stamps a fresh per-claim lease_token (UUID). Every
mutation of a claimed row (heartbeat / mark_done / mark_failed) is guarded by
`AND status = 'running' AND lease_token = <token>`. If orphan recovery has requeued
the job (clearing the token) and another worker has re-claimed it under a NEW token,
the original worker's writes match zero rows — it has lost permission to mutate the
record, so it can no longer mark a job done/failed out from under the worker that now
owns it. This can't undo duplicate SIDE EFFECTS already written by the agent, but it
keeps the queue record itself consistent and single-owner.
"""
import json
import socket
import os
import uuid
from datetime import datetime, timedelta

from database import get_connection

# A job is considered orphaned if it's been 'running' with no heartbeat for this
# long (worker likely died). Tune vs. your longest expected job.
ORPHAN_AFTER = timedelta(minutes=15)

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

# Retry backoff: a requeued (retryable) job waits base * 2**(attempts-1) seconds,
# capped, before it can be claimed again — so a persistently-failing job doesn't
# spin fail -> queued -> claim -> fail with no pause.
RETRY_BASE_DELAY = 10    # seconds
RETRY_MAX_DELAY = 300    # seconds (5 minutes)


def _retry_delay(attempts):
    """Seconds to wait before the NEXT attempt. `attempts` is how many have already
    been made (>=1): 1 -> 10s, 2 -> 20s, 3 -> 40s, ... capped at RETRY_MAX_DELAY."""
    delay = RETRY_BASE_DELAY * (2 ** max(0, attempts - 1))
    return min(delay, RETRY_MAX_DELAY)


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
              AND (available_at IS NULL OR available_at <= NOW())
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
        # Fresh per-claim lease token. Every later mutation of this row must present
        # it (plus status='running'), so a stale prior owner can't touch the record
        # after it's been reclaimed and re-claimed under a new token.
        lease_token = str(uuid.uuid4())
        cur.execute("""
            UPDATE job_queue
            SET status = 'running', attempts = attempts + 1,
                claimed_at = %s, heartbeat_at = %s, worker_id = %s, lease_token = %s
            WHERE id = %s
        """, (now, now, worker_id, lease_token, job_id))
        conn.commit()
        return {
            "id": job_id, "kind": kind,
            "payload": payload if isinstance(payload, dict) else json.loads(payload),
            "run_id": run_id, "attempts": attempts + 1, "max_attempts": max_attempts,
            "lease_token": lease_token,
        }
    finally:
        conn.close()


# -------------------------------------------------------------- heartbeat -----

def heartbeat(job_id, lease_token):
    """Mark the running job alive (so orphan recovery doesn't reclaim a long but
    healthy job). The worker calls this periodically during long work.

    Guarded by the lease: returns True if this worker STILL owns the job, False if
    it has lost the lease (reclaimed as an orphan and taken by another worker). A
    False return is the worker's signal to stop touching the queue record."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE job_queue SET heartbeat_at = %s
        WHERE id = %s AND status = 'running' AND lease_token = %s
    """, (datetime.now(), job_id, lease_token))
    owned = cur.rowcount == 1
    conn.commit()
    conn.close()
    return owned


# ------------------------------------------------------- complete / fail -----

def mark_done(job_id, lease_token):
    """Mark the job done — only if this worker still holds the lease. Returns True
    if it did (and the row was updated), False if the lease was lost (another worker
    owns the job now, so we must NOT mark it done)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE job_queue SET status = 'done', finished_at = %s
        WHERE id = %s AND status = 'running' AND lease_token = %s
    """, (datetime.now(), job_id, lease_token))
    owned = cur.rowcount == 1
    conn.commit()
    conn.close()
    return owned


def mark_failed(job_id, error, attempts, max_attempts, lease_token, terminal=False):
    """
    Record a job failure — only if this worker still holds the lease.

    terminal=False (a RETRYABLE failure): if attempts remain, put the job BACK to
      'queued' for retry — clearing the lease and setting available_at to NOW() + an
      exponential backoff so it isn't re-claimed immediately; out of attempts it
      becomes 'failed'.
    terminal=True (a TERMINAL failure — bad input, parse failure, budget spent, ...):
      mark it 'failed' straight away, no retry, regardless of remaining attempts.

    Returns one of: "requeued", "failed", or "lost" (the lease was lost, so another
    worker already owns the job and we changed nothing — the caller must not retry
    or fail it itself).
    """
    conn = get_connection()
    cur = conn.cursor()
    if not terminal and attempts < max_attempts:
        available_at = datetime.now() + timedelta(seconds=_retry_delay(attempts))
        cur.execute("""
            UPDATE job_queue
            SET status = 'queued', last_error = %s, claimed_at = NULL,
                heartbeat_at = NULL, worker_id = NULL, lease_token = NULL,
                available_at = %s
            WHERE id = %s AND status = 'running' AND lease_token = %s
        """, (str(error)[:2000], available_at, job_id, lease_token))
        outcome = "requeued" if cur.rowcount == 1 else "lost"
    else:
        cur.execute("""
            UPDATE job_queue
            SET status = 'failed', last_error = %s, finished_at = %s, lease_token = NULL
            WHERE id = %s AND status = 'running' AND lease_token = %s
        """, (str(error)[:2000], datetime.now(), job_id, lease_token))
        outcome = "failed" if cur.rowcount == 1 else "lost"
    conn.commit()
    conn.close()
    return outcome


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
            lease_token = NULL, available_at = NULL,
            last_error = COALESCE(last_error, '') || ' [reclaimed orphan]'
        WHERE status = 'running'
          AND attempts < max_attempts
          AND (heartbeat_at IS NULL OR heartbeat_at < %s)
    """, (cutoff,))
    requeued = cur.rowcount
    # Fail orphans that are out of attempts.
    cur.execute("""
        UPDATE job_queue
        SET status = 'failed', finished_at = %s, lease_token = NULL,
            last_error = COALESCE(last_error, '') || ' [orphan, out of attempts]'
        WHERE status = 'running'
          AND attempts >= max_attempts
          AND (heartbeat_at IS NULL OR heartbeat_at < %s)
    """, (datetime.now(), cutoff))
    failed = cur.rowcount
    conn.commit()
    conn.close()
    return requeued + failed