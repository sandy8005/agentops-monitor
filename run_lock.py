"""
Run-level execution lock + ownership guard.

The queue's lease token answers "who owns this QUEUE ROW?". It cannot stop two
workers from EXECUTING the same run at once: if worker A stalls (a long LLM call,
or it can't reach the DB to heartbeat) past ORPHAN_AFTER, orphan recovery requeues
the job, worker B claims it, and both would write steps / LLM calls / rankings /
checkpoints for the same run — duplicate spend and a corrupted trace.

This module answers the other question: "who may EXECUTE this run right now?"

  * RunLock takes a PostgreSQL SESSION-level advisory lock keyed on run_id, on a
    DEDICATED connection (not the pool — a pooled connection returned with the lock
    still held would leak it to the next borrower). While A's session holds it, B
    cannot execute the run. If A's process or connection dies, PostgreSQL releases
    the lock automatically when the session ends, so a dead worker never blocks the
    run forever.

  * The ownership guard (mark_lost / is_lost / check_owner) is the cooperative
    half. When a worker learns it no longer owns the run — its lease was taken, or
    its lock connection died — it marks the run LOST. The agent checks between jobs
    and aborts (ExecutionLost), and the graph entrypoints skip finalization, so a
    stale worker never overwrites the status written by the worker that now owns it.
"""
import threading

import psycopg2

from logging_config import get_logger

log = get_logger(__name__)

# Advisory-lock namespace (first int4 key) so these locks can't collide with any
# other advisory lock the database might use. 0x4147 = "AG" (AgentOps).
LOCK_NAMESPACE = 0x4147


class ExecutionLost(RuntimeError):
    """This worker no longer owns the run it was executing; stop without finalizing."""


# --------------------------------------------------------------- ownership guard

_lost = set()
_lost_lock = threading.Lock()


def mark_lost(run_id):
    with _lost_lock:
        _lost.add(run_id)


def is_lost(run_id):
    with _lost_lock:
        return run_id in _lost


def clear(run_id):
    with _lost_lock:
        _lost.discard(run_id)


def check_owner(run_id):
    """Raise ExecutionLost if this process has lost ownership of run_id. Called by the
    agent at job boundaries (next to the cooperative cancel check)."""
    if run_id is not None and is_lost(run_id):
        raise ExecutionLost(f"run {run_id}: execution ownership lost — stopping")


# -------------------------------------------------------------------- run lock

class RunLock:
    """Session-level advisory lock on one run, held on a dedicated connection."""

    def __init__(self, run_id, conn):
        self.run_id = run_id
        self._conn = conn

    @classmethod
    def try_acquire(cls, run_id):
        """Return a held RunLock, or None if another session already executes run_id."""
        from settings import settings
        conn = psycopg2.connect(**settings.db_kwargs())
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s, %s)", (LOCK_NAMESPACE, run_id))
                got = cur.fetchone()[0]
        except Exception:
            conn.close()
            raise
        if not got:
            conn.close()
            return None
        return cls(run_id, conn)

    def alive(self):
        """True if the lock's session is still usable (so the lock is still held)."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def release(self):
        """Unlock and close. Closing the session releases the lock even if the
        explicit unlock fails, so this is always safe."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s, %s)", (LOCK_NAMESPACE, self.run_id))
        except Exception:
            pass
        finally:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False