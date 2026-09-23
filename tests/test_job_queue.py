"""
Concurrency + durability tests for job_queue.py — the most safety-critical code in
the project (SKIP LOCKED claiming, lease-token fencing, heartbeats, orphan recovery,
retry backoff, and transactional enqueue). These need a live Postgres.

The worker is NOT running during tests, so each test starts from an empty queue
(claim_next picks the oldest queued row across the whole table, so leftover rows from
other tests would make claims non-deterministic).
"""
import threading
import uuid
from datetime import datetime, timedelta

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection
from job_queue import (
    enqueue, enqueue_tx, claim_next, heartbeat, mark_done, mark_failed,
    reclaim_orphans, _retry_delay,
)
from llm import create_run_tx
from auth import create_user


@pytest.fixture(autouse=True)
def _clean_queue():
    with get_connection() as conn:
        conn.cursor().execute("DELETE FROM job_queue")
    yield


def _row(job_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT status, lease_token, available_at, attempts FROM job_queue WHERE id=%s",
            (job_id,),
        )
        return cur.fetchone()


def _make_user():
    return create_user("jqtest_" + uuid.uuid4().hex[:10], "password123", role="user")


# --- claiming -----------------------------------------------------------------

def test_a_job_is_claimed_exactly_once():
    jid = enqueue("start_run", {"x": 1})
    first = claim_next()
    second = claim_next()
    assert first is not None and first["id"] == jid
    assert second is None                       # the only job is now 'running'
    status, lease, _, attempts = _row(jid)
    assert status == "running" and lease and attempts == 1


def test_concurrent_workers_never_double_claim():
    # N jobs, several threads racing claim_next(): SKIP LOCKED must ensure every job
    # is claimed by exactly one thread (no id claimed twice, none dropped).
    n = 12
    for i in range(n):
        enqueue("start_run", {"i": i})
    claimed, lock = [], threading.Lock()

    def worker():
        while True:
            job = claim_next()
            if job is None:
                return
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == n                     # every job claimed
    assert len(set(claimed)) == n                # and none claimed twice


# --- lease fencing ------------------------------------------------------------

def test_stale_lease_cannot_mark_done():
    jid = enqueue("start_run", {"x": 1})
    job = claim_next()
    stale = job["lease_token"]
    # simulate orphan-reclaim + re-claim by another worker under a NEW lease
    with get_connection() as conn:
        conn.cursor().execute("UPDATE job_queue SET lease_token=%s WHERE id=%s", ("NEWLEASE", jid))
    assert mark_done(jid, stale) is False        # stale worker locked out
    assert _row(jid)[0] == "running"             # record unchanged
    assert heartbeat(jid, stale) is False        # heartbeat also fenced
    assert mark_done(jid, "NEWLEASE") is True     # the owner succeeds
    assert _row(jid)[0] == "done"


# --- orphan recovery ----------------------------------------------------------

def test_orphan_reclaim_requeues_and_clears_lease():
    jid = enqueue("start_run", {"x": 1})
    job = claim_next()
    assert job["lease_token"]
    # heartbeat older than ORPHAN_AFTER -> looks like a dead worker
    with get_connection() as conn:
        conn.cursor().execute(
            "UPDATE job_queue SET heartbeat_at=%s WHERE id=%s",
            (datetime.now() - timedelta(hours=1), jid),
        )
    reclaim_orphans()
    status, lease, _, _ = _row(jid)
    assert status == "queued" and lease is None   # requeued, lease invalidated


# --- retry backoff ------------------------------------------------------------

def test_retry_backoff_gate_and_curve():
    jid = enqueue("start_run", {"x": 1}, max_attempts=3)
    job = claim_next()                            # attempts -> 1
    out = mark_failed(jid, "boom", job["attempts"], job["max_attempts"], job["lease_token"])
    assert out == "requeued"
    status, lease, available_at, _ = _row(jid)
    assert status == "queued" and lease is None
    assert available_at is not None and available_at > datetime.now()
    assert claim_next() is None                    # backoff gate blocks immediate re-claim
    # exponential curve, capped
    assert _retry_delay(1) == 10 and _retry_delay(2) == 20 and _retry_delay(10) == 300


# --- attempt accounting -------------------------------------------------------

def test_out_of_attempts_becomes_failed():
    jid = enqueue("start_run", {"x": 1}, max_attempts=1)
    job = claim_next()                            # attempts -> 1 == max
    out = mark_failed(jid, "boom", job["attempts"], job["max_attempts"], job["lease_token"])
    assert out == "failed"
    assert _row(jid)[0] == "failed"


def test_terminal_failure_does_not_retry():
    jid = enqueue("start_run", {"x": 1}, max_attempts=3)  # attempts remain
    job = claim_next()
    out = mark_failed(jid, "bad input", job["attempts"], job["max_attempts"],
                      job["lease_token"], terminal=True)
    assert out == "failed"                        # terminal -> no requeue despite attempts left
    assert _row(jid)[0] == "failed"


# --- transactional enqueue (atomicity) ----------------------------------------

def test_create_run_and_enqueue_roll_back_together():
    uid = _make_user()
    conn = get_connection()
    run_id = job_id = None
    try:
        cur = conn.cursor()
        run_id = create_run_tx(cur, "rollback test", user_id=uid)
        job_id = enqueue_tx(cur, "start_run", {"run_id": run_id}, run_id=run_id)
        raise RuntimeError("simulated failure after both writes")
    except RuntimeError:
        conn.rollback()
    finally:
        conn.close()
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT count(*) FROM runs WHERE id=%s", (run_id,))
        assert cur.fetchone()[0] == 0             # run rolled back
        cur.execute("SELECT count(*) FROM job_queue WHERE id=%s", (job_id,))
        assert cur.fetchone()[0] == 0             # job rolled back too


def test_create_run_and_enqueue_commit_together():
    uid = _make_user()
    with get_connection() as conn:               # commits on clean exit
        cur = conn.cursor()
        run_id = create_run_tx(cur, "commit test", user_id=uid)
        job_id = enqueue_tx(cur, "start_run", {"run_id": run_id}, run_id=run_id)
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT count(*) FROM runs WHERE id=%s", (run_id,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT status FROM job_queue WHERE id=%s", (job_id,))
        assert cur.fetchone()[0] == "queued"


def test_resume_flip_and_enqueue_roll_back_together():
    uid = _make_user()
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(
            "INSERT INTO runs (started_at,status,input_summary,user_id) "
            "VALUES (NOW(),'waiting_for_human','t',%s) RETURNING id", (uid,))
        run_id = cur.fetchone()[0]
    conn = get_connection()
    job_id = None
    try:
        cur = conn.cursor()
        cur.execute("UPDATE runs SET status='queued' WHERE id=%s", (run_id,))
        job_id = enqueue_tx(cur, "resume_run", {"run_id": run_id}, run_id=run_id)
        raise RuntimeError("simulated failure")
    except RuntimeError:
        conn.rollback()
    finally:
        conn.close()
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT status FROM runs WHERE id=%s", (run_id,))
        assert cur.fetchone()[0] == "waiting_for_human"   # flip rolled back
        cur.execute("SELECT count(*) FROM job_queue WHERE id=%s", (job_id,))
        assert cur.fetchone()[0] == 0                      # no resume job left