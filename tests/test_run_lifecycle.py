"""
Run-lifecycle consistency between job_queue and runs:

  * orphan requeued            -> run becomes 'retrying' (ended_at cleared)
  * orphan out of attempts     -> run becomes 'failed', ended_at set, worker_lost
  * retrying / running again   -> ended_at cleared, previous end kept separately
  * every execution attempt is numbered and trace rows carry it
  * a run can't be EXECUTED by two workers at once (run-level lock); the second
    worker hands the job back WITHOUT consuming an attempt
  * a job whose run already moved on is closed without re-executing the run
  * a worker that lost ownership stops at the next node boundary

Needs Postgres. The worker is not running during tests.
"""
import uuid
from datetime import timedelta

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection
from auth import create_user
from timeutil import utcnow
import job_queue
import run_lock
import worker


@pytest.fixture(autouse=True)
def _clean_queue():
    with get_connection() as conn:
        conn.cursor().execute("DELETE FROM job_queue")
    yield


def _run(status="running", ended=False):
    uid = create_user("lc_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (started_at, ended_at, status, input_summary, user_id, attempt) "
                    "VALUES (NOW(), %s, %s, 't', %s, 1) RETURNING id",
                    (utcnow() if ended else None, status, uid))
        return cur.fetchone()[0]


def _run_row(run_id):
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT status, ended_at, error_code, last_attempt_ended_at, attempt "
                    "FROM runs WHERE id = %s", (run_id,))
        return cur.fetchone()


def _orphan(run_id, attempts, max_attempts=3):
    """Enqueue + claim a job for run_id, then age its heartbeat past ORPHAN_AFTER."""
    jid = job_queue.enqueue("start_run", {"run_id": run_id}, run_id=run_id,
                            max_attempts=max_attempts)
    job = job_queue.claim_next()
    assert job["id"] == jid
    stale = utcnow() - job_queue.ORPHAN_AFTER - timedelta(minutes=1)
    with get_connection() as c:
        c.cursor().execute("UPDATE job_queue SET attempts = %s, heartbeat_at = %s WHERE id = %s",
                           (attempts, stale, jid))
    return jid


# ------------------------------------------------------------ orphan recovery --

def test_orphan_reclaimed_marks_run_retrying():
    run_id = _run("running")
    _orphan(run_id, attempts=1)
    assert job_queue.reclaim_orphans() == 1
    status, ended_at, _, _, _ = _run_row(run_id)
    assert status == "retrying" and ended_at is None


def test_orphan_out_of_attempts_fails_the_run():
    run_id = _run("running")
    _orphan(run_id, attempts=3, max_attempts=3)
    assert job_queue.reclaim_orphans() == 1
    status, ended_at, error_code, _, _ = _run_row(run_id)
    assert status == "failed" and ended_at is not None and error_code == "worker_lost"


def test_orphan_recovery_leaves_finalized_runs_alone():
    run_id = _run("success", ended=True)          # a surviving worker already finished it
    _orphan(run_id, attempts=3, max_attempts=3)
    job_queue.reclaim_orphans()
    assert _run_row(run_id)[0] == "success"


# ------------------------------------------------------ retry / attempt state --

def test_retrying_clears_ended_at_and_keeps_last_attempt_end():
    run_id = _run("failed", ended=True)
    worker._mark_run_retrying(run_id)
    status, ended_at, _, last_end, _ = _run_row(run_id)
    assert status == "retrying" and ended_at is None and last_end is not None


def test_running_again_clears_ended_at_and_numbers_the_attempt():
    from autonomous_graph import _mark_run_running
    from llm import create_step, log_tool_call
    run_id = _run("failed", ended=True)                  # attempt 1 failed
    s1 = create_step(run_id, "parse_resume", 0)
    _mark_run_running(run_id, new_attempt=True)          # retry -> attempt 2
    status, ended_at, _, last_end, attempt = _run_row(run_id)
    assert status == "running" and ended_at is None and last_end is not None
    assert attempt == 2
    s2 = create_step(run_id, "parse_resume", 0)          # same step_order as attempt 1
    log_tool_call(run_id, s2, "t", {}, {}, 1, "success", None, "t")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT id, run_attempt FROM steps WHERE id IN (%s, %s) ORDER BY id", (s1, s2))
        assert [r[1] for r in cur.fetchall()] == [1, 2]
        cur.execute("SELECT run_attempt FROM tool_calls WHERE step_id = %s", (s2,))
        assert cur.fetchone()[0] == 2
    # A resume continues the SAME attempt.
    _mark_run_running(run_id, new_attempt=False)
    assert _run_row(run_id)[4] == 2


# ------------------------------------------------------- run execution lock --

def test_run_lock_is_exclusive_and_released():
    run_id = _run("running")
    a = run_lock.RunLock.try_acquire(run_id)
    assert a is not None
    assert run_lock.RunLock.try_acquire(run_id) is None      # B can't execute it
    a.release()
    b = run_lock.RunLock.try_acquire(run_id)
    assert b is not None
    b.release()


def test_worker_does_not_execute_a_run_another_worker_holds(monkeypatch):
    run_id = _run("retrying")
    jid = job_queue.enqueue("start_run", {"run_id": run_id, "resume_id": 1}, run_id=run_id)
    held = run_lock.RunLock.try_acquire(run_id)          # "worker A" still executing
    try:
        executed = []
        monkeypatch.setattr(worker, "_run_job", lambda job: executed.append(job))
        job = job_queue.claim_next()
        worker._process(job)
        assert executed == [], "a second worker executed the same run concurrently"
        with get_connection() as c:
            cur = c.cursor()
            cur.execute("SELECT status, attempts, available_at FROM job_queue WHERE id = %s", (jid,))
            status, attempts, available_at = cur.fetchone()
        assert status == "queued" and attempts == 0          # handed back, not consumed
        assert available_at > utcnow()
    finally:
        held.release()


def test_stale_job_for_finished_run_is_closed_without_executing(monkeypatch):
    run_id = _run("success", ended=True)
    jid = job_queue.enqueue("start_run", {"run_id": run_id, "resume_id": 1}, run_id=run_id)
    monkeypatch.setattr(worker, "_run_job", lambda job: pytest.fail("must not re-run"))
    worker._process(job_queue.claim_next())
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT status FROM job_queue WHERE id = %s", (jid,))
        assert cur.fetchone()[0] == "done"
    assert _run_row(run_id)[0] == "success"


def test_lost_ownership_stops_at_the_next_node_boundary():
    from autonomous_graph import _hydrate
    run_id = 10_000_000 + uuid.uuid4().int % 1_000_000
    run_lock.mark_lost(run_id)
    try:
        with pytest.raises(run_lock.ExecutionLost):
            _hydrate({"run_id": run_id, "goal": "x", "resume_id": 1})
    finally:
        run_lock.clear(run_id)