"""
Cancellation behaviour. A run that ISN'T executing (queued/retrying) is cancelled
immediately — the queued job is marked cancelled and the run finalized now, so the
user never waits through retry backoff. An actively-running run stays cooperative.
Needs Postgres.
"""
import uuid
from datetime import datetime, timedelta

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection
from auth import create_user
import api


def _mk(uid, run_status, available_at=None):
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (started_at,status,input_summary,user_id) "
                    "VALUES (NULL,%s,'t',%s) RETURNING id", (run_status, uid))
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO job_queue (kind,payload,run_id,status,max_attempts,"
                    "enqueued_at,available_at) VALUES "
                    "('start_run','{}',%s,'queued',3,NOW(),%s) RETURNING id",
                    (run_id, available_at))
        job_id = cur.fetchone()[0]
    return run_id, job_id


def _status(run_id, job_id):
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT status, ended_at FROM runs WHERE id=%s", (run_id,))
        run_status, ended_at = cur.fetchone()
        cur.execute("SELECT status FROM job_queue WHERE id=%s", (job_id,))
        job_status = cur.fetchone()[0]
    return run_status, ended_at is not None, job_status


def test_queued_run_cancels_immediately():
    uid = create_user("cx1_" + uuid.uuid4().hex[:8], "password123")
    run_id, job_id = _mk(uid, "queued")
    res = api.cancel_run(run_id, user={"id": uid}, _csrf=None)
    run_status, ended, job_status = _status(run_id, job_id)
    assert res.get("cancelled") is True
    assert run_status == "cancelled" and ended and job_status == "cancelled"


def test_retrying_run_with_future_backoff_cancels_immediately():
    # The key win: a retrying run whose job is gated behind a long available_at is
    # still cancelled now, not after the backoff elapses.
    uid = create_user("cx2_" + uuid.uuid4().hex[:8], "password123")
    run_id, job_id = _mk(uid, "retrying", datetime.now() + timedelta(minutes=5))
    res = api.cancel_run(run_id, user={"id": uid}, _csrf=None)
    run_status, ended, job_status = _status(run_id, job_id)
    assert res.get("cancelled") is True
    assert run_status == "cancelled" and job_status == "cancelled"


def test_running_run_is_cooperative_not_immediate():
    # An actively-running run isn't force-finalized here; it gets the cooperative flag.
    uid = create_user("cx3_" + uuid.uuid4().hex[:8], "password123")
    run_id, job_id = _mk(uid, "running")
    # mark the job claimed/running to reflect a real in-flight run
    with get_connection() as c:
        c.cursor().execute("UPDATE job_queue SET status='running' WHERE id=%s", (job_id,))
    res = api.cancel_run(run_id, user={"id": uid}, _csrf=None)
    run_status, _ended, _job = _status(run_id, job_id)
    assert res.get("cancel_requested") is True and "cancelled" not in res
    assert run_status == "running"   # not force-finalized; the loop will stop it