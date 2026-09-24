"""
Remotive live-fetch is RUN-SCOPED, like the Adzuna layer: fetched postings are
associated with the run's search (job_search_results) so a live_only run actually
sees them, and the external call is observed as a tool_call. Needs Postgres.
The network call is mocked (Remotive is remote-only and not reachable in CI).
"""
import uuid

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection
from auth import create_user
import live_jobs
from job_source import search_jobs


def _run_and_step(uid):
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (started_at,status,input_summary,user_id) "
                    "VALUES (NOW(),'running','t',%s) RETURNING id", (uid,))
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO steps (run_id,step_name,status,started_at) "
                    "VALUES (%s,'search_jobs','running',NOW()) RETURNING id", (run_id,))
        step_id = cur.fetchone()[0]
    return run_id, step_id


def test_remotive_fetch_is_run_scoped_and_visible_in_live_only(monkeypatch):
    uid = create_user("rmt_" + uuid.uuid4().hex[:10], "password123")
    run_id, step_id = _run_and_step(uid)

    eid = "remotive:test-" + uuid.uuid4().hex[:10]
    sample = [{"external_id": eid, "title": "Remote Python Engineer", "company": "RemoteCo",
               "description": "Python Django AWS remote role with plenty of real content here",
               "location": "Worldwide", "work_mode": "remote",
               "employment_type": "full-time", "source": "live"}]
    monkeypatch.setattr(live_jobs, "fetch_live_jobs",
                        lambda role, location=None, limit=10: (list(sample), "success", None))

    inserted, _skipped, status = live_jobs.fetch_and_upsert_remotive(
        "python engineer", "USA", run_id=run_id, step_id=step_id)
    assert status == "success" and inserted >= 1

    # the run-scoped live_only search now includes the Remotive posting
    jobs = search_jobs("python engineer", "USA", None, None, live_only=True, run_id=run_id)
    assert any("Remote Python Engineer" in j["title"] for j in jobs)

    # a remotive search row + a remotive_fetch trace were recorded (parity with Adzuna)
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT count(*) FROM job_searches WHERE run_id=%s AND source='remotive'", (run_id,))
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT count(*) FROM tool_calls WHERE run_id=%s AND tool_name='remotive_fetch'", (run_id,))
        assert cur.fetchone()[0] == 1


def test_remotive_fetch_failure_reports_classified_status(monkeypatch):
    uid = create_user("rmtf_" + uuid.uuid4().hex[:10], "password123")
    run_id, step_id = _run_and_step(uid)
    monkeypatch.setattr(live_jobs, "fetch_live_jobs",
                        lambda role, location=None, limit=10: ([], "network_error", "boom"))
    inserted, _skipped, status = live_jobs.fetch_and_upsert_remotive(
        "x", None, run_id=run_id, step_id=step_id)
    assert status == "network_error" and inserted == 0