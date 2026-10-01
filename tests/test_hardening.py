"""
API hardening, data erasure, live-posting persistence, and LLM retry policy.

  * /logout requires the CSRF token
  * closed vocabularies (work_mode / employment_type / decision) reject junk with 422
  * /me no longer advertises a role (there is no RBAC)
  * deleting a resume ERASES its text and every trace embedding it
  * a resume review carries the reviewer's identity into the queue payload
  * Remotive postings get last_seen_at / fetched_at / apply_url / posted_at, and
    last_seen_at refreshes on the next sighting
  * a live posting with no last_seen_at is not "fresh forever"
  * Remotive's search association is not treated as a location match
  * the LLM HTTP layer retries rate limits (with the provider's delay) but never an
    exhausted quota

Needs Postgres for the DB/API parts.
"""
import json
import uuid

import pytest

pytestmark = [pytest.mark.db]

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient

from database import get_connection
from auth import create_user


def _client(logged_in=True):
    import api
    c = TestClient(api.app)
    token = c.get("/csrf").json()["csrf_token"]
    c.headers.update({"X-CSRF-Token": token})
    uid = None
    if logged_in:
        name, pw = "hd_" + uuid.uuid4().hex[:10], "pw-" + uuid.uuid4().hex[:12]
        uid = create_user(name, pw)
        assert c.post("/login", data={"username": name, "password": pw}).status_code == 200
    return c, uid


def _resume(uid, text="Jane Q. Candidate, 555-0100, Python engineer"):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO resumes (name, resume_text, created_at, user_id) "
                    "VALUES ('cv.pdf', %s, NOW(), %s) RETURNING id", (text, uid))
        return cur.fetchone()[0]


# ------------------------------------------------------------------- API ------

def test_logout_requires_csrf():
    c, _ = _client()
    del c.headers["X-CSRF-Token"]
    assert c.post("/logout").status_code == 403
    assert c.get("/me").status_code == 200                   # still logged in
    c.headers["X-CSRF-Token"] = c.get("/csrf").json()["csrf_token"]
    assert c.post("/logout").status_code == 200
    assert c.get("/me").status_code == 401


def test_closed_vocabularies_are_enforced():
    c, uid = _client()
    rid = _resume(uid)
    bad = c.post("/runs", json={"resume_id": rid, "target_role": "engineer", "work_mode": "whatever-I-want"})
    assert bad.status_code == 422
    bad = c.post("/runs", json={"resume_id": rid, "target_role": "engineer", "employment_type": "gig"})
    assert bad.status_code == 422
    assert c.post("/runs/1/resume", json={"decision": "Definitely"}).status_code == 422
    ok = c.post("/runs", json={"resume_id": rid, "target_role": "engineer",
                               "work_mode": "remote", "employment_type": "full-time"})
    assert ok.status_code == 200


def test_me_does_not_advertise_a_role():
    c, _ = _client()
    assert "role" not in c.get("/me").json()


def test_deleting_a_resume_erases_its_text_and_traces():
    c, uid = _client()
    rid = _resume(uid)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, resume_id, ended_at) "
                    "VALUES ('success', 't', %s, %s, NOW()) RETURNING id", (uid, rid))
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO llm_calls (run_id, prompt, response) VALUES "
                    "(%s, 'RESUME: Jane Q. Candidate 555-0100', 'Apply')", (run_id,))
    assert c.delete(f"/resumes/{rid}").status_code == 200
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT resume_text, name, is_deleted FROM resumes WHERE id = %s", (rid,))
        text, name, deleted = cur.fetchone()
        cur.execute("SELECT prompt, response FROM llm_calls WHERE run_id = %s", (run_id,))
        prompt, response = cur.fetchone()
    assert "Jane" not in text and "Jane" not in name and deleted is True
    assert prompt is None and response is None


def test_resume_of_an_active_run_cannot_be_erased():
    c, uid = _client()
    rid = _resume(uid)
    with get_connection() as conn:
        conn.cursor().execute("INSERT INTO runs (status, input_summary, user_id, resume_id) "
                              "VALUES ('running', 't', %s, %s)", (uid, rid))
    assert c.delete(f"/resumes/{rid}").status_code == 409


def test_resume_review_payload_carries_reviewer_identity():
    c, uid = _client()
    rid = _resume(uid)
    with get_connection() as conn:
        cur = conn.cursor()
        rev = "r-" + uuid.uuid4().hex[:8]
        cur.execute("INSERT INTO runs (status, input_summary, user_id, resume_id, mode, "
                    "pending_review) VALUES ('waiting_for_human', 't', %s, %s, 'agent', %s) "
                    "RETURNING id", (uid, rid, json.dumps({"type": "review_request",
                                                           "review_id": rev})))
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO review_requests (review_id, run_id, kind, payload) "
                    "VALUES (%s, %s, 'job_review', '{}')", (rev, run_id))
    assert c.post(f"/runs/{run_id}/resume", json={"decision": "Apply", "comment": "ok",
                                                   "review_id": rev}).status_code == 200
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT payload FROM job_queue WHERE run_id = %s", (run_id,))
        payload = cur.fetchone()[0]
    payload = payload if isinstance(payload, dict) else json.loads(payload)
    assert payload["reviewer_user_id"] == uid and payload["reviewer"]


def test_legacy_pipeline_run_cannot_be_resumed():
    c, uid = _client()
    rid = _resume(uid)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, resume_id, mode) "
                    "VALUES ('waiting_for_human', 't', %s, %s, 'pipeline') RETURNING id",
                    (uid, rid))
        run_id = cur.fetchone()[0]
    r = c.post(f"/runs/{run_id}/resume", json={"decision": "Apply", "comment": "ok"})
    assert r.status_code == 409 and "retired" in r.json()["detail"]


# ------------------------------------------------------ live persistence ------

def _remotive_job(ext):
    return {"external_id": ext, "title": "Python Engineer", "company": "Remote Co",
            "description": "Python role", "location": "Worldwide", "work_mode": "remote",
            "employment_type": "full-time", "source": "remotive",
            "apply_url": "https://remotive.com/remote-jobs/software-dev/python-engineer-1",
            "posted_at": None}


def _posting(ext):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, fetched_at, last_seen_at, apply_url FROM job_postings "
                    "WHERE external_id = %s", (ext,))
        return cur.fetchone()


def test_remotive_upsert_writes_and_refreshes_freshness_fields():
    from live_jobs import upsert_live_jobs
    ext = "remotive:t" + uuid.uuid4().hex[:10]
    upsert_live_jobs([_remotive_job(ext)])
    _, fetched1, seen1, url = _posting(ext)
    assert fetched1 is not None and seen1 is not None and url.startswith("https://remotive.com/")
    with get_connection() as conn:   # pretend it was last seen long ago
        conn.cursor().execute("UPDATE job_postings SET last_seen_at = NOW() - INTERVAL '40 days' "
                              "WHERE external_id = %s", (ext,))
    upsert_live_jobs([_remotive_job(ext)])
    _, fetched2, seen2, _ = _posting(ext)
    assert fetched2 == fetched1                       # first-seen kept
    assert seen2 >= seen1                             # last-seen refreshed


def test_remotive_payload_fields_are_extracted(monkeypatch):
    import live_jobs

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"jobs": [{"id": 77, "title": "Python Engineer", "company_name": "RC",
                              "description": "<p>Python</p>", "url": "https://remotive.com/x/77",
                              "publication_date": "2026-09-20T14:03:11",
                              "candidate_required_location": "USA", "job_type": "full_time"}]}
    monkeypatch.setattr(live_jobs.requests, "get", lambda *a, **k: _Resp())
    jobs, status, _ = live_jobs.fetch_live_jobs("python")
    assert status == "success"
    assert jobs[0]["apply_url"] == "https://remotive.com/x/77"
    assert jobs[0]["posted_at"].year == 2026 and jobs[0]["posted_at"].tzinfo is not None


def _agent_run_with(title):
    """A run whose own search returned the posting with this title (if given)."""
    from job_search import create_search, associate_jobs
    _c, uid = _client()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id", (uid,))
        run_id = cur.fetchone()[0]
        ids = []
        if title:
            cur.execute("SELECT id FROM job_postings WHERE title = %s", (title,))
            ids = [r[0] for r in cur.fetchall()]
    if ids:
        sid = create_search(run_id, "zyxwv", None, "remotive", location_filter_applied=False)
        associate_jobs(sid, ids)
    return run_id


def test_live_posting_without_last_seen_is_not_fresh_forever():
    from job_source import search_jobs
    title = "Zyxwv Engineer " + uuid.uuid4().hex[:6]
    with get_connection() as conn:
        conn.cursor().execute(
            "INSERT INTO job_postings (title, company, description, source, external_id) "
            "VALUES (%s, 'X', 'zyxwv role', 'remotive', %s) RETURNING id",
            (title, "remotive:" + uuid.uuid4().hex))
    run_id = _agent_run_with(title)
    assert not [j for j in search_jobs("zyxwv", run_id=run_id) if j["title"] == title]


def test_remotive_association_is_not_a_location_match():
    """A remote-only provider's association doesn't make a job 'returned for Paris'."""
    from job_search import create_search, associate_jobs
    from job_persistence import upsert_postings
    from job_source import search_jobs
    tag = "qwrty" + uuid.uuid4().hex[:6]
    job = {**_remotive_job("remotive:" + uuid.uuid4().hex), "title": f"{tag} Engineer",
           "description": f"{tag} role"}
    _, _, ids = upsert_postings([job])
    run_id = _agent_run_with(None)
    sid = create_search(run_id, tag, None, "remotive", location_filter_applied=False)
    associate_jobs(sid, ids)
    # Remote job with no GEO-filtered association: kept (location-agnostic), and its
    # association is not reported as a match for any location.
    hits = [j for j in search_jobs(tag, location="Paris", run_id=run_id) if j["id"] == ids[0]]
    assert hits and hits[0]["assoc_locations"] == set()


# ------------------------------------------------------- LLM retry policy ------

@pytest.fixture
def _llm_stubs(monkeypatch):
    import llm
    sleeps = []
    monkeypatch.setattr(llm, "_log_llm_attempt", lambda *a, **k: None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    return llm, sleeps


def test_exhausted_quota_is_not_retried(_llm_stubs, monkeypatch):
    llm, sleeps = _llm_stubs
    calls = []
    def boom(prompt):
        calls.append(1)
        raise RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProject")
    monkeypatch.setattr(llm, "real_llm_once", boom)
    with pytest.raises(RuntimeError):
        llm.logged_llm_call("p", None, None)
    assert len(calls) == 1 and sleeps == []


def test_rate_limit_is_retried_honoring_provider_delay(_llm_stubs, monkeypatch):
    llm, sleeps = _llm_stubs
    outcomes = [RuntimeError("429 RESOURCE_EXHAUSTED {'retryDelay': '7s'}"),
                {"text": "ok", "prompt_tokens": 1, "completion_tokens": 1}]
    def flaky(prompt):
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(llm, "real_llm_once", flaky)
    assert llm.logged_llm_call("p", None, None) == "ok"
    assert len(sleeps) == 1 and 7.0 <= sleeps[0] <= 8.0


def test_backoff_is_jittered(_llm_stubs):
    llm, _ = _llm_stubs
    delays = {round(llm._backoff_seconds(3), 6) for _ in range(20)}
    assert len(delays) > 1 and all(0 <= d <= 4 for d in delays)