"""
job_source.search_jobs: live-only and run-scoped. Needs Postgres.

  * apply_url is carried through from job_postings (it becomes the application
    link in run_rankings and the dashboard);
  * a posting is returned only if THIS run's own provider search returned it;
  * rows from a non-live source (old seed / csv / scraped data) are never
    returned, even when associated with the run.
"""
import uuid

import pytest

pytestmark = [pytest.mark.db]

from auth import create_user
from database import get_connection
from job_search import associate_jobs, create_search
from job_source import search_jobs


def _run():
    uid = create_user("sf_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id", (uid,))
        return cur.fetchone()[0]


def _posting(source, title="Python Engineer", url=None):
    eid = f"{source}:t-" + uuid.uuid4().hex[:10]
    with get_connection() as c:
        cur = c.cursor()
        cur.execute(
            """INSERT INTO job_postings
               (title, company, description, location, work_mode, employment_type,
                source, external_id, apply_url, last_seen_at, created_at)
               VALUES (%s, 'Acme', 'Python role', 'USA', 'onsite', 'full-time',
                       %s, %s, %s, NOW(), NOW()) RETURNING id""",
            (title, source, eid, url or f"https://apply.example/{uuid.uuid4().hex[:8]}"))
        return cur.fetchone()[0], eid


def _associate(run_id, job_ids, provider="adzuna", location="USA"):
    conn = get_connection()
    try:
        sid = create_search(run_id, "python engineer", location, provider, conn=conn)
        associate_jobs(sid, job_ids, conn=conn)
        conn.commit()
    finally:
        conn.close()


def test_search_jobs_preserves_apply_url():
    run_id = _run()
    url = "https://apply.example/" + uuid.uuid4().hex[:8]
    jid, eid = _posting("adzuna", url=url)
    _associate(run_id, [jid])
    jobs = search_jobs("python engineer", "USA", None, None, run_id=run_id)
    match = [j for j in jobs if j.get("external_id") == eid]
    assert match, "the run's own posting was not returned"
    assert match[0]["apply_url"] == url


def test_practice_sources_are_never_returned():
    run_id = _run()
    live, _ = _posting("adzuna", title="Python Engineer One")
    seeded, _ = _posting("seed", title="Python Engineer Two")
    scraped, _ = _posting("scraped", title="Python Engineer Three")
    _associate(run_id, [live, seeded, scraped])
    ids = {j["id"] for j in search_jobs("python engineer", "USA", None, None, run_id=run_id)}
    assert live in ids and seeded not in ids and scraped not in ids


def test_other_runs_postings_are_not_returned():
    mine, theirs = _run(), _run()
    a, _ = _posting("adzuna", title="Python Engineer Mine")
    b, _ = _posting("adzuna", title="Python Engineer Theirs")
    _associate(mine, [a])
    _associate(theirs, [b])
    ids = {j["id"] for j in search_jobs("python engineer", "USA", None, None, run_id=mine)}
    assert a in ids and b not in ids


def test_search_requires_a_run():
    with pytest.raises(ValueError):
        search_jobs("python engineer", "USA", None, None)


def test_nonsense_role_matches_nothing_and_role_filter_applies():
    run_id = _run()
    jid, _ = _posting("adzuna", title="Python Engineer")
    _associate(run_id, [jid])
    assert search_jobs("xyzzy plugh frobnicate", "USA", None, None, run_id=run_id) == []
    assert [j["id"] for j in search_jobs("engineer", "USA", None, None, run_id=run_id)] == [jid]