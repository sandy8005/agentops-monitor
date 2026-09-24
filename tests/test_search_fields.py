"""
search_jobs must carry apply_url through from job_postings — it's the actual
application link that ends up in run_rankings and the dashboard. Regression: the
SELECT dropped apply_url, so every returned job had apply_url=None. Needs Postgres.
"""
import uuid

import pytest

pytestmark = [pytest.mark.db]

from database import get_connection
from job_source import search_jobs


def test_search_jobs_preserves_apply_url():
    eid = "seed:apply-" + uuid.uuid4().hex[:10]
    url = "https://apply.example/" + uuid.uuid4().hex[:8]
    with get_connection() as c:
        c.cursor().execute(
            """INSERT INTO job_postings
               (title, company, description, location, work_mode, employment_type,
                source, external_id, apply_url, last_seen_at, created_at)
               VALUES ('Python Engineer','Acme','Python role','USA','onsite',
                       'full-time','seed',%s,%s,NOW(),NOW())""",
            (eid, url))
    jobs = search_jobs("python engineer", "USA", None, None, live_only=False)
    match = [j for j in jobs if j.get("external_id") == eid]
    assert match, "seeded job not returned by search_jobs"
    assert match[0]["apply_url"] == url   # exact URL retained, not None