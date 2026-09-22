"""
Per-search job association.

This is the write side of the model introduced in migrate_job_searches.py. A live
fetch (Adzuna, Remotive, ...) records ONE `job_searches` row describing the search
(run_id, role, location, source), then associates every posting it returned to
that search in `job_search_results`.

This REPLACES the old approach of stamping `search_location` permanently onto each
job row. A job can now belong to many searches; the tie between a job and a
location lives per-search, in the join, not as a fixed property of the job.

`record_search` is best-effort: if run_id is missing (standalone CLI use) it still
creates a search row (run_id NULL) so associations remain queryable, and it never
raises into the agent.
"""
import os
from database import get_connection as _get_connection
from datetime import datetime


def create_search(run_id, target_role, location, source, conn=None):
    """
    Insert a job_searches row and return its id. Reuses an open connection when
    given one (so the whole fetch+associate is a single transaction); otherwise
    opens and commits its own.
    """
    own = conn is None
    if own:
        conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO job_searches (run_id, target_role, location, source, created_at)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """,
        (run_id, (target_role or "").strip() or None,
         (location or "").strip() or None, source, datetime.now()),
    )
    search_id = cur.fetchone()[0]
    if own:
        conn.commit()
        cur.close()
        conn.close()
    return search_id


def associate_jobs(search_id, job_ids, conn=None):
    """
    Link a set of job_postings.id values to a search. Idempotent: re-associating
    the same (search_id, job_id) is a no-op (ON CONFLICT DO NOTHING).
    """
    ids = [i for i in (job_ids or []) if i is not None]
    if not search_id or not ids:
        return 0
    own = conn is None
    if own:
        conn = _get_connection()
    cur = conn.cursor()
    for job_id in ids:
        cur.execute(
            """
            INSERT INTO job_search_results (search_id, job_id)
            VALUES (%s, %s) ON CONFLICT (search_id, job_id) DO NOTHING
            """,
            (search_id, job_id),
        )
    if own:
        conn.commit()
        cur.close()
        conn.close()
    return len(ids)


def job_ids_for_run(run_id, conn=None):
    """
    All job_postings.id associated with any search issued by this run. This is how
    Live Mode scopes to 'the jobs THIS run fetched live', instead of the whole pool.
    Returns a set (empty if the run did no live search).
    """
    if run_id is None:
        return set()
    own = conn is None
    if own:
        conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT r.job_id
        FROM job_search_results r
        JOIN job_searches s ON s.id = r.search_id
        WHERE s.run_id = %s
        """,
        (run_id,),
    )
    ids = {row[0] for row in cur.fetchall()}
    if own:
        cur.close()
        conn.close()
    return ids