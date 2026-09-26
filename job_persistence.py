"""
Shared persistence for LIVE job postings (Adzuna, Remotive, ...).

Every live provider writes through upsert_postings(), so they all get the SAME
freshness semantics: fetched_at (first seen), last_seen_at (refreshed on every
sighting), posted_at and apply_url. Previously Remotive had its own INSERT that
wrote none of these, so its rows had last_seen_at = NULL and were treated as fresh
forever — one code path means the providers can't drift apart again.
"""
from timeutil import utcnow


def _get_connection():
    from database import get_connection
    return get_connection()


def upsert_postings(jobs, conn=None):
    """
    Insert postings not already in the pool (dedup on external_id); on conflict
    refresh mutable fields and ALWAYS advance last_seen_at. Returns
    (inserted, skipped, job_ids) where job_ids are the job_postings.id of EVERY
    posting touched (inserted or updated), in input order.

    Reuses an open connection when given one (caller owns the transaction);
    otherwise opens, commits and returns its own (even on error).
    """
    if not jobs:
        return (0, 0, [])
    if conn is None:
        with _get_connection() as own:
            return upsert_postings(jobs, conn=own)
    cur = conn.cursor()
    now = utcnow()
    inserted = 0
    job_ids = []
    for j in jobs:
        cur.execute("""
            INSERT INTO job_postings
            (title, company, description, location, work_mode, employment_type,
             source, external_id, fetched_at, last_seen_at, posted_at, apply_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id) WHERE external_id IS NOT NULL
            DO UPDATE SET
                -- Refresh mutable fields from the newest fetch, but NEVER overwrite
                -- an existing good value with a blank/NULL one.
                title           = COALESCE(NULLIF(EXCLUDED.title, ''),          job_postings.title),
                company         = COALESCE(NULLIF(EXCLUDED.company, ''),        job_postings.company),
                description     = COALESCE(NULLIF(EXCLUDED.description, ''),    job_postings.description),
                location        = COALESCE(NULLIF(EXCLUDED.location, ''),       job_postings.location),
                work_mode       = COALESCE(NULLIF(EXCLUDED.work_mode, ''),      job_postings.work_mode),
                employment_type = COALESCE(NULLIF(EXCLUDED.employment_type, ''),job_postings.employment_type),
                posted_at       = COALESCE(EXCLUDED.posted_at,                  job_postings.posted_at),
                apply_url       = COALESCE(NULLIF(EXCLUDED.apply_url, ''),      job_postings.apply_url),
                -- first-seen time is kept (backfilled if a legacy row lacked it);
                -- last_seen_at always advances to the newest sighting.
                fetched_at      = COALESCE(job_postings.fetched_at, EXCLUDED.fetched_at),
                last_seen_at    = EXCLUDED.last_seen_at
            RETURNING id, (xmax = 0) AS was_inserted
        """, (j["title"], j["company"], j["description"], j["location"],
              j["work_mode"], j["employment_type"], j["source"], j["external_id"],
              now, now, j.get("posted_at") or None, j.get("apply_url") or None))
        row = cur.fetchone()
        job_ids.append(row[0])
        if row[1]:                 # xmax = 0 → fresh INSERT, not an UPDATE
            inserted += 1
    return (inserted, len(jobs) - inserted, job_ids)