"""
Live job fetching from the Remotive API, per-search.

Unlike import_api.py (which batch-imports a generic set), fetch_live_jobs passes
the user's actual target role as Remotive's `search` parameter, so it returns
jobs matching THIS search. Results are normalized to the job_postings shape and
deduplicated on upsert (see upsert_live_jobs). Remotive is remote-only and free;
that's the honest limit of what this source provides.
"""
import requests
import hashlib
from logging_config import get_logger
log = get_logger(__name__)
REMOTIVE_API = "https://remotive.com/api/remote-jobs"


def _clean_description(raw):
    """Remotive returns HTML; strip to plain text so the requirement-extractor
    sees readable content, not markup."""
    if not raw:
        return ""
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        import re
        text = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(text.split())[:4000]


def _map_employment_type(job_type):
    jt = (job_type or "").lower().replace("_", "-").strip()
    if "intern" in jt:
        return "internship"
    if "part-time" in jt:
        return "part-time"
    if "contract" in jt or "freelance" in jt:
        return "contract"
    return "full-time"


def _external_id(job):
    """Stable dedup key for a live job. Prefer Remotive's own id; else hash
    title+company so the same posting isn't inserted twice."""
    rid = job.get("id")
    if rid:
        return f"remotive:{rid}"
    basis = f"{job.get('title','')}|{job.get('company_name','')}"
    return "remotive:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def fetch_live_jobs(role, location=None, limit=10):
    """
    Fetch live jobs from Remotive matching `role`. Returns (jobs, status, error) —
    a CLASSIFIED status (success / empty / rate_limited / http_error /
    network_error), mirroring the Adzuna layer so the run trace shows WHY a fetch
    produced nothing. `location` is accepted but Remotive is remote-only, so it's
    informational. Never raises into the agent.
    """
    try:
        params = {"limit": limit}
        if role and role.strip():
            params["search"] = role.strip()
        headers = {"User-Agent": "AgentOpsMonitor/1.0 (educational project)"}
        resp = requests.get(REMOTIVE_API, params=params, headers=headers, timeout=20)
        if resp.status_code == 429:
            return ([], "rate_limited", "Remotive rate limit (HTTP 429)")
        resp.raise_for_status()
        raw_jobs = resp.json().get("jobs", [])[:limit]
    except requests.exceptions.HTTPError as e:
        return ([], "http_error", str(e))
    except Exception as e:
        log.warning("live fetch failed (%s) — continuing with existing pool", e)
        return ([], "network_error", str(e))

    out = []
    for j in raw_jobs:
        title = (j.get("title") or "").strip()
        description = _clean_description(j.get("description"))
        if not title or not description:
            continue
        out.append({
            "external_id": _external_id(j),
            "title": title,
            "company": (j.get("company_name") or "").strip(),
            "description": description,
            "location": (j.get("candidate_required_location") or "").strip(),
            "work_mode": "remote",
            "employment_type": _map_employment_type(j.get("job_type")),
            "source": "live",
        })
    return (out, "success" if out else "empty", None)



def _get_connection():
    # Draw from the shared pool (database.py) rather than opening a fresh socket.
    from database import get_connection
    return get_connection()


def upsert_live_jobs(jobs, conn=None):
    """
    Insert live jobs not already in the pool (dedup on external_id) and return
    (inserted, skipped, job_ids). job_ids are ALL upserted postings (new AND
    already-present), so the caller can associate every fetched posting with the
    current run's search. Accepts an existing `conn` (caller owns the transaction);
    opens and commits its own when none is given.
    """
    if not jobs:
        return (0, 0, [])
    own = conn is None
    if own:
        conn = _get_connection()
    cur = conn.cursor()
    inserted = 0
    job_ids = []
    for j in jobs:
        cur.execute("""
            INSERT INTO job_postings
            (title, company, description, location, work_mode, employment_type, source, external_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id) WHERE external_id IS NOT NULL DO NOTHING
            RETURNING id
        """, (j["title"], j["company"], j["description"], j["location"],
              j["work_mode"], j["employment_type"], j["source"], j["external_id"]))
        row = cur.fetchone()
        if row:
            inserted += 1
            job_ids.append(row[0])
        else:  # conflict -> fetch the existing id so it's still associated to this run
            cur.execute("SELECT id FROM job_postings WHERE external_id = %s", (j["external_id"],))
            r2 = cur.fetchone()
            if r2:
                job_ids.append(r2[0])
    if own:
        conn.commit()
        cur.close()
        conn.close()
    skipped = len(jobs) - inserted
    return (inserted, skipped, job_ids)


def fetch_and_upsert(role, location=None, limit=10):
    """Fetch live jobs and upsert them in their own transaction (no run scope).
    Returns (inserted, skipped)."""
    jobs, _status, _err = fetch_live_jobs(role, location, limit)
    inserted, skipped, _ = upsert_live_jobs(jobs)
    if jobs:
        log.info("live: fetched %d, added %d new, %d already known", len(jobs), inserted, skipped)
    return (inserted, skipped)


def _log_remotive_call(run_id, step_id, role, location, latency_ms,
                       fetched, inserted, duplicates, status, error_message):
    """Write an AgentOps trace row for the Remotive external API call, mirroring the
    Adzuna one, so its latency/results/errors are observed like every other tool.
    No-op if run_id/step_id aren't provided (standalone CLI use)."""
    if run_id is None or step_id is None:
        return
    try:
        import json
        from datetime import datetime
        from database import get_connection
        with get_connection() as conn:
            conn.cursor().execute("""
                INSERT INTO tool_calls
                (run_id, step_id, tool_name, input_json, output_json, latency_ms,
                 status, error_message, created_at, operation_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, step_id, "remotive_fetch",
                  json.dumps({"role": role, "location": location}),
                  json.dumps({"fetched": fetched, "inserted": inserted, "duplicates": duplicates}),
                  latency_ms, status, error_message, datetime.now(), "live_fetch"))
    except Exception as log_err:
        log.warning("remotive trace log failed: %s", log_err)


def fetch_and_upsert_remotive(role, location=None, limit=10, run_id=None, step_id=None):
    """
    Fetch live Remotive jobs for `role` and upsert them, RUN-SCOPED like the Adzuna
    layer: the fetched postings are associated with THIS run's search (via
    job_search_results) so a `live_only` run actually sees them. Returns
    (inserted, skipped, status).
    """
    import time
    from job_search import create_search, associate_jobs
    start = time.time()
    inserted = skipped = 0
    jobs, status, error_message = fetch_live_jobs(role, location, limit)
    fetched = len(jobs)
    try:
        if jobs:
            conn = _get_connection()
            try:
                inserted, skipped, job_ids = upsert_live_jobs(jobs, conn=conn)
                if job_ids:
                    search_id = create_search(run_id, role, location, "remotive", conn=conn)
                    associate_jobs(search_id, job_ids, conn=conn)
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        status = "failed"
        error_message = str(e)
    latency_ms = int((time.time() - start) * 1000)
    _log_remotive_call(run_id, step_id, role, location, latency_ms,
                       fetched, inserted, skipped, status, error_message)
    if fetched:
        log.info("remotive: fetched %d, added %d new, %d already known (%dms)",
                 fetched, inserted, skipped, latency_ms)
    return (inserted, skipped, status)


if __name__ == "__main__":
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "engineer"
    jobs, _status, _err = fetch_live_jobs(role)
    print(f"Fetched {len(jobs)} live jobs for '{role}':")
    for j in jobs:
        print(f"  - {j['title']} @ {j['company']} [{j['employment_type']}] ({j['external_id']})")
    if len(sys.argv) > 2 and sys.argv[2] == "--upsert":
        ins, skip = upsert_live_jobs(jobs)
        print(f"\nUpsert: {ins} inserted, {skip} skipped (already in pool)")