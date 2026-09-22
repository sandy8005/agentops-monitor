"""
Live job fetching from the Adzuna API — REAL jobs with REAL search.

Unlike Remotive (remote-only, ignores the search term), Adzuna filters by BOTH
role (`what`) and location (`where`), so "AI Engineer" in "Texas" returns actual
AI Engineer jobs in Texas. Free tier; requires ADZUNA_APP_ID + ADZUNA_APP_KEY
in .env. Same fetch → normalize → dedup → upsert shape as live_jobs.py.
"""
import hashlib
import requests
from datetime import datetime
from settings import settings
from logging_config import get_logger
log = get_logger(__name__)

ADZUNA_APP_ID = settings.adzuna_app_id
ADZUNA_APP_KEY = settings.adzuna_app_key
ADZUNA_COUNTRY = settings.adzuna_country   # us, gb, au, etc.


def _external_id(job):
    """Stable dedup key. Adzuna gives each job an 'id'; else hash title+company."""
    rid = job.get("id")
    if rid:
        return f"adzuna:{rid}"
    basis = f"{job.get('title','')}|{(job.get('company') or {}).get('display_name','')}"
    return "adzuna:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _infer_employment_type(contract_time, contract_type):
    # Adzuna gives contract_time (full_time/part_time) and contract_type
    # (permanent/contract). Both are frequently ABSENT. When neither is present we
    # return "" (unknown) rather than guessing "full-time" — the employment_type
    # filter is soft and keeps unknown-type jobs, so a real signal is only asserted
    # when Adzuna actually provided one.
    ct = (contract_time or "").lower()
    cty = (contract_type or "").lower()
    if "part" in ct:
        return "part-time"
    if "full" in ct:
        return "full-time"
    if "contract" in cty:
        return "contract"
    if "permanent" in cty:
        return "full-time"
    return ""   # no signal → unknown, not a guessed full-time


def fetch_adzuna_jobs(role, location=None, limit=10):
    """
    Fetch REAL jobs from Adzuna matching role + location.

    Returns a 3-tuple (jobs, status, error_message) so the caller can distinguish
    WHY a fetch produced no jobs instead of collapsing everything to "empty":
      - "missing_keys"  : ADZUNA_APP_ID / ADZUNA_APP_KEY not configured
      - "auth_error"    : HTTP 401/403 — bad/expired credentials
      - "rate_limited"  : HTTP 429 — quota exhausted (retry later)
      - "http_error"    : any other non-2xx HTTP response
      - "network_error" : connection/timeout/DNS — never reached the API
      - "empty"         : API responded OK but returned zero postings
      - "success"       : one or more postings returned
    Never raises into the agent — failures are returned as status, not exceptions.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        log.warning("Adzuna keys missing (ADZUNA_APP_ID / ADZUNA_APP_KEY in .env) — skipping")
        return ([], "missing_keys", "ADZUNA_APP_ID / ADZUNA_APP_KEY not set")

    url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": limit,
        "content-type": "application/json",
    }
    if role and role.strip():
        params["what"] = role.strip()
    if location and location.strip():
        params["where"] = location.strip()

    try:
        resp = requests.get(url, params=params, timeout=20)
    except requests.exceptions.RequestException as e:
        # DNS / connection refused / timeout — request never got an HTTP response.
        msg = f"network error: {e}"
        log.warning("Adzuna fetch failed (%s) — continuing with existing pool", msg)
        return ([], "network_error", msg)

    # Classify by HTTP status BEFORE trying to parse the body.
    if resp.status_code in (401, 403):
        msg = f"auth error: HTTP {resp.status_code}"
        log.warning("Adzuna %s — check ADZUNA_APP_ID / ADZUNA_APP_KEY", msg)
        return ([], "auth_error", msg)
    if resp.status_code == 429:
        msg = "rate limited: HTTP 429"
        log.warning("Adzuna %s — quota exhausted, try later", msg)
        return ([], "rate_limited", msg)
    if not resp.ok:
        msg = f"http error: HTTP {resp.status_code}"
        log.warning("Adzuna %s — continuing with existing pool", msg)
        return ([], "http_error", msg)

    try:
        raw_jobs = resp.json().get("results", [])[:limit]
    except ValueError as e:
        # 2xx but unparseable body — treat as an HTTP-level problem, not "empty".
        msg = f"http error: bad JSON ({e})"
        log.warning("Adzuna %s — continuing with existing pool", msg)
        return ([], "http_error", msg)

    out = []
    for j in raw_jobs:
        title = (j.get("title") or "").strip()
        description = (j.get("description") or "").strip()
        if not title or not description:
            continue
        company = (j.get("company") or {}).get("display_name", "")
        loc = (j.get("location") or {}).get("display_name", "")
        out.append({
            "external_id": _external_id(j),
            "title": title,
            "company": company.strip(),
            "description": description[:4000],
            "location": loc.strip(),
            "work_mode": "",   # Adzuna doesn't cleanly label remote; leave unknown
            "employment_type": _infer_employment_type(
                j.get("contract_time"), j.get("contract_type")),
            "source": "adzuna",
            "posted_at": j.get("created"),   # Adzuna's posting timestamp (ISO string)
            "apply_url": j.get("redirect_url"),
        })

    if not out:
        # API answered fine, there just weren't any (usable) postings.
        return ([], "empty", None)
    return (out, "success", None)


def _get_connection():
    # Draw from the shared pool (database.py) rather than opening a fresh socket.
    from database import get_connection
    return get_connection()


def upsert_adzuna_jobs(jobs, conn=None):
    """
    Insert jobs not already in the pool (dedup on external_id); refresh last_seen_at
    on ones already known. No longer writes a permanent search_location onto the row
    — the job↔search↔location tie now lives in job_search_results (see job_search.py).

    Returns (inserted, skipped, job_ids) where job_ids are the job_postings.id of
    EVERY posting touched (inserted or updated), in input order — the caller uses
    these to associate the postings to the search that produced them.

    Reuses an open connection when given one so fetch+associate is one transaction.
    """
    if not jobs:
        return (0, 0, [])
    own = conn is None
    if own:
        conn = _get_connection()
    cur = conn.cursor()
    now = datetime.now()
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
                -- an existing good value with a blank/NULL one. NULLIF(...,'') turns
                -- an empty incoming string into NULL so COALESCE falls back to the
                -- stored value; a non-empty incoming value wins.
                title           = COALESCE(NULLIF(EXCLUDED.title, ''),          job_postings.title),
                company         = COALESCE(NULLIF(EXCLUDED.company, ''),        job_postings.company),
                description     = COALESCE(NULLIF(EXCLUDED.description, ''),    job_postings.description),
                location        = COALESCE(NULLIF(EXCLUDED.location, ''),       job_postings.location),
                work_mode       = COALESCE(NULLIF(EXCLUDED.work_mode, ''),      job_postings.work_mode),
                employment_type = COALESCE(NULLIF(EXCLUDED.employment_type, ''),job_postings.employment_type),
                posted_at       = COALESCE(EXCLUDED.posted_at,                  job_postings.posted_at),
                apply_url       = COALESCE(NULLIF(EXCLUDED.apply_url, ''),      job_postings.apply_url),
                -- last_seen_at always advances to the newest sighting; fetched_at
                -- is left untouched so it keeps the ORIGINAL first-seen time.
                last_seen_at    = EXCLUDED.last_seen_at
            RETURNING id, (xmax = 0) AS was_inserted
        """, (j["title"], j["company"], j["description"], j["location"],
              j["work_mode"], j["employment_type"], j["source"], j["external_id"],
              now, now, j.get("posted_at"), j.get("apply_url")))
        row = cur.fetchone()
        job_ids.append(row[0])
        if row[1]:                 # xmax = 0 → this row was a fresh INSERT, not an UPDATE
            inserted += 1
    if own:
        conn.commit()
        cur.close()
        conn.close()
    return (inserted, len(jobs) - inserted, job_ids)


def _log_adzuna_call(run_id, step_id, role, location, latency_ms,
                     fetched, inserted, duplicates, status, error_message):
    """
    Write an AgentOps trace row for the Adzuna external API call — so its latency,
    jobs returned, inserts, duplicates, and errors are observed like every other
    tool. No-op if run_id/step_id aren't provided (e.g. standalone CLI use).
    """
    if run_id is None or step_id is None:
        return
    try:
        import json
        from datetime import datetime
        from llm import get_connection
        conn = get_connection()
        cur = conn.cursor()
        input_json = json.dumps({"role": role, "location": location})
        output_json = json.dumps({
            "fetched": fetched, "inserted": inserted, "duplicates": duplicates,
        })
        cur.execute("""
            INSERT INTO tool_calls
            (run_id, step_id, tool_name, input_json, output_json, latency_ms,
             status, error_message, created_at, operation_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (run_id, step_id, "adzuna_fetch", input_json, output_json, latency_ms,
              status, error_message, datetime.now(), "live_fetch"))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as log_err:
        # Never let trace-logging break the run.
        log.warning("adzuna trace log failed: %s", log_err)


def fetch_and_upsert_adzuna(role, location=None, limit=10, run_id=None, step_id=None):
    """
    Fetch real Adzuna jobs for role+location and upsert them. The single call the
    agent makes. Now OBSERVED with a CLASSIFIED status (success / empty /
    missing_keys / auth_error / rate_limited / http_error / network_error) rather
    than collapsing every non-result into "empty", so the trace shows WHY a fetch
    produced nothing. Returns (inserted, skipped).
    """
    import time
    from job_search import create_search, associate_jobs
    start = time.time()
    status = "success"
    error_message = None
    fetched = inserted = skipped = 0
    try:
        jobs, fetch_status, fetch_error = fetch_adzuna_jobs(role, location, limit)
        fetched = len(jobs)
        # Carry the fetch's classified status/message straight through to the trace.
        status = fetch_status
        error_message = fetch_error
        if jobs:
            # One transaction: upsert the postings, record THIS search, associate the
            # returned postings to it. The search (not the job row) now carries the
            # role/location intent, so the same posting can join multiple searches.
            conn = _get_connection()
            try:
                inserted, skipped, job_ids = upsert_adzuna_jobs(jobs, conn=conn)
                if job_ids:
                    search_id = create_search(run_id, role, location, "adzuna", conn=conn)
                    associate_jobs(search_id, job_ids, conn=conn)
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        # Only reaches here for UNEXPECTED errors (e.g. DB failure during upsert);
        # fetch-level problems are already classified and returned as status above.
        status = "failed"
        error_message = str(e)
    latency_ms = int((time.time() - start) * 1000)

    _log_adzuna_call(run_id, step_id, role, location, latency_ms,
                     fetched, inserted, skipped, status, error_message)

    if fetched:
        log.info("adzuna: fetched %s, added %s new, %s already known (%sms)", fetched, inserted, skipped, latency_ms)
    return (inserted, skipped)


if __name__ == "__main__":
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "engineer"
    loc = sys.argv[2] if len(sys.argv) > 2 else None
    jobs, status, error = fetch_adzuna_jobs(role, loc)
    where = f" in '{loc}'" if loc else ""
    print(f"Fetched {len(jobs)} Adzuna jobs for '{role}'{where} [status: {status}"
          + (f", {error}" if error else "") + "]:")
    for j in jobs:
        print(f"  - {j['title']} @ {j['company']} "
              f"[{j['employment_type'] or 'unknown'}] @ {j['location']} ({j['external_id']})")