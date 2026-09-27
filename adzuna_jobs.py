"""
Live job fetching from the Adzuna API — REAL jobs with REAL search.

Unlike Remotive (remote-only, ignores the search term), Adzuna filters by BOTH
role (`what`) and location (`where`), so "AI Engineer" in "Texas" returns actual
AI Engineer jobs in Texas. Free tier; requires ADZUNA_APP_ID + ADZUNA_APP_KEY
in .env. Same fetch → normalize → dedup → upsert shape as live_jobs.py.
"""
import hashlib
import time
import requests
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


def _parse_created(value):
    """Adzuna 'created' (e.g. '2026-09-20T14:03:11Z') -> aware UTC datetime, or None."""
    if not value:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fetch_adzuna_jobs(role, location=None, limit=10):
    """
    Fetch REAL jobs from Adzuna matching role + location.

    Returns a 3-tuple (jobs, status, error_message) so the caller can distinguish
    WHY a fetch produced no jobs instead of collapsing everything to "empty":
      - "missing_keys"  : ADZUNA_APP_ID / ADZUNA_APP_KEY not configured
      - "auth_error"    : HTTP 401/403 — bad/expired credentials
      - "rate_limited"  : HTTP 429 — quota exhausted (retry later)
      - "server_error"  : HTTP 5xx — provider outage (retryable)
      - "http_error"    : any other non-2xx HTTP response
      - "invalid_response": 2xx, but the body is not the expected JSON
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
    if resp.status_code >= 500:
        msg = f"server error: HTTP {resp.status_code}"
        log.warning("Adzuna %s — continuing with existing pool", msg)
        return ([], "server_error", msg)
    if not resp.ok:
        msg = f"http error: HTTP {resp.status_code}"
        log.warning("Adzuna %s — continuing with existing pool", msg)
        return ([], "http_error", msg)

    try:
        body = resp.json()
    except ValueError as e:
        # 2xx but unparseable body — the request worked; the RESPONSE is bad.
        msg = f"invalid response: malformed JSON ({e})"
        log.warning("Adzuna %s — continuing with existing pool", msg)
        return ([], "invalid_response", msg)
    if not isinstance(body, dict) or not isinstance(body.get("results", []), list):
        return ([], "invalid_response", "invalid response: unexpected JSON shape")
    raw_jobs = [j for j in body.get("results", []) if isinstance(j, dict)][:limit]

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
            "posted_at": _parse_created(j.get("created")),   # aware UTC datetime or None
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
    Upsert Adzuna postings through the SHARED persistence path
    (job_persistence.upsert_postings) — the same code Remotive uses, so both live
    providers get identical freshness semantics (fetched_at / last_seen_at /
    posted_at / apply_url) and timezone-aware UTC timestamps. Returns
    (inserted, skipped, job_ids).
    """
    from job_persistence import upsert_postings
    return upsert_postings(jobs, conn=conn)


def _log_adzuna_call(run_id, step_id, role, location, latency_ms,
                     fetched, inserted, duplicates, status, error_message):
    """
    Write an AgentOps trace row for the Adzuna external API call through the SHARED
    llm.log_tool_call(), so it is tagged with the run's execution attempt and uses
    UTC timestamps exactly like every other tool call. No-op without run/step ids
    (standalone CLI use). Never lets trace logging break the run.
    """
    if run_id is None or step_id is None:
        return
    try:
        from llm import log_tool_call
        log_tool_call(run_id, step_id, "adzuna_fetch", {"role": role, "location": location},
                      {"fetched": fetched, "inserted": inserted, "duplicates": duplicates,
                       "location_filter_applied": bool(location and location.strip())},
                      latency_ms, status, error_message, "live_fetch")
    except Exception as log_err:
        log.warning("adzuna trace log failed: %s", log_err)


def fetch_and_upsert_adzuna(role, location=None, limit=10, run_id=None, step_id=None):
    """
    Fetch real Adzuna jobs for role+location and upsert them. The single call the
    agent makes. Now OBSERVED with a CLASSIFIED status (success / empty /
    missing_keys / auth_error / rate_limited / http_error / network_error) rather
    than collapsing every non-result into "empty", so the trace shows WHY a fetch
    produced nothing. Returns (inserted, skipped, status).
    """
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
    return (inserted, skipped, status)


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