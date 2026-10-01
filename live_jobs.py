"""
Live job fetching from the Remotive API, per-search.

Unlike import_api.py (which batch-imports a generic set), fetch_live_jobs passes
the user's actual target role as Remotive's `search` parameter, so it returns
jobs matching THIS search. Results are normalized to the job_postings shape and
deduplicated on upsert (see upsert_live_jobs). Remotive is remote-only and free;
that's the honest limit of what this source provides.
"""
from timeutil import utcnow
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
    """Map Remotive's job_type to our vocabulary. An UNKNOWN or unrecognized value
    returns "" (unknown) — never an invented "full-time". The employment-type search
    filter is soft and already keeps unknown-type jobs, so nothing is lost by being
    honest here."""
    jt = (job_type or "").lower().replace("_", "-").strip()
    if not jt:
        return ""
    if "intern" in jt:
        return "internship"
    if "part-time" in jt or "part time" in jt:
        return "part-time"
    if "contract" in jt or "freelance" in jt:
        return "contract"
    if "full-time" in jt or "full time" in jt or "permanent" in jt:
        return "full-time"
    return ""


def _parse_published(value):
    """Remotive publication_date (e.g. '2026-09-20T14:03:11') -> aware UTC datetime,
    or None if absent/unparseable (never raises into the fetch)."""
    if not value:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


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
    a CLASSIFIED status, mirroring the Adzuna layer so the run trace shows WHY a
    fetch produced nothing:
      - "rate_limited"     : HTTP 429 (retryable)
      - "server_error"     : HTTP 5xx — provider outage (retryable)
      - "http_error"       : any other non-2xx
      - "network_error"    : connection / timeout / DNS — never got an HTTP answer
      - "invalid_response" : 2xx, but the body was not the JSON shape we expect
      - "empty" / "success"
    `location` is accepted but Remotive is remote-only, so it's informational.
    Never raises into the agent.
    """
    params = {"limit": limit}
    if role and role.strip():
        params["search"] = role.strip()
    headers = {"User-Agent": "AgentOpsMonitor/1.0 (educational project)"}
    try:
        resp = requests.get(REMOTIVE_API, params=params, headers=headers, timeout=20)
    except requests.exceptions.RequestException as e:
        from sanitize import safe_exception_summary
        msg = "network error: " + safe_exception_summary(e)
        log.warning("remotive fetch failed (%s)", msg)
        return ([], "network_error", msg)
    if resp.status_code == 429:
        return ([], "rate_limited", "Remotive rate limit (HTTP 429)")
    if resp.status_code >= 500:
        return ([], "server_error", f"server error: HTTP {resp.status_code}")
    if not (200 <= resp.status_code < 300):
        return ([], "http_error", f"http error: HTTP {resp.status_code}")
    # The request itself SUCCEEDED from here on: a body we can't parse is a
    # provider-response problem, not a network failure.
    try:
        body = resp.json()
    except ValueError as e:
        log.warning("remotive returned malformed JSON (%s)", type(e).__name__)
        return ([], "invalid_response", f"invalid response: malformed JSON ({e})")
    if not isinstance(body, dict) or not isinstance(body.get("jobs", []), list):
        return ([], "invalid_response", "invalid response: unexpected JSON shape")
    raw_jobs = [j for j in body.get("jobs", []) if isinstance(j, dict)][:limit]

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
            # Provider name, consistent with job_searches.source and Adzuna's
            # "adzuna" — per-provider metrics are a plain GROUP BY source.
            "source": "remotive",
            # Remotive's posting URL (the apply entry point) and publication date.
            "apply_url": (j.get("url") or "").strip() or None,
            "posted_at": _parse_published(j.get("publication_date")),
        })
    return (out, "success" if out else "empty", None)



def _get_connection():
    # Draw from the shared pool (database.py) rather than opening a fresh socket.
    from database import get_connection
    return get_connection()


def upsert_live_jobs(jobs, conn=None):
    """
    Upsert Remotive postings through the SAME persistence path as Adzuna
    (job_persistence.upsert_postings): fetched_at / last_seen_at / posted_at /
    apply_url are all written, and last_seen_at is refreshed on every sighting, so
    Remotive postings age out of search like any other live posting. Returns
    (inserted, skipped, job_ids).
    """
    from job_persistence import upsert_postings
    return upsert_postings(jobs, conn=conn)


def _log_remotive_call(run_id, step_id, role, location, latency_ms,
                       fetched, inserted, duplicates, status, error_message):
    """Write an AgentOps trace row for the Remotive external API call, mirroring the
    Adzuna one, so its latency/results/errors are observed like every other tool.
    No-op if run_id/step_id aren't provided (standalone CLI use)."""
    if run_id is None or step_id is None:
        return
    try:
        from llm import log_tool_call
        log_tool_call(run_id, step_id, "remotive_fetch", {"role": role, "location": location},
                      {"fetched": fetched, "inserted": inserted, "duplicates": duplicates,
                       "location_filter_applied": False},
                      latency_ms, status, error_message, "live_fetch")
    except Exception as log_err:
        log.warning("remotive trace log failed (%s)", type(log_err).__name__)


def fetch_and_upsert_remotive(role, location=None, limit=10, run_id=None, step_id=None):
    """
    Fetch live Remotive jobs for `role` and upsert them, RUN-SCOPED: the fetched
    postings are associated with THIS run's search (job_search_results). Returns
    (inserted, skipped, status).

    Provider fetch/parsing problems come back as a classified status. Database
    persistence errors are NOT converted into a provider status: the trace row
    records them, then they propagate (database_unavailable / internal).
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
                    # Remotive is remote-only and ignores location: record the search
                    # WITHOUT a location and mark that no geographic filter was
                    # applied, so the association is never read as "this job was
                    # returned for <location>".
                    search_id = create_search(run_id, role, None, "remotive", conn=conn,
                                              location_filter_applied=False)
                    associate_jobs(search_id, job_ids, conn=conn)
                conn.commit()
            finally:
                conn.close()
    except Exception as e:
        from sanitize import safe_exception_summary
        _log_remotive_call(run_id, step_id, role, location, int((time.time() - start) * 1000),
                           fetched, 0, 0, "persistence_failed", safe_exception_summary(e))
        raise
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
        ins, skip, _ids = upsert_live_jobs(jobs)
        print(f"\nUpsert: {ins} inserted, {skip} skipped (already in pool)")