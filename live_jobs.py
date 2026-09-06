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
    Fetch live jobs from Remotive matching `role`. Returns a list of normalized
    job dicts (NOT yet in the DB). `location` is accepted but Remotive is
    remote-only, so it's informational (matches the project's location policy).
    Returns [] on any API failure — never raises into the agent.
    """
    try:
        params = {"limit": limit}
        if role and role.strip():
            params["search"] = role.strip()
        headers = {"User-Agent": "AgentOpsMonitor/1.0 (educational project)"}
        resp = requests.get(REMOTIVE_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        raw_jobs = resp.json().get("jobs", [])[:limit]
    except Exception as e:
        print(f"    live fetch failed ({e}) — continuing with existing pool")
        return []

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
    return out



def _get_connection():
    import psycopg2, os
    from dotenv import load_dotenv
    load_dotenv()
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))


def upsert_live_jobs(jobs):
    """
    Insert live jobs that aren't already in the pool (dedup on external_id).
    ON CONFLICT DO NOTHING means re-fetching the same posting is a no-op — no
    duplicates. Returns (inserted_count, skipped_count).
    """
    if not jobs:
        return (0, 0)
    conn = _get_connection()
    cur = conn.cursor()
    inserted = 0
    for j in jobs:
        cur.execute("""
            INSERT INTO job_postings
            (title, company, description, location, work_mode, employment_type, source, external_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id) WHERE external_id IS NOT NULL DO NOTHING
        """, (j["title"], j["company"], j["description"], j["location"],
              j["work_mode"], j["employment_type"], j["source"], j["external_id"]))
        inserted += cur.rowcount  # 1 if inserted, 0 if skipped (conflict)
    conn.commit()
    cur.close()
    conn.close()
    skipped = len(jobs) - inserted
    return (inserted, skipped)


def fetch_and_upsert(role, location=None, limit=10):
    """Fetch live jobs for `role` and upsert them. Returns (inserted, skipped).
    This is the single call the agent makes before searching."""
    jobs = fetch_live_jobs(role, location, limit)
    inserted, skipped = upsert_live_jobs(jobs)
    if jobs:
        print(f"    live: fetched {len(jobs)}, added {inserted} new, {skipped} already known")
    return (inserted, skipped)


if __name__ == "__main__":
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "engineer"
    jobs = fetch_live_jobs(role)
    print(f"Fetched {len(jobs)} live jobs for '{role}':")
    for j in jobs:
        print(f"  - {j['title']} @ {j['company']} [{j['employment_type']}] ({j['external_id']})")
    if len(sys.argv) > 2 and sys.argv[2] == "--upsert":
        ins, skip = upsert_live_jobs(jobs)
        print(f"\nUpsert: {ins} inserted, {skip} skipped (already in pool)")