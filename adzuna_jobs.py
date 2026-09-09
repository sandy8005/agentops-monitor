"""
Live job fetching from the Adzuna API — REAL jobs with REAL search.

Unlike Remotive (remote-only, ignores the search term), Adzuna filters by BOTH
role (`what`) and location (`where`), so "AI Engineer" in "Texas" returns actual
AI Engineer jobs in Texas. Free tier; requires ADZUNA_APP_ID + ADZUNA_APP_KEY
in .env. Same fetch → normalize → dedup → upsert shape as live_jobs.py.
"""
import os
import hashlib
import requests
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")   # us, gb, au, etc.


def _external_id(job):
    """Stable dedup key. Adzuna gives each job an 'id'; else hash title+company."""
    rid = job.get("id")
    if rid:
        return f"adzuna:{rid}"
    basis = f"{job.get('title','')}|{(job.get('company') or {}).get('display_name','')}"
    return "adzuna:" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _infer_employment_type(contract_time, contract_type):
    # Adzuna gives contract_time (full_time/part_time) and contract_type (permanent/contract)
    ct = (contract_time or "").lower()
    cty = (contract_type or "").lower()
    if "part" in ct:
        return "part-time"
    if "contract" in cty:
        return "contract"
    return "full-time"


def fetch_adzuna_jobs(role, location=None, limit=10):
    """
    Fetch REAL jobs from Adzuna matching role + location. Returns normalized job
    dicts (not yet in DB). Returns [] on any failure — never raises into the agent.
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        print("    Adzuna keys missing (ADZUNA_APP_ID / ADZUNA_APP_KEY in .env) — skipping")
        return []
    try:
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
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        raw_jobs = resp.json().get("results", [])[:limit]
    except Exception as e:
        print(f"    Adzuna fetch failed ({e}) — continuing with existing pool")
        return []

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
            "search_location": (location or "").strip() or None,
        })
    return out


def _get_connection():
    import psycopg2
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))


def upsert_adzuna_jobs(jobs):
    """Insert jobs not already in the pool (dedup on external_id). Returns (inserted, skipped)."""
    if not jobs:
        return (0, 0)
    conn = _get_connection()
    cur = conn.cursor()
    inserted = 0
    for j in jobs:
        cur.execute("""
            INSERT INTO job_postings
            (title, company, description, location, work_mode, employment_type, source, external_id, search_location)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (external_id) WHERE external_id IS NOT NULL DO NOTHING
        """, (j["title"], j["company"], j["description"], j["location"],
              j["work_mode"], j["employment_type"], j["source"], j["external_id"],
              j.get("search_location")))
        inserted += cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return (inserted, len(jobs) - inserted)


def fetch_and_upsert_adzuna(role, location=None, limit=10):
    """Fetch real Adzuna jobs for role+location and upsert them. The single call
    the agent makes. Returns (inserted, skipped)."""
    jobs = fetch_adzuna_jobs(role, location, limit)
    inserted, skipped = upsert_adzuna_jobs(jobs)
    if jobs:
        print(f"    adzuna: fetched {len(jobs)}, added {inserted} new, {skipped} already known")
    return (inserted, skipped)


if __name__ == "__main__":
    import sys
    role = sys.argv[1] if len(sys.argv) > 1 else "engineer"
    loc = sys.argv[2] if len(sys.argv) > 2 else None
    jobs = fetch_adzuna_jobs(role, loc)
    print(f"Fetched {len(jobs)} Adzuna jobs for '{role}'" + (f" in '{loc}'" if loc else "") + ":")
    for j in jobs:
        print(f"  - {j['title']} @ {j['company']} [{j['employment_type']}] @ {j['location']} ({j['external_id']})")