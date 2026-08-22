import requests
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

REMOTIVE_API = "https://remotive.com/api/remote-jobs"


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def map_remotive_type(job_type):
    """Map Remotive's job_type (e.g. 'full_time', 'contract') to our vocabulary."""
    jt = (job_type or "").lower().replace("_", "-").strip()
    if "intern" in jt:
        return "internship"
    if "part-time" in jt:
        return "part-time"
    if "contract" in jt or "freelance" in jt:
        return "contract"
    return "full-time"


def clean_description(raw):
    """
    Remotive returns HTML in the description. Strip tags to plain text so the
    requirement-extractor sees readable content, not markup.
    """
    if not raw:
        return ""
    try:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    except Exception:
        # Fallback: crude tag strip if BeautifulSoup isn't available for any reason.
        import re
        text = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace and cap length so we don't store huge blobs.
    text = " ".join(text.split())
    return text[:4000]


def fetch_remotive_jobs(limit=5, search=None):
    """Fetch remote jobs from the Remotive public API."""
    params = {"limit": limit}
    if search:
        params["search"] = search
    headers = {"User-Agent": "AgentOpsMonitor/1.0 (educational project)"}
    resp = requests.get(REMOTIVE_API, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("jobs", [])[:limit]


def import_api_jobs(limit=5, search=None):
    raw_jobs = fetch_remotive_jobs(limit=limit, search=search)
    if not raw_jobs:
        print("No jobs returned from Remotive API.")
        return

    conn = get_connection()
    cur = conn.cursor()
    # Idempotent: clear this source's previous rows before re-importing.
    cur.execute("DELETE FROM job_postings WHERE source = 'api'")

    imported = 0
    for j in raw_jobs:
        title = (j.get("title") or "").strip()
        company = (j.get("company_name") or "").strip()
        description = clean_description(j.get("description"))
        location = (j.get("candidate_required_location") or "").strip()
        employment_type = map_remotive_type(j.get("job_type"))
        # Remotive jobs are remote by definition.
        work_mode = "remote"

        if not title or not description:
            print(f"  skipping incomplete job: {title or '(no title)'}")
            continue

        cur.execute("""
            INSERT INTO job_postings
            (title, company, description, location, work_mode, employment_type, source)
            VALUES (%s, %s, %s, %s, %s, %s, 'api')
        """, (title, company, description, location, work_mode, employment_type))
        imported += 1

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM job_postings")
    total = cur.fetchone()[0]
    conn.close()
    print(f"Imported {imported} jobs from Remotive API. Total in table: {total}")


if __name__ == "__main__":
    import_api_jobs()