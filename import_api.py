import requests
import re
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def import_jobs_from_api(search_term="python", limit=5):
    url = "https://remotive.com/api/remote-jobs"
    params = {"search": search_term, "limit": limit}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    jobs = data.get("jobs", [])[:limit]
    if not jobs:
        print(f"No jobs returned for '{search_term}'")
        return

    conn = get_connection()
    cur = conn.cursor()

    # idempotent: clear previous API imports so re-running doesn't duplicate
    cur.execute("DELETE FROM job_postings WHERE source = 'api'")

    imported = 0
    for job in jobs:
        title = job.get("title", "").strip()
        company = job.get("company_name", "").strip()
        description = job.get("description", "")
        description = re.sub(r"<[^>]+>", " ", description)          # strip HTML tags
        description = re.sub(r"\s+", " ", description).strip()[:2000]  # collapse whitespace, cap length

        if not title or not description:
            continue

        cur.execute("""
            INSERT INTO job_postings (title, company, description, location, work_mode, source)
            VALUES (%s, %s, %s, %s, %s, 'api')
        """, (
            title, company, description,
            job.get("candidate_required_location", ""), "remote"
        ))
        imported += 1

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM job_postings")
    total = cur.fetchone()[0]
    conn.close()
    print(f"Imported {imported} jobs from Remotive API. Total in table: {total}")


if __name__ == "__main__":
    import_jobs_from_api("python", 5)