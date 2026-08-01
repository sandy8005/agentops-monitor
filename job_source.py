import psycopg2, os
from dotenv import load_dotenv
load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def search_jobs(target_role=None, location=None, min_results=3):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, company, description, location, work_mode
        FROM job_postings ORDER BY id
    """)
    rows = cur.fetchall()
    conn.close()

    all_jobs = [
        {"id": r[0], "title": r[1], "company": r[2], "description": r[3],
         "location": r[4], "work_mode": r[5]}
        for r in rows
    ]

    # no target role given → return everything
    if not target_role:
        return all_jobs

    # build loose keyword list from the target role
    keywords = [w.lower() for w in target_role.replace("/", " ").split() if len(w) > 2]

    def matches(job):
        haystack = f"{job['title']} {job['description']}".lower()
        return any(kw in haystack for kw in keywords)

    filtered = [j for j in all_jobs if matches(j)]

    # safety net: if filtering is too aggressive, fall back to all
    if len(filtered) < min_results:
        print(f"  (only {len(filtered)} matched '{target_role}' — returning all {len(all_jobs)} jobs)")
        return all_jobs

    print(f"  ({len(filtered)} of {len(all_jobs)} jobs matched '{target_role}')")
    return filtered