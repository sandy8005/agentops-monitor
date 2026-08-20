import psycopg2, os
from dotenv import load_dotenv
load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def search_jobs(target_role=None, location=None, work_mode=None,
                employment_type=None, min_results=3):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, company, description, location, work_mode, employment_type
        FROM job_postings ORDER BY id
    """)
    rows = cur.fetchall()
    conn.close()

    all_jobs = [
        {"id": r[0], "title": r[1], "company": r[2], "description": r[3],
         "location": r[4], "work_mode": r[5], "employment_type": r[6]}
        for r in rows
    ]

    if not target_role:
        return all_jobs

    # --- role filter (keyword match on title + description) ---
    keywords = [w.lower() for w in target_role.replace("/", " ").split()]

    def role_matches(job):
        haystack = f"{job['title']} {job['description']}".lower()
        return any(kw in haystack for kw in keywords)

    filtered = [j for j in all_jobs if role_matches(j)]

    # --- work_mode filter (only excludes jobs that HAVE a mode and clearly conflict) ---
    # Location is intentionally NOT used to filter: the user sees jobs from all
    # locations. Location is collected and displayed but does not filter or score.
    if work_mode:
        wm = work_mode.lower()

        def mode_ok(job):
            jm = (job.get("work_mode") or "").lower()
            jl = (job.get("location") or "").lower()
            if not jm and "remote" not in jl:
                return True
            if "remote" in jm or "remote" in jl:
                return True
            return wm in jm

        filtered = [j for j in filtered if mode_ok(j)]

    # --- employment_type filter (SOFT: exclude only jobs that HAVE a type that
    # clearly conflicts; keep jobs with no type data so the pool isn't wiped out) ---
    if employment_type:
        et = employment_type.lower().strip()

        def type_ok(job):
            jt = (job.get("employment_type") or "").lower().strip()
            if not jt:
                return True          # no type data → don't exclude (missing != mismatch)
            return jt == et          # keep only jobs whose type matches the request

        filtered = [j for j in filtered if type_ok(j)]

    # --- results handling ---
    # Return the genuine matches, even if few. Do NOT dilute with irrelevant jobs
    # — that wastes LLM tokens/quota evaluating things the user didn't ask for.
    if len(filtered) == 0:
        sample = all_jobs[:min_results]
        print(f"  (no jobs matched '{target_role}' — showing {len(sample)} sample job(s); "
              f"try a broader role or different filters)")
        return sample

    print(f"  ({len(filtered)} of {len(all_jobs)} job(s) matched your criteria)")
    return filtered