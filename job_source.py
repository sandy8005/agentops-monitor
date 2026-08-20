import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

# Generic role words that are too common to distinguish a role on their own.
# A search for "AI/ML Engineer" should require a SPECIALIZING term (ai, ml, ...),
# not merely "engineer" — otherwise Civil/QA/Network Engineer all match.
GENERIC_ROLE_WORDS = {
    "engineer", "developer", "analyst", "manager", "specialist", "consultant",
    "administrator", "architect", "designer", "coordinator", "lead", "senior",
    "junior", "staff", "principal", "associate", "intern", "assistant",
    "of", "the", "and", "or", "a", "an", "i", "ii", "iii"
}


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def _role_matcher(target_role):
    """
    Build a matcher that requires at least one SPECIALIZING term from the query
    (e.g. 'ai', 'ml', 'backend'), so a generic word like 'engineer' alone does
    NOT pull in unrelated roles (Civil Engineer, QA Engineer, ...).
    If the query has only generic words, fall back to matching those.
    """
    words = [w.lower() for w in target_role.replace("/", " ").split() if w.strip()]
    specializing = [w for w in words if w not in GENERIC_ROLE_WORDS]
    generic = [w for w in words if w in GENERIC_ROLE_WORDS]

    def matches(job):
        haystack = f"{job['title']} {job['description']}".lower()
        if specializing:
            # Require at least one specializing term to be present.
            return any(term in haystack for term in specializing)
        # Query was only generic words — match on those instead.
        return any(term in haystack for term in generic)

    return matches


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

    # --- role filter: require a specializing term, not just a generic word ---
    role_matches = _role_matcher(target_role)
    filtered = [j for j in all_jobs if role_matches(j)]

    # --- work_mode filter (only excludes jobs that HAVE a mode and clearly conflict) ---
    # Location is intentionally NOT used to filter (informational only).
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

    # --- employment_type filter (SOFT: keep unknown-type jobs, exclude known mismatches) ---
    if employment_type:
        et = employment_type.lower().strip()

        def type_ok(job):
            jt = (job.get("employment_type") or "").lower().strip()
            if not jt:
                return True
            return jt == et

        filtered = [j for j in filtered if type_ok(j)]

    # --- results handling: honest empty result, never manufactured jobs ---
    if len(filtered) == 0:
        print(f"  (no jobs matched '{target_role}' with the given filters)")
        return []

    print(f"  ({len(filtered)} of {len(all_jobs)} job(s) matched your criteria)")
    return filtered