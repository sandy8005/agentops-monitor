import psycopg2, os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# Generic role words too common to distinguish a role on their own.
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


STALE_AFTER_DAYS = 14   # jobs not seen in this many days drop out of search


def _role_matcher(target_role):
    """
    Require at least one SPECIALIZING term from the query (e.g. 'ai', 'ml',
    'backend'), matched as a WHOLE WORD — so short terms like 'ai'/'ml' don't
    false-match inside 'airline'/'HTML'. Multi-word terms phrase-match. If the
    query is only generic words, match on those.
    """
    words = [w.lower() for w in target_role.replace("/", " ").split() if w.strip()]
    specializing = [w for w in words if w not in GENERIC_ROLE_WORDS]
    generic = [w for w in words if w in GENERIC_ROLE_WORDS]

    def _term_present(term, haystack_lower, haystack_tokens):
        t = term.strip().lower()
        if not t:
            return False
        if " " in t:                       # multi-word: phrase match
            return t in haystack_lower
        return t in haystack_tokens        # single word: WHOLE-WORD (token) match

    def matches(job):
        haystack_lower = f"{job['title']} {job['description']}".lower()
        haystack_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", haystack_lower))
        terms = specializing if specializing else generic
        return any(_term_present(t, haystack_lower, haystack_tokens) for t in terms)

    return matches


# Source preference: when two rows are the same posting, keep the better one.
# Adzuna (real search) > live/api (Remotive) > scraped > csv > seed.
_SOURCE_RANK = {"adzuna": 5, "live": 4, "api": 3, "scraped": 2, "csv": 1, "seed": 0}


def _dedupe_key(job):
    """
    Strongest available identity for a posting, in priority order:
      1. external_id                 (stable per-posting id from the source)
      2. apply/source URL            (unique per posting when present)
      3. title + company + LOCATION  (same role in two cities stays distinct)
    Including LOCATION in the fallback means 'AI Engineer @ Wipro' in Texas and
    in California are treated as DIFFERENT jobs, not merged.
    """
    ext = (job.get("external_id") or "").strip().lower()
    if ext:
        return ("ext", ext)
    url = (job.get("url") or job.get("source_url") or "").strip().lower()
    if url:
        return ("url", url)
    return ("tcl",
            (job.get("title") or "").strip().lower(),
            (job.get("company") or "").strip().lower(),
            (job.get("location") or "").strip().lower())


def _dedupe_jobs(jobs):
    """
    Collapse duplicate postings using the strongest available identity
    (external_id -> URL -> title+company+location). On a collision, keep the row
    from the PREFERRED source (Adzuna > Remotive > scraped > ...), so a better/
    newer record wins over an older one instead of just 'first seen'.
    """
    best = {}
    order = []
    for j in jobs:
        key = _dedupe_key(j)
        if key not in best:
            best[key] = j
            order.append(key)
        else:
            cur = best[key]
            if (_SOURCE_RANK.get((j.get("source") or "").lower(), -1) >
                    _SOURCE_RANK.get((cur.get("source") or "").lower(), -1)):
                best[key] = j
    return [best[k] for k in order]


def search_jobs(target_role=None, location=None, work_mode=None,
                employment_type=None, min_results=3):
    conn = get_connection()
    cur = conn.cursor()
    # Select provenance (source) + search_location (what location a job was
    # FETCHED for) — both are valuable trace/filter context.
    cur.execute("""
        SELECT id, title, company, description, location, work_mode, employment_type, source, search_location, external_id, last_seen_at
        FROM job_postings ORDER BY id
    """)
    rows = cur.fetchall()
    conn.close()

    all_jobs = [
        {"id": r[0], "title": r[1], "company": r[2], "description": r[3],
         "location": r[4], "work_mode": r[5], "employment_type": r[6], "source": r[7],
         "search_location": r[8], "external_id": r[9], "last_seen_at": r[10]}
        for r in rows
    ]

    # --- freshness filter: drop jobs not seen recently ---
    # Jobs with no last_seen_at (seed/csv/scraped — not time-based) are always fresh.
    cutoff = datetime.now() - timedelta(days=STALE_AFTER_DAYS)
    def _is_fresh(job):
        ls = job.get("last_seen_at")
        return ls is None or ls >= cutoff
    all_jobs = [j for j in all_jobs if _is_fresh(j)]

    if not target_role:
        return all_jobs

    # --- role filter: whole-word specializing-term match ---
    role_matches = _role_matcher(target_role)
    filtered = [j for j in all_jobs if role_matches(j)]

    # --- location filter (on search_location = fetch intent, not messy display address) ---
    # Keep a job if it has no search_location (seed/csv/scraped — location-agnostic)
    # OR its search_location matches the requested location.
    if location and location.strip():
        loc = location.strip().lower()

        def location_ok(job):
            sl = (job.get("search_location") or "").strip().lower()
            if not sl:
                return True
            return sl == loc

        filtered = [j for j in filtered if location_ok(j)]

    # --- work_mode filter (compatibility, not "remote is always OK") ---
    # A job matches if: its mode is unknown (soft — don't exclude), OR equals the
    # request, OR hybrid is involved (partial match either way). A remote job is
    # correctly EXCLUDED from an onsite request (and vice versa).
    if work_mode:
        wm = work_mode.lower().strip()

        def mode_ok(job):
            jm = (job.get("work_mode") or "").lower().strip()
            jl = (job.get("location") or "").lower()
            if not jm and "remote" in jl:
                jm = "remote"
            if not jm:
                return True          # unknown mode → don't exclude (soft filter)
            if wm == jm:
                return True          # exact match
            if "hybrid" in (wm, jm):
                return True          # hybrid is a partial match either direction
            return False             # clear conflict (e.g. remote job, onsite request)

        filtered = [j for j in filtered if mode_ok(j)]

    # --- employment_type filter (SOFT: keep unknown-type, exclude known mismatch) ---
    if employment_type:
        et = employment_type.lower().strip()

        def type_ok(job):
            jt = (job.get("employment_type") or "").lower().strip()
            if not jt:
                return True
            return jt == et

        filtered = [j for j in filtered if type_ok(j)]

    # Collapse same-posting duplicates that entered via multiple sources.
    filtered = _dedupe_jobs(filtered)

    # --- results handling: honest empty result, never manufactured jobs ---
    if len(filtered) == 0:
        print(f"  (no jobs matched '{target_role}' with the given filters)")
        return []

    print(f"  ({len(filtered)} of {len(all_jobs)} job(s) matched your criteria)")
    return filtered