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

# Which sources count as LIVE (real-time feeds) vs. practice data (seed/CSV/scraped).
# Live Mode reads ONLY live sources, so a live search is never polluted by the
# built-in sample/practice pool.
LIVE_SOURCES = {"adzuna", "live"}
PRACTICE_SOURCES = {"seed", "csv", "scraped", "api"}


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

# Seniority / qualifier words stripped from a title before fingerprinting, so
# "Senior AI Engineer" and "AI Engineer" from two providers fingerprint the same.
_TITLE_NOISE = {
    "senior", "sr", "junior", "jr", "staff", "principal", "lead", "associate",
    "entry", "level", "mid", "i", "ii", "iii", "iv",
}
# Company-suffix noise stripped so "Wipro Inc." == "Wipro" == "Wipro, LLC".
_COMPANY_NOISE = {
    "inc", "inc.", "llc", "ltd", "ltd.", "limited", "corp", "corp.",
    "corporation", "co", "co.", "company", "gmbh", "plc", "pvt", "private",
}


def _norm_token_string(text, drop=frozenset()):
    """
    Lowercase, split on non-alphanumerics (keeping + and # so 'c++'/'c#' survive),
    drop noise tokens, and rejoin with single spaces. Provider-independent — the
    same posting from two feeds normalizes to the same string regardless of
    punctuation, casing, or decorative words.
    """
    if not text:
        return ""
    tokens = re.findall(r"[a-z0-9\+\#]+", text.lower())
    kept = [t for t in tokens if t not in drop]
    return " ".join(kept)


def _fingerprint(job):
    """
    Provider-INDEPENDENT identity for a posting: normalized title + company +
    location. This is the whole redesign — external_id and apply URLs are
    provider-namespaced ('adzuna:1' vs 'remotive:2') and NEVER match across feeds,
    so the same job on Adzuna and Remotive used to slip through as two rows. A
    content fingerprint collapses them.

    LOCATION stays in the key (normalized), so 'AI Engineer @ Wipro' in Texas and
    in California remain DISTINCT — matching the existing location-aware intent.
    """
    return (
        _norm_token_string(job.get("title"), drop=_TITLE_NOISE),
        _norm_token_string(job.get("company"), drop=_COMPANY_NOISE),
        _norm_token_string(job.get("location")),
    )


# Fields worth salvaging from a discarded duplicate when the winner lacks them.
_MERGE_FILL = ("apply_url", "url", "source_url", "external_id",
               "posted_at", "work_mode", "employment_type")


def _merge_duplicate(winner, loser):
    """
    Winner (higher source-rank) keeps its identity, but we backfill any USEFUL
    field it's missing from the loser, and carry the freshest last_seen_at. So a
    high-rank Adzuna row that lacks an apply_url can inherit one from a Remotive
    duplicate instead of losing it. Never overwrites a value the winner already has.
    """
    for f in _MERGE_FILL:
        if not winner.get(f) and loser.get(f):
            winner[f] = loser.get(f)
    # Freshness is a max, not a preference — the most recent sighting wins.
    lw, ll = winner.get("last_seen_at"), loser.get("last_seen_at")
    if ll is not None and (lw is None or ll > lw):
        winner["last_seen_at"] = ll
    return winner


def _rank(job):
    return _SOURCE_RANK.get((job.get("source") or "").lower(), -1)


def _dedupe_jobs(jobs):
    """
    Collapse cross-provider duplicates on a provider-independent content
    fingerprint (normalized title + company + location). On a collision the
    HIGHER-source-rank row wins (Adzuna > Remotive > scraped > csv > seed), but
    useful fields from the loser (apply_url, dates, external_id, ...) are merged
    into the winner so nothing valuable is dropped. First-seen order is preserved.
    """
    best = {}
    order = []
    for j in jobs:
        key = _fingerprint(j)
        if key not in best:
            best[key] = dict(j)          # copy — merging mutates the kept row
            order.append(key)
            continue
        cur = best[key]
        if _rank(j) > _rank(cur):
            winner, loser = dict(j), cur   # new row outranks: it becomes the keeper
        else:
            winner, loser = cur, j          # existing row stays the keeper
        best[key] = _merge_duplicate(winner, loser)
    return [best[k] for k in order]


def search_jobs(target_role=None, location=None, work_mode=None,
                employment_type=None, min_results=3, live_only=False, run_id=None):
    """
    Search the job pool.

    Sources are now separated:
      - live_only=True  → LIVE MODE: read ONLY live-sourced jobs (Adzuna/Remotive),
        never the seed/CSV/scraped practice pool. When run_id is given, scope further
        to just the postings THIS run fetched live (its per-search associations), so
        a live search returns exactly what this run pulled — nothing stale, nothing
        from other runs, no practice data.
      - live_only=False → MIXED MODE (default/back-compat): the combined pool, with
        practice data treated as location-agnostic (kept for every location).

    Location filtering uses the PER-SEARCH ASSOCIATION (job_search_results), not a
    permanent search_location column: a job matches a location if some search for
    that location returned it. Practice jobs (no association) stay location-agnostic
    in mixed mode.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Pull provenance (source) and, via the association join, the set of locations
    # any search has ever returned this posting for. search_location is no longer
    # read for filtering (kept only until a later migration drops it).
    cur.execute("""
        SELECT p.id, p.title, p.company, p.description, p.location, p.work_mode,
               p.employment_type, p.source, p.external_id, p.last_seen_at,
               COALESCE(
                   ARRAY_AGG(DISTINCT lower(s.location))
                   FILTER (WHERE s.location IS NOT NULL),
                   '{}'
               ) AS assoc_locations
        FROM job_postings p
        LEFT JOIN job_search_results r ON r.job_id = p.id
        LEFT JOIN job_searches s ON s.id = r.search_id
        GROUP BY p.id
        ORDER BY p.id
    """)
    rows = cur.fetchall()

    # For Live Mode scoped to a single run, which postings did THIS run fetch?
    run_job_ids = set()
    if live_only and run_id is not None:
        cur.execute("""
            SELECT DISTINCT r.job_id
            FROM job_search_results r
            JOIN job_searches s ON s.id = r.search_id
            WHERE s.run_id = %s
        """, (run_id,))
        run_job_ids = {row[0] for row in cur.fetchall()}
    conn.close()

    all_jobs = [
        {"id": r[0], "title": r[1], "company": r[2], "description": r[3],
         "location": r[4], "work_mode": r[5], "employment_type": r[6], "source": r[7],
         "external_id": r[8], "last_seen_at": r[9],
         "assoc_locations": set(r[10] or [])}
        for r in rows
    ]

    # --- source separation: Live Mode excludes all practice data ---
    if live_only:
        all_jobs = [j for j in all_jobs if (j.get("source") or "").lower() in LIVE_SOURCES]
        # When a run_id is given, restrict to postings THIS run actually fetched.
        if run_id is not None:
            all_jobs = [j for j in all_jobs if j["id"] in run_job_ids]

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

    # --- location filter (via PER-SEARCH association, not a permanent column) ---
    # A job matches the requested location if SOME search for that location
    # returned it (its assoc_locations contains the request). A job with no
    # association at all (seed/csv/scraped practice data) is location-agnostic and
    # kept — but only in mixed mode; in Live Mode there is no practice data and
    # every live job carries the association from the search that fetched it.
    if location and location.strip():
        loc = location.strip().lower()

        def location_ok(job):
            assoc = job.get("assoc_locations") or set()
            if not assoc:
                return True          # unassociated → location-agnostic (practice data)
            return loc in assoc      # matched by at least one search for this location

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