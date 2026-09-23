from database import get_connection
import re
from datetime import datetime, timedelta

from logging_config import get_logger
log = get_logger(__name__)

# Generic role words too common to distinguish a role on their own.
GENERIC_ROLE_WORDS = {
    "engineer", "developer", "analyst", "manager", "specialist", "consultant",
    "administrator", "architect", "designer", "coordinator", "lead", "senior",
    "junior", "staff", "principal", "associate", "intern", "assistant",
    "of", "the", "and", "or", "a", "an", "i", "ii", "iii"
}


STALE_AFTER_DAYS = 14   # jobs not seen in this many days drop out of search

# Which sources count as LIVE (real-time feeds) vs. practice data (seed/CSV/scraped).
# Live Mode reads ONLY live sources, so a live search is never polluted by the
# built-in sample/practice pool.
LIVE_SOURCES = {"adzuna", "live"}
PRACTICE_SOURCES = {"seed", "csv", "scraped", "api"}


# --- Role aliases: expand a query term into equivalent phrases/abbreviations. ---
# ONE-WAY expansion: a query for any phrase in a group also searches for every
# other phrase in that group, so "ML Engineer" matches "machine learning" jobs and
# vice-versa. A job's wording never pulls in an UNRELATED query — only the typed
# query is expanded, not the job text. To extend, add a group (list of equivalent
# phrases, any casing); every phrase in the group becomes a mutual alias.
ROLE_ALIAS_GROUPS = [
    ["ml", "machine learning"],
    ["ai", "artificial intelligence"],
    ["nlp", "natural language processing"],
    ["cv", "computer vision"],
    ["frontend", "front end", "front-end"],
    ["backend", "back end", "back-end"],
    ["fullstack", "full stack", "full-stack"],
    ["devops", "sre", "site reliability"],
    ["qa", "quality assurance", "test engineer", "sdet"],
    ["data scientist", "ml scientist"],
    ["data engineer", "data engineering"],
    ["ios", "swift developer"],
    ["android", "kotlin developer"],
    ["pm", "product manager"],
    ["ux", "user experience"],
    ["ui", "user interface"],
    ["k8s", "kubernetes"],
]


def _build_alias_index(groups):
    """
    Flatten the alias groups into a lookup: normalized phrase -> set of ALL
    phrases in its group (including itself). Lets us expand a query term to every
    equivalent phrase in one dict hit. Built once at import.
    """
    index = {}
    for group in groups:
        normalized = [p.strip().lower() for p in group if p and p.strip()]
        phrase_set = set(normalized)
        for phrase in normalized:
            # If a phrase appears in two groups, union them (rare, but safe).
            index.setdefault(phrase, set()).update(phrase_set)
    return index


_ALIAS_INDEX = _build_alias_index(ROLE_ALIAS_GROUPS)

# Multi-word alias phrases, longest first, so we detect "machine learning" in a
# raw query BEFORE it's split into single tokens (otherwise its alias 'ml' is
# never triggered). Single-word aliases are handled by the normal token path.
_MULTIWORD_ALIASES = sorted(
    (p for p in _ALIAS_INDEX if " " in p),
    key=lambda p: -len(p),
)


def _extract_query_terms(target_role):
    """
    Turn a raw query into the specializing/generic term lists, but FIRST pull out
    any known multi-word alias phrases as single units (e.g. 'machine learning'),
    so they can be alias-expanded. Remaining words are split and bucketed as
    before. Multi-word phrases are always specializing.
    """
    raw = " " + target_role.replace("/", " ").lower() + " "
    phrases = []
    for phrase in _MULTIWORD_ALIASES:
        pad = f" {phrase} "
        if pad in raw:
            phrases.append(phrase)
            raw = raw.replace(pad, " ")   # consume it so its words aren't re-bucketed
    leftover = [w for w in raw.split() if w.strip()]
    specializing = phrases + [w for w in leftover if w not in GENERIC_ROLE_WORDS]
    generic = [w for w in leftover if w in GENERIC_ROLE_WORDS]
    return specializing, generic


def _expand_terms(terms):
    """
    ONE-WAY query expansion: for each query term, add every alias in its group.
    Order-stable and de-duplicated. Terms with no alias pass through unchanged.
    """
    expanded = []
    seen = set()
    for t in terms:
        key = t.strip().lower()
        group = _ALIAS_INDEX.get(key, {key})
        for phrase in [key] + sorted(group - {key}):
            if phrase and phrase not in seen:
                seen.add(phrase)
                expanded.append(phrase)
    return expanded


def _role_matcher(target_role):
    """
    Require at least one SPECIALIZING term from the query (e.g. 'ai', 'ml',
    'backend'), matched as a WHOLE WORD — so short terms like 'ai'/'ml' don't
    false-match inside 'airline'/'HTML'. Multi-word terms phrase-match. If the
    query is only generic words, match on those.

    Each query term is first EXPANDED through the role-alias table (one-way), so
    e.g. "ML Engineer" also matches "machine learning" postings AND "machine
    learning engineer" matches "ML" postings — while keeping the same whole-word/
    phrase rigor (an alias like 'ai' still won't hit 'airline').
    """
    specializing, generic = _extract_query_terms(target_role)

    # Expand whichever set we'll actually match on, through the alias table.
    base_terms = specializing if specializing else generic
    terms = _expand_terms(base_terms)

    def _term_present(term, haystack_lower, haystack_tokens):
        t = term.strip().lower()
        if not t:
            return False
        # Multi-word OR hyphenated phrases ('front-end', 'machine learning') are
        # phrase-matched against the raw text, since the tokenizer splits on space
        # and hyphen and would never surface them as a single token.
        if " " in t or "-" in t:
            return t in haystack_lower
        return t in haystack_tokens        # single word: WHOLE-WORD (token) match

    def matches(job):
        haystack_lower = f"{job['title']} {job['description']}".lower()
        haystack_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", haystack_lower))
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


def _canonical_url(job):
    """
    A normalized apply/source URL used as the PRIMARY dedup identity. Two postings
    with the same canonical URL are the same job (even across providers); postings
    with DIFFERENT URLs are kept DISTINCT — so two real requisitions that merely share
    a title+company+location are never merged into one. Returns None when the posting
    has no usable URL (then we fall back to the content fingerprint).
    """
    for f in ("apply_url", "url", "source_url"):
        u = job.get(f)
        if u and isinstance(u, str):
            # drop #fragment and ?query (tracking params), trailing slash, and case,
            # so the same link matches regardless of ?utm=... etc.
            u = u.strip().split("#", 1)[0].split("?", 1)[0].rstrip("/")
            if u:
                return u.lower()
    return None


def _dedupe_key(job):
    """
    (Retained for callers/tests that want the primary identity signal of a single
    posting: its canonical URL if it has one, else its content fingerprint.)
    """
    cu = _canonical_url(job)
    if cu:
        return ("url", cu)
    return ("content",) + _fingerprint(job)


def _rank(job):
    return _SOURCE_RANK.get((job.get("source") or "").lower(), -1)


def _merge_group(rows):
    """Collapse rows that are the SAME job into one: the highest source-rank row wins,
    and useful fields (apply_url, dates, external_id, ...) are salvaged from the rest."""
    best_i = max(range(len(rows)), key=lambda i: _rank(rows[i]))
    winner = dict(rows[best_i])
    for i, r in enumerate(rows):
        if i != best_i:
            _merge_duplicate(winner, r)
    return winner


def _dedupe_jobs(jobs):
    """
    Collapse duplicates in TWO tiers, so genuinely distinct requisitions are never
    silently merged (the reviewer's concern), while true duplicates still collapse:

      1. Group by content fingerprint (title+company+location).
      2. WITHIN a group, split by DISTINCT canonical URL. Rows that share a URL — or
         have no URL — are the same job and merge (higher source-rank wins; apply_url
         etc. salvaged from the losers). Rows with DIFFERENT URLs are DISTINCT
         requisitions and each survive. A url-less row inside a group that has
         multiple distinct URLs is ambiguous, so it's folded into the highest-rank URL
         bucket rather than dropped or turned into a phantom row.

    So "AI Engineer @ Google, NYC" for two different teams (two apply URLs) stays TWO
    rows, while the SAME posting seen on two feeds (same URL, or one feed missing the
    URL) collapses to one. First-seen order is preserved.
    """
    groups = {}
    order = []
    for j in jobs:
        fp = _fingerprint(j)
        if fp not in groups:
            groups[fp] = []
            order.append(fp)
        groups[fp].append(j)

    result = []
    for fp in order:
        members = groups[fp]
        by_url = {}
        urlless = []
        for j in members:
            cu = _canonical_url(j)
            if cu:
                by_url.setdefault(cu, []).append(j)
            else:
                urlless.append(j)

        if len(by_url) <= 1:
            # 0 or 1 distinct URL in this content group -> ONE job; merge every member.
            result.append(_merge_group(members))
        else:
            # Multiple distinct URLs -> multiple distinct requisitions: one row each.
            buckets = list(by_url.values())
            if urlless:
                target = max(buckets, key=lambda b: max(_rank(x) for x in b))
                target.extend(urlless)
            for b in buckets:
                result.append(_merge_group(b))
    return result


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
    with get_connection() as conn:
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
            log.info("no jobs matched '%s' with the given filters", target_role)
            return []

        log.info("%d of %d job(s) matched your criteria", len(filtered), len(all_jobs))
        return filtered