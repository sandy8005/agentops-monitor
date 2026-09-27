from timeutil import utcnow
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
# "live" is the LEGACY name Remotive rows were stored under; migration 0009 renames
# them to "remotive". It stays recognised so an un-migrated row is still classified
# correctly (never as practice data).
LIVE_SOURCES = {"adzuna", "remotive", "live"}
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
# Adzuna (real search) > Remotive > api > scraped > csv > seed.
_SOURCE_RANK = {"adzuna": 5, "remotive": 4, "live": 4, "api": 3, "scraped": 2,
                "csv": 1, "seed": 0}

# Seniority is BUSINESS-SIGNIFICANT: "Senior AI Engineer" and "Junior AI Engineer" at
# the same company/location are different requisitions, so seniority stays in the
# strict identity. Only spelling variants are normalized (sr -> senior, ...).
_TITLE_SYNONYMS = {"sr": "senior", "jr": "junior", "snr": "senior"}
# Words removed only for the RELAXED fingerprint, which merely nominates POSSIBLE
# duplicates (e.g. one feed dropping "Senior" from the title). A relaxed-only match is
# merged only with strong confirmation — near-identical descriptions.
_TITLE_NOISE = {
    "senior", "sr", "junior", "jr", "staff", "principal", "lead", "associate",
    "entry", "level", "mid", "i", "ii", "iii", "iv",
}
RELAXED_MATCH_MIN_DESC_SIMILARITY = 0.9
# Company-suffix noise stripped so "Wipro Inc." == "Wipro" == "Wipro, LLC".
_COMPANY_NOISE = {
    "inc", "inc.", "llc", "ltd", "ltd.", "limited", "corp", "corp.",
    "corporation", "co", "co.", "company", "gmbh", "plc", "pvt", "private",
}


def _norm_token_string(text, drop=frozenset(), synonyms=None):
    """
    Lowercase, split on non-alphanumerics (keeping + and # so 'c++'/'c#' survive),
    drop noise tokens, and rejoin with single spaces. Provider-independent — the
    same posting from two feeds normalizes to the same string regardless of
    punctuation, casing, or decorative words.
    """
    if not text:
        return ""
    tokens = re.findall(r"[a-z0-9\+\#]+", text.lower())
    if synonyms:
        tokens = [synonyms.get(t, t) for t in tokens]
    kept = [t for t in tokens if t not in drop]
    return " ".join(kept)


def _fingerprint(job):
    """
    STRICT provider-independent identity for a URL-less posting: normalized title
    (seniority KEPT) + company + location. external_id and apply URLs are
    provider-namespaced and never match across feeds, so a content fingerprint is
    the fallback; LOCATION stays in the key so the same role in two cities stays
    distinct, and SENIORITY stays so "Senior X" and "Junior X" stay distinct.
    """
    return (
        _norm_token_string(job.get("title"), synonyms=_TITLE_SYNONYMS),
        _norm_token_string(job.get("company"), drop=_COMPANY_NOISE),
        _norm_token_string(job.get("location")),
    )


def _relaxed_fingerprint(job):
    """Seniority-stripped fingerprint. Only NOMINATES possible duplicates; see
    _dedupe_jobs for the description-similarity confirmation required to merge."""
    return (
        _norm_token_string(job.get("title"), drop=_TITLE_NOISE, synonyms=_TITLE_SYNONYMS),
        _norm_token_string(job.get("company"), drop=_COMPANY_NOISE),
        _norm_token_string(job.get("location")),
    )


def _desc_similarity(a, b):
    """Jaccard similarity of the two descriptions' token sets (0..1). Empty -> 0."""
    ta = set(re.findall(r"[a-z0-9\+\#]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z0-9\+\#]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


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


# Query parameters that only TRACK a click and never identify a job. Everything else
# in the query string is preserved, because many ATSs carry the job's identity there
# (/jobs?jobId=123 vs /jobs?jobId=456 are two different postings).
_TRACKING_PARAMS = {
    "gclid", "fbclid", "msclkid", "dclid", "yclid", "mc_cid", "mc_eid", "_hsenc",
    "_hsmi", "ref", "ref_src", "referrer", "src", "source", "trk", "trackingid",
    "igshid", "si",
}


def _is_tracking_param(key):
    k = key.lower()
    return k.startswith("utm") or k in _TRACKING_PARAMS


def _canonical_url(job):
    """
    A normalized apply/source URL used as the PRIMARY dedup identity. Two postings
    with the same canonical URL are the same job (even across providers); postings
    with DIFFERENT URLs are kept DISTINCT. Returns None when the posting has no usable
    URL (then we fall back to the content fingerprint).

    Normalization is deliberately conservative (urllib.parse, not string surgery):
      * scheme and HOST are lowercased (they're case-insensitive); default ports and
        the #fragment are dropped;
      * the PATH keeps its case (paths can be case-sensitive) and only loses a
        trailing slash;
      * the QUERY keeps every identity-bearing parameter and drops only known
        tracking parameters (utm_*, gclid, fbclid, ...), sorted for stability.
    """
    from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
    for f in ("apply_url", "url", "source_url"):
        u = job.get(f)
        if not (u and isinstance(u, str) and u.strip()):
            continue
        try:
            parts = urlsplit(u.strip())
        except ValueError:
            continue
        if not parts.netloc:
            continue
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        port = parts.port
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = parts.path.rstrip("/")
        query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                                 if not _is_tracking_param(k)))
        return urlunsplit((scheme, host, path, query, ""))
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
         multiple distinct URLs is ambiguous: it joins a URL bucket only on strong
         evidence (near-identical description), otherwise it is kept separate.

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
            # A URL-less row cannot be PROVEN to be any one of them. It is attached
            # to a URL bucket only when its description is near-identical to that
            # bucket's (strong evidence); otherwise it stays a separate row rather
            # than being arbitrarily merged into (and hidden behind) one of them.
            buckets = list(by_url.values())
            unresolved = []
            for u in urlless:
                best, best_sim = None, 0.0
                for b in buckets:
                    sim = max(_desc_similarity(u.get("description"), x.get("description"))
                              for x in b)
                    if sim > best_sim:
                        best, best_sim = b, sim
                if best is not None and best_sim >= RELAXED_MATCH_MIN_DESC_SIMILARITY:
                    best.append(u)
                else:
                    unresolved.append(u)
            for b in buckets:
                result.append(_merge_group(b))
            if unresolved:
                # URL-less rows that match no URL bucket are still the same content
                # as EACH OTHER (same strict fingerprint) — collapse among themselves.
                result.append(_merge_group(unresolved))
    return _merge_relaxed_duplicates(result)


def _merge_relaxed_duplicates(rows):
    """
    Second, CONSERVATIVE pass: rows that differ only by seniority wording (same
    relaxed fingerprint) are merged only when neither carries a conflicting URL AND
    their descriptions are near-identical — i.e. the same posting where one feed
    dropped "Senior" from the title. Different seniority with different descriptions
    (a real Senior vs Junior opening) stays separate.
    """
    out = []
    for row in rows:
        merged = False
        for i, kept in enumerate(out):
            if _fingerprint(kept) == _fingerprint(row):
                continue   # already handled by the strict pass (distinct URLs)
            if _relaxed_fingerprint(kept) != _relaxed_fingerprint(row):
                continue
            ku, ru = _canonical_url(kept), _canonical_url(row)
            if ku and ru and ku != ru:
                continue   # two distinct requisitions
            if _desc_similarity(kept.get("description"), row.get("description")) \
                    >= RELAXED_MATCH_MIN_DESC_SIMILARITY:
                out[i] = _merge_group([kept, row])
                merged = True
                break
        if not merged:
            out.append(row)
    return out


# --- Geographic eligibility of REMOTE postings ---------------------------------
# A remote-only provider (Remotive) doesn't filter by location, but its postings
# often restrict WHERE the candidate may live ("USA only", "Europe", "UK"). Work mode
# (remote) and geographic eligibility are different things. The check below is
# deliberately CONSERVATIVE: it only reports "ineligible" when both the posting's
# region and the requested location map to KNOWN, non-overlapping regions. Anything
# it can't resolve (a city, a state, free text) is "unknown" and kept.
_WORLDWIDE = {"worldwide", "anywhere", "global", "globally", "international",
              "any location", "everywhere"}
_REGION_TERMS = {
    "us": {"us", "usa", "u.s.", "u.s.a.", "united states", "america",
           "united states of america"},
    "canada": {"canada"},          # NOT "ca" — that is California in "San Jose, CA"
    "north_america": {"north america", "americas"},
    "latam": {"latam", "latin america", "south america", "brazil", "mexico",
              "argentina", "colombia", "chile"},
    "uk": {"uk", "u.k.", "united kingdom", "great britain", "britain", "england",
           "scotland", "wales"},
    "europe": {"europe", "eu", "european union", "eea", "emea", "germany", "france",
               "spain", "italy", "netherlands", "poland", "portugal", "ireland",
               "sweden", "switzerland", "austria", "belgium", "denmark", "norway",
               "finland"},
    "apac": {"apac", "asia", "asia pacific", "india", "japan", "singapore",
             "australia", "new zealand", "philippines", "indonesia", "vietnam",
             "china", "korea"},
    "africa": {"africa", "nigeria", "kenya", "south africa", "egypt"},
    "middle_east": {"middle east", "uae", "saudi arabia", "israel", "qatar"},
}
# Regions that CONTAIN others (a posting open to "North America" accepts a US
# candidate; "EMEA"/"Europe" accept the UK).
_REGION_PARENTS = {
    "us": {"north_america"}, "canada": {"north_america"},
    "uk": {"europe"}, "middle_east": {"europe"}, "africa": {"europe"},
}


def _regions_in(text):
    low = f" {' '.join(re.split(r'[^a-z0-9.]+', (text or '').lower()))} "
    found = set()
    for region, terms in _REGION_TERMS.items():
        for t in terms:
            if f" {t} " in low:
                found.add(region)
                break
    return found


def geo_eligibility(job, requested_location):
    """
    'eligible' / 'ineligible' / 'unknown' for a remote posting vs. the location the
    user searched from. Only a clear conflict between KNOWN regions is 'ineligible'.
    """
    req = (requested_location or "").strip()
    if not req:
        return "unknown"
    region_text = (job.get("location") or "").strip().lower()
    if not region_text:
        return "unknown"
    job_regions = _regions_in(region_text)
    if not job_regions:
        return "eligible" if any(w in region_text for w in _WORLDWIDE) else "unknown"
    req_regions = _regions_in(req)
    if not req_regions:
        return "unknown"
    expanded_req = set(req_regions)
    for r in req_regions:
        expanded_req |= _REGION_PARENTS.get(r, set())
    expanded_job = set(job_regions)
    for r in job_regions:
        expanded_job |= _REGION_PARENTS.get(r, set())
    if expanded_req & job_regions or req_regions & expanded_job:
        return "eligible"
    return "ineligible"


def search_jobs(target_role=None, location=None, work_mode=None,
                employment_type=None, live_only=False, run_id=None):
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
    in mixed mode. Remote postings from a provider that doesn't geo-filter are kept
    unless their stated candidate region clearly excludes the requested location
    (geo_eligibility); each returned job is annotated with "geo_eligibility".

    target_role is optional: without it, every fresh job passes the role filter, but
    the location / work-mode / employment-type filters and de-duplication still run.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        # Pull provenance (source) and, via the association join, the set of locations
        # any search has ever returned this posting for. search_location is no longer
        # read for filtering (kept only until a later migration drops it).
        cur.execute("""
            SELECT p.id, p.title, p.company, p.description, p.location, p.work_mode,
                   p.employment_type, p.source, p.external_id, p.last_seen_at, p.apply_url,
                   -- Only searches whose provider actually GEO-FILTERED count as
                   -- "returned for this location". Remotive is remote-only and
                   -- ignores location, so its searches carry
                   -- location_filter_applied = FALSE and are not a location match.
                   COALESCE(
                       ARRAY_AGG(DISTINCT lower(s.location))
                       FILTER (WHERE s.location IS NOT NULL
                                 AND s.location_filter_applied),
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
             "external_id": r[8], "last_seen_at": r[9], "apply_url": r[10],
             "assoc_locations": set(r[11] or [])}
            for r in rows
        ]

        # --- source separation: Live Mode excludes all practice data ---
        if live_only:
            all_jobs = [j for j in all_jobs if (j.get("source") or "").lower() in LIVE_SOURCES]
            # When a run_id is given, restrict to postings THIS run actually fetched.
            if run_id is not None:
                all_jobs = [j for j in all_jobs if j["id"] in run_job_ids]

        # --- freshness filter: drop jobs not seen recently ---
        # Practice data (seed/csv/scraped) has no sighting time and is always kept.
        # A LIVE posting must have been seen within the window; a live row with no
        # last_seen_at is treated as STALE, never as "fresh forever".
        cutoff = utcnow() - timedelta(days=STALE_AFTER_DAYS)
        def _is_fresh(job):
            ls = job.get("last_seen_at")
            if ls is None:
                return (job.get("source") or "").lower() not in LIVE_SOURCES
            if ls.tzinfo is None:   # defensive: legacy naive value
                ls = ls.replace(tzinfo=cutoff.tzinfo)
            return ls >= cutoff
        all_jobs = [j for j in all_jobs if _is_fresh(j)]

        # --- role filter: whole-word specializing-term match ---
        # OPTIONAL: with no target role every (fresh) job is a role match, but the
        # location / work-mode / employment-type filters and de-duplication below
        # still apply — the function's contract doesn't change with the role.
        if target_role and target_role.strip():
            role_matches = _role_matcher(target_role)
            filtered = [j for j in all_jobs if role_matches(j)]
        else:
            filtered = list(all_jobs)

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
                if assoc:
                    return loc in assoc  # matched by at least one geo-filtered search
                # No geo-filtered association: practice data, or a remote-only
                # provider that doesn't filter by location. "Remote" is a WORK MODE,
                # not worldwide eligibility: exclude only when the posting's stated
                # candidate region clearly excludes the requested location.
                return geo_eligibility(job, location) != "ineligible"

            filtered = [j for j in filtered if location_ok(j)]
            for j in filtered:
                j["geo_eligibility"] = ("matched_search" if j.get("assoc_locations")
                                        else geo_eligibility(j, location))

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