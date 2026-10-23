from timeutil import utcnow
from database import get_connection
import re
from datetime import datetime, timedelta

from logging_config import get_logger
log = get_logger(__name__)

# Generic role words too common to distinguish a role on their own.
GENERIC_ROLE_WORDS = {
    "engineer", "developer", "analyst", "manager", "specialist", "consultant",
    "administrator", "architect", "designer", "coordinator", "scientist",
    "researcher", "lead", "senior",
    "junior", "staff", "principal", "associate", "intern", "assistant",
    "of", "the", "and", "or", "a", "an", "i", "ii", "iii"
}


# The ROLE FAMILY words among them — what kind of job it is. Seniority words and
# stop-words are generic too, but they say nothing about the job family.
ROLE_FAMILY_WORDS = {
    "engineer", "developer", "analyst", "manager", "specialist", "consultant",
    "administrator", "architect", "designer", "coordinator", "assistant",
    "scientist", "researcher",
}
# Family words that name the same job family in titles ("AI Engineer" ==
# "AI Developer"). One-way lookup: query family word -> acceptable title words.
_FAMILY_EQUIVALENTS = {
    "engineer": {"engineer", "developer"},
    "developer": {"developer", "engineer"},
    # "ML Researcher" == "Research Scientist, ML"; a scientist query still needs a
    # scientist/researcher TITLE, so "Soil Scientist" never matches "ML Scientist"
    # by specialization alone, nor "Marketing Manager" via "our ML scientists".
    "scientist": {"scientist", "researcher"},
    "researcher": {"researcher", "scientist"},
}
_RESEARCH_FAMILIES = {"scientist", "researcher"}
# Title words that do not name a scientific FIELD ("Research Scientist II",
# "Applied Scientist", "Scientists").
_GENERIC_RESEARCH_TITLE_WORDS = {
    "research", "applied", "scientists", "researchers", "postdoctoral", "postdoc",
    "fellow", "iv", "v", "sr", "jr", "mid", "level", "remote", "hybrid", "onsite",
}


STALE_AFTER_DAYS = 14   # jobs not seen in this many days drop out of search

# The only job sources in scope are the two live providers. Seed / CSV / scraped
# "practice" data and the mixed "pool" mode were removed: every search reads live
# postings that THIS run fetched, nothing else. Rows with any other source (left
# over from an old database) are never returned.
from agent_goal import LIVE_PROVIDERS  # noqa: E402  (single definition)
LIVE_SOURCES = frozenset(LIVE_PROVIDERS)


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
    # ("data scientist" / "ml scientist" are no longer one alias phrase: consuming
    # the phrase whole skipped the scientist family check. "ML Scientist" is now the
    # specialization "ml" + the scientist family, which matches "Machine Learning
    # Scientist" and "Research Scientist, ML" and rejects unrelated scientists.)
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
    Local validation that a posting is the requested KIND of job. Providers are
    not trusted to obey the query, so this runs on everything they return.

    The query is split into SPECIALIZING terms ('ai', 'ml', 'backend', 'security')
    and generic words, among them ROLE FAMILY words ('engineer', 'analyst').

      * specialization: at least one specializing term (alias-expanded: "ML" also
        matches "machine learning") in the title OR description, whole-word —
        'ai' never hits 'airline', 'ml' never hits 'HTML';
      * role family: when the query names one, the TITLE must name it too
        (engineer and developer are treated as one family; plurals count).

    Both dimensions are required. Without the family check, "AI Engineer"
    matched a "Product Manager" whose description says "work with AI engineers",
    a recruiter posting for "AI Engineering", or "Sales Executive, AI Solutions".
    A query with only a specialization ("ML") or only generic words keeps the
    single-dimension rule.
    """
    specializing, generic = _extract_query_terms(target_role)

    # Expand whichever set we'll actually match on, through the alias table.
    base_terms = specializing if specializing else generic
    terms = _expand_terms(base_terms)
    families = [w for w in generic if w in ROLE_FAMILY_WORDS] if specializing else []
    title_family_words = set()
    for f in families:
        for w in _FAMILY_EQUIVALENTS.get(f, {f}):
            title_family_words.update({w, w + "s"})

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

    # Scientist / researcher titles name a FIELD ("Soil Scientist", "Clinical
    # Researcher"), so for these families an incidental mention of the
    # specialization in the description ("we use machine learning tools") is not
    # enough: the specialization must be in the TITLE, or the title must be a
    # generic research title with no other field ("Research Scientist",
    # "Senior Scientist") whose description names the specialization.
    research_family = any(f in _RESEARCH_FAMILIES for f in families)

    def matches(job):
        title_lower = str(job.get("title") or "").lower()
        haystack_lower = f"{title_lower} {job.get('description') or ''}".lower()
        haystack_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", haystack_lower))
        if not any(_term_present(t, haystack_lower, haystack_tokens) for t in terms):
            return False
        if title_family_words:
            title_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", title_lower))
            if not title_tokens & title_family_words:
                return False
            if research_family:
                if any(_term_present(t, title_lower, title_tokens) for t in terms):
                    return True
                # the title HEAD only: "Research Scientist - London (Remote)"
                head = re.split(r"\s[-–—|]\s|[(,|/]", title_lower)[0]
                other = {w for w in re.findall(r"[a-z]+", head)
                         if w not in GENERIC_ROLE_WORDS and w not in _GENERIC_RESEARCH_TITLE_WORDS}
                return not other
        return True

    return matches


# Source preference: when two rows are the same posting, keep the better one.
# Adzuna (real role+location search) > Remotive (remote-only feed).
_SOURCE_RANK = {"adzuna": 1, "remotive": 0}

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
# often restrict WHERE the candidate may live ("USA only", "Europe", "UK, Germany").
# Work mode (remote) and geographic eligibility are different things. Resolution is
# STRUCTURED (geo.py: ISO countries + a region hierarchy) and CONSERVATIVE: only a
# clear conflict between resolved scopes is "ineligible"; anything unresolved,
# ambiguous or phrased as an exclusion is "unknown" and kept.

def geo_eligibility(job, requested_location):
    """'eligible' / 'ineligible' / 'unknown' for a remote posting vs. the location
    the user searched from (see geo.eligibility)."""
    from geo import eligibility
    return eligibility((job.get("location") or "").strip(), (requested_location or "").strip())


def work_mode_status(job, work_mode, allow_hybrid=False):
    """How a posting relates to the work-mode constraint:
      'verified' — its stated mode (or a "remote" location) satisfies it;
      'unknown'  — it states no mode: kept (work_mode_ok) but NEVER presented as a
                   confirmed match (work_mode_verified = False);
      'mismatch' — it states a different mode.
    With no constraint every posting is 'verified'."""
    if not work_mode:
        return "verified"
    return ("verified" if work_mode_ok(job, work_mode, allow_hybrid, unknown=False)
            else "unknown" if _stated_work_mode(job) is None else "mismatch")


def _stated_work_mode(job):
    jm = (job.get("work_mode") or "").lower().strip()
    if not jm and "remote" in (job.get("location") or "").lower():
        jm = "remote"
    return jm or None


def work_mode_ok(job, work_mode, allow_hybrid=False, unknown=True):
    """
    Work mode is a FIXED constraint, so the match is STRICT: a remote search
    returns remote postings, onsite returns onsite, hybrid returns hybrid.

    allow_hybrid=True is the explicit, user-chosen relaxation "remote (or onsite)
    preferred, hybrid acceptable": it additionally accepts hybrid postings for a
    remote or onsite search. It never turns a hybrid search into "anything".

    A posting with NO stated mode (and no "remote" in its location) is kept: the
    providers often omit the field, and excluding it would drop most onsite jobs.
    It is labeled unverified (work_mode_status); unknown=False treats it as no match.
    """
    if not work_mode:
        return True
    wm = work_mode.lower().strip()
    jm = _stated_work_mode(job)
    if not jm:
        return unknown
    if wm == jm:
        return True
    return bool(allow_hybrid) and jm == "hybrid" and wm in ("remote", "onsite")


def search_jobs(target_role=None, location=None, work_mode=None,
                employment_type=None, run_id=None, allow_hybrid=False,
                provider=None, query_norm=None):
    """
    Search the postings THIS run fetched from the live providers (Adzuna/Remotive).

    run_id is required: a search returns exactly what this run's own provider
    searches returned (job_search_results), filtered by role / location / work
    mode / employment type, freshness-checked and de-duplicated. Nothing from
    other runs and nothing from a non-live source is ever returned.

    Location uses the PER-SEARCH ASSOCIATION: a posting matches a location if one
    of this run's GEO-FILTERED searches (Adzuna) returned it for that location.
    Remote postings from a provider that does not geo-filter (Remotive) are kept
    unless their stated candidate region clearly excludes the requested location
    (geo_eligibility); each returned job is annotated with "geo_eligibility".

    provider + query_norm scope the result to ONE search: only the job_searches
    rows that provider recorded for that (normalized) query in this run — every
    generation's retry of it, nothing another query returned. Run-wide
    de-duplication against earlier searches is the caller's separate step.
    """
    if run_id is None:
        raise ValueError("search_jobs requires the run_id whose searches to read")
    scope_sql, scope_args = "", ()
    if provider is not None or query_norm is not None:
        if not provider or not query_norm:
            raise ValueError("a search scope needs both provider and query_norm")
        scope_sql = (" AND s.source = %s AND lower(regexp_replace("
                     "btrim(COALESCE(s.target_role, '')), '\\s+', ' ', 'g')) = %s")
        scope_args = (provider, query_norm)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.id, p.title, p.company, p.description, p.location, p.work_mode,
                   p.employment_type, p.source, p.external_id, p.last_seen_at, p.apply_url,
                   -- Only searches whose provider actually GEO-FILTERED count as
                   -- "returned for this location" (Remotive ignores location).
                   COALESCE(
                       ARRAY_AGG(DISTINCT lower(s.location))
                       FILTER (WHERE s.location IS NOT NULL
                                 AND s.location_filter_applied),
                       '{}'
                   ) AS assoc_locations
            FROM job_postings p
            JOIN job_search_results r ON r.job_id = p.id
            JOIN job_searches s ON s.id = r.search_id
            WHERE s.run_id = %s AND lower(p.source) = ANY(%s)""" + scope_sql + """
            GROUP BY p.id
            ORDER BY p.id
        """, (run_id, sorted(LIVE_SOURCES)) + scope_args)
        rows = cur.fetchall()

    all_jobs = [
        {"id": r[0], "title": r[1], "company": r[2], "description": r[3],
         "location": r[4], "work_mode": r[5], "employment_type": r[6], "source": r[7],
         "external_id": r[8], "last_seen_at": r[9], "apply_url": r[10],
         "assoc_locations": set(r[11] or [])}
        for r in rows
    ]

    # --- freshness: a live posting must have been seen within the window; a row
    # with no last_seen_at is treated as STALE, never as "fresh forever".
    cutoff = utcnow() - timedelta(days=STALE_AFTER_DAYS)

    def _is_fresh(job):
        ls = job.get("last_seen_at")
        if ls is None:
            return False
        if ls.tzinfo is None:   # defensive: legacy naive value
            ls = ls.replace(tzinfo=cutoff.tzinfo)
        return ls >= cutoff
    all_jobs = [j for j in all_jobs if _is_fresh(j)]

    # --- role filter: whole-word specializing-term match (optional) ---
    if target_role and target_role.strip():
        role_matches = _role_matcher(target_role)
        filtered = [j for j in all_jobs if role_matches(j)]
    else:
        filtered = list(all_jobs)

    # --- location filter (via PER-SEARCH association) ---
    if location and location.strip():
        loc = location.strip().lower()

        def location_ok(job):
            assoc = job.get("assoc_locations") or set()
            if assoc:
                return loc in assoc
            # A remote-only provider that doesn't filter by location. "Remote" is a
            # WORK MODE, not worldwide eligibility: exclude only when the posting's
            # stated candidate region clearly excludes the requested location.
            return geo_eligibility(job, location) != "ineligible"

        filtered = [j for j in filtered if location_ok(j)]
        for j in filtered:
            j["geo_eligibility"] = ("matched_search" if j.get("assoc_locations")
                                    else geo_eligibility(j, location))

    # --- work_mode filter (strict; hybrid only by explicit allow_hybrid) ---
    if work_mode:
        filtered = [j for j in filtered if work_mode_ok(j, work_mode, allow_hybrid)]

    # --- employment_type filter (SOFT: keep unknown-type, exclude known mismatch) ---
    if employment_type:
        et = employment_type.lower().strip()
        filtered = [j for j in filtered
                    if not (j.get("employment_type") or "").strip()
                    or (j.get("employment_type") or "").lower().strip() == et]

    filtered = _dedupe_jobs(filtered)
    # Kept postings are LABELED (after dedup, which can backfill a mode): an
    # unstated mode is "unknown", never a confirmed match of the constraint.
    if work_mode:
        for j in filtered:
            j["work_mode_verified"] = work_mode_status(j, work_mode, allow_hybrid) == "verified"
    # Counts only — the query text is user input and stays out of the logs.
    log.info("search matched %d of %d fetched posting(s)", len(filtered), len(all_jobs),
             extra={"run_id": run_id})
    return filtered