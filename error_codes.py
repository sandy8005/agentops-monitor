"""
Run-level error codes.

A machine-readable classification of WHY a run ended in a non-success state, stored
in runs.error_code (queryable). Distinct from:
  - runs.status       : the coarse outcome (success/failed/cancelled/...).
  - runs.stop_reason  : the human-readable detail string (e.g. the exception text).

error_code is the stable, enumerable vocabulary you can group/alert on:
    SELECT error_code, count(*) FROM runs WHERE error_code IS NOT NULL GROUP BY 1;

It is a real Enum so the DB, API, monitor and tests all operate against one explicit
vocabulary instead of scattered string literals. The `str` mixin means a member IS
its value: it compares equal to the plain string ("cancelled"), is a member of a set
of such strings, persists to Postgres as that string, and JSON-encodes directly — so
existing string-based call sites and stored rows keep working unchanged.

Add new codes here as new failure modes are identified; keep values stable once used
(they may be persisted in the DB and queried).
"""
from enum import Enum


class ErrorCode(str, Enum):
    # Pipeline stage failures.
    PARSE_FAILED = "parse_failed"                 # resume parsing failed
    SEARCH_FAILED = "search_failed"               # job search / fetch failed
    NO_MATCHES = "no_matches"                     # search returned nothing (not an error per se)

    # LLM failures. A 429 is NOT one thing: a per-minute RATE limit clears in
    # seconds (retry, honoring the provider's retry delay), while an exhausted
    # daily/project QUOTA will not recover on a 10-second backoff (terminal — retrying
    # only burns attempts and money).
    LLM_RATE_LIMITED = "llm_rate_limited"         # 429 per-minute / burst limit — transient
    LLM_QUOTA_EXHAUSTED = "llm_quota_exhausted"   # 429 daily/project quota spent — terminal
    LLM_UNAVAILABLE = "llm_unavailable"           # 503 / timeout / provider down
    LLM_INVALID_RESPONSE = "llm_invalid_response" # provider answered without text/usage
                                                  # (blocked / safety-filtered) — terminal
    LLM_NOT_CONFIGURED = "llm_not_configured"     # no GEMINI_API_KEY — terminal; callers
                                                  # take their rules fallback

    # External job-provider failures (Adzuna / Remotive). Same philosophy as the LLM
    # taxonomy: a rate limit or an outage is worth retrying, bad credentials are not,
    # and a provider that answered "0 jobs" is not a failure at all (no_matches).
    JOB_SOURCE_RATE_LIMITED = "job_source_rate_limited"   # provider 429 — transient
    JOB_SOURCE_UNAVAILABLE = "job_source_unavailable"     # network / timeout / 5xx — transient
    JOB_SOURCE_AUTH_FAILED = "job_source_auth_failed"     # 401/403 / missing keys — terminal
    JOB_SOURCE_INVALID_RESPONSE = "job_source_invalid_response"  # malformed body — terminal
    JOB_SEARCH_EMPTY = "job_search_empty"                 # provider OK, zero postings

    # Control / lifecycle.
    CANCELLED = "cancelled"                       # user cancelled
    BUDGET_EXCEEDED = "budget_exceeded"           # per-run LLM request budget spent
    COST_UNKNOWN = "cost_unknown"                 # a hard USD cap is set but the model's
                                                  # price is unknown — fail closed
    WORKER_LOST = "worker_lost"                   # worker died / stopped heartbeating
                                                  # and the job ran out of attempts
    DATABASE_UNAVAILABLE = "database_unavailable" # connection / operational DB error —
                                                  # transient, the job is retried

    # Catch-all.
    INTERNAL = "internal_error"                   # unclassified exception

    def __str__(self):
        # Yield the value ("cancelled"), not "ErrorCode.CANCELLED", so logs, f-strings,
        # and str() coercion all produce the stored vocabulary term.
        return self.value


# All defined code VALUES — for validation/enumeration by the API, monitor, and tests.
# (No NONE member: "no error" is represented by NULL in the DB and None in Python.)
ALL_CODES = frozenset(c.value for c in ErrorCode)


# Markers that a 429 is a spent QUOTA (daily / project / zero allowance), not a
# short-lived rate limit. Gemini reports the violated quota id (e.g.
# "GenerateRequestsPerDayPerProjectPerModel") and "limit: 0" for disabled tiers.
_QUOTA_MARKERS = ("perday", "per day", "per_day", "daily", "limit: 0",
                  "billing account", "billing has not been enabled")


def _is_429(msg):
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def classify_exception(exc):
    """
    Best-effort mapping of an exception (or an error string) to an ErrorCode, for the
    graph's generic except blocks. Specific call sites should set a precise code
    directly; this is the fallback so a raw exception still gets a useful classification.
    """
    msg = str(exc)
    low = msg.lower()
    name = type(exc).__name__
    if name == "ModelNotConfigured" or "gemini_api_key is not set" in low:
        return ErrorCode.LLM_NOT_CONFIGURED
    if name in ("OperationalError", "InterfaceError", "DatabaseUnavailable", "PoolError") \
            or "database_unavailable" in low:
        return ErrorCode.DATABASE_UNAVAILABLE
    if name == "InvalidProviderResponse" or "provider returned no text" in low:
        return ErrorCode.LLM_INVALID_RESPONSE
    # Job-provider failures are tagged by the search stage ("job_source_...:") — honour
    # the tag before the generic HTTP heuristics below.
    for code in (ErrorCode.JOB_SOURCE_RATE_LIMITED, ErrorCode.JOB_SOURCE_UNAVAILABLE,
                 ErrorCode.JOB_SOURCE_AUTH_FAILED, ErrorCode.JOB_SOURCE_INVALID_RESPONSE):
        if code.value in low:
            return code
    if type(exc).__name__ == "QuotaCircuitOpen" or "circuit open" in low:
        return ErrorCode.LLM_QUOTA_EXHAUSTED
    if _is_429(msg):
        if any(m in low for m in _QUOTA_MARKERS):
            return ErrorCode.LLM_QUOTA_EXHAUSTED
        return ErrorCode.LLM_RATE_LIMITED
    if "503" in msg or "UNAVAILABLE" in msg or "timeout" in low or "timed out" in low:
        return ErrorCode.LLM_UNAVAILABLE
    if name == "CostUnknown" or "cost_unknown" in low:
        return ErrorCode.COST_UNKNOWN
    if "budget" in low:
        return ErrorCode.BUDGET_EXCEEDED
    return ErrorCode.INTERNAL


_QUOTA_ID_RE = None


def summarize_error(exc, max_len=160):
    """One short, log-friendly line for a provider error: the classified code, the
    violated quota id when Gemini reports one, and a truncated message. Provider
    bodies can be kilobytes of JSON; logging them per job floods the console."""
    global _QUOTA_ID_RE
    import re
    if _QUOTA_ID_RE is None:
        _QUOTA_ID_RE = re.compile(r"quotaId['\"]?\s*:\s*['\"]([A-Za-z0-9_\-]+)")
    code = classify_exception(exc)
    text = str(exc)
    m = _QUOTA_ID_RE.search(text)
    first = text.split("{", 1)[0].strip() or type(exc).__name__
    out = f"{code}: {first}"
    if m:
        out += f" [quota={m.group(1)}]"
    return out if len(out) <= max_len else out[:max_len] + "…"


_RETRY_DELAY_RE = None


def provider_retry_after(exc):
    """
    The provider's own retry hint, in seconds, if the error carries one — Gemini puts
    a google.rpc.RetryInfo `retryDelay: "31s"` in 429 details; HTTP errors may carry a
    Retry-After header. Returns None when there's no hint.
    """
    global _RETRY_DELAY_RE
    import re
    if _RETRY_DELAY_RE is None:
        _RETRY_DELAY_RE = re.compile(r"retry[_ ]?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s",
                                     re.IGNORECASE)
    # 1. Structured Retry-After header on an HTTP response, if present. HTTP allows
    #    both delay-seconds ("31") and an HTTP-date ("Wed, 21 Oct 2026 07:28:00 GMT").
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        try:
            ra = headers.get("Retry-After") or headers.get("retry-after")
        except (AttributeError, TypeError):
            ra = None
        parsed = parse_retry_after(ra)
        if parsed is not None:
            return parsed
    # 2. RetryInfo embedded in the error details / message.
    m = _RETRY_DELAY_RE.search(str(getattr(exc, "details", "") or "") + " " + str(exc))
    return float(m.group(1)) if m else None


def parse_retry_after(value, now=None):
    """
    Parse an HTTP Retry-After value into seconds (>= 0), or None if unusable.
    Accepts delay-seconds ("31", "2.5") and HTTP-date (RFC 9110 IMF-fixdate and the
    obsolete forms email.utils understands). A date in the past yields 0.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        secs = float(text)
        return max(0.0, secs)
    except ValueError:
        pass
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())