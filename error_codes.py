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

    # Control / lifecycle.
    CANCELLED = "cancelled"                       # user cancelled
    BUDGET_EXCEEDED = "budget_exceeded"           # per-run LLM request budget spent
    WORKER_LOST = "worker_lost"                   # worker died / stopped heartbeating
                                                  # and the job ran out of attempts

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
    if _is_429(msg):
        if any(m in low for m in _QUOTA_MARKERS):
            return ErrorCode.LLM_QUOTA_EXHAUSTED
        return ErrorCode.LLM_RATE_LIMITED
    if "503" in msg or "UNAVAILABLE" in msg or "timeout" in low or "timed out" in low:
        return ErrorCode.LLM_UNAVAILABLE
    if "budget" in low:
        return ErrorCode.BUDGET_EXCEEDED
    return ErrorCode.INTERNAL


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
    # 1. Structured Retry-After header on an HTTP response, if present.
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers:
        try:
            ra = headers.get("Retry-After") or headers.get("retry-after")
            if ra is not None:
                return float(ra)
        except (TypeError, ValueError):
            pass
    # 2. RetryInfo embedded in the error details / message.
    m = _RETRY_DELAY_RE.search(str(getattr(exc, "details", "") or "") + " " + str(exc))
    return float(m.group(1)) if m else None