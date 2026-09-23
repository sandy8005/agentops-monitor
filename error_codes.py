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

    # LLM failures.
    LLM_QUOTA_EXHAUSTED = "llm_quota_exhausted"   # 429 / RESOURCE_EXHAUSTED
    LLM_UNAVAILABLE = "llm_unavailable"           # 503 / timeout / provider down

    # Control / lifecycle.
    CANCELLED = "cancelled"                       # user cancelled
    BUDGET_EXCEEDED = "budget_exceeded"           # per-run LLM request budget spent

    # Catch-all.
    INTERNAL = "internal_error"                   # unclassified exception

    def __str__(self):
        # Yield the value ("cancelled"), not "ErrorCode.CANCELLED", so logs, f-strings,
        # and str() coercion all produce the stored vocabulary term.
        return self.value


# All defined code VALUES — for validation/enumeration by the API, monitor, and tests.
# (No NONE member: "no error" is represented by NULL in the DB and None in Python.)
ALL_CODES = frozenset(c.value for c in ErrorCode)


def classify_exception(exc):
    """
    Best-effort mapping of an exception (or an error string) to an ErrorCode, for the
    graph's generic except blocks. Specific call sites should set a precise code
    directly; this is the fallback so a raw exception still gets a useful classification.
    """
    msg = str(exc)
    low = msg.lower()
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return ErrorCode.LLM_QUOTA_EXHAUSTED
    if "503" in msg or "UNAVAILABLE" in msg or "timeout" in low or "timed out" in low:
        return ErrorCode.LLM_UNAVAILABLE
    if "budget" in low:
        return ErrorCode.BUDGET_EXCEEDED
    return ErrorCode.INTERNAL