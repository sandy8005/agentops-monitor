"""
Secret sanitization for anything that leaves the process as text: log lines,
trace rows, runs.stop_reason and API responses (review finding R05).
Adzuna authenticates with ?app_id=...&app_key=... and requests' ConnectionError
messages embed the failing URL.
"""
import re

_SECRET_KEYS = ("app_key", "app_id", "api_key", "apikey", "key", "token",
                "access_token", "client_secret", "password", "secret", "signature")
_QUERY_RE = re.compile(
    r"(?i)([?&;](?:%s)=)([^&#\s'\"]+)" % "|".join(re.escape(k) for k in _SECRET_KEYS))
_HEADER_RE = re.compile(r"(?i)((?:authorization|x-api-key|x-goog-api-key)\s*[:=]\s*)(\S+)")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-\._~\+/]+=*")
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z\-_]{30,}")
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")
REDACTED = "[REDACTED]"


def redact_secrets(text, max_len=2000):
    if text is None:
        return None
    try:
        s = str(text)
    except Exception:
        return "[unprintable error]"
    s = _QUERY_RE.sub(lambda m: m.group(1) + REDACTED, s)
    s = _HEADER_RE.sub(lambda m: m.group(1) + REDACTED, s)
    s = _BEARER_RE.sub(lambda m: m.group(1) + REDACTED, s)
    s = _GOOGLE_KEY_RE.sub(REDACTED, s)
    s = _JWT_RE.sub(REDACTED, s)
    if max_len and len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def safe_exception_summary(exc, provider=None):
    prefix = f"{provider}: " if provider else ""
    return f"{prefix}{type(exc).__name__}: {redact_secrets(exc, max_len=300)}"