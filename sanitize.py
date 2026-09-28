"""
Secret sanitization for anything that leaves the process as text: log lines,
trace rows (tool_calls.error_message, steps.error_message), runs.stop_reason and
API responses.

Adzuna authenticates with ?app_id=...&app_key=... query parameters, and requests'
ConnectionError / Timeout messages embed the full failing URL. Any exception text
from a provider call must therefore pass through redact_secrets() before it is
logged or persisted (review finding R05).
"""
import re

# Query / form parameters whose VALUES are credentials.
_SECRET_KEYS = ("app_key", "app_id", "api_key", "apikey", "key", "token",
                "access_token", "client_secret", "password", "secret", "signature")

_QUERY_RE = re.compile(
    r"(?i)([?&;](?:%s)=)([^&#\s'\"]+)" % "|".join(re.escape(k) for k in _SECRET_KEYS))
# The WHOLE credential after the header name is removed, including an auth scheme
# ("Bearer abc", "Basic xyz=") and optional quotes.
_HEADER_RE = re.compile(
    r"(?i)((?:proxy-)?authorization|x-api-key|x-goog-api-key|api-key)(['\"]?\s*[:=]\s*['\"]?)"
    r"(?:(?:bearer|basic|token|apikey)\s+)?[^\s'\",}]+")
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-\._~\+/]+=*")
# Google API keys and JWT-looking blobs, wherever they appear.
_GOOGLE_KEY_RE = re.compile(r"AIza[0-9A-Za-z\-_]{30,}")
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")

REDACTED = "[REDACTED]"


def redact_secrets(text, max_len=2000):
    """Return `text` with credential values replaced by [REDACTED], truncated to
    max_len characters. None stays None. Never raises."""
    if text is None:
        return None
    try:
        s = str(text)
    except Exception:
        return "[unprintable error]"
    s = _QUERY_RE.sub(lambda m: m.group(1) + REDACTED, s)
    s = _HEADER_RE.sub(lambda m: m.group(1) + m.group(2) + REDACTED, s)
    s = _BEARER_RE.sub(lambda m: m.group(1) + REDACTED, s)
    s = _GOOGLE_KEY_RE.sub(REDACTED, s)
    s = _JWT_RE.sub(REDACTED, s)
    if max_len and len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def safe_exception_summary(exc, provider=None):
    """Short, secret-free description of an exception: class name plus sanitized
    message, optionally prefixed with the provider name."""
    name = type(exc).__name__
    msg = redact_secrets(exc, max_len=300)
    prefix = f"{provider}: " if provider else ""
    return f"{prefix}{name}: {msg}"