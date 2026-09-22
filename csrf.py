"""
CSRF protection — synchronizer (session-bound) token.

Rather than the naive double-submit cookie (an independent, unsigned cookie compared
to a header), the token is bound to the server-signed session: it lives in
request.session['csrf_token'], carried in Starlette's SameSite=strict, signed
SessionMiddleware cookie. The frontend echoes it in the 'X-CSRF-Token' header on
every state-changing request, and require_csrf compares the header to the token held
in the session. A cross-site attacker can neither read nor forge a signed same-site
session cookie, so they cannot recover or set the token — this is the OWASP-recommended
approach for a stateful app and removes the independent unsigned cookie entirely.

  1. GET /csrf mints a token (if the session lacks one), stores it in the session, and
     returns it in the JSON body so the frontend can read it.
  2. The frontend echoes it in 'X-CSRF-Token' on POST/PUT/PATCH/DELETE.
  3. require_csrf compares the header to the session token (constant-time).

Exemptions: safe methods (GET/HEAD/OPTIONS) and POST /login only. Login necessarily
runs before any session or token exists and is protected by rate limiting instead.
Logout is NOT exempt — by then the client holds a token, and an attacker-forced
logout should be rejected like any other state-changing request. The token lifetime
is the session's (it's stored in the session), so there is no separate max_age to keep
in sync with settings.session_max_age.
"""
import secrets
from fastapi import Request, HTTPException

CSRF_HEADER = "x-csrf-token"
SESSION_KEY = "csrf_token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# POST /login is the only state-changing path allowed without a token: it runs before
# a session/token exists, and is guarded by rate limiting instead.
_EXEMPT_PATHS = {"/login"}


def issue_token():
    """A fresh, unguessable CSRF token."""
    return secrets.token_urlsafe(32)


def get_or_create_token(request: Request):
    """
    Return the session's CSRF token, minting and storing one if the session doesn't
    have it yet. Because it lives in the session, it stays stable for the session's
    lifetime and is discarded when the session is cleared (logout).
    """
    token = request.session.get(SESSION_KEY)
    if not token:
        token = issue_token()
        request.session[SESSION_KEY] = token
    return token


def require_csrf(request: Request):
    """
    Dependency for state-changing endpoints. Rejects the request unless the
    X-CSRF-Token header matches the token bound to the caller's session. Constant-time
    comparison; the error is identical whether the session token or the header was the
    missing/mismatched piece.
    """
    if request.method in _SAFE_METHODS or request.url.path in _EXEMPT_PATHS:
        return
    session_token = request.session.get(SESSION_KEY)
    header = request.headers.get(CSRF_HEADER)
    if not session_token or not header or not secrets.compare_digest(session_token, header):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")