"""
CSRF protection via the double-submit-cookie pattern.

Defense-in-depth on top of SameSite=strict cookies. Even though a strict-samesite
session cookie is not sent on cross-site requests (which already blocks classic
CSRF), we add a second, independent check so a single misconfiguration doesn't
open a hole:

  1. GET /csrf issues a random token in a NON-HttpOnly cookie ('csrf_token') so the
     frontend JS can read it.
  2. The frontend echoes that value in the 'X-CSRF-Token' header on every
     state-changing request (POST/PUT/PATCH/DELETE).
  3. require_csrf compares the header to the cookie. A cross-site attacker can't
     read the victim's cookie (same-origin policy) and so can't forge the header —
     the request is rejected even if the session cookie somehow rode along.

Safe methods (GET/HEAD/OPTIONS) and the auth bootstrap (/login, /logout, /csrf)
are exempt: /login has no session to protect yet and is covered by rate limiting;
/csrf issues the token.
"""
import secrets
from fastapi import Request, HTTPException

CSRF_COOKIE = "csrf_token"
CSRF_HEADER = "x-csrf-token"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Paths that don't require a CSRF token (auth bootstrap + token issuance).
_EXEMPT_PATHS = {"/login", "/logout", "/csrf"}


def issue_token():
    """A fresh, unguessable CSRF token."""
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response, token, secure):
    """Attach the CSRF token as a readable (NON-HttpOnly) cookie so the frontend
    can echo it in a header. HttpOnly would defeat the double-submit pattern."""
    response.set_cookie(
        CSRF_COOKIE, token,
        httponly=False,        # must be readable by JS for the double-submit echo
        secure=secure,         # HTTPS-only in production
        samesite="strict",
        max_age=8 * 60 * 60,
        path="/",
    )


def require_csrf(request: Request):
    """
    Dependency for state-changing endpoints. Rejects the request unless the
    X-CSRF-Token header matches the csrf_token cookie. Constant-time comparison.
    """
    if request.method in _SAFE_METHODS or request.url.path in _EXEMPT_PATHS:
        return
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if not cookie or not header or not secrets.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")