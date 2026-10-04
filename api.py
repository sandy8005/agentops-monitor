from timeutil import utcnow
from database import get_connection as _db_get_connection
from settings import settings
from fastapi import (FastAPI, HTTPException, UploadFile, File,
                     Form, Query, Depends, Request)
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import os, tempfile
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
from llm import create_run, create_run_tx, request_cancel
from job_queue import enqueue, enqueue_tx
from pdf_reader import read_resume_file_isolated, PdfExtractionError
from auth import authenticate
from error_codes import classify_exception
from csrf import get_or_create_token, require_csrf

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Auto-generated API docs are handy in dev but expose the full API schema, so they
# are turned off in production (ENV=production). Set ENV to anything else to re-enable
# them locally.
_docs = None if settings.is_production else "/docs"
app = FastAPI(title="AgentOps Monitor", docs_url=_docs,
              redoc_url=(None if settings.is_production else "/redoc"),
              openapi_url=(None if settings.is_production else "/openapi.json"))

# --- Rate limiting ----------------------------------------------------------
# Per-client-IP limits to blunt login brute-force and enqueue abuse. Configurable via
# settings (RATE_LIMIT_LOGIN / RATE_LIMIT_RUNS).
#
# Storage: the default memory:// store is PER PROCESS — with 4 API workers the
# effective limit is 4x. Set RATE_LIMIT_STORAGE_URI (e.g. redis://redis:6379/0) to
# share counters before scaling horizontally.
#
# Client IP: get_remote_address reads request.client.host. Behind a reverse proxy
# that is the PROXY's address unless Uvicorn is started with --proxy-headers and
# --forwarded-allow-ips=<the proxy's IP> (never '*' on an internet-facing host, or
# clients can spoof X-Forwarded-For and dodge the limit).
limiter = Limiter(key_func=get_remote_address,
                  storage_uri=settings.rate_limit_storage_uri)
app.state.limiter = limiter


# --- HTTP security headers --------------------------------------------------
# The dashboard is a cookie-authenticated browser app, so it gets a strict CSP (all
# scripts/styles are served from /static; no inline script is needed), framing is
# forbidden (clickjacking), MIME sniffing is off, and referrers don't leak run URLs.
# HSTS is sent only in production (HTTPS-only deployments).
_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
_DOCS_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
             "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
             "img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'")


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    is_docs = path in ("/docs", "/redoc") or path.startswith("/docs/")
    response.headers.setdefault("Content-Security-Policy", _DOCS_CSP if is_docs else _CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    if settings.is_production:
        response.headers.setdefault("Strict-Transport-Security",
                                    "max-age=63072000; includeSubDomains")
    return response


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429,
                        content={"detail": "Too many requests — slow down."})

MAX_UPLOAD_BYTES = settings.max_upload_bytes
UPLOAD_CHUNK_BYTES = 64 * 1024

from logging_config import get_logger
log = get_logger("api")

# Closed vocabularies. The browser dropdowns are NOT validation — a client can call
# the API directly — so the server declares the allowed values and FastAPI rejects
# anything else with 422 before the handler runs. "" means "any".
WorkMode = Literal["", "remote", "hybrid", "onsite"]
EmploymentType = Literal["", "full-time", "part-time", "contract", "internship"]
Decision = Literal["Apply", "Maybe", "Skip"]

# Free-text input limits — guard memory, LLM cost, latency, and DB size. A 5 MB PDF
# can still expand to a lot of text, so the extracted resume is capped too.
MAX_NAME_CHARS = 200
MAX_ROLE_CHARS = 200
MAX_LOCATION_CHARS = 200
MAX_COMMENT_CHARS = 2000
MAX_USERNAME_CHARS = 150
MAX_PASSWORD_CHARS = 200
MAX_RESUME_CHARS = 60000


def _check_len(field, value, limit):
    if value and len(value) > limit:
        raise HTTPException(status_code=400,
                            detail=f"{field} is too long (max {limit} characters)")

# --- Session auth -----------------------------------------------------------
# Signed, HttpOnly session cookie via Starlette's SessionMiddleware. The signing
# key MUST be set (SESSION_SECRET in .env) for a public deploy; we refuse to boot
# with a default in production so sessions can't be forged with a known key.
# max_age gives the session a lifetime (refreshed per response = idle timeout);
# same_site='strict' + https_only(prod) + HttpOnly harden the cookie against CSRF
# and interception.
_SESSION_SECRET = settings.session_secret
if not _SESSION_SECRET:
    if settings.is_production:
        raise RuntimeError("SESSION_SECRET must be set when ENV=production")
    _SESSION_SECRET = "dev-only-insecure-session-secret-change-me"
app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET,
    session_cookie="agentops_session",
    https_only=settings.is_production,
    same_site="strict",
    max_age=settings.session_max_age,
)

# Trace-payload redaction (REDACT_TRACE_PAYLOADS, default ON; the old name
# REDACT_SENSITIVE is still honoured). EXACTLY what it covers, in API responses:
#   * LLM prompts and responses                     (embed resume + job text)
#   * tool-call inputs and outputs                  (embed resume + job text)
#   * step retrieved_context                        (resume-derived evidence)
#   * evaluation notes and hallucinated claims      (quote resume/job content)
#   * LLM / tool / step ERROR MESSAGES               (providers can echo input)
#   * agent-action free text (errors, rejections, model reasons) — recursively
# What it deliberately does NOT hide — these are the product's OUTPUT to the run's
# owner, shown only to that authenticated owner:
#   * generated application advice (/runs/{id}/rankings)
#   * the owner's own review comments and the review reason codes
#   * scores, decisions, job titles/companies, metrics
# It is a DISPLAY control, not deletion: the database still holds the data until
# erasure or the retention purge (privacy.py).
REDACT_TRACE_PAYLOADS = settings.redact_trace_payloads
REDACT_SENSITIVE = REDACT_TRACE_PAYLOADS   # deprecated alias
_REDACTED = "[redacted]"


def _redact(value):
    """Redact one trace payload when redaction is on (None stays None)."""
    if not REDACT_TRACE_PAYLOADS or value is None:
        return value
    return _REDACTED


def get_connection():
    # Delegate to the centralized pooled connection (see database.py).
    return _db_get_connection()
def require_auth(request: Request):
    """
    Dependency: every gated endpoint requires a logged-in session. Returns the
    session user dict; raises 401 otherwise. Applied to ALL data endpoints — the
    only open routes are GET / (the empty shell), /static/*, and /login.
    """
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Serve the static frontend assets (app.js, style.css, index.html) at /static/*.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the static dashboard shell (open). It contains no data — every
    data fetch it makes hits a gated endpoint, so an unauthenticated visitor sees
    an empty page and the JS redirects them to sign in."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/csrf")
def get_csrf(request: Request):
    """Return the session-bound CSRF token (minting one into the session if needed).
    The frontend reads it from this JSON body and echoes it in the X-CSRF-Token
    header on state-changing requests; require_csrf checks it against the session."""
    return {"csrf_token": get_or_create_token(request)}


@app.post("/login")
@limiter.limit(settings.rate_limit_login)
def login(request: Request, username: str = Form(...), password: str = Form(...),
          _csrf: None = Depends(require_csrf)):
    """Verify credentials via auth.authenticate and start a signed session.
    Rate-limited per IP to blunt brute-force."""
    _check_len("username", username, MAX_USERNAME_CHARS)
    _check_len("password", password, MAX_PASSWORD_CHARS)
    user = authenticate(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    # Session carries identity only. users.role exists in the schema but there is no
    # role-based authorization in this app yet (every endpoint is owner-scoped), so it
    # is deliberately NOT exposed — advertising a role implies RBAC that doesn't exist.
    request.session["user"] = {"id": user["id"], "username": user["username"]}
    return {"ok": True, "username": user["username"]}


@app.post("/logout")
def logout(request: Request, _csrf: None = Depends(require_csrf)):
    """Clear the session cookie. CSRF-protected (the frontend already sends the token)
    so a cross-site page can't force a logout."""
    request.session.clear()
    return {"ok": True}


@app.get("/me")
def whoami(user: dict = Depends(require_auth)):
    """Who am I — used by the frontend to decide whether to show the login form."""
    return {"username": user["username"]}


class _UploadTooLarge(Exception):
    pass


async def _read_capped(file, limit):
    """Read an upload in CHUNKS and stop as soon as it exceeds `limit`, so a huge
    body is never held in memory in full. (Starlette spools the multipart part to
    a temporary file; this bounds what WE buffer. The reverse proxy's body limit —
    e.g. nginx client_max_body_size — must be set as well: it is the only layer
    that can refuse the bytes before they are received.)"""
    chunks, total = [], 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise _UploadTooLarge()
        chunks.append(chunk)


@app.post("/upload")
@limiter.limit(settings.rate_limit_upload)
async def upload_resume(request: Request, file: UploadFile = File(...), name: str = Form(""),
                        user: dict = Depends(require_auth),
                        _csrf: None = Depends(require_csrf)):
    filename = file.filename or ""          # multipart parts may omit the filename
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    _check_len("name", name, MAX_NAME_CHARS)

    # Cheap early reject from the declared length (a client can lie, so the
    # streamed cap below is what actually enforces the limit).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES + 64 * 1024:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        contents = await _read_capped(file, MAX_UPLOAD_BYTES)
    except _UploadTooLarge:
        raise HTTPException(status_code=413, detail="File too large")
    if not contents[:5].startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Parsed in a resource-limited child process, not in the API process.
        resume_text = read_resume_file_isolated(tmp_path)
    except PdfExtractionError as e:
        log.warning("pdf extraction failed (%s)", type(e).__name__)
        raise HTTPException(status_code=400, detail="Could not read that PDF")
    finally:
        os.remove(tmp_path)

    name = (name or filename or "resume.pdf")[:MAX_NAME_CHARS]

    if not resume_text or not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF")

    # Cap the extracted text before storing / feeding it to the LLM — a big PDF can
    # expand into a very large amount of text. Truncation is never SILENT: the row
    # records it and the response tells the caller, because skills or experience
    # after the cut are simply not seen by the matcher.
    original_chars = len(resume_text)
    truncated = original_chars > MAX_RESUME_CHARS
    if truncated:
        log.info("resume text truncated from %d to %d chars", original_chars, MAX_RESUME_CHARS)
        resume_text = resume_text[:MAX_RESUME_CHARS]

    with get_connection() as conn:   # guaranteed release even if the INSERT throws
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO resumes (name, resume_text, created_at, user_id,
                                 original_chars, truncated)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
        """, (name, resume_text, utcnow(), user["id"], original_chars, truncated))
        resume_id = cur.fetchone()[0]
        # A NEW upload of a previously erased text is fresh consent to store it: lift
        # the parse-cache tombstone for this content (erasure_guards.py).
        from router import resume_content_hash
        cur.execute("DELETE FROM erased_resume_hashes WHERE content_hash = %s",
                    (resume_content_hash(resume_text),))

    message = f"Resume stored as #{resume_id}"
    if truncated:
        message += (f" — WARNING: the extracted text was {original_chars:,} characters; "
                    f"only the first {MAX_RESUME_CHARS:,} were kept and will be matched.")
    return {"resume_id": resume_id, "name": name,
            "chars": len(resume_text), "truncated": truncated,
            "original_chars": original_chars, "stored_chars": len(resume_text),
            "message": message}


@app.get("/resumes")
def list_resumes(user: dict = Depends(require_auth)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, length(resume_text), created_at, truncated, original_chars
            FROM resumes WHERE is_deleted = FALSE AND user_id = %s ORDER BY id DESC
        """, (user["id"],))
        rows = cur.fetchall()
        return [
            {"id": r[0], "name": r[1], "chars": r[2],
             "created_at": r[3].isoformat() if r[3] else None,
             "truncated": bool(r[4]), "original_chars": r[5]}
            for r in rows
        ]


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int, user: dict = Depends(require_auth),
                  _csrf: None = Depends(require_csrf)):
    """
    Delete a resume — for real. The row is kept so historical runs keep a valid
    foreign key and their metrics stay auditable, but the resume TEXT and name are
    erased, together with every trace that embeds them (LLM prompts/responses, tool
    I/O, context, advice, cached parse, LangGraph checkpoints). See privacy.py.
    """
    from privacy import erase_resume, ErasureConflict
    try:
        found = erase_resume(resume_id, user["id"])
    except ErasureConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not found:
        # 404 (not 403) so we don't reveal that the id exists for another owner.
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return {"resume_id": resume_id, "deleted": True, "erased": True}


# Per-run cost, one definition (matches agent_store.cost_summary): priced calls,
# calls whose cost is unknown (usage missing / failed after dispatch) plus
# reservations abandoned by a dead worker, and the conservative bound of those.
_COST_LATERAL_SQL = """
    LEFT JOIN LATERAL (
        SELECT (SELECT SUM(cost_usd) FROM llm_calls
                WHERE run_id = r.id AND cost_status = 'priced') AS known,
               (SELECT COUNT(*) FROM llm_calls
                WHERE run_id = r.id AND cost_status = 'unknown')
             + (SELECT COUNT(*) FROM llm_cost_reservations
                WHERE run_id = r.id AND status = 'abandoned') AS unknown_calls,
               COALESCE((SELECT SUM(cost_upper_bound_usd) FROM llm_calls
                         WHERE run_id = r.id AND cost_status = 'unknown'), 0)
             + COALESCE((SELECT SUM(amount_usd) FROM llm_cost_reservations
                         WHERE run_id = r.id AND status IN ('abandoned', 'open')), 0)
                   AS unknown_bound
    ) c ON TRUE"""


def _cost_fields(known, unknown_calls, unknown_bound=0.0):
    """One cost shape for every endpoint. total_cost is only a number when it is
    COMPLETE; otherwise the known part, the number of calls with unknown cost and
    a conservative upper bound are explicit (a bare SUM() of a partly-unknown run
    would look complete)."""
    known = float(known or 0)
    unknown_calls = int(unknown_calls or 0)
    bound = float(unknown_bound or 0)
    complete = unknown_calls == 0
    return {"total_cost": known if complete else None, "known_cost_usd": known,
            "unknown_cost_calls": unknown_calls, "cost_complete": complete,
            "cost_upper_bound_usd": known + bound,
            "cost_basis": "estimated_paid_tier"}


@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100), user: dict = Depends(require_auth)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT r.id, r.status, r.started_at, r.total_tokens,
                   r.target_role, r.location, r.work_mode, r.error_code,
                   COALESCE(c.known, 0), c.unknown_calls, c.unknown_bound
            FROM runs r
            {_COST_LATERAL_SQL}
            WHERE r.user_id = %s ORDER BY r.id DESC LIMIT %s
        """, (user["id"], limit))
        rows = cur.fetchall()
        return [{"id": r[0], "status": r[1],
                 "started_at": r[2].isoformat() if r[2] else None,
                 "total_tokens": r[3], "target_role": r[4], "location": r[5],
                 "work_mode": r[6], "error_code": r[7], **_cost_fields(r[8], r[9], r[10])}
                for r in rows]


class AgentGoalRequest(BaseModel):
    """Goal options for mode='agent'. Location, work mode and employment type come
    from the run's own fields and become FIXED constraints the controller cannot
    change."""
    model_config = ConfigDict(extra="forbid")
    target_count: int = Field(10, ge=1, le=50)
    alternative_titles: list[str] = Field(default_factory=list, max_length=8)
    seniority: Literal["", "intern", "entry", "junior", "mid", "senior"] = ""
    include_maybe: bool = False
    providers: list[Literal["adzuna", "remotive"]] = Field(
        default_factory=lambda: ["adzuna", "remotive"])
    max_iterations: int = Field(20, ge=1, le=60)
    max_searches: int = Field(6, ge=1, le=20)
    max_llm_calls: int = Field(40, ge=0, le=200)
    max_runtime_seconds: int = Field(1800, ge=60, le=14400)
    max_cost_usd: float = Field(0.5, ge=0.0, le=20.0)
    use_llm_controller: bool = True
    use_llm_advice: bool = False
    on_model_unavailable: Literal["rules", "pause"] = "rules"
    require_verified_matches: bool = False


class StartRunRequest(BaseModel):
    """
    JSON body for POST /runs. Run configuration travels in the BODY, not the query
    string: query strings end up in reverse-proxy/access logs, APM tools and browser
    history. Unknown fields are rejected (422) so a typo can't be silently ignored.
    """
    model_config = ConfigDict(extra="forbid")
    resume_id: Optional[int] = None
    target_role: str = ""
    location: str = ""
    work_mode: WorkMode = ""
    employment_type: EmploymentType = ""
    evaluate: bool = False
    # The controller agent is the ONLY execution engine (the fixed "pipeline"
    # engine was retired). "agent" is still accepted so existing clients keep
    # working; anything else is rejected with 422.
    mode: Literal["agent"] = "agent"
    goal: Optional[AgentGoalRequest] = None
    # "rules_only" = zero model calls for the whole run.
    model_policy: Literal["auto", "rules_only"] = "auto"


class ResumeRunRequest(BaseModel):
    """JSON body for POST /runs/{id}/resume (the reviewer's comment is free text and
    must not travel in the URL either). review_id binds the decision to the exact
    review card that was shown (R01): a stale tab gets 409, never another job."""
    model_config = ConfigDict(extra="forbid")
    decision: Decision = "Maybe"
    comment: str = ""
    review_id: Optional[str] = Field(None, max_length=100)
    answer: Optional[str] = Field(None, max_length=60)


def _build_agent_goal(body):
    """Validated AgentGoal from the request (raises HTTPException 422 on conflict)."""
    from agent_goal import AgentGoal
    g = body.goal or AgentGoalRequest()
    try:
        goal = AgentGoal(
            description=f"Find {g.target_count} suitable {g.seniority + ' ' if g.seniority else ''}"
                        f"{body.target_role.strip()} roles",
            target_role=body.target_role.strip(),
            alternative_titles=g.alternative_titles,
            target_count=g.target_count,
            qualifying_decisions=["Apply", "Maybe"] if g.include_maybe else ["Apply"],
            providers=g.providers,
            constraints={"location": body.location.strip(), "work_mode": body.work_mode,
                         "employment_type": body.employment_type, "seniority": g.seniority},
            limits={"max_iterations": g.max_iterations, "max_searches": g.max_searches,
                    "max_llm_calls": g.max_llm_calls,
                    "max_runtime_seconds": g.max_runtime_seconds,
                    "max_cost_usd": g.max_cost_usd},
            evaluate_quality=body.evaluate,
            use_llm_controller=g.use_llm_controller,
            use_llm_advice=g.use_llm_advice,
            on_model_unavailable=g.on_model_unavailable,
            model_policy=body.model_policy,
            require_verified_matches=g.require_verified_matches,
        )
    except ValueError as e:
        # pydantic.ValidationError is a ValueError: a conflicting / out-of-range goal
        # is the CLIENT's error (422). Anything else is a bug in our code and must
        # surface as a 500, not be reported to the user as "invalid goal".
        raise HTTPException(status_code=422, detail=f"invalid agent goal: {e}")
    from agent_loop import cost_preflight
    blocked = cost_preflight(goal)
    if blocked:
        # Fail closed at submission: a hard USD cap cannot be enforced for a model
        # whose price is unknown (see pricing.py).
        raise HTTPException(status_code=422, detail=blocked["reason"])
    return goal


@app.get("/reviews/pending")
def pending_reviews(user: dict = Depends(require_auth)):
    """
    Lightweight feed for the "Runs Awaiting Your Review" panel: ONE query returning
    only what a review card shows (run id, role, status, the pending_review
    payload). The dashboard polls this every few seconds, so it must not expand the
    full trace of every paused run the way GET /runs/{id} does.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, target_role, location, status, pending_review
            FROM runs
            WHERE user_id = %s AND status = 'waiting_for_human'
            ORDER BY id DESC LIMIT 50
        """, (user["id"],))
        rows = cur.fetchall()
    return [{"run_id": r[0], "target_role": r[1], "location": r[2], "status": r[3],
             "pending_review": r[4] or {}} for r in rows]


@app.post("/runs")
@limiter.limit(settings.rate_limit_runs)
def start_run(request: Request, body: StartRunRequest,
              user: dict = Depends(require_auth),
              _csrf: None = Depends(require_csrf)):
    resume_id, target_role, location = body.resume_id, body.target_role, body.location
    work_mode, employment_type = body.work_mode, body.employment_type
    evaluate = body.evaluate
    if resume_id is None:
        raise HTTPException(status_code=400, detail="resume_id is required; upload or pick a resume first")
    if not target_role.strip():
        raise HTTPException(status_code=400, detail="target_role is required for a search")
    _check_len("target_role", target_role, MAX_ROLE_CHARS)
    _check_len("location", location, MAX_LOCATION_CHARS)
    agent_goal = _build_agent_goal(body)

    conn = get_connection()
    try:
        cur = conn.cursor()
        # The resume must belong to THIS user — otherwise a user could run against
        # someone else's resume by guessing its id. Checked in the SAME transaction
        # that creates the run and enqueues its job.
        cur.execute("SELECT is_deleted FROM resumes WHERE id = %s AND user_id = %s",
                    (resume_id, user["id"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
        if row[0]:
            raise HTTPException(status_code=400, detail="That resume was deleted; upload it again to run new searches")

        # R22: lock the resume row (compatible with erase_resume's FOR UPDATE) and
        # re-check deletion under the lock, so a concurrent erase either wins (and
        # this start is rejected) or waits until the run exists (and the erase then
        # sees an active run and returns 409).
        cur.execute("SELECT is_deleted FROM resumes WHERE id = %s AND user_id = %s FOR SHARE",
                    (resume_id, user["id"]))
        locked = cur.fetchone()
        if not locked or locked[0]:
            raise HTTPException(status_code=409, detail="That resume was deleted; upload it again")

        # ATOMIC: create the run AND enqueue its worker job in ONE transaction, so
        # they commit or roll back together. Previously create_run() and enqueue()
        # committed on separate connections: if enqueue failed, the run was left
        # 'running' with no queue job and could sit forever. Now a failure rolls the
        # run back too, so there is nothing orphaned to recover.
        run_id = create_run_tx(cur, "job search run (dashboard)", resume_id=resume_id,
                               target_role=target_role, location=location,
                               work_mode=work_mode, employment_type=employment_type,
                               user_id=user["id"], goal=agent_goal)
        # The goal (with every setting) lives on the run row; the job only names it.
        job_id = enqueue_tx(cur, "start_run", {"run_id": run_id, "resume_id": resume_id},
                            run_id=run_id)
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        # Type + code only: DB error details can echo row values (goal text, roles).
        log.error("start_run failed (%s, code=%s)", type(e).__name__, classify_exception(e))
        raise HTTPException(status_code=500, detail="Unable to start the run.")
    finally:
        conn.close()
    return {"run_id": run_id, "resume_id": resume_id, "target_role": target_role,
            "job_id": job_id, "mode": "agent",
            "message": f"Run {run_id} enqueued (job {job_id})."}


@app.delete("/runs/{run_id}")
def delete_run_endpoint(run_id: int, user: dict = Depends(require_auth),
                        _csrf: None = Depends(require_csrf)):
    """Hard-delete one of the caller's finished runs and all of its traces."""
    from privacy import delete_run, ErasureConflict
    try:
        found = delete_run(run_id, user["id"])
    except ErasureConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not found:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"run_id": run_id, "deleted": True}


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int, user: dict = Depends(require_auth),
               _csrf: None = Depends(require_csrf)):
    with get_connection() as conn:
        cur = conn.cursor()
        # Lock the run row so concurrent cancels (double-click) serialize — otherwise
        # two cancels of a waiting_for_human run could each enqueue a resume job for
        # the same LangGraph checkpoint. Same exactly-once pattern as /resume.
        cur.execute("SELECT status FROM runs WHERE id = %s AND user_id = %s FOR UPDATE",
                    (run_id, user["id"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        status = row[0]

        if status == "running":
            # Actively executing. Set the flag in THIS locked transaction. The loop
            # checks it at every decision/tool boundary, and every external dispatch
            # (model reservation, provider token count, job search) REQUIRES
            # cancel_requested = FALSE under this same row lock — so once this
            # commits, the worker cannot send another request.
            cur.execute("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
            return {"run_id": run_id, "cancel_requested": True}

        if status in ("queued", "retrying"):
            # Not executing yet. Cancel the queued job BEFORE a worker claims it, so the
            # user doesn't wait through retry backoff; if we cancel it, finalize the run
            # IMMEDIATELY.
            #
            # This run-row lock does NOT serialize against job_queue.claim_next, which
            # locks only the QUEUE row: a worker may already have claimed the job while
            # the run still reads 'queued'. Then the UPDATE below matches nothing and we
            # fall back to the flag. That is safe because the worker's begin_execution
            # (an UPDATE of this run row) waits for this lock, and run_agent_loop
            # re-reads cancel_requested AFTER begin_execution — plus every dispatch
            # refuses a cancelled run (see the 'running' branch).
            cur.execute("UPDATE job_queue SET status = 'cancelled', finished_at = NOW() "
                        "WHERE run_id = %s AND status = 'queued'", (run_id,))
            if cur.rowcount > 0:
                cur.execute("UPDATE runs SET status = 'cancelled', ended_at = NOW(), "
                            "cancel_requested = TRUE, error_code = 'cancelled', "
                            "stop_reason = 'cancelled by user' WHERE id = %s", (run_id,))
                return {"run_id": run_id, "cancelled": True}
            # A worker claimed the queue row in the meantime — set the flag; the worker
            # sees it right after begin_execution, before setup or any external request.
            cur.execute("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
            return {"run_id": run_id, "cancel_requested": True}

        if status == "waiting_for_human":
            # Paused at an interrupt — not executing, so no loop is checking the flag.
            # Flip to queued, set the cancel flag, AND enqueue the resume-to-cancel job
            # in ONE locked transaction (same pattern as /resume): the row lock makes
            # this exactly-once, so two concurrent cancels can't create duplicate
            # resume jobs for the same checkpoint. The graph wakes, sees the cancel,
            # and terminates through its normal 'cancelled' routing (no orphaned
            # checkpoint).
            cur.execute("UPDATE runs SET status = 'queued', cancel_requested = TRUE WHERE id = %s",
                        (run_id,))
            enqueue_tx(cur, "resume_run",
                       {"run_id": run_id, "decision": "Skip", "comment": "cancelled by user",
                        "reviewer_user_id": user["id"], "reviewer": user["username"]},
                       run_id=run_id)
            return {"run_id": run_id, "cancel_requested": True, "resumed_to_cancel": True}

        raise HTTPException(status_code=400,
                            detail=f"Run {run_id} cannot be cancelled (status: {status})")


@app.get("/runs/{run_id}")
def get_run(run_id: int, user: dict = Depends(require_auth)):
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            SELECT id, status, started_at, ended_at, input_summary, total_tokens, total_cost,
                   resume_id, target_role, location, work_mode, employment_type, pending_review,
                   stop_reason, error_code, attempt, last_attempt_ended_at
            FROM runs WHERE id = %s AND user_id = %s
        """, (run_id, user["id"]))
        run = cur.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        cur.execute(f"""SELECT COALESCE(c.known, 0), c.unknown_calls, c.unknown_bound
                        FROM runs r {_COST_LATERAL_SQL} WHERE r.id = %s""", (run_id,))
        cost_row = cur.fetchone()

        # Bulk-load the whole trace in a CONSTANT number of queries (steps, tool
        # calls, LLM calls, evaluations — each filtered by run_id and served by the
        # *_run_idx indexes) and group by step in Python. The previous version ran
        # three queries PER STEP (~92 queries for a 30-step run).
        cur.execute("""
            SELECT id, step_name, status, match_score, score_decision,
                   llm_decision, needs_human_review, review_status, error_message,
                   retrieved_context, reviewer, review_comment, review_reason, score_breakdown,
                   final_decision, run_attempt, security_flag, security_reason,
                   judge_status, judge_skip_reason, reviewed_at
            FROM steps WHERE run_id = %s ORDER BY run_attempt NULLS FIRST, step_order, id
        """, (run_id,))
        step_rows = cur.fetchall()

        cur.execute("""
            SELECT step_id, tool_name, status, latency_ms, input_json, output_json,
                   error_message, operation_name, run_attempt
            FROM tool_calls WHERE run_id = %s ORDER BY id
        """, (run_id,))
        tools_by_step = {}
        for t in cur.fetchall():
            tools_by_step.setdefault(t[0], []).append({
                "tool_name": t[1], "status": t[2], "latency_ms": t[3],
                "input_json": _redact(t[4]), "output_json": _redact(t[5]),
                "error_message": _redact(t[6]), "operation": t[7], "run_attempt": t[8]})

        cur.execute("""
            SELECT step_id, prompt_tokens, completion_tokens, latency_ms, cost_usd, status,
                   prompt, response, error_message, operation_name, attempt_number,
                   retry_count, provider_request_id, logical_call_id, pricing_version,
                   model, run_attempt, cost_status, cost_upper_bound_usd, usage_missing
            FROM llm_calls WHERE run_id = %s ORDER BY id
        """, (run_id,))
        llm_by_step = {}
        for l in cur.fetchall():
            llm_by_step.setdefault(l[0], []).append({
                "prompt_tokens": l[1], "completion_tokens": l[2], "latency_ms": l[3],
                # ESTIMATED paid-tier cost (pricing.py); None = model price unknown.
                "cost_usd": float(l[4]) if l[4] is not None else None,
                "cost_basis": "estimated_paid_tier",
                "status": l[5], "prompt": _redact(l[6]), "response": _redact(l[7]),
                "error_message": _redact(l[8]), "operation": l[9],
                # retry explainability: logical call -> HTTP attempt -> run attempt
                "attempt_number": l[10], "retry_count": l[11], "provider_request_id": l[12],
                "logical_call_id": l[13], "pricing_version": l[14], "model": l[15],
                "run_attempt": l[16],
                # priced | unknown (usage missing / failed after dispatch; bounded) |
                # not_billed (rejected by the provider before any work)
                "cost_status": l[17], "cost_upper_bound_usd": l[18], "usage_missing": l[19]})

        cur.execute("""
            SELECT DISTINCT ON (step_id) step_id, relevance_score, faithfulness_score,
                   completeness_score, hallucination_detected, hallucinated_claims, notes
            FROM evaluations WHERE run_id = %s ORDER BY step_id, id DESC
        """, (run_id,))
        eval_by_step = {}
        for ev in cur.fetchall():
            eval_by_step[ev[0]] = {
                "relevance_score": ev[1], "faithfulness_score": ev[2],
                "completeness_score": ev[3], "hallucination_detected": ev[4],
                "hallucinated_claims": _redact(ev[5]), "notes": _redact(ev[6])}

        steps = []
        for s in step_rows:
            step_id = s[0]
            steps.append({
                "id": step_id, "step_name": s[1], "status": s[2],
                "match_score": float(s[3]) if s[3] is not None else None,
                "score_decision": s[4], "llm_decision": s[5],
                "needs_human_review": s[6], "review_status": s[7],
                "error_message": _redact(s[8]),
                "retrieved_context": _redact(s[9]),
                "reviewer": s[10], "review_comment": s[11], "review_reason": s[12],
                "score_breakdown": s[13],
                "final_decision": s[14],
                "run_attempt": s[15],
                "security_flag": s[16], "security_reason": s[17],
                "judge_status": s[18], "judge_skip_reason": s[19],
                "reviewed_at": s[20].isoformat() if s[20] else None,
                "tool_calls": tools_by_step.get(step_id, []),
                "llm_calls": llm_by_step.get(step_id, []),
                "evaluation": eval_by_step.get(step_id)
            })

        return {
            "id": run[0], "status": run[1],
            "started_at": run[2].isoformat() if run[2] else None,
            "ended_at": run[3].isoformat() if run[3] else None,
            "input_summary": run[4],
            "total_tokens": run[5],
            # total_cost is NULL unless complete; the partial known sum and the
            # number of unpriced calls are always explicit (R15).
            **_cost_fields(*cost_row),
            "resume_id": run[7], "target_role": run[8], "location": run[9],
            "work_mode": run[10], "employment_type": run[11],
            "pending_review": run[12],
            # R04: stop_reason can embed exception text derived from user input.
            "stop_reason": _redact(run[13]), "error_code": run[14],
            "attempt": run[15],
            "last_attempt_ended_at": run[16].isoformat() if run[16] else None,
            "steps": steps
        }


@app.post("/runs/{run_id}/resume")
def resume_run(run_id: int, body: ResumeRunRequest,
               user: dict = Depends(require_auth),
               _csrf: None = Depends(require_csrf)):
    """Resume a paused (waiting_for_human) run with the human's decision. The
    reviewer's identity travels with the decision into the audit trail."""
    decision, comment = body.decision, body.comment
    _check_len("comment", comment, MAX_COMMENT_CHARS)
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Lock the run row for the duration of this transaction. A concurrent
        # resume / double-click serializes behind this lock, so exactly one request
        # flips the run — the equivalent of the old rowcount compare-and-swap, but
        # now the flip and the enqueue live in the SAME transaction.
        cur.execute("SELECT status, pending_review, mode FROM runs "
                    "WHERE id = %s AND user_id = %s FOR UPDATE", (run_id, user["id"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if row[0] != "waiting_for_human":
            raise HTTPException(status_code=409,
                                detail=f"Run {run_id} is not awaiting review (already {row[0]})")
        pending = row[1] or {}
        expected = pending.get("review_id")
        # R01: the decision must name the review card currently pending. A stale tab
        # (the run moved on to another job) is rejected instead of deciding that job.
        if expected is not None and body.review_id != expected:
            raise HTTPException(status_code=409,
                                detail="This review is no longer current; refresh and try again")
        if pending.get("type") == "input_request":
            if body.answer not in (pending.get("options") or []):
                raise HTTPException(status_code=422, detail="answer must be one of the offered options")
        if row[2] != "agent":
            # A retired legacy run (migration 0013 closes these): nothing resumes it.
            raise HTTPException(status_code=409,
                                detail="This run belongs to the retired pipeline engine and "
                                       "cannot be resumed; cancel it and start a new run")
        cur.execute("""
            UPDATE review_requests SET status = 'submitted', decision = %s, answer = %s,
                   comment = %s, reviewer_user_id = %s, reviewer = %s, submitted_at = NOW()
            WHERE review_id = %s AND run_id = %s AND status = 'pending'
        """, (decision if pending.get("type") != "input_request" else None, body.answer,
              comment, user["id"], user["username"], expected, run_id))
        if cur.rowcount != 1:
            raise HTTPException(status_code=409, detail="This review was already answered")

        # ATOMIC: move waiting_for_human -> queued AND enqueue the resume job
        # together. The run goes back to 'queued' (not straight to 'running') because
        # it is only waiting in the queue until a worker claims the resume job and
        # actually resumes execution — _mark_run_running flips it to 'running' then.
        # If the enqueue fails, this status change rolls back, so the run stays
        # 'waiting_for_human' and the user can retry — no lost work.
        cur.execute("UPDATE runs SET status = 'queued' WHERE id = %s", (run_id,))
        enqueue_tx(cur, "resume_run",
                   {"run_id": run_id, "decision": decision, "comment": comment,
                    "reviewer_user_id": user["id"], "reviewer": user["username"],
                    "review_id": expected, "answer": body.answer},
                   run_id=run_id)
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        log.error("resume_run failed (%s, code=%s)", type(e).__name__, classify_exception(e))
        raise HTTPException(status_code=500, detail="Unable to resume the run.")
    finally:
        conn.close()
    return {"run_id": run_id, "resumed_with": decision}


@app.get("/runs/{run_id}/rankings")
def get_run_rankings(run_id: int, user: dict = Depends(require_auth)):
    """
    The persisted final ranked list for a run (self-contained snapshot rows from
    run_rankings), with any generated advice joined in from run_advice. Ordered by
    rank_position. Returns [] if the run produced no ranking.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM runs WHERE id = %s AND user_id = %s", (run_id, user["id"]))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # R20: advice joins by (run_id, job_id). A title fallback is used ONLY for
        # legacy advice rows that have no job_id AND whose title is unambiguous.
        cur.execute("SELECT job_id, title, advice FROM run_advice WHERE run_id = %s", (run_id,))
        advice_by_job = {}
        legacy_by_title = {}
        for job_id, title, advice in cur.fetchall():
            if job_id is not None:
                advice_by_job[job_id] = advice
            elif title:
                legacy_by_title.setdefault(title, []).append(advice)

        # resume_suggestions is part of the required schema (migration 0010). Any
        # error here — outage, permissions, a missing table on an un-migrated DB —
        # is a real failure and surfaces as one; it is never shown as "no
        # suggestions".
        suggestions_by_job = {}
        cur.execute("""
            SELECT job_id, kind, original_text, suggested_text, reason, evidence, method,
                   status, validation_notes
            FROM resume_suggestions WHERE run_id = %s AND status <> 'rejected'
            ORDER BY job_id, position
        """, (run_id,))
        for r in cur.fetchall():
            suggestions_by_job.setdefault(r[0], []).append({
                "kind": r[1], "original_text": r[2], "suggested_text": r[3],
                "reason": r[4], "evidence": r[5], "method": r[6], "status": r[7],
                "validation_notes": r[8]})

        cur.execute("""
            SELECT rank_position, job_id, title, company, score, final_decision, apply_url
            FROM run_rankings WHERE run_id = %s ORDER BY rank_position
        """, (run_id,))
        rows = cur.fetchall()

    title_counts = {}
    for r in rows:
        title_counts[r[2]] = title_counts.get(r[2], 0) + 1
    rankings = []
    for pos, job_id, title, company, score, final_decision, apply_url in rows:
        advice = advice_by_job.get(job_id) if job_id is not None else None
        if advice is None and job_id is None and title_counts.get(title) == 1 \
                and len(legacy_by_title.get(title, [])) == 1:
            advice = legacy_by_title[title][0]
        rankings.append({
            "rank": pos, "job_id": job_id, "title": title, "company": company,
            "score": float(score) if score is not None else None,
            "final_decision": final_decision, "apply_url": _safe_url(apply_url),
            "advice": advice,
            "suggestions": suggestions_by_job.get(job_id, []) if job_id is not None else [],
        })
    return {"run_id": run_id, "rankings": rankings}


# Free-text fields inside action traces that can quote resume/job/provider/user
# content. Everything else in an observation is counts, ids, enums, statuses or a
# backend-validated search title.
_FREE_TEXT_KEYS = {"failed", "rejected", "note", "error", "stopped_before_execution",
                   "answer", "question", "reason", "provider_detail", "generation"}


def _redact_tree(value, key=None):
    """Apply the trace-redaction policy RECURSIVELY (N09): free-text values anywhere
    in the structure are redacted, structural data is kept."""
    if isinstance(value, dict):
        return {k: _redact_tree(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_tree(v, key) for v in value]
    if key in _FREE_TEXT_KEYS and value is not None:
        return _REDACTED
    return value


def _safe_action(a):
    """Explicit response schema for one controller action."""
    if not REDACT_TRACE_PAYLOADS:
        return a
    out = dict(a)
    out["error"] = _redact(a.get("error"))
    out["observation"] = _redact_tree(a.get("observation"))
    out["arguments"] = _redact_tree(a.get("arguments"))
    # A model-written justification is free text; backend/rules reasons are ours.
    if a.get("decided_by") == "llm":
        out["reason"] = _redact(a.get("reason"))
    out["attempts"] = [{**t, "error": _redact(t.get("error")),
                        "observation": _redact_tree(t.get("observation"))}
                       for t in (a.get("attempts") or [])]
    return out


def _safe_url(url):
    """Only http(s) provider URLs are returned as clickable links (R07)."""
    u = (url or "").strip()
    return u if u.lower().startswith(("https://", "http://")) else None


@app.get("/runs/{run_id}/agent")
def get_agent_timeline(run_id: int, user: dict = Depends(require_auth)):
    """AgentOps view of an autonomous run: goal, fixed constraints, limits, verified
    progress, which policy chose actions, and every action with its short reason,
    validation outcome and compact observation."""
    import agent_store
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT mode, goal_json, controller_mode, goal_progress, llm_calls_reserved,
                              llm_call_budget, status, error_code
                       FROM runs WHERE id = %s AND user_id = %s""", (run_id, user["id"]))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if row[0] != "agent":
        return {"run_id": run_id, "mode": "pipeline_retired", "actions": []}
    actions = [_safe_action(a) for a in agent_store.list_actions(run_id)]
    usage = agent_store.run_usage(run_id)
    return {"run_id": run_id, "mode": "agent", "goal": row[1], "controller_mode": row[2],
            "progress": row[3], "status": row[6], "error_code": row[7],
            "llm_calls": {"reserved": row[4], "budget": row[5]},
            "runtime": {"active_seconds": usage["active_runtime_seconds"],
                        "wall_clock_seconds": usage["elapsed_seconds"]},
            "cost": {"known_estimated_usd": usage["known_cost_usd"],
                     "calls_with_unknown_cost": usage["unknown_cost_calls"],
                     "unknown_cost_upper_bound_usd": usage["unknown_cost_bound_usd"],
                     "in_flight_reserved_usd": usage["reserved_open_usd"],
                     "committed_usd": usage["committed_usd"],
                     "limit_usd": ((row[1] or {}).get("limits") or {}).get("max_cost_usd"),
                     "complete": usage["unknown_cost_calls"] == 0,
                     "basis": "estimated_paid_tier"},
            "actions": actions}