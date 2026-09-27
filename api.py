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

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

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


@app.post("/upload")
async def upload_resume(file: UploadFile = File(...), name: str = Form(""),
                        user: dict = Depends(require_auth),
                        _csrf: None = Depends(require_csrf)):
    filename = file.filename or ""          # multipart parts may omit the filename
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    _check_len("name", name, MAX_NAME_CHARS)

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")
    if not contents[:5].startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Parsed in a resource-limited child process, not in the API process.
        resume_text = read_resume_file_isolated(tmp_path)
    except PdfExtractionError as e:
        log.warning("pdf extraction failed: %s", e)
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


@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100), user: dict = Depends(require_auth)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, status, started_at, total_tokens, total_cost,
                   target_role, location, work_mode, error_code
            FROM runs WHERE user_id = %s ORDER BY id DESC LIMIT %s
        """, (user["id"], limit))
        rows = cur.fetchall()
        return [
            {"id": r[0], "status": r[1],
             "started_at": r[2].isoformat() if r[2] else None,
             "total_tokens": r[3], "total_cost": float(r[4]) if r[4] else 0,
             "cost_basis": "estimated_paid_tier",
             "target_role": r[5], "location": r[6], "work_mode": r[7],
             "error_code": r[8]}
            for r in rows
        ]


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
    live_only: bool = False


class ResumeRunRequest(BaseModel):
    """JSON body for POST /runs/{id}/resume (the reviewer's comment is free text and
    must not travel in the URL either)."""
    model_config = ConfigDict(extra="forbid")
    decision: Decision = "Maybe"
    comment: str = ""


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
    evaluate, live_only = body.evaluate, body.live_only
    if resume_id is None:
        raise HTTPException(status_code=400, detail="resume_id is required; upload or pick a resume first")
    if not target_role.strip():
        raise HTTPException(status_code=400, detail="target_role is required for a search")
    _check_len("target_role", target_role, MAX_ROLE_CHARS)
    _check_len("location", location, MAX_LOCATION_CHARS)

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

        # ATOMIC: create the run AND enqueue its worker job in ONE transaction, so
        # they commit or roll back together. Previously create_run() and enqueue()
        # committed on separate connections: if enqueue failed, the run was left
        # 'running' with no queue job and could sit forever. Now a failure rolls the
        # run back too, so there is nothing orphaned to recover.
        run_id = create_run_tx(cur, "job search run (dashboard)", resume_id=resume_id,
                               target_role=target_role, location=location,
                               work_mode=work_mode, employment_type=employment_type,
                               user_id=user["id"])
        job_id = enqueue_tx(cur, "start_run", {
            "resume_id": resume_id, "target_role": target_role, "location": location,
            "work_mode": work_mode, "employment_type": employment_type,
            "evaluate": evaluate, "run_id": run_id, "live_only": live_only,
        }, run_id=run_id)
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        log.exception("start_run failed: %s", e)   # full detail to the server log only
        raise HTTPException(status_code=500, detail="Unable to start the run.")
    finally:
        conn.close()
    return {"run_id": run_id, "resume_id": resume_id, "target_role": target_role,
            "live_only": live_only, "job_id": job_id,
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
            # Actively executing — cooperative cancel: the loop checks is_cancel_requested
            # between jobs. Set the flag in THIS locked transaction.
            cur.execute("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
            return {"run_id": run_id, "cancel_requested": True}

        if status in ("queued", "retrying"):
            # NOT executing. Cancel the queued job BEFORE a worker claims it, so the user
            # doesn't wait through retry backoff. The row lock serializes against
            # claim_next: if we cancel the queued job first, finalize the run IMMEDIATELY;
            # if a worker just claimed it (now 'running'), fall back to cooperative cancel.
            cur.execute("UPDATE job_queue SET status = 'cancelled', finished_at = NOW() "
                        "WHERE run_id = %s AND status = 'queued'", (run_id,))
            if cur.rowcount > 0:
                cur.execute("UPDATE runs SET status = 'cancelled', ended_at = NOW(), "
                            "cancel_requested = TRUE, error_code = 'cancelled', "
                            "stop_reason = 'cancelled by user' WHERE id = %s", (run_id,))
                return {"run_id": run_id, "cancelled": True}
            # A worker claimed it in the meantime — cooperative cancel instead.
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
                   model, run_attempt
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
                "run_attempt": l[16]})

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
            "total_tokens": run[5], "total_cost": float(run[6]) if run[6] else 0,
            "cost_basis": "estimated_paid_tier",
            "resume_id": run[7], "target_role": run[8], "location": run[9],
            "work_mode": run[10], "employment_type": run[11],
            "pending_review": run[12],
            "stop_reason": run[13], "error_code": run[14],
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
        cur.execute("SELECT status FROM runs WHERE id = %s AND user_id = %s FOR UPDATE",
                    (run_id, user["id"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        if row[0] != "waiting_for_human":
            raise HTTPException(status_code=409,
                                detail=f"Run {run_id} is not awaiting review (already {row[0]})")

        # ATOMIC: move waiting_for_human -> queued AND enqueue the resume job
        # together. The run goes back to 'queued' (not straight to 'running') because
        # it is only waiting in the queue until a worker claims the resume job and
        # actually resumes execution — _mark_run_running flips it to 'running' then.
        # If the enqueue fails, this status change rolls back, so the run stays
        # 'waiting_for_human' and the user can retry — no lost work.
        cur.execute("UPDATE runs SET status = 'queued' WHERE id = %s", (run_id,))
        enqueue_tx(cur, "resume_run",
                   {"run_id": run_id, "decision": decision, "comment": comment,
                    "reviewer_user_id": user["id"], "reviewer": user["username"]},
                   run_id=run_id)
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        log.exception("resume_run failed: %s", e)
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

        # advice keyed by job_id (and by title as a fallback for legacy rows)
        cur.execute("SELECT job_id, title, advice FROM run_advice WHERE run_id = %s", (run_id,))
        advice_by_job = {}
        advice_by_title = {}
        for job_id, title, advice in cur.fetchall():
            if job_id is not None:
                advice_by_job[job_id] = advice
            if title:
                advice_by_title[title] = advice

        cur.execute("""
            SELECT rank_position, job_id, title, company, score, final_decision, apply_url
            FROM run_rankings WHERE run_id = %s ORDER BY rank_position
        """, (run_id,))
        rows = cur.fetchall()

    rankings = []
    for pos, job_id, title, company, score, final_decision, apply_url in rows:
        advice = advice_by_job.get(job_id) or advice_by_title.get(title)
        rankings.append({
            "rank": pos, "job_id": job_id, "title": title, "company": company,
            "score": float(score) if score is not None else None,
            "final_decision": final_decision, "apply_url": apply_url,
            "advice": advice,
        })
    return {"run_id": run_id, "rankings": rankings}