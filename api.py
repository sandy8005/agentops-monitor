from database import get_connection as _db_get_connection
from settings import settings
from fastapi import (FastAPI, HTTPException, UploadFile, File,
                     Form, Query, Depends, Request)
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from datetime import datetime
import psycopg2, os, tempfile
from llm import create_run, create_run_tx, request_cancel
from job_queue import enqueue, enqueue_tx
from pdf_reader import read_resume_file
from auth import authenticate
from csrf import get_or_create_token, require_csrf

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(title="AgentOps Monitor")

# --- Rate limiting ----------------------------------------------------------
# Per-client-IP limits (in-memory) to blunt login brute-force and enqueue abuse.
# Configurable via settings (RATE_LIMIT_LOGIN / RATE_LIMIT_RUNS).
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429,
                        content={"detail": "Too many requests — slow down."})

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

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

# Redact resume-bearing / trace fields from API responses when deploying
# publicly. Toggle with REDACT_SENSITIVE=1 (on for public deploys, off locally).
REDACT_SENSITIVE = settings.redact_sensitive
_REDACTED = "[redacted]"


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
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Verify credentials via auth.authenticate and start a signed session.
    Rate-limited per IP to blunt brute-force."""
    user = authenticate(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    request.session["user"] = {"id": user["id"], "username": user["username"],
                               "role": user["role"]}
    return {"ok": True, "username": user["username"], "role": user["role"]}


@app.post("/logout")
def logout(request: Request):
    """Clear the session cookie."""
    request.session.clear()
    return {"ok": True}


@app.get("/me")
def whoami(user: dict = Depends(require_auth)):
    """Who am I — used by the frontend to decide whether to show the login form."""
    return {"username": user["username"], "role": user["role"]}


@app.post("/upload")
async def upload_resume(file: UploadFile = File(...), name: str = Form(""),
                        user: dict = Depends(require_auth),
                        _csrf: None = Depends(require_csrf)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB)")
    if not contents[:5].startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        resume_text = read_resume_file(tmp_path)
    finally:
        os.remove(tmp_path)

    if not resume_text or not resume_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the PDF")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO resumes (name, resume_text, created_at, user_id)
        VALUES (%s, %s, %s, %s) RETURNING id
    """, (name or file.filename, resume_text, datetime.now(), user["id"]))
    resume_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    return {"resume_id": resume_id, "name": name or file.filename,
            "chars": len(resume_text), "message": f"Resume stored as #{resume_id}"}


@app.get("/resumes")
def list_resumes(user: dict = Depends(require_auth)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, length(resume_text), created_at
            FROM resumes WHERE is_deleted = FALSE AND user_id = %s ORDER BY id DESC
        """, (user["id"],))
        rows = cur.fetchall()
        return [
            {"id": r[0], "name": r[1], "chars": r[2],
             "created_at": r[3].isoformat() if r[3] else None}
            for r in rows
        ]


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int, user: dict = Depends(require_auth),
                  _csrf: None = Depends(require_csrf)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM resumes WHERE id = %s AND user_id = %s", (resume_id, user["id"]))
        if not cur.fetchone():
            # 404 (not 403) so we don't reveal that the id exists for another owner.
            raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
        # Soft delete: hide from the library but keep the row, so historical runs
        # that reference this resume keep an intact link. A hard DELETE would orphan
        # those runs (runs.resume_id would point at a missing row).
        cur.execute("UPDATE resumes SET is_deleted = TRUE WHERE id = %s AND user_id = %s",
                    (resume_id, user["id"]))
    return {"resume_id": resume_id, "deleted": True}


@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100), user: dict = Depends(require_auth)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, status, started_at, total_tokens, total_cost,
                   target_role, location, work_mode
            FROM runs WHERE user_id = %s ORDER BY id DESC LIMIT %s
        """, (user["id"], limit))
        rows = cur.fetchall()
        return [
            {"id": r[0], "status": r[1],
             "started_at": r[2].isoformat() if r[2] else None,
             "total_tokens": r[3], "total_cost": float(r[4]) if r[4] else 0,
             "target_role": r[5], "location": r[6], "work_mode": r[7]}
            for r in rows
        ]


@app.post("/runs")
@limiter.limit(settings.rate_limit_runs)
def start_run(request: Request, resume_id: int = None,
              target_role: str = "", location: str = "", work_mode: str = "",
              employment_type: str = "", evaluate: bool = False,
              live_only: bool = False, user: dict = Depends(require_auth),
              _csrf: None = Depends(require_csrf)):
    if resume_id is None:
        raise HTTPException(status_code=400, detail="resume_id is required; upload or pick a resume first")
    if not target_role.strip():
        raise HTTPException(status_code=400, detail="target_role is required for a search")

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
        raise HTTPException(status_code=500, detail=f"Failed to start run: {e}")
    finally:
        conn.close()
    return {"run_id": run_id, "resume_id": resume_id, "target_role": target_role,
            "live_only": live_only, "job_id": job_id,
            "message": f"Run {run_id} enqueued (job {job_id})."}


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int, user: dict = Depends(require_auth),
               _csrf: None = Depends(require_csrf)):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM runs WHERE id = %s AND user_id = %s",
                    (run_id, user["id"]))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        status = row[0]

        if status in ("running", "queued"):
            # Cooperative cancel. 'running': the loop checks is_cancel_requested between
            # jobs. 'queued': not claimed yet — set the flag now so that when a worker
            # picks the job up and begins the run, it sees the cancel and stops early.
            request_cancel(run_id)
            return {"run_id": run_id, "cancel_requested": True}

        if status == "waiting_for_human":
            # Paused at an interrupt — not executing, so no loop is checking the flag.
            # Set the flag AND resume the graph so it wakes, sees the cancel, and
            # terminates through its normal 'cancelled' routing (no orphaned checkpoint).
            request_cancel(run_id)
            enqueue("resume_run",
                    {"run_id": run_id, "decision": "Skip", "comment": "cancelled by user"},
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
                   stop_reason, error_code
            FROM runs WHERE id = %s AND user_id = %s
        """, (run_id, user["id"]))
        run = cur.fetchone()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        cur.execute("""
            SELECT id, step_name, status, match_score, score_decision,
                   llm_decision, needs_human_review, review_status, error_message,
                   retrieved_context, reviewer, review_comment, review_reason, score_breakdown,
                   final_decision
            FROM steps WHERE run_id = %s ORDER BY step_order
        """, (run_id,))
        step_rows = cur.fetchall()

        steps = []
        for s in step_rows:
            step_id = s[0]
            cur.execute("""
                SELECT tool_name, status, latency_ms, input_json, output_json, error_message, operation_name
                FROM tool_calls WHERE step_id = %s ORDER BY id
            """, (step_id,))
            tool_calls = [
                {"tool_name": t[0], "status": t[1], "latency_ms": t[2],
                 "input_json": _REDACTED if REDACT_SENSITIVE else t[3],
                 "output_json": _REDACTED if REDACT_SENSITIVE else t[4],
                 "error_message": t[5], "operation": t[6]}
                for t in cur.fetchall()
            ]
            cur.execute("""
                SELECT prompt_tokens, completion_tokens, latency_ms, cost_usd, status, prompt, response,
                       error_message, operation_name, attempt_number, retry_count, provider_request_id
                FROM llm_calls WHERE step_id = %s ORDER BY id
            """, (step_id,))
            llm_calls = [
                {"prompt_tokens": l[0], "completion_tokens": l[1], "latency_ms": l[2],
                 "cost_usd": float(l[3]) if l[3] is not None else 0, "status": l[4],
                 "prompt": _REDACTED if REDACT_SENSITIVE else l[5],
                 "response": _REDACTED if REDACT_SENSITIVE else l[6],
                 "error_message": l[7], "operation": l[8],
                 "attempt_number": l[9], "retry_count": l[10], "provider_request_id": l[11]}
                for l in cur.fetchall()
            ]
            cur.execute("""
                SELECT relevance_score, faithfulness_score, completeness_score,
                       hallucination_detected, hallucinated_claims, notes
                FROM evaluations WHERE step_id = %s ORDER BY id DESC LIMIT 1
            """, (step_id,))
            ev = cur.fetchone()
            evaluation = None
            if ev:
                evaluation = {
                    "relevance_score": ev[0], "faithfulness_score": ev[1],
                    "completeness_score": ev[2], "hallucination_detected": ev[3],
                    "hallucinated_claims": ev[4], "notes": ev[5]
                }

            steps.append({
                "id": step_id, "step_name": s[1], "status": s[2],
                "match_score": float(s[3]) if s[3] is not None else None,
                "score_decision": s[4], "llm_decision": s[5],
                "needs_human_review": s[6], "review_status": s[7],
                "error_message": s[8],
                "retrieved_context": _REDACTED if REDACT_SENSITIVE else s[9],
                "reviewer": s[10], "review_comment": s[11], "review_reason": s[12],
                "score_breakdown": s[13],
                "final_decision": s[14],
                "tool_calls": tool_calls, "llm_calls": llm_calls,
                "evaluation": evaluation
            })


        return {
            "id": run[0], "status": run[1],
            "started_at": run[2].isoformat() if run[2] else None,
            "ended_at": run[3].isoformat() if run[3] else None,
            "input_summary": run[4],
            "total_tokens": run[5], "total_cost": float(run[6]) if run[6] else 0,
            "resume_id": run[7], "target_role": run[8], "location": run[9],
            "work_mode": run[10], "employment_type": run[11],
            "pending_review": run[12],
            "stop_reason": run[13], "error_code": run[14],
            "steps": steps
        }


@app.post("/runs/{run_id}/resume")
def resume_run(run_id: int, decision: str = "Maybe", comment: str = "",
               user: dict = Depends(require_auth),
               _csrf: None = Depends(require_csrf)):
    """Resume a paused (waiting_for_human) run with the human's decision."""
    if decision not in ("Apply", "Maybe", "Skip"):
        raise HTTPException(status_code=400, detail="decision must be Apply, Maybe, or Skip")
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
                   {"run_id": run_id, "decision": decision, "comment": comment},
                   run_id=run_id)
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to resume run: {e}")
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