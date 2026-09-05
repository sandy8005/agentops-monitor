from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
import psycopg2, os, tempfile
from dotenv import load_dotenv
from autonomous_agent import run_agent_autonomous
from llm import create_run, request_cancel
from pdf_reader import read_resume_file

load_dotenv()
app = FastAPI(title="AgentOps Monitor")
app.mount("/static", StaticFiles(directory="static"), name="static")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the dashboard shell. HTML/CSS/JS now live in static/ files."""
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()



@app.post("/upload")
async def upload_resume(file: UploadFile = File(...), name: str = Form("")):
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
        INSERT INTO resumes (name, resume_text, created_at)
        VALUES (%s, %s, %s) RETURNING id
    """, (name or file.filename, resume_text, datetime.now()))
    resume_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    return {"resume_id": resume_id, "name": name or file.filename,
            "chars": len(resume_text), "message": f"Resume stored as #{resume_id}"}


@app.get("/resumes")
def list_resumes():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, length(resume_text), created_at
        FROM resumes WHERE is_deleted = FALSE ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "name": r[1], "chars": r[2],
         "created_at": r[3].isoformat() if r[3] else None}
        for r in rows
    ]


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM resumes WHERE id = %s", (resume_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    # Soft delete: hide from the library but keep the row, so historical runs
    # that reference this resume keep an intact link. A hard DELETE would orphan
    # those runs (runs.resume_id would point at a missing row).
    cur.execute("UPDATE resumes SET is_deleted = TRUE WHERE id = %s", (resume_id,))
    conn.commit()
    conn.close()
    return {"resume_id": resume_id, "deleted": True}


@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, status, started_at, total_tokens, total_cost,
               target_role, location, work_mode
        FROM runs ORDER BY id DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "status": r[1],
         "started_at": r[2].isoformat() if r[2] else None,
         "total_tokens": r[3], "total_cost": float(r[4]) if r[4] else 0,
         "target_role": r[5], "location": r[6], "work_mode": r[7]}
        for r in rows
    ]


@app.post("/runs")
def start_run(background_tasks: BackgroundTasks, resume_id: int = None,
              target_role: str = "", location: str = "", work_mode: str = "",
              employment_type: str = "", evaluate: bool = False):
    if resume_id is None:
        raise HTTPException(status_code=400, detail="resume_id is required; upload or pick a resume first")
    if not target_role.strip():
        raise HTTPException(status_code=400, detail="target_role is required for a search")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT is_deleted FROM resumes WHERE id = %s", (resume_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    if row[0]:
        raise HTTPException(status_code=400, detail="That resume was deleted; upload it again to run new searches")

    run_id = create_run("job search run (dashboard)", resume_id=resume_id,
                        target_role=target_role, location=location,
                        work_mode=work_mode, employment_type=employment_type)
    background_tasks.add_task(
        run_agent_autonomous,
        resume_id=resume_id, target_role=target_role, location=location,
        work_mode=work_mode, employment_type=employment_type,
        evaluate=evaluate, run_id=run_id
    )
    return {"run_id": run_id, "resume_id": resume_id, "target_role": target_role,
            "message": f"Run {run_id} started in background."}


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT status FROM runs WHERE id = %s", (run_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if row[0] != "running":
        conn.close()
        raise HTTPException(status_code=400, detail=f"Run {run_id} is not running (status: {row[0]})")
    conn.close()
    request_cancel(run_id)
    return {"run_id": run_id, "cancel_requested": True}


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, status, started_at, ended_at, input_summary, total_tokens, total_cost,
               resume_id, target_role, location, work_mode, employment_type
        FROM runs WHERE id = %s
    """, (run_id,))
    run = cur.fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    cur.execute("""
        SELECT id, step_name, status, match_score, score_decision,
               llm_decision, needs_human_review, review_status, error_message,
               retrieved_context, reviewer, review_comment, review_reason, score_breakdown
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
             "input_json": t[3], "output_json": t[4], "error_message": t[5], "operation": t[6]}
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
             "prompt": l[5], "response": l[6], "error_message": l[7], "operation": l[8],
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
            "error_message": s[8], "retrieved_context": s[9],
            "reviewer": s[10], "review_comment": s[11], "review_reason": s[12],
            "score_breakdown": s[13],
            "tool_calls": tool_calls, "llm_calls": llm_calls,
            "evaluation": evaluation
        })

    conn.close()

    return {
        "id": run[0], "status": run[1],
        "started_at": run[2].isoformat() if run[2] else None,
        "ended_at": run[3].isoformat() if run[3] else None,
        "input_summary": run[4],
        "total_tokens": run[5], "total_cost": float(run[6]) if run[6] else 0,
        "resume_id": run[7], "target_role": run[8], "location": run[9],
        "work_mode": run[10], "employment_type": run[11],
        "steps": steps
    }


@app.get("/reviews/pending")
def pending_reviews():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.run_id, s.step_name, s.match_score,
               s.score_decision, s.llm_decision, s.review_reason
        FROM steps s
        WHERE s.needs_human_review = TRUE AND s.review_status IS NULL
        ORDER BY s.id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"step_id": r[0], "run_id": r[1], "step_name": r[2],
         "match_score": float(r[3]) if r[3] is not None else None,
         "score_decision": r[4], "llm_decision": r[5], "review_reason": r[6]}
        for r in rows
    ]


@app.post("/steps/{step_id}/review")
def submit_review(step_id: int, decision: str, reviewer: str = "anonymous", comment: str = ""):
    if decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT needs_human_review, review_status FROM steps WHERE id = %s", (step_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Step {step_id} not found")
    if not row[0]:
        conn.close()
        raise HTTPException(status_code=400, detail="This step was not flagged for review")
    if row[1] is not None:
        conn.close()
        raise HTTPException(status_code=409, detail=f"This step was already reviewed ({row[1]})")

    cur.execute("""
        UPDATE steps
        SET review_status = %s, reviewed_at = %s, reviewer = %s, review_comment = %s
        WHERE id = %s
    """, (decision, datetime.now(), reviewer or "anonymous", comment, step_id))
    conn.commit()
    conn.close()
    return {"step_id": step_id, "review_status": decision, "reviewer": reviewer or "anonymous"}