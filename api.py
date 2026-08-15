from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse
from datetime import datetime
import psycopg2, os, tempfile
from dotenv import load_dotenv
from agent import run_agent
from llm import create_run, request_cancel
from pdf_reader import read_resume_file

load_dotenv()
app = FastAPI(title="AgentOps Monitor")

MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB cap on resume uploads


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>AgentOps Monitor</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #0f1117; color: #e4e6eb; }
    header { background: #1a1d27; padding: 20px 32px; border-bottom: 1px solid #2a2e3a; }
    h1 { margin: 0; font-size: 20px; }
    .sub { color: #8b8f9c; font-size: 13px; margin-top: 4px; }
    .container { padding: 24px 32px; }
    button { background: #2563eb; color: white; border: none; padding: 10px 18px; border-radius: 6px; font-size: 14px; cursor: pointer; margin-bottom: 20px; }
    button:disabled { background: #374151; cursor: default; }
    .btn-sm { padding: 4px 12px; font-size: 12px; margin: 0 4px 0 0; }
    .btn-approve { background: #15803d; }
    .btn-reject { background: #b91c1c; }
    table { width: 100%; border-collapse: collapse; }
    th { text-align: left; padding: 10px 12px; color: #8b8f9c; font-size: 12px; text-transform: uppercase; border-bottom: 1px solid #2a2e3a; }
    td { padding: 10px 12px; border-bottom: 1px solid #1e2129; font-size: 14px; }
    tr:hover td { background: #171a22; cursor: pointer; }
    .status { padding: 3px 8px; border-radius: 4px; font-size: 12px; }
    .success { background: #14361f; color: #4ade80; }
    .errors { background: #3a2814; color: #fbbf24; }
    .failed { background: #3a1518; color: #f87171; }
    .running { background: #1e2a4a; color: #60a5fa; }
    .cancelled { background: #2a2e3a; color: #b0b4c0; }
    .detail { margin-top: 24px; }
    .step { background: #171a22; border: 1px solid #2a2e3a; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; }
    .review { border-color: #fbbf24; }
    .flag { color: #fbbf24; font-size: 12px; font-weight: 600; }
    .agree { color: #4ade80; }
    .reviewed { color: #60a5fa; font-size: 12px; font-weight: 600; }
    .call { color: #8b8f9c; font-size: 12px; margin-left: 16px; margin-top: 4px; }
    .call-failed { color: #f87171; }
    .io { background: #0d0f15; border: 1px solid #2a2e3a; border-radius: 4px; padding: 8px; font-size: 11px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; color: #b0b4c0; margin-top: 4px; }
    details summary { color: #60a5fa; font-size: 11px; cursor: pointer; margin-top: 4px; }
    h2 { font-size: 16px; margin: 8px 0 16px; }
    .section-title { font-size: 14px; color: #8b8f9c; text-transform: uppercase; margin: 24px 0 8px; }
    .upload-form { background: #171a22; border: 1px solid #2a2e3a; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; max-width: 640px; }
    .upload-form label { display: block; font-size: 12px; color: #8b8f9c; margin: 8px 0 4px; }
    .upload-form input, .upload-form select { width: 100%; box-sizing: border-box; background: #0d0f15; border: 1px solid #2a2e3a; border-radius: 4px; padding: 8px; color: #e4e6eb; font-size: 13px; }
    .upload-form .row { display: flex; gap: 12px; }
    .upload-form .row > div { flex: 1; }
    .err { color: #f87171; font-size: 12px; margin-top: 8px; }
    .progress { border-color: #60a5fa !important; }
  </style>
</head>
<body>
  <header>
    <h1>AgentOps Monitor</h1>
    <div class="sub">AI job-search agent observability</div>
  </header>
  <div class="container">

    <div class="section-title">New Run — Upload Resume</div>
    <div class="upload-form">
      <label>Resume PDF</label>
      <input type="file" id="resumeFile" accept="application/pdf">
      <label>Resume label (optional)</label>
      <input type="text" id="resumeName" placeholder="e.g. Backend-focused resume">
      <div class="row">
        <div><label>Target role</label><input type="text" id="targetRole" placeholder="AI/ML Engineer"></div>
        <div><label>Location</label><input type="text" id="location" placeholder="Michigan"></div>
      </div>
      <div class="row">
        <div><label>Work mode</label>
          <select id="workMode">
            <option value="">(any)</option>
            <option value="remote">remote</option>
            <option value="hybrid">hybrid</option>
            <option value="onsite">onsite</option>
          </select>
        </div>
        <div><label>Employment type</label>
          <select id="employmentType">
            <option value="">(any)</option>
            <option value="full-time">full-time</option>
            <option value="part-time">part-time</option>
            <option value="contract">contract</option>
            <option value="internship">internship</option>
          </select>
        </div>
      </div>
      <div style="margin-top:14px;">
        <button id="uploadBtn" onclick="uploadAndRun()">&#9654; Upload &amp; Start Run</button>
      </div>
      <div id="uploadMsg" class="err"></div>
    </div>

    <div class="section-title">Saved Resumes</div>
    <div id="resumeLibrary"></div>

    <div class="section-title">Pending Human Review</div>
    <div id="pending"></div>

    <div class="section-title">Runs</div>
    <table id="runs">
      <thead><tr><th>Run</th><th>Status</th><th>Started</th><th>Tokens</th><th>Cost</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="detail" id="detail"></div>
  </div>

  <script>
    var NL = String.fromCharCode(10);
    var pollTimer = null;

    function escapeHtml(s) {
      return String(s == null ? '' : s)
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
    }

    async function uploadAndRun() {
      const fileInput = document.getElementById('resumeFile');
      const role = document.getElementById('targetRole').value.trim();
      const msg = document.getElementById('uploadMsg');
      msg.textContent = '';

      if (!fileInput.files.length) { msg.textContent = 'Please choose a PDF file.'; return; }
      if (!role) { msg.textContent = 'Please enter a target role.'; return; }
      const file = fileInput.files[0];
      if (file.size > 5 * 1024 * 1024) { msg.textContent = 'File too large (max 5 MB).'; return; }

      const btn = document.getElementById('uploadBtn');
      btn.disabled = true; btn.textContent = 'Uploading...';
      msg.textContent = 'Extracting and storing resume...';

      const fd = new FormData();
      fd.append('file', file);
      fd.append('name', document.getElementById('resumeName').value.trim());
      fd.append('target_role', role);
      fd.append('location', document.getElementById('location').value.trim());
      fd.append('work_mode', document.getElementById('workMode').value);
      fd.append('employment_type', document.getElementById('employmentType').value);

      try {
        const up = await fetch('/upload', { method: 'POST', body: fd });
        if (!up.ok) { const e = await up.json(); throw new Error(e.detail || 'upload failed'); }
        const upData = await up.json();
        msg.textContent = 'Stored resume #' + upData.resume_id + ' (' + upData.chars + ' chars). Starting run...';

        const run = await fetch('/runs?resume_id=' + upData.resume_id, { method: 'POST' });
        if (!run.ok) { const e = await run.json(); throw new Error(e.detail || 'run failed to start'); }
        const runData = await run.json();
        msg.textContent = 'Resume #' + upData.resume_id + ' -> Run #' + runData.run_id + ' started.';
        loadResumes();
        setTimeout(function() { loadRuns(); loadDetail(runData.run_id); }, 1500);
      } catch (err) {
        msg.textContent = 'Error: ' + err.message;
      } finally {
        btn.disabled = false; btn.innerHTML = '&#9654; Upload &amp; Start Run';
      }
    }

    async function loadResumes() {
      const div = document.getElementById('resumeLibrary');
      try {
        const res = await fetch('/resumes');
        if (!res.ok) throw new Error('failed to load resumes');
        const resumes = await res.json();
        if (resumes.length === 0) { div.innerHTML = '<div class="step">No saved resumes yet.</div>'; return; }
        div.innerHTML = '';
        for (const r of resumes) {
          const el = document.createElement('div');
          el.className = 'step';
          el.innerHTML = '<strong>' + escapeHtml(r.name) + '</strong> - ' + escapeHtml(r.target_role || 'no role') +
            ' <span style="color:#8b8f9c;">(' + escapeHtml(r.chars) + ' chars, ' + escapeHtml(r.work_mode || 'any') + ')</span>' +
            '<div style="margin-top:8px;">' +
              '<button class="btn-sm btn-approve" onclick="runResume(' + r.id + ')">Run</button>' +
              '<button class="btn-sm btn-reject" onclick="deleteResume(' + r.id + ')">Delete</button>' +
            '</div>';
          div.appendChild(el);
        }
      } catch (err) {
        div.innerHTML = '<div class="step" style="color:#f87171;">Could not load resumes: ' + escapeHtml(err.message) + '</div>';
      }
    }

    async function runResume(id) {
      try {
        const run = await fetch('/runs?resume_id=' + id, { method: 'POST' });
        if (!run.ok) { const e = await run.json(); throw new Error(e.detail || 'run failed'); }
        const runData = await run.json();
        setTimeout(function() { loadRuns(); loadDetail(runData.run_id); }, 1500);
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function cancelRun(id) {
      if (!confirm('Cancel run #' + id + '?')) return;
      try {
        const res = await fetch('/runs/' + id + '/cancel', { method: 'POST' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'cancel failed'); }
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function deleteResume(id) {
      if (!confirm('Delete resume #' + id + '?')) return;
      try {
        const res = await fetch('/resumes/' + id, { method: 'DELETE' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'delete failed'); }
        loadResumes();
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function loadPending() {
      const div = document.getElementById('pending');
      try {
        const res = await fetch('/reviews/pending');
        if (!res.ok) throw new Error('failed to load reviews');
        const items = await res.json();
        if (items.length === 0) { div.innerHTML = '<div class="step">No steps pending review.</div>'; return; }
        div.innerHTML = '';
        for (const it of items) {
          const el = document.createElement('div');
          el.className = 'step review';
          el.innerHTML = '<strong>' + escapeHtml(it.step_name) + '</strong> (run #' + escapeHtml(it.run_id) + ') - ' +
            'Score: ' + escapeHtml(it.match_score) + ' (' + escapeHtml(it.score_decision) + ') | LLM: ' + escapeHtml(it.llm_decision) +
            '<div style="margin-top:8px;">' +
              '<button class="btn-sm btn-approve" onclick="review(' + it.step_id + ', \\'approved\\')">Approve</button>' +
              '<button class="btn-sm btn-reject" onclick="review(' + it.step_id + ', \\'rejected\\')">Reject</button>' +
            '</div>';
          div.appendChild(el);
        }
      } catch (err) {
        div.innerHTML = '<div class="step" style="color:#f87171;">Could not load reviews: ' + escapeHtml(err.message) + '</div>';
      }
    }

    async function review(stepId, decision) {
      try {
        const res = await fetch('/steps/' + stepId + '/review?decision=' + decision, { method: 'POST' });
        if (res.ok) { loadPending(); }
        else { const err = await res.json(); alert('Error: ' + (err.detail || 'failed')); }
      } catch (err) { alert('Network error: ' + err.message); }
    }

    async function loadRuns() {
      const tbody = document.querySelector('#runs tbody');
      try {
        const res = await fetch('/runs');
        if (!res.ok) throw new Error('failed to load runs');
        const runs = await res.json();
        tbody.innerHTML = '';
        for (const r of runs) {
          let cls = 'errors';
          if (r.status === 'success') cls = 'success';
          else if (r.status === 'failed') cls = 'failed';
          else if (r.status === 'running') cls = 'running';
          else if (r.status === 'cancelled') cls = 'cancelled';
          const tr = document.createElement('tr');
          tr.innerHTML = '<td>#' + escapeHtml(r.id) + '</td>' +
            '<td><span class="status ' + cls + '">' + escapeHtml(r.status) + '</span></td>' +
            '<td>' + escapeHtml((r.started_at || '').replace('T', ' ').slice(0, 16)) + '</td>' +
            '<td>' + escapeHtml(r.total_tokens || 0) + '</td>' +
            '<td>$' + escapeHtml((r.total_cost || 0).toFixed(6)) + '</td>';
          tr.onclick = function() { loadDetail(r.id); };
          tbody.appendChild(tr);
        }
      } catch (err) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:#f87171;">Could not load runs: ' + escapeHtml(err.message) + '</td></tr>';
      }
    }

    async function loadDetail(id) {
      const d = document.getElementById('detail');
      try {
        const res = await fetch('/runs/' + id);
        if (!res.ok) throw new Error('failed to load run ' + id);
        const run = await res.json();

        const total = run.steps.length;
        const done = run.steps.filter(function(s) {
          return s.status === 'success' || s.status === 'failed';
        }).length;
        const running = run.status === 'running';

        let html = '<h2>Run #' + escapeHtml(run.id) + ' - ' + escapeHtml(run.status) + '</h2>';

        if (running) {
          const active = run.steps.filter(function(s){ return s.status === 'running'; });
          const current = active.length ? active[active.length - 1].step_name : 'starting...';
          html += '<div class="step progress">' +
            '<strong style="color:#60a5fa;">&#9679; RUNNING</strong> &nbsp; ' +
            done + '/' + total + ' steps done &nbsp; | &nbsp; current: ' + escapeHtml(current) +
            ' &nbsp; <button class="btn-sm btn-reject" onclick="cancelRun(' + run.id + ')">Cancel</button>' +
            '</div>';
        }

        for (const s of run.steps) {
          const review = s.needs_human_review;
          html += '<div class="step ' + (review ? 'review' : '') + '">' +
            '<strong>' + escapeHtml(s.step_name) + '</strong> [' + escapeHtml(s.status) + ']';
          if (s.match_score !== null) {
            html += ' - Score: ' + escapeHtml(s.match_score) + ' (' + escapeHtml(s.score_decision) + ') | LLM: ' + escapeHtml(s.llm_decision);
            if (review && s.review_status) {
              html += ' <span class="reviewed">&#9679; ' + escapeHtml(s.review_status.toUpperCase()) + '</span>';
            } else if (review) {
              html += ' <span class="flag">&#9888; NEEDS REVIEW</span>';
            } else {
              html += ' <span class="agree">&#10003;</span>';
            }
          }
          if (s.error_message) {
            html += '<div class="call call-failed">error: ' + escapeHtml(s.error_message) + '</div>';
          }
          for (const t of (s.tool_calls || [])) {
            const fc = t.status === 'failed' ? 'call-failed' : '';
            let io = 'IN: ' + escapeHtml(t.input_json || '') + NL + NL + 'OUT: ' + escapeHtml(t.output_json || '');
            if (t.error_message) io += NL + NL + 'ERROR: ' + escapeHtml(t.error_message);
            html += '<div class="call ' + fc + '">tool: ' + escapeHtml(t.tool_name) + ' [' + escapeHtml(t.status) + '] ' + escapeHtml(t.latency_ms) + 'ms' +
              '<details><summary>view i/o</summary><pre class="io">' + io + '</pre></details></div>';
          }
          for (const l of (s.llm_calls || [])) {
            const fc = l.status === 'failed' ? 'call-failed' : '';
            let io = 'PROMPT:' + NL + escapeHtml(l.prompt || '') + NL + NL + 'RESPONSE:' + NL + escapeHtml(l.response || '');
            if (l.error_message) io += NL + NL + 'ERROR: ' + escapeHtml(l.error_message);
            html += '<div class="call ' + fc + '">llm: ' + escapeHtml(l.prompt_tokens) + '+' + escapeHtml(l.completion_tokens) + ' tok, ' + escapeHtml(l.latency_ms) + 'ms, $' + escapeHtml(l.cost_usd) + ' [' + escapeHtml(l.status) + ']' +
              '<details><summary>view prompt/response</summary><pre class="io">' + io + '</pre></details></div>';
          }
          if (s.evaluation) {
            const ev = s.evaluation;
            html += '<div class="call">eval: rel=' + escapeHtml(ev.relevance_score) + ' faith=' + escapeHtml(ev.faithfulness_score) + ' complete=' + escapeHtml(ev.completeness_score) + ' halluc=' + escapeHtml(ev.hallucination_detected) + '</div>';
          }
          html += '</div>';
        }
        d.innerHTML = html;

        if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
        if (running) {
          pollTimer = setTimeout(function() { loadDetail(id); }, 3000);
        } else {
          loadRuns();
        }
      } catch (err) {
        d.innerHTML = '<div style="color:#f87171;">Could not load run detail: ' + escapeHtml(err.message) + '</div>';
      }
    }

    loadResumes();
    loadPending();
    loadRuns();
  </script>
</body>
</html>
    """


@app.post("/upload")
async def upload_resume(
    file: UploadFile = File(...),
    name: str = Form(""),
    target_role: str = Form(...),
    location: str = Form(""),
    work_mode: str = Form(""),
    employment_type: str = Form("")
):
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
        INSERT INTO resumes (name, resume_text, target_role, location, work_mode, employment_type, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        name or file.filename,
        resume_text, target_role, location, work_mode, employment_type,
        datetime.now()
    ))
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
        SELECT id, name, target_role, location, work_mode, length(resume_text), created_at
        FROM resumes ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "name": r[1], "target_role": r[2], "location": r[3],
         "work_mode": r[4], "chars": r[5],
         "created_at": r[6].isoformat() if r[6] else None}
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
    cur.execute("DELETE FROM resumes WHERE id = %s", (resume_id,))
    conn.commit()
    conn.close()
    return {"resume_id": resume_id, "deleted": True}


@app.get("/runs")
def list_runs(limit: int = Query(20, ge=1, le=100)):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, status, started_at, total_tokens, total_cost
        FROM runs ORDER BY id DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "status": r[1],
         "started_at": r[2].isoformat() if r[2] else None,
         "total_tokens": r[3], "total_cost": float(r[4]) if r[4] else 0}
        for r in rows
    ]


@app.post("/runs")
def start_run(background_tasks: BackgroundTasks, resume_id: int = None):
    if resume_id is None:
        raise HTTPException(status_code=400, detail="resume_id is required; upload or pick a resume first")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM resumes WHERE id = %s", (resume_id,))
    exists = cur.fetchone()
    conn.close()
    if not exists:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")

    run_id = create_run("job search run (dashboard)", resume_id=resume_id)
    background_tasks.add_task(run_agent, run_id, False, resume_id)
    return {"run_id": run_id, "resume_id": resume_id,
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
        SELECT id, status, started_at, ended_at, input_summary, total_tokens, total_cost, resume_id
        FROM runs WHERE id = %s
    """, (run_id,))
    run = cur.fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    cur.execute("""
        SELECT id, step_name, status, match_score, score_decision,
               llm_decision, needs_human_review, review_status, error_message, retrieved_context
        FROM steps WHERE run_id = %s ORDER BY step_order
    """, (run_id,))
    step_rows = cur.fetchall()

    steps = []
    for s in step_rows:
        step_id = s[0]
        cur.execute("""
            SELECT tool_name, status, latency_ms, input_json, output_json, error_message
            FROM tool_calls WHERE step_id = %s ORDER BY id
        """, (step_id,))
        tool_calls = [
            {"tool_name": t[0], "status": t[1], "latency_ms": t[2],
             "input_json": t[3], "output_json": t[4], "error_message": t[5]}
            for t in cur.fetchall()
        ]
        cur.execute("""
            SELECT prompt_tokens, completion_tokens, latency_ms, cost_usd, status, prompt, response, error_message
            FROM llm_calls WHERE step_id = %s ORDER BY id
        """, (step_id,))
        llm_calls = [
            {"prompt_tokens": l[0], "completion_tokens": l[1], "latency_ms": l[2],
             "cost_usd": float(l[3]) if l[3] is not None else 0, "status": l[4],
             "prompt": l[5], "response": l[6], "error_message": l[7]}
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
        "resume_id": run[7],
        "steps": steps
    }


@app.get("/reviews/pending")
def pending_reviews():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.run_id, s.step_name, s.match_score,
               s.score_decision, s.llm_decision
        FROM steps s
        WHERE s.needs_human_review = TRUE AND s.review_status IS NULL
        ORDER BY s.id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"step_id": r[0], "run_id": r[1], "step_name": r[2],
         "match_score": float(r[3]) if r[3] is not None else None,
         "score_decision": r[4], "llm_decision": r[5]}
        for r in rows
    ]


@app.post("/steps/{step_id}/review")
def submit_review(step_id: int, decision: str):
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
        UPDATE steps SET review_status = %s, reviewed_at = %s WHERE id = %s
    """, (decision, datetime.now(), step_id))
    conn.commit()
    conn.close()
    return {"step_id": step_id, "review_status": decision}