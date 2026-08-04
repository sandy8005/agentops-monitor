from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from datetime import datetime
import psycopg2, os
from dotenv import load_dotenv
from agent import run_agent
from llm import create_run

load_dotenv()
app = FastAPI(title="AgentOps Monitor")


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
    .detail { margin-top: 24px; }
    .step { background: #171a22; border: 1px solid #2a2e3a; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; }
    .review { border-color: #fbbf24; }
    .flag { color: #fbbf24; font-size: 12px; font-weight: 600; }
    .agree { color: #4ade80; }
    .reviewed { color: #60a5fa; font-size: 12px; font-weight: 600; }
    .call { color: #8b8f9c; font-size: 12px; margin-left: 16px; margin-top: 4px; }
    .call-failed { color: #f87171; }
    h2 { font-size: 16px; margin: 8px 0 16px; }
    .section-title { font-size: 14px; color: #8b8f9c; text-transform: uppercase; margin: 24px 0 8px; }
  </style>
</head>
<body>
  <header>
    <h1>AgentOps Monitor</h1>
    <div class="sub">AI job-search agent observability</div>
  </header>
  <div class="container">
    <button id="startBtn" onclick="startRun()">&#9654; Start New Run</button>

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
    async function loadPending() {
      const res = await fetch('/reviews/pending');
      const items = await res.json();
      const div = document.getElementById('pending');
      if (items.length === 0) {
        div.innerHTML = '<div class="step">No steps pending review.</div>';
        return;
      }
      div.innerHTML = '';
      for (const it of items) {
        const el = document.createElement('div');
        el.className = 'step review';
        el.innerHTML = `<strong>${it.step_name}</strong> (run #${it.run_id}) —
          Score: ${it.match_score} (${it.score_decision}) | LLM: ${it.llm_decision}
          <div style="margin-top:8px;">
            <button class="btn-sm btn-approve" onclick="review(${it.step_id}, 'approved')">Approve</button>
            <button class="btn-sm btn-reject" onclick="review(${it.step_id}, 'rejected')">Reject</button>
          </div>`;
        div.appendChild(el);
      }
    }

    async function review(stepId, decision) {
      const res = await fetch('/steps/' + stepId + '/review?decision=' + decision, { method: 'POST' });
      if (res.ok) {
        loadPending();
      } else {
        const err = await res.json();
        alert('Error: ' + (err.detail || 'failed'));
      }
    }

    async function loadRuns() {
      const res = await fetch('/runs');
      const runs = await res.json();
      const tbody = document.querySelector('#runs tbody');
      tbody.innerHTML = '';
      for (const r of runs) {
        let cls = 'errors';
        if (r.status === 'success') cls = 'success';
        else if (r.status === 'failed') cls = 'failed';
        else if (r.status === 'running') cls = 'running';
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>#${r.id}</td>
          <td><span class="status ${cls}">${r.status}</span></td>
          <td>${(r.started_at || '').replace('T', ' ').slice(0, 16)}</td>
          <td>${r.total_tokens || 0}</td>
          <td>$${(r.total_cost || 0).toFixed(6)}</td>`;
        tr.onclick = () => loadDetail(r.id);
        tbody.appendChild(tr);
      }
    }

    async function loadDetail(id) {
      const res = await fetch('/runs/' + id);
      const run = await res.json();
      const d = document.getElementById('detail');
      let html = `<h2>Run #${run.id} — ${run.status}</h2>`;
      for (const s of run.steps) {
        const review = s.needs_human_review;
        html += `<div class="step ${review ? 'review' : ''}">
          <strong>${s.step_name}</strong> [${s.status}]`;
        if (s.match_score !== null) {
          html += ` — Score: ${s.match_score} (${s.score_decision}) | LLM: ${s.llm_decision}`;
          if (review && s.review_status) {
            html += ` <span class="reviewed">&#9679; ${s.review_status.toUpperCase()}</span>`;
          } else if (review) {
            html += ` <span class="flag">&#9888; NEEDS REVIEW</span>`;
          } else {
            html += ` <span class="agree">&#10003;</span>`;
          }
        }
        // tool calls
        for (const t of (s.tool_calls || [])) {
          const fc = t.status === 'failed' ? 'call-failed' : '';
          html += `<div class="call ${fc}">tool: ${t.tool_name} [${t.status}] ${t.latency_ms}ms</div>`;
        }
        // llm calls
        for (const l of (s.llm_calls || [])) {
          const fc = l.status === 'failed' ? 'call-failed' : '';
          html += `<div class="call ${fc}">llm: ${l.prompt_tokens}+${l.completion_tokens} tok, ${l.latency_ms}ms, $${l.cost_usd} [${l.status}]</div>`;
        }
        html += `</div>`;
      }
      d.innerHTML = html;
      d.scrollIntoView({ behavior: 'smooth' });
    }

    async function startRun() {
      const btn = document.getElementById('startBtn');
      btn.disabled = true;
      btn.textContent = 'Starting...';
      const res = await fetch('/runs', { method: 'POST' });
      const data = await res.json();
      alert(data.message);
      btn.disabled = false;
      btn.innerHTML = '&#9654; Start New Run';
      setTimeout(() => { loadRuns(); loadDetail(data.run_id); }, 2000);
    }

    loadPending();
    loadRuns();
  </script>
</body>
</html>
    """


@app.get("/runs")
def list_runs(limit: int = 20):
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
def start_run(background_tasks: BackgroundTasks):
    run_id = create_run("job search run (dashboard)")
    background_tasks.add_task(run_agent, run_id)
    return {"run_id": run_id, "message": f"Run {run_id} started in background."}


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, status, started_at, ended_at, input_summary, total_tokens, total_cost
        FROM runs WHERE id = %s
    """, (run_id,))
    run = cur.fetchone()
    if not run:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    cur.execute("""
        SELECT id, step_name, status, match_score, score_decision,
               llm_decision, needs_human_review, review_status
        FROM steps WHERE run_id = %s ORDER BY step_order
    """, (run_id,))
    step_rows = cur.fetchall()

    steps = []
    for s in step_rows:
        step_id = s[0]
        # tool calls for this step
        cur.execute("""
            SELECT tool_name, status, latency_ms
            FROM tool_calls WHERE step_id = %s ORDER BY id
        """, (step_id,))
        tool_calls = [
            {"tool_name": t[0], "status": t[1], "latency_ms": t[2]}
            for t in cur.fetchall()
        ]
        # llm calls for this step
        cur.execute("""
            SELECT prompt_tokens, completion_tokens, latency_ms, cost_usd, status
            FROM llm_calls WHERE step_id = %s ORDER BY id
        """, (step_id,))
        llm_calls = [
            {"prompt_tokens": l[0], "completion_tokens": l[1], "latency_ms": l[2],
             "cost_usd": float(l[3]) if l[3] is not None else 0, "status": l[4]}
            for l in cur.fetchall()
        ]

        steps.append({
            "id": step_id, "step_name": s[1], "status": s[2],
            "match_score": float(s[3]) if s[3] is not None else None,
            "score_decision": s[4], "llm_decision": s[5],
            "needs_human_review": s[6], "review_status": s[7],
            "tool_calls": tool_calls, "llm_calls": llm_calls
        })

    conn.close()

    return {
        "id": run[0], "status": run[1],
        "started_at": run[2].isoformat() if run[2] else None,
        "ended_at": run[3].isoformat() if run[3] else None,
        "input_summary": run[4],
        "total_tokens": run[5], "total_cost": float(run[6]) if run[6] else 0,
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
    cur.execute("SELECT needs_human_review FROM steps WHERE id = %s", (step_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Step {step_id} not found")
    if not row[0]:
        conn.close()
        raise HTTPException(status_code=400, detail="This step was not flagged for review")

    cur.execute("""
        UPDATE steps SET review_status = %s, reviewed_at = %s WHERE id = %s
    """, (decision, datetime.now(), step_id))
    conn.commit()
    conn.close()
    return {"step_id": step_id, "review_status": decision}