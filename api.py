from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
import psycopg2, os
from dotenv import load_dotenv
from agent import run_agent

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
    h2 { font-size: 16px; margin: 8px 0 16px; }
  </style>
</head>
<body>
  <header>
    <h1>AgentOps Monitor</h1>
    <div class="sub">AI job-search agent observability</div>
  </header>
  <div class="container">
    <button id="startBtn" onclick="startRun()">&#9654; Start New Run</button>
    <table id="runs">
      <thead><tr><th>Run</th><th>Status</th><th>Started</th><th>Tokens</th><th>Cost</th></tr></thead>
      <tbody></tbody>
    </table>
    <div class="detail" id="detail"></div>
  </div>

  <script>
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
          html += ` — Score: ${s.match_score} (${s.score_decision}) | LLM: ${s.llm_decision}
            ${review ? '<span class="flag">&#9888; NEEDS REVIEW</span>' : '<span class="agree">&#10003;</span>'}`;
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
      setTimeout(loadRuns, 2000);
    }

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
    background_tasks.add_task(run_agent)
    return {"message": "Run started in background. Refresh the run list to watch it appear."}


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
               llm_decision, needs_human_review
        FROM steps WHERE run_id = %s ORDER BY step_order
    """, (run_id,))
    steps = [
        {"id": s[0], "step_name": s[1], "status": s[2],
         "match_score": float(s[3]) if s[3] is not None else None,
         "score_decision": s[4], "llm_decision": s[5],
         "needs_human_review": s[6]}
        for s in cur.fetchall()
    ]
    conn.close()

    return {
        "id": run[0], "status": run[1],
        "started_at": run[2].isoformat() if run[2] else None,
        "ended_at": run[3].isoformat() if run[3] else None,
        "input_summary": run[4],
        "total_tokens": run[5], "total_cost": float(run[6]) if run[6] else 0,
        "steps": steps
    }