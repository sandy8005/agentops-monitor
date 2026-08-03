# AgentOps Monitor

An AI job-search agent with a built-in observability, evaluation, and human-review layer — plus a web dashboard to watch and control it.

The project has two halves:

- **The Job Search Agent** — reads a resume, pulls jobs from multiple sources, and recommends which to apply to, with tailored application strategy and resume-edit advice.
- **AgentOps Monitor** — records everything the agent does (every step, LLM call, tool call, cost, latency, error, and retrieved context), evaluates the quality of its decisions, and surfaces disagreements for human approval.

The agent is the worker; the Monitor is the observer. The observability and evaluation layer is the real focus — it turns "an LLM that gives answers" into a system you can inspect, evaluate, and trust.

---

## Why this exists

Most agent projects build the agent and stop. The harder, more interesting problem is knowing *what the agent actually did and whether its output can be trusted.* AgentOps Monitor answers:

- How much did this run cost, and where was the time spent?
- What evidence did the agent use to reach each decision?
- Where does the agent's reasoning disagree with a deterministic check — and which decisions need a human to sign off?

---

## How it works

Each **run** is one full execution. A run contains ordered **steps**; each step contains **LLM calls** and **tool calls**. Everything is stored in PostgreSQL as a linked trace and can be reconstructed after the fact.

### The agent pipeline

1. **receive_user_input** — reads and validates `user_input.json` (resume file, target role, location, work mode, employment type).
2. **read_resume_file** — extracts text from the resume PDF.
3. **parse_resume** — an LLM structures the resume into JSON (skills, projects, education, experience), validated with Pydantic.
4. **search_jobs** — pulls postings from the Job Source Service, filtered by target role.
5. For each job: **extract_requirements** (required vs. preferred skills, min experience) then **keyword_overlap** (deterministic skill check) then **judge** (LLM Apply/Maybe/Skip) then **match_score** (deterministic 100-point score) then **disagreement flag** (needs_human_review when score and LLM differ) then **application_strategy** and **resume_edit_advice** (for viable jobs only).
6. **rank_jobs** — sorts evaluated jobs by match score, best first.

### The Monitor

Every LLM and tool call is wrapped so timing, tokens, cost, and errors are recorded automatically — including **failed** calls. It tracks run/step lifecycle, retry with exponential backoff on transient failures, a quota pre-check, match scores, retrieved context (the evidence behind each decision), and the human-review workflow.

### Human approval workflow

When a step is flagged needs_human_review, it enters a pending queue. A reviewer can **approve** or **reject** it from the dashboard; the decision and a timestamp are recorded, giving an audit trail. The flag is an actionable decision point, not just a label.

---

## The core idea: two signals, and their disagreement

Each job gets two independent verdicts:

- A **deterministic match score** — consistent and explainable, but context-blind (it can't detect overqualification, and weights all required skills equally).
- An **LLM judgment** — context-aware, but inconsistent between runs.

Neither is trustworthy alone. Where they disagree is exactly where a human should look — and the Monitor flags those cases automatically. A separate **LLM-as-judge evaluator** grades the agent's reasoning for relevance, faithfulness, and hallucination; it correctly handles negation (understanding "the candidate lacks Kubernetes" is not a false claim), which a naive keyword check cannot.

---

## The dashboard

A web UI (FastAPI + HTML/JS) that:

- Lists all runs with status, tokens, and cost
- Shows a full trace of any run, with review flags and decisions
- Has a **pending-review queue** with approve/reject buttons
- Can **start a new agent run** in the background from a button

FastAPI also auto-generates interactive API docs at /docs.

---

## Job Source Service

The agent doesn't depend on a single source. Jobs flow into one job_postings table from multiple feeds, each tagged by origin, and the agent reads them all through one search_jobs() interface:

- **Seed** — built-in sample postings
- **CSV** — imported from a spreadsheet
- **API** — live remote jobs from the Remotive API

New feeds can be added without changing the agent.

---

## Tech stack

Python, PostgreSQL, FastAPI, Google Gemini API, Pydantic, pypdf, requests.

---

## Setup

1. Install PostgreSQL and create a database named agentops.
2. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a .env file:
   ```
   DB_NAME=agentops
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   GEMINI_API_KEY=your_key
   ```
4. Build the schema, run migrations, and load jobs:
   ```
   python db_pg.py
   python migrate_jobs_table.py
   python migrate_add_score.py
   python migrate_add_context.py
   python migrate_llm_status.py
   python migrate_approval.py
   python seed_jobs.py
   python import_csv.py        # optional CSV feed
   python import_api.py        # optional live API feed
   ```
5. Put your resume PDF in the project folder and point user_input.json at it.

## Running

```
python agent.py                          # run the agent from the command line
python view_run.py <run_id>              # inspect a run in the terminal
uvicorn api:app --reload                 # then open http://localhost:8000 for the dashboard
```

---

## Engineering notes

- **Failed LLM calls are logged**, not just the step error — so no call escapes the trace.
- **Pydantic validation** on all structured LLM output, so malformed responses fail loudly at the boundary.
- **Schema migrations** are idempotent and safe to re-run.
- **Orphaned-run cleanup** reconciles runs left "running" after a hard crash.
- Secrets live in .env and are kept out of version control.

## Limitations

- The deterministic match score can't detect overqualification or weight "dealbreaker" skills — the disagreement flag and human-review queue surface these instead.
- A purely mechanical hallucination check gives false positives on negation, which is why hallucination detection uses an LLM judge that understands meaning.
- On the Gemini free tier, a full multi-job run can exceed the daily request quota; role filtering, smaller job sets, or a paid key keep runs within limits.
- Web scraping (a possible fourth job source) is intentionally not implemented: scraping major job boards violates their terms of service and is actively blocked.