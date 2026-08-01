# AgentOps Monitor

An AI job-search agent with a built-in observability and evaluation layer.

The project has two halves that work together:

- **The Job Search Agent** — reads a resume, pulls jobs from a source, and recommends which to apply to.
- **AgentOps Monitor** — records everything the agent does (every step, LLM call, tool call, cost, latency, and error) and evaluates the quality of its decisions.

The agent is the worker; the Monitor is the observer that watches it. The observability layer is the real focus — it's what turns "an LLM that gives answers" into "a system you can inspect, debug, and trust."

---

## Why this exists

Most agent projects build the agent and stop there. The hard, interesting problem is knowing *what the agent actually did and whether its output can be trusted*. AgentOps Monitor answers questions like:

- How much did this run cost, and where was the time spent?
- What evidence did the agent use to reach each decision?
- When does the agent's reasoning disagree with a deterministic check — and which decisions need a human's eyes?

---

## How it works

Each **run** is one full execution. A run contains ordered **steps**. Each step can contain **LLM calls** and **tool calls**. Everything is stored in PostgreSQL as a linked trace, so any run can be reconstructed and inspected after the fact.

### The agent pipeline (per run)

1. **receive_user_input** — reads and validates `user_input.json` (resume file, target role, location, work mode, employment type).
2. **read_resume_file** — extracts text from the resume PDF.
3. **parse_resume** — an LLM structures the resume into JSON (skills, projects, education, experience), validated with Pydantic.
4. **search_jobs** — pulls postings from the job source (a PostgreSQL table), filtered loosely by target role.
5. For each job:
   - **extract_requirements** — an LLM pulls required vs. preferred skills and minimum experience from the posting (Pydantic-validated).
   - **keyword_overlap** — a deterministic check of which required skills appear in the resume.
   - **judge** — an LLM recommends Apply / Maybe / Skip with reasoning.
   - **match_score** — a deterministic 100-point score (required skills, preferred skills, project relevance, experience, location).
   - **disagreement flag** — if the deterministic score and the LLM decision disagree, the step is flagged `needs_human_review`.
   - **application_strategy** and **resume_edit_advice** — for viable jobs only, an LLM suggests how to apply and how to tailor the resume.
6. **rank_jobs** — sorts the evaluated jobs by match score, best first.

### The Monitor

Every LLM and tool call is wrapped so that timing, token counts, cost, and errors are recorded automatically — the agent code can't accidentally skip logging. The Monitor tracks:

- Run and step lifecycle (running → success / failed / completed_with_errors)
- Per-call latency, tokens, and estimated cost, rolled up to the run
- Errors, with automatic retry and exponential backoff on transient failures
- Match scores, LLM decisions, and the `needs_human_review` disagreement flag
- Retrieved context — the evidence (skills matched/missing, requirements) behind each decision

---

## The core idea: two signals, and their disagreement

The agent produces two independent verdicts for each job:

- A **deterministic match score** — consistent and explainable, but context-blind. It can't tell that a candidate is *overqualified*, and it treats every required skill as equal weight.
- An **LLM judgment** — context-aware, but inconsistent between runs.

Neither is trustworthy alone. Where they disagree is exactly where a human should look — and the Monitor flags those cases automatically. This disagreement flag is the heart of the evaluation layer.

---

## Tech stack

- **Python**
- **PostgreSQL** — trace storage (runs, steps, LLM calls, tool calls, job postings)
- **Google Gemini API** — the LLM behind parsing, extraction, judgment, and advice
- **Pydantic** — validation of structured LLM output
- **pypdf** — resume text extraction

---

## Project layout

| File | Purpose |
|------|---------|
| `agent.py` | Orchestrates the full pipeline |
| `llm.py` | LLM/tool call wrappers, run/step lifecycle, logging, retry |
| `db_pg.py` | PostgreSQL schema |
| `input_handler.py` | Reads and validates user input |
| `pdf_reader.py` | Extracts text from the resume PDF |
| `parser.py` | Structures the resume via LLM |
| `job_parser.py` | Extracts job requirements via LLM |
| `job_source.py` | Reads and filters jobs from the source |
| `tools.py` | Deterministic keyword-overlap check |
| `scorer.py` | Deterministic 100-point match score |
| `ranker.py` | Sorts jobs by score |
| `advisor.py` | Application strategy and resume-edit advice |
| `evaluator.py` | LLM-as-judge evaluation of the agent's decisions |
| `schemas.py` | Pydantic models for structured output |
| `view_run.py` | Inspects a stored run trace |
| `seed_jobs.py`, `import_csv.py` | Load jobs into the source (seed and CSV feeds) |
| `migrate_*.py` | Schema migrations |

---

## Setup

1. Install PostgreSQL and create a database named `agentops`.
2. Create a virtual environment and install dependencies:
   ```
   pip install psycopg2-binary python-dotenv google-genai pydantic pypdf
   ```
3. Create a `.env` file with your database credentials and Gemini API key:
   ```
   DB_NAME=agentops
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   GEMINI_API_KEY=your_key
   ```
4. Build the schema and load jobs:
   ```
   python db_pg.py
   python migrate_jobs_table.py
   python seed_jobs.py
   python import_csv.py        # optional: adds jobs from sample_jobs.csv
   ```
5. Put your resume PDF in the project folder and point `user_input.json` at it.

## Running

```
python agent.py                 # run the full pipeline
python view_run.py <run_id>     # inspect a stored run
```

---

## Status

**Working:** the full agent pipeline (input → parse → search → evaluate → score → rank → advise), the observability layer (traces, cost, latency, errors, retry), match scoring with the disagreement flag, retrieved-context capture, and Pydantic validation of LLM output. Two job-source feeds (seed + CSV).

**In progress:** LLM-as-judge evaluation scores (relevance, faithfulness, hallucination detection) and additional job-source feeds (API, scraping).

---

## Notes & limitations

- The deterministic match score can't detect overqualification or weight "dealbreaker" skills — by design, the disagreement flag surfaces these cases for review instead.
- A purely mechanical hallucination check gives false positives on negation (it can't tell "has Kubernetes" from "lacks Kubernetes"), which is why hallucination detection moves to an LLM judge that understands meaning.
- On the Gemini free tier, a full multi-job run can exceed the daily request quota; role-based filtering and smaller job sets keep runs within limits.
