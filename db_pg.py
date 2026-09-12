"""
Complete schema builder for AgentOps Monitor.

Running `python db_pg.py` on a FRESH database creates every table and column the
current code expects — ZERO migrations needed. This is the authoritative schema,
rebuilt to match the working database exactly (all live-job, review, judge-signal,
freshness, and cache columns included).

Note: the checkpoint_* tables (LangGraph's Postgres checkpointer) are created
automatically by PostgresSaver.setup() at first graph run — not here.
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# runs: one row per agent run. Search config + evaluation/stop tracking + paused review.
cur.execute("""
CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT,
    input_summary TEXT,
    total_tokens INTEGER DEFAULT 0,
    total_cost NUMERIC(12,6) DEFAULT 0,
    resume_id INTEGER,
    cancel_requested BOOLEAN DEFAULT FALSE,
    target_role TEXT,
    location TEXT,
    work_mode TEXT,
    employment_type TEXT,
    evaluation_status TEXT DEFAULT 'not_requested',
    stop_reason TEXT,
    pending_review JSONB
)
""")

# steps: one row per conceptual stage. Includes review + judge-signal + final_decision.
cur.execute("""
CREATE TABLE IF NOT EXISTS steps (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    step_name TEXT,
    step_order INTEGER,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    status TEXT,
    error_message TEXT,
    match_score NUMERIC(5,1),
    score_decision TEXT,
    llm_decision TEXT,
    needs_human_review BOOLEAN DEFAULT FALSE,
    retrieved_context JSONB,
    review_status TEXT,
    reviewed_at TIMESTAMP,
    reviewer TEXT,
    review_comment TEXT,
    review_reason TEXT,
    score_breakdown JSONB,
    judge_status TEXT,
    judge_skip_reason TEXT,
    cache_hit BOOLEAN,
    final_decision TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")

# llm_calls: one row per HTTP attempt.
cur.execute("""
CREATE TABLE IF NOT EXISTS llm_calls (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    step_id INTEGER,
    model TEXT,
    prompt TEXT,
    response TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER,
    cost_usd NUMERIC(12,6),
    created_at TIMESTAMP,
    status TEXT DEFAULT 'success',
    error_message TEXT,
    operation_name TEXT,
    attempt_number INTEGER DEFAULT 1,
    retry_count INTEGER DEFAULT 0,
    provider_request_id TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

# tool_calls: one row per tool invocation.
cur.execute("""
CREATE TABLE IF NOT EXISTS tool_calls (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    step_id INTEGER,
    tool_name TEXT,
    input_json TEXT,
    output_json TEXT,
    latency_ms INTEGER,
    status TEXT,
    error_message TEXT,
    created_at TIMESTAMP,
    operation_name TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

# job_postings: the searchable pool. Includes live-job + freshness + apply_url columns.
cur.execute("""
CREATE TABLE IF NOT EXISTS job_postings (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT,
    description TEXT NOT NULL,
    location TEXT,
    work_mode TEXT,
    source TEXT DEFAULT 'seed',
    created_at TIMESTAMP DEFAULT NOW(),
    employment_type TEXT,
    external_id TEXT,
    search_location TEXT,
    fetched_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    posted_at TIMESTAMP,
    apply_url TEXT
)
""")
# unique index for dedup upserts (partial: only where external_id is set)
cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS job_postings_external_id_uniq
    ON job_postings (external_id) WHERE external_id IS NOT NULL
""")

# resumes: stored resume document. Soft-delete via is_deleted.
cur.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    name TEXT,
    resume_text TEXT NOT NULL,
    target_role TEXT,
    location TEXT,
    work_mode TEXT,
    employment_type TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
)
""")

# evaluations: LLM-as-judge grades for a step's decision.
cur.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    step_id INTEGER,
    relevance_score INTEGER,
    faithfulness_score INTEGER,
    completeness_score INTEGER,
    hallucination_detected BOOLEAN,
    hallucinated_claims JSONB,
    notes TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

# users: auth (bcrypt password hashes).
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW()
)
""")

# --- Autonomous-agent caches (call reduction), versioned. ---
cur.execute("""
CREATE TABLE IF NOT EXISTS parsed_resume_cache (
    resume_hash TEXT PRIMARY KEY,
    parsed_json TEXT NOT NULL,
    cache_version TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS job_reqs_cache (
    desc_hash TEXT PRIMARY KEY,
    reqs_json TEXT NOT NULL,
    cache_version TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
""")

conn.commit()
conn.close()
print("Postgres schema ready — complete current schema (10 app tables, all columns). "
      "Fresh install needs zero migrations. (LangGraph checkpoint_* tables auto-create "
      "on first graph run.)")