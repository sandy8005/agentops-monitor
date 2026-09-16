"""
Complete current schema — one-step fresh install.

This is the SINGLE source of truth for the full, current database schema. It
creates every table with every column the runtime code expects (the state you'd
reach by running the initial schema plus ALL migrate_*.py migrations in order).

A fresh install runs THIS, then records the baseline migration as already-applied
via the migration runner (see migrate.py / migrations/0001_baseline.py), so the
runner knows the baseline is in place and only future migrations (0002+) run.

Keep this file and migrations/0001_baseline.py in lockstep: they define the same
schema. New schema changes after the baseline go in NEW numbered migrations
(0002_, 0003_, ...), NOT here.
"""
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# --- runs ---
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
    pending_review JSONB,
    stop_reason TEXT,
    evaluation_status TEXT DEFAULT 'not_run'
)
""")

# --- steps ---
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
    final_decision TEXT,
    needs_human_review BOOLEAN DEFAULT FALSE,
    retrieved_context JSONB,
    score_breakdown JSONB,
    review_status TEXT,
    review_reason TEXT,
    reviewed_at TIMESTAMP,
    reviewer TEXT,
    review_comment TEXT,
    judge_status TEXT,
    judge_skip_reason TEXT,
    cache_hit BOOLEAN,
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")

# --- llm_calls ---
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

# --- tool_calls ---
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

# --- job_postings ---
cur.execute("""
CREATE TABLE IF NOT EXISTS job_postings (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT,
    description TEXT NOT NULL,
    location TEXT,
    work_mode TEXT,
    employment_type TEXT,
    source TEXT DEFAULT 'seed',
    external_id TEXT,
    search_location TEXT,
    fetched_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    posted_at TIMESTAMP,
    apply_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS job_postings_external_id_uniq
    ON job_postings (external_id) WHERE external_id IS NOT NULL
""")

# --- job_searches + job_search_results (per-search association) ---
cur.execute("""
CREATE TABLE IF NOT EXISTS job_searches (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    target_role TEXT,
    location TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS job_search_results (
    search_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    PRIMARY KEY (search_id, job_id),
    FOREIGN KEY (search_id) REFERENCES job_searches(id) ON DELETE CASCADE,
    FOREIGN KEY (job_id) REFERENCES job_postings(id) ON DELETE CASCADE
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS job_search_results_search_idx ON job_search_results (search_id)")
cur.execute("CREATE INDEX IF NOT EXISTS job_search_results_job_idx ON job_search_results (job_id)")
cur.execute("CREATE INDEX IF NOT EXISTS job_searches_run_idx ON job_searches (run_id)")

# --- resumes ---
cur.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    name TEXT,
    resume_text TEXT NOT NULL,
    created_at TIMESTAMP,
    is_deleted BOOLEAN DEFAULT FALSE
)
""")

# --- evaluations ---
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

# --- caches ---
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
    extraction_method TEXT,
    source_model TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
""")

# --- persisted ranked results + advice ---
cur.execute("""
CREATE TABLE IF NOT EXISTS run_rankings (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL,
    job_id INTEGER,
    rank_position INTEGER NOT NULL,
    title TEXT,
    company TEXT,
    score DOUBLE PRECISION,
    final_decision TEXT,
    apply_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS run_rankings_run_idx ON run_rankings (run_id)")
cur.execute("""
CREATE TABLE IF NOT EXISTS run_advice (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL,
    job_id INTEGER,
    title TEXT,
    advice TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")
cur.execute("CREATE INDEX IF NOT EXISTS run_advice_run_idx ON run_advice (run_id)")

# --- auth users ---
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW()
)
""")

# --- migration tracking + record the baseline as already-applied ---
# So the migration runner (migrate.py) knows this fresh DB is at the baseline and
# only runs NEW migrations (0002+).
cur.execute("""
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    migration_name TEXT,
    applied_at TIMESTAMP DEFAULT NOW()
)
""")
cur.execute("""
    INSERT INTO schema_migrations (version, migration_name)
    VALUES ('0001', 'baseline')
    ON CONFLICT (version) DO NOTHING
""")

conn.commit()
conn.close()
print("Postgres tables ready (complete schema: 14 tables + schema_migrations; baseline recorded)")