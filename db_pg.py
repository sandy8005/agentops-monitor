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
    cancel_requested BOOLEAN DEFAULT FALSE
)
""")

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
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")

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
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

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
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

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
    search_location TEXT,          -- legacy; superseded by job_search_results (kept nullable)
    fetched_at TIMESTAMP,
    last_seen_at TIMESTAMP,
    posted_at TIMESTAMP,
    apply_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
# Live jobs dedup on external_id (partial unique: seeded rows keep NULL).
cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS job_postings_external_id_uniq
    ON job_postings (external_id) WHERE external_id IS NOT NULL
""")

# --- Per-search association (Live Mode + many-to-many job↔search) ---
# One row per live search performed; carries the role/location intent that used
# to be stamped permanently onto each job as search_location.
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
# Many-to-many between a search and the postings it returned.
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

cur.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id SERIAL PRIMARY KEY,
    name TEXT,
    resume_text TEXT NOT NULL,
    target_role TEXT,
    location TEXT,
    work_mode TEXT,
    employment_type TEXT,
    created_at TIMESTAMP DEFAULT NOW()
)
""")

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

# Persisted final ranked list (self-contained snapshot rows) + generated advice.
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

conn.commit()
conn.close()
print("Postgres tables ready (complete schema: 11 tables)")