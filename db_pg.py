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
    source TEXT DEFAULT 'seed',
    created_at TIMESTAMP DEFAULT NOW()
)
""")

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

conn.commit()
conn.close()
print("Postgres tables ready (complete schema: 7 tables)")