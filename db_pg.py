import os
import psycopg2
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
    total_cost NUMERIC(12,6) DEFAULT 0
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

conn.commit()
conn.close()
print("Postgres tables ready")