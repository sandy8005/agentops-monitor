import sqlite3

conn = sqlite3.connect("agentops.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT,
    ended_at TEXT,
    status TEXT,
    input_summary TEXT,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    step_name TEXT,
    step_order INTEGER,
    started_at TEXT,
    ended_at TEXT,
    status TEXT,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    step_id INTEGER,
    model TEXT,
    prompt TEXT,
    response TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER,
    cost_usd REAL,
    created_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    step_id INTEGER,
    tool_name TEXT,
    input_json TEXT,
    output_json TEXT,
    latency_ms INTEGER,
    status TEXT,
    error_message TEXT,
    created_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

conn.commit()
conn.close()
print("Database ready")