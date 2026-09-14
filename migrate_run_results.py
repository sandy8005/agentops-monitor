"""
Migration: persist ranked results and advice.

- run_rankings : the final ranked list for a run, ONE self-contained snapshot row
                 per ranked job. Snapshot fields (title/company/score/final_decision/
                 apply_url) mean the ranking is readable without joining job_postings,
                 which may change or be pruned later. rank_position is 1-based.
- run_advice   : the generated application/resume advice text per (run, job).

Both key on run_id (and job_id where known). Idempotent.
Run AFTER migrate_job_searches.py (job_postings must exist for the FK).
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

cur.execute("""
    CREATE TABLE IF NOT EXISTS run_rankings (
        id SERIAL PRIMARY KEY,
        run_id INTEGER NOT NULL,
        job_id INTEGER,                    -- job_postings.id when known (nullable for legacy)
        rank_position INTEGER NOT NULL,    -- 1-based order in the ranked list
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
print("Migration complete: run_rankings + run_advice added")