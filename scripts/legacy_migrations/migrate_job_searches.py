"""
Migration: per-search job association + Live Mode separation.
Creates job_searches (one row per live search) and job_search_results (the
many-to-many join between a search and the postings it returned), replacing the
permanent search_location identity. Idempotent.
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

conn.commit()
conn.close()
print("Migration complete: job_searches + job_search_results added")