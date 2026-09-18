"""
Migration: job_postings.apply_url
Adds the apply_url column (Adzuna's redirect link) to existing databases.
Idempotent.
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
cur.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS apply_url TEXT")
conn.commit()
conn.close()
print("Migration complete: job_postings.apply_url added")