import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
cur = conn.cursor()
cur.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMP")
cur.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP")
cur.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS posted_at TIMESTAMP")
conn.commit(); conn.close()
print("Migration complete: fetched_at, last_seen_at, posted_at added")