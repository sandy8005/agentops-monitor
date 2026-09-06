import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
cur = conn.cursor()
# external_id: stable per-posting key for live jobs, so re-fetching doesn't duplicate.
cur.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS external_id TEXT")
# unique index (partial: only where external_id is set, so seeded jobs with NULL are unaffected)
cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS job_postings_external_id_uniq
    ON job_postings (external_id) WHERE external_id IS NOT NULL
""")
conn.commit(); conn.close()
print("Migration complete: job_postings.external_id + unique index added")