import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
cur = conn.cursor()
# search_location: the location a job was FETCHED FOR (Adzuna's `where`). NULL for
# jobs not tied to a location search (seed/csv/scraped) — those are always eligible.
cur.execute("ALTER TABLE job_postings ADD COLUMN IF NOT EXISTS search_location TEXT")
conn.commit(); conn.close()
print("Migration complete: job_postings.search_location added")