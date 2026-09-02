import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE parsed_resume_cache ADD COLUMN IF NOT EXISTS cache_version TEXT")
cur.execute("ALTER TABLE job_reqs_cache ADD COLUMN IF NOT EXISTS cache_version TEXT")
conn.commit()
conn.close()
print("Migration complete: cache_version added to parsed_resume_cache and job_reqs_cache")