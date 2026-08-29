import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS job_reqs_cache (
    desc_hash TEXT PRIMARY KEY,
    reqs_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
conn.commit()
conn.close()
print("Migration complete: job_reqs_cache created")