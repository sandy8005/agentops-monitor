import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS job_postings (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT,
    description TEXT NOT NULL,
    location TEXT,
    work_mode TEXT,
    source TEXT DEFAULT 'seed',
    created_at TIMESTAMP DEFAULT NOW()
)
""")

conn.commit()
conn.close()
print("job_postings table ready")