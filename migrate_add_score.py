import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS match_score NUMERIC(5,1)")
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS score_decision TEXT")
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS llm_decision TEXT")
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS needs_human_review BOOLEAN DEFAULT FALSE")

conn.commit()
conn.close()
print("Migration complete: score columns added to steps")