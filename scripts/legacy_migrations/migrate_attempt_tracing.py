import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS attempt_number INTEGER DEFAULT 1")
cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0")
cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS provider_request_id TEXT")
conn.commit()
conn.close()
print("Migration complete: attempt_number, retry_count, provider_request_id added to llm_calls")