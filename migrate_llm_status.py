import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'success'")
cur.execute("ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS error_message TEXT")
conn.commit()
conn.close()
print("Migration complete: status + error_message added to llm_calls")