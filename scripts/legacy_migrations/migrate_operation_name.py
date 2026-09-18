import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE llm_calls  ADD COLUMN IF NOT EXISTS operation_name TEXT")
cur.execute("ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS operation_name TEXT")
conn.commit()
conn.close()
print("Migration complete: operation_name added to llm_calls and tool_calls")