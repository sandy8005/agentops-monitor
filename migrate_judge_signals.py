import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS judge_status TEXT")
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS judge_skip_reason TEXT")
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS cache_hit BOOLEAN")
conn.commit()
conn.close()
print("Migration complete: steps.judge_status, judge_skip_reason, cache_hit added")