import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS target_role TEXT")
cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS location TEXT")
cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS work_mode TEXT")
cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS employment_type TEXT")
conn.commit()
conn.close()
print("Migration complete: search config columns added to runs")