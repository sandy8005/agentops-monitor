import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS review_status TEXT")
cur.execute("ALTER TABLE steps ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP")
conn.commit()
conn.close()
print("Migration complete: review_status + reviewed_at added to steps")