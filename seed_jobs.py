import psycopg2, os
from dotenv import load_dotenv
from jobs import JOBS
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# clear existing seed rows so re-running doesn't duplicate
cur.execute("DELETE FROM job_postings WHERE source = 'seed'")

for job in JOBS:
    cur.execute("""
        INSERT INTO job_postings (title, company, description, source)
        VALUES (%s, %s, %s, 'seed')
    """, (job["title"], job["company"], job["description"]))

conn.commit()
cur.execute("SELECT COUNT(*) FROM job_postings")
count = cur.fetchone()[0]
conn.close()
print(f"Seeded jobs. Total in table: {count}")