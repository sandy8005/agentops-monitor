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
    # employment_type is optional in the JOBS data; default to full-time.
    emp_type = job.get("employment_type", "full-time")
    cur.execute("""
        INSERT INTO job_postings (title, company, description, employment_type, source)
        VALUES (%s, %s, %s, %s, 'seed')
    """, (job["title"], job["company"], job["description"], emp_type))

conn.commit()
cur.execute("SELECT COUNT(*) FROM job_postings")
count = cur.fetchone()[0]
conn.close()
print(f"Seeded jobs. Total in table: {count}")