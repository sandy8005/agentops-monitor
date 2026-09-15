import os, psycopg2
from dotenv import load_dotenv
load_dotenv()
c = psycopg2.connect(dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
cur = c.cursor()
cur.execute("DELETE FROM job_reqs_cache")
print("cleared", cur.rowcount, "rows")
c.commit()