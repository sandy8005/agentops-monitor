import sys, os, psycopg2
from dotenv import load_dotenv
load_dotenv()
run_id = int(sys.argv[1])
c = psycopg2.connect(dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"))
cur = c.cursor()
cur.execute("UPDATE runs SET status='failed', ended_at=NOW() WHERE id=%s AND status='running'", (run_id,))
print("updated", cur.rowcount, "run(s)")
c.commit()