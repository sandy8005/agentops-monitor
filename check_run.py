import sys, os, psycopg2
from dotenv import load_dotenv

load_dotenv()
run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 107

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("""
    SELECT step_name, status, error_message
    FROM steps WHERE run_id = %s ORDER BY step_order
""", (run_id,))
for name, status, err in cur.fetchall():
    print(f"[{status}] {name}")
    if err:
        print(f"    error: {err}")
conn.close()