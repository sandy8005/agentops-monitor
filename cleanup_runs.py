import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# find orphans: still "running", started more than 10 minutes ago
cur.execute("""
    SELECT id, started_at FROM runs
    WHERE status = 'running'
      AND started_at < NOW() - INTERVAL '10 minutes'
    ORDER BY id
""")
orphans = cur.fetchall()

if not orphans:
    print("No orphaned runs found.")
else:
    print(f"Found {len(orphans)} orphaned run(s):")
    for run_id, started in orphans:
        print(f"  Run {run_id} (started {started})")

    cur.execute("""
        UPDATE runs
        SET status = 'failed', ended_at = NOW()
        WHERE status = 'running'
          AND started_at < NOW() - INTERVAL '10 minutes'
    """)
    conn.commit()
    print(f"Marked {len(orphans)} run(s) as failed.")

conn.close()