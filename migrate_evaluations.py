import psycopg2, os
from dotenv import load_dotenv
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    run_id INTEGER,
    step_id INTEGER,
    relevance_score INTEGER,
    faithfulness_score INTEGER,
    completeness_score INTEGER,
    hallucination_detected BOOLEAN,
    hallucinated_claims JSONB,
    notes TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(id),
    FOREIGN KEY (step_id) REFERENCES steps(id)
)
""")

conn.commit()
conn.close()
print("Migration complete: evaluations table created")