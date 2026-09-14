import json, psycopg2, os
from dotenv import load_dotenv
from hallucination import check_hallucination_risk
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

RUN_ID = 26

# resume text
cur.execute("""
    SELECT t.output_json FROM tool_calls t JOIN steps s ON t.step_id = s.id
    WHERE s.run_id = %s AND s.step_name = 'read_resume_file'
""", (RUN_ID,))
resume_text = json.loads(cur.fetchone()[0])

# for each job step, grab requirements (2nd llm call) and decision (3rd)
cur.execute("""
    SELECT s.step_name, s.id FROM steps s
    WHERE s.run_id = %s AND s.step_order >= 4 ORDER BY s.step_order
""", (RUN_ID,))
steps = cur.fetchall()

for step_name, step_id in steps:
    cur.execute("SELECT response FROM llm_calls WHERE step_id = %s ORDER BY id", (step_id,))
    calls = [r[0] for r in cur.fetchall()]
    if len(calls) < 2:
        continue
    try:
        requirements = json.loads(calls[0])   # first llm call = requirements
    except:
        continue
    decision_response = calls[1]              # second = the judgment prose

    result = check_hallucination_risk(decision_response, requirements, resume_text)
    print(f"\n{step_name}: risk={result['risk']}")
    print(f"  mentioned: {result['mentioned_skills']}")
    if result['hallucinated_skills']:
        print(f"  ⚠ HALLUCINATED: {result['hallucinated_skills']}")

conn.close()