import json, psycopg2, os
from dotenv import load_dotenv
from scorer import calculate_match_score
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# grab the parsed resume (llm_call in the parse_resume step of run 22)
cur.execute("""
    SELECT l.response FROM llm_calls l
    JOIN steps s ON l.step_id = s.id
    WHERE s.run_id = 22 AND s.step_name = 'parse_resume'
""")
parsed = json.loads(cur.fetchone()[0])

# grab the resume text (tool_call in the read_resume_file step of run 22)
cur.execute("""
    SELECT t.output_json FROM tool_calls t
    JOIN steps s ON t.step_id = s.id
    WHERE s.run_id = 22 AND s.step_name = 'read_resume_file'
""")
resume_text = json.loads(cur.fetchone()[0])

# grab each job's extracted requirements (first valid JSON llm_call per job step)
cur.execute("""
    SELECT s.step_name, l.response FROM llm_calls l
    JOIN steps s ON l.step_id = s.id
    WHERE s.run_id = 22 AND s.step_name NOT IN ('receive_user_input','read_resume_file','parse_resume')
    ORDER BY s.step_order, l.id
""")

seen = set()
for step_name, response in cur.fetchall():
    if step_name in seen:
        continue
    try:
        reqs = json.loads(response)
        if "required_skills" not in reqs:
            continue
    except:
        continue
    seen.add(step_name)
    result = calculate_match_score(parsed, reqs, resume_text)
    print(f"\n{step_name}: {result['score']}/100 → {result['decision']}")
    print(f"  {result['breakdown']}")

conn.close()