import json, psycopg2, os
from dotenv import load_dotenv
from evaluator import evaluate_decision
from llm import create_run, create_step, finish_step, finish_run
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

RUN_ID = 26

cur.execute("""
    SELECT t.output_json FROM tool_calls t JOIN steps s ON t.step_id = s.id
    WHERE s.run_id = %s AND s.step_name = 'read_resume_file'
""", (RUN_ID,))
resume_text = json.loads(cur.fetchone()[0])

# grab DevOps (the false-positive case) and Backend Engineer (a clean case)
targets = ["DevOps Engineer", "Backend Engineer (Python)"]

eval_run = create_run("evaluation test")

for title in targets:
    cur.execute("""
        SELECT s.id FROM steps s WHERE s.run_id = %s AND s.step_name = %s
    """, (RUN_ID, title))
    src_step = cur.fetchone()
    if not src_step:
        continue
    cur.execute("SELECT response FROM llm_calls WHERE step_id = %s ORDER BY id", (src_step[0],))
    calls = [r[0] for r in cur.fetchall()]
    if len(calls) < 2:
        continue
    agent_response = calls[1]   # the judgment

    step_id = create_step(eval_run, f"eval_{title}", 0)
    result = evaluate_decision(resume_text, {"title": title, "description": ""}, agent_response, eval_run, step_id)
    finish_step(step_id)

    print(f"\n{title}:")
    print(f"  relevance: {result['relevance_score']}/10  faithfulness: {result['faithfulness_score']}/10")
    print(f"  hallucination: {result['hallucination_detected']}  claims: {result['hallucinated_claims']}")
    print(f"  notes: {result['notes']}")

finish_run(eval_run)
conn.close()