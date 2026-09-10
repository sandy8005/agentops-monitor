# force_paused_run.py — creates a run stuck at waiting_for_human for UI testing
import os, json
from dotenv import load_dotenv
load_dotenv()
from llm import create_run, get_connection

run_id = create_run("FORCED review test", resume_id=1, target_role="engineer")
conn = get_connection(); cur = conn.cursor()
cur.execute("UPDATE runs SET status='waiting_for_human', pending_review=%s WHERE id=%s",
            (json.dumps({"job_title": "Senior Python Developer", "score": 74.0,
                         "score_decision": "Maybe", "llm_decision": "Apply",
                         "step_id": 999}), run_id))
conn.commit(); conn.close()
print(f"Created paused run {run_id} — check the dashboard's review panel.")