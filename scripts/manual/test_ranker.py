import psycopg2, os
from dotenv import load_dotenv
from ranker import rank_jobs
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

RUN_ID = 23   # change to whichever run has scores

cur.execute("""
    SELECT step_name, match_score, score_decision, llm_decision, needs_human_review
    FROM steps
    WHERE run_id = %s AND match_score IS NOT NULL
    ORDER BY step_order
""", (RUN_ID,))

results = []
for step_name, score, score_dec, llm_dec, needs_review in cur.fetchall():
    results.append({
        "title": step_name,
        "score": float(score),
        "decision": score_dec,
        "llm_decision": llm_dec,
        "needs_review": needs_review
    })

conn.close()

ranked = rank_jobs(results)

print(f"\nRanked jobs for run {RUN_ID}:\n")
for i, r in enumerate(ranked, 1):
    flag = "  ** REVIEW **" if r["needs_review"] else ""
    print(f"{i}. {r['title']}  —  {r['score']}/100 ({r['decision']}) | LLM: {r['llm_decision']}{flag}")