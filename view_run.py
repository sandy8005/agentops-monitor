import sys
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


def list_runs(limit=15):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, status, started_at, total_tokens, total_cost
        FROM runs ORDER BY id DESC LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    print("=" * 70)
    print(f"{'ID':>4}  {'STATUS':<22} {'STARTED':<20} {'TOKENS':>7} {'COST':>10}")
    print("=" * 70)
    for run_id, status, started, tokens, cost in rows:
        started_str = started.strftime("%Y-%m-%d %H:%M") if started else "-"
        print(f"{run_id:>4}  {status:<22} {started_str:<20} {tokens or 0:>7} ${cost or 0:>9}")
    print("\nRun 'python view_run.py <id>' to inspect a specific run.")


def view_run(run_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, status, started_at, ended_at, input_summary, total_tokens, total_cost
        FROM runs WHERE id = %s
    """, (run_id,))
    run = cur.fetchone()

    if not run:
        print(f"No run found with id {run_id}")
        conn.close()
        return

    print("=" * 70)
    print(f"RUN {run[0]}  |  status: {run[1]}")
    print(f"input:   {run[4]}")
    print(f"started: {run[2]}")
    print(f"ended:   {run[3]}")
    print(f"tokens:  {run[5]}   cost: ${run[6]}")
    print("=" * 70)

    # get the steps
    cur.execute("""
        SELECT id, step_name, status, error_message,
               match_score, score_decision, llm_decision, needs_human_review
        FROM steps WHERE run_id = %s ORDER BY step_order
    """, (run_id,))
    steps = cur.fetchall()

    for step in steps:
        (step_id, step_name, status, error_message,
         match_score, score_decision, llm_decision, needs_review) = step

        review_flag = "  ** NEEDS REVIEW **" if needs_review else ""
        print(f"\nStep: {step_name}  [{status}]{review_flag}")

        # show score comparison if this step was scored
        if match_score is not None:
            print(f"  SCORE: {match_score}/100 → {score_decision}   |   LLM: {llm_decision}")

        # tool calls for this step
        cur.execute("""
            SELECT tool_name, output_json, status, latency_ms
            FROM tool_calls WHERE step_id = %s ORDER BY id
        """, (step_id,))
        for tool_name, output_json, tstatus, latency in cur.fetchall():
            print(f"  tool: {tool_name} [{tstatus}]  ({latency}ms)")
            print(f"        {output_json}")

        # llm calls for this step
        cur.execute("""
            SELECT response, prompt_tokens, completion_tokens, latency_ms, cost_usd
            FROM llm_calls WHERE step_id = %s ORDER BY id
        """, (step_id,))
        for response, p_tok, c_tok, latency, cost in cur.fetchall():
            print(f"  llm: {p_tok}+{c_tok} tokens, {latency}ms, ${cost}")
            print(f"       {response}")

        if status == "failed":
            print(f"  error: {error_message}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_run(int(sys.argv[1]))
    else:
        list_runs()