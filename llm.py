import time
import random
from datetime import datetime
import psycopg2
import os
from dotenv import load_dotenv
from google import genai
import json

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )


def quota_available():
    """Cheap probe: returns True if the API responds, False if rate-limited."""
    try:
        client.models.generate_content(
            model="gemini-flash-latest",
            contents="hi"
        )
        return True
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return False
        raise


def fake_llm(prompt):
    """Mock LLM — used for testing the logging pipeline without real API calls."""
    time.sleep(0.5)
    return {
        "text": "Apply",
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": random.randint(5, 20)
    }


def real_llm(prompt, max_retries=3):
    """Real LLM call via Gemini, with retry on transient failures."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            usage = response.usage_metadata
            return {
                "text": response.text,
                "prompt_tokens": usage.prompt_token_count,
                "completion_tokens": usage.candidates_token_count
            }
        except Exception as e:
            is_last = attempt == max_retries - 1
            transient = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            if is_last or not transient:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{max_retries - 1} after {wait}s...")
            time.sleep(wait)


def create_run(input_summary):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (started_at, status, input_summary)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (datetime.now(), "running", input_summary))
    run_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return run_id


def create_step(run_id, step_name, step_order):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO steps (run_id, step_name, step_order, started_at, status)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (run_id, step_name, step_order, datetime.now(), "running"))
    step_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return step_id


def finish_step(step_id, status="success"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE steps SET ended_at = %s, status = %s WHERE id = %s
    """, (datetime.now(), status, step_id))
    conn.commit()
    conn.close()


def fail_step(step_id, error_message):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE steps SET ended_at = %s, status = %s, error_message = %s WHERE id = %s
    """, (datetime.now(), "failed", str(error_message), step_id))
    conn.commit()
    conn.close()


def record_score(step_id, match_score, score_decision, llm_decision):
    needs_review = score_decision != llm_decision
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE steps
        SET match_score = %s, score_decision = %s, llm_decision = %s, needs_human_review = %s
        WHERE id = %s
    """, (match_score, score_decision, llm_decision, needs_review, step_id))
    conn.commit()
    conn.close()
    return needs_review


def record_context(step_id, context):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE steps SET retrieved_context = %s WHERE id = %s
    """, (json.dumps(context), step_id))
    conn.commit()
    conn.close()


def finish_run(run_id, status="success"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE runs
        SET ended_at = %s,
            status = %s,
            total_tokens = (
                SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                FROM llm_calls WHERE run_id = %s
            ),
            total_cost = (
                SELECT COALESCE(SUM(cost_usd), 0)
                FROM llm_calls WHERE run_id = %s
            )
        WHERE id = %s
    """, (datetime.now(), status, run_id, run_id, run_id))
    conn.commit()
    conn.close()


def logged_llm_call(prompt, run_id, step_id):
    start = time.time()
    result = real_llm(prompt)
    end = time.time()

    latency_ms = int((end - start) * 1000)

    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    cost_usd = (prompt_tokens + completion_tokens) * 0.000001

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO llm_calls
        (run_id, step_id, model, prompt, response,
         prompt_tokens, completion_tokens, latency_ms, cost_usd, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id, step_id, "gemini-flash-latest", prompt, result["text"],
        prompt_tokens, completion_tokens, latency_ms, cost_usd, datetime.now()
    ))
    conn.commit()
    conn.close()

    return result["text"]


def logged_tool_call(tool_name, tool_func, tool_input, run_id, step_id):
    start = time.time()
    try:
        result = tool_func(tool_input)
        status, error_message = "success", None
    except Exception as e:
        result, status, error_message = None, "failed", str(e)
    end = time.time()
    latency_ms = int((end - start) * 1000)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tool_calls
        (run_id, step_id, tool_name, input_json, output_json, latency_ms, status, error_message, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id, step_id, tool_name, json.dumps(tool_input),
        json.dumps(result) if result is not None else None,
        latency_ms, status, error_message, datetime.now()
    ))
    conn.commit()
    conn.close()
    return result


if __name__ == "__main__":
    run_id = create_run("test resume vs test job")
    step_id = create_step(run_id, "score_job", 1)
    answer = logged_llm_call("Does this resume match this job?", run_id, step_id)
    print("Agent got back:", answer)