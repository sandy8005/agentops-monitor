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

# Estimated pricing (USD per token), based on published Gemini Flash rates.
# ESTIMATES for observability, not the actual provider invoice.
INPUT_TOKEN_RATE = 0.075 / 1_000_000     # ~$0.075 per 1M input tokens
OUTPUT_TOKEN_RATE = 0.30 / 1_000_000     # ~$0.30 per 1M output tokens


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
        client.models.generate_content(model="gemini-flash-latest", contents="hi")
        return True
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return False
        raise


def fake_llm(prompt):
    time.sleep(0.5)
    return {"text": "Apply", "prompt_tokens": len(prompt.split()),
            "completion_tokens": random.randint(5, 20)}


def real_llm_once(prompt):
    """Single LLM attempt — no retry. Raises on failure. Retry lives in logged_llm_call."""
    response = client.models.generate_content(
        model="gemini-flash-latest", contents=prompt
    )
    usage = response.usage_metadata
    request_id = None
    try:
        request_id = getattr(response, "response_id", None) or getattr(response, "_request_id", None)
    except Exception:
        request_id = None
    return {
        "text": response.text,
        "prompt_tokens": usage.prompt_token_count,
        "completion_tokens": usage.candidates_token_count,
        "provider_request_id": request_id
    }


def create_run(input_summary, resume_id=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO runs (started_at, status, input_summary, resume_id)
        VALUES (%s, %s, %s, %s) RETURNING id
    """, (datetime.now(), "running", input_summary, resume_id))
    run_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return run_id


def create_step(run_id, step_name, step_order):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO steps (run_id, step_name, step_order, started_at, status)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
    """, (run_id, step_name, step_order, datetime.now(), "running"))
    step_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return step_id


def finish_step(step_id, status="success"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE steps SET ended_at = %s, status = %s WHERE id = %s",
                (datetime.now(), status, step_id))
    conn.commit()
    conn.close()


def fail_step(step_id, error_message):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE steps SET ended_at = %s, status = %s, error_message = %s WHERE id = %s",
                (datetime.now(), "failed", str(error_message), step_id))
    conn.commit()
    conn.close()


def record_score(step_id, match_score, score_decision, llm_decision):
    needs_review = score_decision != llm_decision
    conn = get_connection()
    cur = conn.cursor()
    if needs_review:
        cur.execute("""
            UPDATE steps SET match_score = %s, score_decision = %s, llm_decision = %s,
                needs_human_review = TRUE, review_reason = 'score_disagreement' WHERE id = %s
        """, (match_score, score_decision, llm_decision, step_id))
    else:
        cur.execute("""
            UPDATE steps SET match_score = %s, score_decision = %s, llm_decision = %s,
                needs_human_review = FALSE WHERE id = %s
        """, (match_score, score_decision, llm_decision, step_id))
    conn.commit()
    conn.close()
    return needs_review


def flag_for_review(step_id, reason="unspecified"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT review_reason FROM steps WHERE id = %s", (step_id,))
    row = cur.fetchone()
    existing = row[0] if row and row[0] else ""
    if existing:
        reasons = [r.strip() for r in existing.split(";")]
        new_reason = existing if reason in reasons else existing + "; " + reason
    else:
        new_reason = reason
    cur.execute("UPDATE steps SET needs_human_review = TRUE, review_reason = %s WHERE id = %s",
                (new_reason, step_id))
    conn.commit()
    conn.close()


def record_context(step_id, context):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE steps SET retrieved_context = %s WHERE id = %s",
                (json.dumps(context), step_id))
    conn.commit()
    conn.close()


def is_cancel_requested(run_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT cancel_requested FROM runs WHERE id = %s", (run_id,))
    row = cur.fetchone()
    conn.close()
    return bool(row and row[0])


def request_cancel(run_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
    conn.commit()
    conn.close()


def save_evaluation(run_id, step_id, evaluation):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO evaluations
        (run_id, step_id, relevance_score, faithfulness_score, completeness_score,
         hallucination_detected, hallucinated_claims, notes, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id, step_id,
        evaluation["relevance_score"], evaluation["faithfulness_score"],
        evaluation["completeness_score"], evaluation["hallucination_detected"],
        json.dumps(evaluation["hallucinated_claims"]), evaluation["notes"], datetime.now()
    ))
    conn.commit()
    conn.close()


def finish_run(run_id, status="success"):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE runs SET ended_at = %s, status = %s,
            total_tokens = (SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                            FROM llm_calls WHERE run_id = %s),
            total_cost = (SELECT COALESCE(SUM(cost_usd), 0)
                          FROM llm_calls WHERE run_id = %s)
        WHERE id = %s
    """, (datetime.now(), status, run_id, run_id, run_id))
    conn.commit()
    conn.close()


def _log_llm_attempt(run_id, step_id, operation, prompt, response_text,
                     prompt_tokens, completion_tokens, latency_ms, cost,
                     status, error_message, attempt_number, retry_count, provider_request_id):
    """Log a single HTTP attempt of an LLM call as its own llm_calls row."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO llm_calls
        (run_id, step_id, model, prompt, response,
         prompt_tokens, completion_tokens, latency_ms, cost_usd, created_at,
         status, error_message, operation_name, attempt_number, retry_count, provider_request_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id, step_id, "gemini-flash-latest", prompt, response_text,
        prompt_tokens, completion_tokens, latency_ms, cost, datetime.now(),
        status, error_message, operation, attempt_number, retry_count, provider_request_id
    ))
    conn.commit()
    conn.close()


def logged_llm_call(prompt, run_id, step_id, operation="llm_call", max_retries=3):
    """
    Run an LLM call with retry, logging EACH HTTP attempt as its own row.
    Failed attempts are logged with status='failed' and their attempt_number;
    the succeeding attempt records retry_count = number of prior failures.
    So two 503s then a success produce three rows (attempts 1, 2, 3).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            result = real_llm_once(prompt)
            latency_ms = int((time.time() - start) * 1000)
            prompt_tokens = result["prompt_tokens"]
            completion_tokens = result["completion_tokens"]
            cost = (prompt_tokens * INPUT_TOKEN_RATE +
                    completion_tokens * OUTPUT_TOKEN_RATE)
            _log_llm_attempt(
                run_id, step_id, operation, prompt, result["text"],
                prompt_tokens, completion_tokens, latency_ms, cost,
                "success", None, attempt, attempt - 1, result.get("provider_request_id")
            )
            return result["text"]
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            last_error = e
            transient = "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)
            _log_llm_attempt(
                run_id, step_id, operation, prompt, None,
                0, 0, latency_ms, 0,
                "failed", str(e), attempt, attempt - 1, None
            )
            if attempt == max_retries or not transient:
                raise
            wait = 2 ** (attempt - 1)
            print(f"  retry {attempt}/{max_retries - 1} after {wait}s...")
            time.sleep(wait)
    if last_error:
        raise last_error


def logged_tool_call(tool_name, tool_func, tool_input, run_id, step_id, operation=None):
    """Run a tool and log it, tagged with the conceptual operation (stage) name."""
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
        (run_id, step_id, tool_name, input_json, output_json, latency_ms, status, error_message, created_at, operation_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        run_id, step_id, tool_name, json.dumps(tool_input),
        json.dumps(result) if result is not None else None,
        latency_ms, status, error_message, datetime.now(), operation or tool_name
    ))
    conn.commit()
    conn.close()
    return result


if __name__ == "__main__":
    run_id = create_run("test resume vs test job")
    step_id = create_step(run_id, "score_job", 1)
    answer = logged_llm_call("Does this resume match this job?", run_id, step_id, operation="test")
    print("Agent got back:", answer)