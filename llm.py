from timeutil import utcnow
import time
import random
import uuid
from database import get_connection
from google import genai
import json
from settings import settings
from error_codes import ErrorCode, classify_exception, provider_retry_after
from logging_config import get_logger
from sanitize import redact_secrets
log = get_logger(__name__)

# timeout is in MILLISECONDS in google-genai's http_options. 30s means a stalled
# Gemini call fails fast (raises) instead of hanging forever — the retry/backoff in
# logged_llm_call then engages, and if it keeps failing the job degrades to the
# rule-based fallback rather than freezing the whole run.
#
# The client is created LAZILY (first real call), so importing llm — which most
# modules and tests do — never requires GEMINI_API_KEY.
_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"timeout": 30_000},   # 30 seconds, in ms
        )
    return _client


class InvalidProviderResponse(RuntimeError):
    """The provider returned a response without the fields we need (no text, no
    usage metadata) — typically a blocked / safety-filtered / truncated response.
    Not transient: retrying the same prompt yields the same block."""


class BudgetExceeded(Exception):
    """Raised when an LLM call is refused because the run's request budget is spent.
    NOT a transient error — must not be retried."""
    pass



# get_connection is imported (pooled) from database at the top of this module, so
# every `from llm import get_connection` (router.py, autonomous_graph.py, ...) now
# draws from the shared ThreadedConnectionPool instead of opening a fresh socket.


def fake_llm(prompt):
    time.sleep(0.5)
    return {"text": "Apply", "prompt_tokens": len(prompt.split()),
            "completion_tokens": random.randint(5, 20)}


def real_llm_once(prompt):
    """Single LLM attempt — no retry. Raises on failure. Retry lives in logged_llm_call.

    Defensive about the SDK response shape: a blocked or abnormal response can have
    text=None, usage_metadata=None, or None token counts. Those become an explicit
    InvalidProviderResponse here instead of a confusing None.strip() far downstream.
    Missing token counts are recorded as 0 (the call still happened)."""
    response = get_client().models.generate_content(
        model=settings.gemini_model, contents=prompt
    )
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        reason = None
        try:
            fb = getattr(response, "prompt_feedback", None)
            reason = getattr(fb, "block_reason", None)
            if reason is None:
                cands = getattr(response, "candidates", None) or []
                reason = getattr(cands[0], "finish_reason", None) if cands else None
        except Exception:
            reason = None
        raise InvalidProviderResponse(
            f"provider returned no text (block/finish reason: {reason or 'unknown'})")
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    request_id = None
    try:
        request_id = getattr(response, "response_id", None) or getattr(response, "_request_id", None)
    except Exception:
        request_id = None
    return {
        "text": text,
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "usage_missing": usage is None or prompt_tokens is None or completion_tokens is None,
        "provider_request_id": request_id
    }


def create_run_tx(cur, input_summary, resume_id=None, target_role=None,
                  location=None, work_mode=None, employment_type=None, user_id=None):
    """
    Transactional create_run: INSERT the run on the CALLER'S cursor and return its
    id WITHOUT committing. Lets the API create the run and enqueue its worker job in
    ONE transaction (both commit or both roll back), so a run is never left
    'running' with no queue job. The caller owns commit / rollback / close.
    """
    # A new run is 'queued', NOT 'running': at creation it is only waiting in
    # job_queue for a worker to claim it. started_at is left NULL and is stamped
    # only when the worker actually begins executing (see _mark_run_running in
    # autonomous_graph), so queue-wait time is never counted as execution latency.
    cur.execute("""
        INSERT INTO runs (started_at, status, input_summary, resume_id,
                          target_role, location, work_mode, employment_type, user_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
    """, (None, "queued", input_summary, resume_id,
          target_role, location, work_mode, employment_type, user_id))
    return cur.fetchone()[0]


def create_run(input_summary, resume_id=None, target_role=None,
               location=None, work_mode=None, employment_type=None, user_id=None):
    """Create a run in its OWN transaction (thin wrapper over create_run_tx)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        run_id = create_run_tx(cur, input_summary, resume_id=resume_id,
                               target_role=target_role, location=location,
                               work_mode=work_mode, employment_type=employment_type,
                               user_id=user_id)
        conn.commit()
        return run_id
    finally:
        conn.close()


def create_step(run_id, step_name, step_order):
    with get_connection() as conn:
        cur = conn.cursor()
        # run_attempt tags the step with the run's CURRENT execution attempt, so a
        # retried run's second pass doesn't interleave ambiguously with the first
        # (step_order restarts at 0 on every attempt).
        cur.execute("""
            INSERT INTO steps (run_id, step_name, step_order, started_at, status, run_attempt)
            VALUES (%s, %s, %s, %s, %s, (SELECT attempt FROM runs WHERE id = %s))
            RETURNING id
        """, (run_id, step_name, step_order, utcnow(), "running", run_id))
        step_id = cur.fetchone()[0]
        return step_id


def finish_step(step_id, status="success"):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE steps SET ended_at = %s, status = %s WHERE id = %s",
                    (utcnow(), status, step_id))


def fail_step(step_id, error_message):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE steps SET ended_at = %s, status = %s, error_message = %s WHERE id = %s",
                    (utcnow(), "failed", redact_secrets(error_message), step_id))


def _breakdown_for_storage(breakdown, breakdown_max=None):
    """
    Persist each category as {"earned": x, "max": y}. The scorer RENORMALIZES weights
    when optional categories are absent, so the maximum for a category is not a fixed
    50/20/15/15 — storing the real max lets the dashboard show the truth.
    """
    if not breakdown:
        return None
    if not breakdown_max:
        return breakdown          # legacy shape (flat numbers) — caller had no maxima
    return {cat: {"earned": breakdown.get(cat, 0.0), "max": breakdown_max.get(cat, 0.0)}
            for cat in breakdown}


def record_score(step_id, match_score, score_decision, llm_decision, breakdown=None,
                 breakdown_max=None):
    """
    Record the match score, decisions, and the per-category breakdown (stored as
    JSONB, each category with its earned points AND its real maximum). The breakdown
    is valuable trace context for a human reviewer: it shows WHERE the score came
    from, not just the total.

    Returns whether THIS call raised a score-disagreement flag. That is ONE review
    trigger among several — callers must route on the authoritative DB flag
    (router._step_needs_review), never on this return value.
    """
    # Disagreement only counts when there is a REAL LLM decision to compare
    # against. A skipped judge ("skipped (...)") or an unparseable one ("Unknown")
    # is the ABSENCE of a second opinion — not a disagreement — so it must not be
    # flagged as score_disagreement. Other review triggers still fire independently.
    real_llm_decisions = {"Apply", "Maybe", "Skip"}
    has_real_judgment = llm_decision in real_llm_decisions
    score_disagreement = has_real_judgment and (score_decision != llm_decision)
    stored = _breakdown_for_storage(breakdown, breakdown_max)
    breakdown_json = json.dumps(stored) if stored else None
    with get_connection() as conn:
        cur = conn.cursor()
        # Record the score fields ONLY. Crucially, do NOT touch needs_human_review
        # here: the review flag is ADDITIVE and owned by flag_for_review(). Blindly
        # writing needs_human_review = FALSE (the old behaviour) would erase a review
        # already raised for another reason on this same step — e.g. a
        # possible_prompt_injection flag set during requirements extraction — silently
        # un-pausing a run that should be reviewed.
        cur.execute("""
            UPDATE steps SET match_score = %s, score_decision = %s, llm_decision = %s,
                score_breakdown = %s
            WHERE id = %s
        """, (match_score, score_decision, llm_decision, breakdown_json, step_id))
    # Score disagreement is one review trigger among several — raise it ADDITIVELY
    # (flag_for_review never clears an existing flag and appends the reason), the
    # same way injection / hallucination / evaluation-failure triggers do.
    if score_disagreement:
        flag_for_review(step_id, reason="score_disagreement")
    return score_disagreement


def flag_for_review(step_id, reason="unspecified"):
    with get_connection() as conn:
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


def flag_security(step_id, reason="unspecified"):
    """
    Record a SECURITY WARNING on a step (e.g. injection-like text in the resume)
    WITHOUT requesting human approval. Deliberately separate from flag_for_review:
    the run continues and the event is monitored — "security warning" is not "human
    approval required", and conflating them makes the Monitor claim a review is
    pending when the graph has no intention of pausing. Additive, like review reasons.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT security_reason FROM steps WHERE id = %s", (step_id,))
        row = cur.fetchone()
        existing = row[0] if row and row[0] else ""
        reasons = [r.strip() for r in existing.split(";") if r.strip()]
        if reason not in reasons:
            reasons.append(reason)
        cur.execute("UPDATE steps SET security_flag = TRUE, security_reason = %s WHERE id = %s",
                    ("; ".join(reasons), step_id))


def record_context(step_id, context):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE steps SET retrieved_context = %s WHERE id = %s",
                    (json.dumps(context, default=str), step_id))


def record_judge_signals(step_id, judge_status, judge_skip_reason=None, cache_hit=None):
    """Structured AgentOps signals per job step (queryable columns)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE steps SET judge_status = %s, judge_skip_reason = %s, cache_hit = %s
            WHERE id = %s
        """, (judge_status, judge_skip_reason, cache_hit, step_id))


def set_stop_reason(run_id, reason):
    """Record WHY a run ended (queryable)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE runs SET stop_reason = %s WHERE id = %s", (reason, run_id))


def set_evaluation_status(run_id, status):
    """Record whether LLM-as-judge evaluation ran for this run."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE runs SET evaluation_status = %s WHERE id = %s", (status, run_id))


def is_cancel_requested(run_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT cancel_requested FROM runs WHERE id = %s", (run_id,))
        row = cur.fetchone()
        return bool(row and row[0])


def request_cancel(run_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))


def save_evaluation(run_id, step_id, evaluation):
    with get_connection() as conn:
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
            json.dumps(evaluation["hallucinated_claims"]), evaluation["notes"], utcnow()
        ))


def finish_run(run_id, status="success", stop_reason=None, error_code=None):
    """
    Finalize a run. stop_reason and error_code are ALWAYS written to reflect THIS
    outcome — including being cleared to NULL on a clean finish — so a run that failed
    on an earlier attempt and then succeeded on retry doesn't keep a stale error_code /
    stop_reason from the failed try.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        # ErrorCode is a str-Enum; store its plain value. (psycopg2 would adapt it to the
        # same string, but be explicit so the stored vocabulary is unambiguous.)
        error_code_val = error_code.value if isinstance(error_code, ErrorCode) else error_code
        cur.execute("""
            UPDATE runs SET ended_at = %s, status = %s,
                stop_reason = %s, error_code = %s,
                total_tokens = (SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                                FROM llm_calls WHERE run_id = %s),
                -- R15: NULL when no call has a known price (unknown is not $0).
                total_cost = (SELECT SUM(cost_usd) FROM llm_calls WHERE run_id = %s)
            WHERE id = %s
        """, (utcnow(), status, redact_secrets(stop_reason), error_code_val, run_id, run_id, run_id))


def _log_llm_attempt(run_id, step_id, operation, prompt, response_text,
                     prompt_tokens, completion_tokens, latency_ms, cost,
                     status, error_message, attempt_number, retry_count, provider_request_id,
                     logical_call_id=None, pricing_version=None):
    """One llm_calls row per HTTP attempt. Three counters make retries explainable
    in the Monitor ("why did this run call Gemini 11 times?"):
      logical_call_id  groups every HTTP attempt of ONE logical LLM call;
      attempt_number   which HTTP attempt this is within that logical call;
      run_attempt      which worker execution attempt of the RUN wrote it.
    cost_usd is the ESTIMATED paid-tier cost (pricing.py), stamped with the model
    and pricing_version so it stays interpretable after prices change."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO llm_calls
            (run_id, step_id, model, prompt, response,
             prompt_tokens, completion_tokens, latency_ms, cost_usd, created_at,
             status, error_message, operation_name, attempt_number, retry_count, provider_request_id,
             logical_call_id, pricing_version, run_attempt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    (SELECT attempt FROM runs WHERE id = %s))
        """, (
            run_id, step_id, settings.gemini_model, prompt, response_text,
            prompt_tokens, completion_tokens, latency_ms, cost, utcnow(),
            status, error_message, operation, attempt_number, retry_count, provider_request_id,
            logical_call_id, pricing_version, run_id
        ))


# HTTP-level retry policy for ONE logical LLM call. Kept deliberately small: the
# worker can additionally retry the WHOLE job, so the two layers multiply. Only
# transient failures are retried here (5xx / timeouts / per-minute rate limits); a
# spent quota or a non-transient error surfaces immediately.
LLM_HTTP_MAX_ATTEMPTS = 4
LLM_BACKOFF_BASE = 1.0      # seconds; exponential: 1, 2, 4 ...
LLM_BACKOFF_CAP = 30.0      # never sleep longer than this between HTTP attempts
_TRANSIENT_CODES = {ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_RATE_LIMITED}


def _backoff_seconds(attempt, exc=None):
    """
    Delay before the next HTTP attempt: the provider's retry hint when it gives one
    (Gemini RetryInfo / Retry-After), else exponential backoff — either way with FULL
    JITTER, so several workers hitting the same outage don't retry in lockstep.
    """
    hint = provider_retry_after(exc) if exc is not None else None
    if hint is not None:
        # honor the provider's floor; add a little jitter on top, respect the cap
        return min(LLM_BACKOFF_CAP, hint + random.uniform(0, 1.0))
    ceiling = min(LLM_BACKOFF_CAP, LLM_BACKOFF_BASE * (2 ** (attempt - 1)))
    return random.uniform(0, ceiling)


def logged_llm_call(prompt, run_id, step_id, operation="llm_call",
                    max_retries=LLM_HTTP_MAX_ATTEMPTS, budget=None):
    from pricing import estimate_cost
    logical_call_id = str(uuid.uuid4())
    last_error = None
    for attempt in range(1, max_retries + 1):
        # Budget enforced HERE at the true unit (one HTTP attempt); retries count.
        if budget is not None:
            if not budget.can_spend():
                raise BudgetExceeded(
                    f"LLM budget reached before attempt {attempt} of {operation}")
            budget.spend()
        start = time.time()
        try:
            result = real_llm_once(prompt)
            latency_ms = int((time.time() - start) * 1000)
            prompt_tokens = result["prompt_tokens"]
            completion_tokens = result["completion_tokens"]
            cost, pricing_version = estimate_cost(settings.gemini_model,
                                                  prompt_tokens, completion_tokens)
            if result.get("usage_missing"):
                log.warning("llm %s: provider response had no usage metadata — "
                            "tokens/cost recorded as 0", operation,
                            extra={"run_id": run_id, "step_id": step_id})
            _log_llm_attempt(
                run_id, step_id, operation, prompt, result["text"],
                prompt_tokens, completion_tokens, latency_ms, cost,
                "success", None, attempt, attempt - 1, result.get("provider_request_id"),
                logical_call_id=logical_call_id, pricing_version=pricing_version,
            )
            return result["text"]
        except BudgetExceeded:
            raise   # budget stop is not transient — propagate, no retry
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            last_error = e
            code = classify_exception(e)
            transient = code in _TRANSIENT_CODES   # quota exhaustion is NOT transient
            _log_llm_attempt(
                run_id, step_id, operation, prompt, None,
                0, 0, latency_ms, 0,
                "failed", redact_secrets(e), attempt, attempt - 1, None,
                logical_call_id=logical_call_id,
            )
            if attempt == max_retries or not transient:
                raise
            wait = _backoff_seconds(attempt, e)
            log.warning("llm %s: %s — retry %s/%s in %.1fs", operation, code,
                        attempt, max_retries - 1, wait)
            time.sleep(wait)
    if last_error:
        raise last_error


def log_tool_call(run_id, step_id, tool_name, tool_input, output, latency_ms,
                  status, error_message, operation_name):
    """Insert one tool_calls trace row (connection always returned to the pool,
    even if the INSERT fails). Tagged with the run's current execution attempt."""
    with get_connection() as conn:
        conn.cursor().execute("""
            INSERT INTO tool_calls
            (run_id, step_id, tool_name, input_json, output_json, latency_ms, status,
             error_message, created_at, operation_name, run_attempt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    (SELECT attempt FROM runs WHERE id = %s))
        """, (
            run_id, step_id, tool_name, json.dumps(tool_input, default=str),
            json.dumps(output, default=str) if output is not None else None,
            latency_ms, status, redact_secrets(error_message), utcnow(), operation_name, run_id
        ))


def logged_tool_call(tool_name, tool_func, tool_input, run_id, step_id,
                     operation=None, swallow_errors=False):
    """
    Run a tool, trace it, and return its result. On error: ALWAYS log the failure
    to the trace, then — by default — RE-RAISE it, so a real failure (e.g. a
    PostgreSQL error) is not silently returned as None and mistaken for an empty
    result. Pass swallow_errors=True only for per-item calls where a failure
    should be caught-and-continued (e.g. one job's tool failing shouldn't kill the
    whole run) — the caller then handles the None.
    """
    start = time.time()
    error = None
    try:
        result = tool_func(tool_input)
        status, error_message = "success", None
    except Exception as e:
        result, status, error_message = None, "failed", str(e)
        error = e
    end = time.time()
    latency_ms = int((end - start) * 1000)

    log_tool_call(run_id, step_id, tool_name, tool_input, result, latency_ms,
                  status, error_message, operation or tool_name)

    # Logged the failure; now surface it unless the caller opted to swallow.
    if error is not None and not swallow_errors:
        raise error
    return result


if __name__ == "__main__":
    run_id = create_run("test resume vs test job")
    step_id = create_step(run_id, "score_job", 1)
    answer = logged_llm_call("Does this resume match this job?", run_id, step_id, operation="test")
    print("Agent got back:", answer)