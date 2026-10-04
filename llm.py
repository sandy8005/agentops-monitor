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
    """The provider returned a response without the fields we need (no text) —
    typically a blocked / safety-filtered / truncated response. Not transient:
    retrying the same prompt yields the same block. The request WAS processed, so
    any usage the provider reported is attached (prompt_tokens/completion_tokens,
    None when missing) and the attempt is priced from it."""

    def __init__(self, message, prompt_tokens=None, completion_tokens=None):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class BudgetExceeded(Exception):
    """Raised when an LLM call is refused because the run's request budget is spent.
    NOT a transient error — must not be retried."""
    pass


class CostLimitReached(BudgetExceeded):
    """The run's hard USD cap would be exceeded by this request's maximum possible
    cost. Refused BEFORE the HTTP request; callers take their rules fallback."""


class CostUnknown(BudgetExceeded):
    """A hard USD cost limit is set but the configured model has no known price, so
    the call's cost could not be accounted for. Refused BEFORE the HTTP request
    (fail closed). Subclasses BudgetExceeded so every caller treats it as a
    non-retryable budget stop and takes its rules fallback."""


class ModelOutputInvalid(ValueError):
    """The model answered, but its output is unusable (not JSON, wrong shape, fails
    schema validation). A degraded-MODEL condition with a rules fallback — distinct
    from a programming error in our own code."""


class QuotaCircuitOpen(RuntimeError):
    """The provider already reported an exhausted (daily/project) quota for this
    model. Calls are refused locally — no HTTP request, no budget reservation —
    until the cooldown passes. Classified as LLM_QUOTA_EXHAUSTED (message says so)."""


# --- Quota circuit breaker ----------------------------------------------------
# A spent DAILY quota does not recover in the provider's 'retryDelay' (Gemini sends
# ~17s even for a per-day limit). Without a breaker every remaining job makes one
# more doomed request, each failing with the same 429. After the first quota
# exhaustion the breaker opens for LLM_QUOTA_COOLDOWN_SECONDS (per worker process);
# callers then take their existing no-LLM fallbacks immediately.
import threading as _threading
_quota_lock = _threading.Lock()
_quota_blocked_until = {}          # model -> epoch seconds


def _quota_cooldown_seconds():
    return float(getattr(settings, "llm_quota_cooldown_seconds", 3600) or 3600)


def quota_blocked(model=None):
    """Seconds remaining on an open quota breaker for `model` (0 if closed)."""
    model = model or settings.gemini_model
    with _quota_lock:
        until = _quota_blocked_until.get(model, 0)
    return max(0.0, until - time.time())


def _open_quota_breaker(model):
    with _quota_lock:
        _quota_blocked_until[model] = time.time() + _quota_cooldown_seconds()
    log.warning("llm quota exhausted for %s — skipping LLM calls for %.0f min (rules fallback)",
                model, _quota_cooldown_seconds() / 60)


class ModelNotConfigured(RuntimeError):
    """No model credentials configured. Terminal for the model path; every caller
    has (or is given) a rules fallback."""


# Exceptions that mean "the MODEL path is degraded" — every caller that has a rules
# fallback may take it for these, and ONLY these. Anything else (a psycopg error, a
# TypeError, a cache-schema regression) is a bug or an infrastructure failure and
# must propagate so it is visible, not silently converted into "rules parser used".
_PROVIDER_MODULES = ("google.genai", "google.api_core", "google.auth", "httpx", "httpcore")


def is_degraded_model_error(exc):
    """True if `exc` is a model-side degradation with a legitimate rules fallback:
    not configured, quota/budget/cost stop, provider unavailable / rate limited, or
    unusable model output. Database and programming errors return False."""
    if isinstance(exc, (ModelNotConfigured, QuotaCircuitOpen, BudgetExceeded,
                        InvalidProviderResponse, ModelOutputInvalid,
                        TimeoutError, ConnectionError)):
        return True
    # Provider SDK / HTTP transport exceptions (google-genai uses httpx). Matched by
    # the defining module so an unrelated error whose MESSAGE happens to contain
    # "timeout" or "503" is never mistaken for a provider outage.
    return (type(exc).__module__ or "").startswith(_PROVIDER_MODULES)


def llm_available():
    """True only if a model call could be attempted now: credentials are configured
    AND the quota breaker is closed. Callers use this to take their rules path
    directly instead of making a doomed request."""
    return bool(settings.gemini_api_key) and quota_blocked() <= 0


def reset_quota_breaker(model=None):
    """For tests / operators."""
    with _quota_lock:
        if model is None:
            _quota_blocked_until.clear()
        else:
            _quota_blocked_until.pop(model, None)



# get_connection is imported (pooled) from database at the top of this module, so
# every `from llm import get_connection` (router.py, agent_store.py, ...) now
# draws from the shared ThreadedConnectionPool instead of opening a fresh socket.


def fake_llm(prompt):
    time.sleep(0.5)
    return {"text": "Apply", "prompt_tokens": len(prompt.split()),
            "completion_tokens": random.randint(5, 20)}


def _usage_tokens(response):
    """(prompt_tokens, completion_tokens) from the provider's usage metadata, each
    None when the provider did not report it. Billing semantics:
      prompt     = prompt_token_count + tool_use_prompt_token_count
      completion = candidates_token_count + thoughts_token_count (thinking tokens
                   are billed at the output rate)
    A MISSING count is never turned into 0 — unknown usage means unknown cost."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    prompt = getattr(usage, "prompt_token_count", None)
    cand = getattr(usage, "candidates_token_count", None)
    thoughts = getattr(usage, "thoughts_token_count", None)
    tool_prompt = getattr(usage, "tool_use_prompt_token_count", None)
    prompt_tokens = None if prompt is None else int(prompt) + int(tool_prompt or 0)
    if cand is None and thoughts is None:
        completion_tokens = None
    else:
        completion_tokens = int(cand or 0) + int(thoughts or 0)
    return prompt_tokens, completion_tokens


def _generation_config():
    """Every request carries a hard output cap. It bounds what one request can cost
    (thinking tokens count against it), which is what makes the pre-dispatch dollar
    reservation a real upper bound."""
    from google.genai import types
    return types.GenerateContentConfig(max_output_tokens=settings.llm_max_output_tokens)


def provider_token_count(prompt):
    """The provider's own input-token count for `prompt` (free endpoint, no
    generation). Raises on any failure — callers fall back to the byte bound."""
    resp = get_client().models.count_tokens(model=settings.gemini_model, contents=prompt)
    total = getattr(resp, "total_tokens", None)
    if total is None:
        raise InvalidProviderResponse("count_tokens returned no total_tokens")
    return int(total)


def input_token_bound(prompt, authorize=None, record=None):
    """(input_token_bound, method) for the pre-dispatch cost reservation.

    Default (LLM_INPUT_TOKEN_BOUND=bytes): the byte bound (pricing.byte_token_bound),
    a PROVEN upper bound computed locally — nothing leaves the process before the
    fenced reservation authorizes the request.

    LLM_INPUT_TOKEN_BOUND=provider tightens it with the provider's count —
    count * (1 + LLM_TOKEN_COUNT_MARGIN) + 16, never above the byte bound. A token
    count SENDS THE PROMPT to the provider, so it is an external request like any
    other and follows the same rule:
      * it needs `authorize` — called first, it proves (durably, under the run row
        lock) that this worker owns the run and that the run is neither erased nor
        cancelled, and raises otherwise. Without an `authorize` callback there is
        no proof, so the provider is NOT contacted and the byte bound is used;
      * it is recorded: `record(status, counted, latency_ms, error)` writes it to
        the trace as an external provider interaction.
    Any count failure, timeout, missing field or implausible value -> the byte
    bound (fail closed). Authorization errors propagate (no request is made)."""
    from pricing import byte_token_bound
    bound = byte_token_bound(len((prompt or "").encode("utf-8")))
    if getattr(settings, "llm_input_token_bound", "bytes") != "provider":
        return bound, "bytes"
    if authorize is None:
        return bound, "bytes"
    authorize()                     # raises ExecutionLost / RunErased / RunCancelled
    start = time.time()
    try:
        counted = provider_token_count(prompt)
    except Exception as e:
        if record is not None:
            record("failed", None, int((time.time() - start) * 1000), e)
        log.warning("count_tokens failed (%s) — reserving with the byte bound",
                    type(e).__name__)
        return bound, "bytes_fallback"
    if record is not None:
        record("success", counted, int((time.time() - start) * 1000), None)
    if counted <= 0 or counted > bound:
        log.warning("count_tokens returned an implausible value — reserving with the "
                    "byte bound")
        return bound, "bytes_fallback"
    margin = float(getattr(settings, "llm_token_count_margin", 0.10) or 0.0)
    tightened = int(counted * (1.0 + margin)) + 16
    return min(bound, tightened), "provider"


def _token_count_recorder(run_id, step_id, operation, prompt_bytes):
    """Trace writer for a provider token count: the request is recorded as a
    tool call (size and outcome only — never the prompt)."""
    def record(status, counted, latency_ms, error):
        log_tool_call(run_id, step_id, "gemini_count_tokens",
                      {"operation": operation, "prompt_bytes": prompt_bytes,
                       "model": settings.gemini_model},
                      None if counted is None else {"total_tokens": counted},
                      latency_ms, status, None if error is None else type(error).__name__,
                      operation)
    return record


def real_llm_once(prompt):
    """Single LLM attempt — no retry. Raises on failure. Retry lives in logged_llm_call.

    Defensive about the SDK response shape: a blocked or abnormal response can have
    text=None or no usage metadata. No text becomes InvalidProviderResponse (with
    whatever usage was reported attached). Missing token counts stay None and are
    flagged usage_missing — the call happened, its cost is UNKNOWN, not zero."""
    response = get_client().models.generate_content(
        model=settings.gemini_model, contents=prompt, config=_generation_config()
    )
    prompt_tokens, completion_tokens = _usage_tokens(response)
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
            f"provider returned no text (block/finish reason: {reason or 'unknown'})",
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    request_id = None
    try:
        request_id = getattr(response, "response_id", None) or getattr(response, "_request_id", None)
    except Exception:
        request_id = None
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_missing": prompt_tokens is None or completion_tokens is None,
        "provider_request_id": request_id
    }


def create_run_tx(cur, input_summary, resume_id=None, target_role=None,
                  location=None, work_mode=None, employment_type=None, user_id=None,
                  goal=None, mode="agent"):
    """
    Transactional create_run: INSERT the run on the CALLER'S cursor and return its
    id WITHOUT committing. Lets the API create the run and enqueue its worker job in
    ONE transaction (both commit or both roll back), so a run is never left
    'running' with no queue job. The caller owns commit / rollback / close.

    Every run is an AGENT run (the only engine; runs_mode_chk rejects anything
    else for a new row). `goal` (an AgentGoal) is stored with its budget and cost
    cap in the same INSERT, so no row ever exists in a half-configured state.
    """
    if mode != "agent":
        raise ValueError(f"mode {mode!r} is retired; the controller agent is the only engine")
    goal_json = budget = max_cost = None
    if goal is not None:
        goal_json = json.dumps(goal.model_dump())
        budget = goal.limits.max_llm_calls
        max_cost = goal.limits.max_cost_usd
    # A new run is 'queued', NOT 'running': at creation it is only waiting in
    # job_queue for a worker to claim it. started_at is left NULL and is stamped
    # only when the worker actually begins executing (see _mark_run_running in
    # agent_store.begin_execution), so queue-wait time is never counted as execution time.
    cur.execute("""
        INSERT INTO runs (started_at, status, input_summary, resume_id,
                          target_role, location, work_mode, employment_type, user_id,
                          mode, goal_json, llm_call_budget, max_cost_usd)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
    """, (None, "queued", input_summary, resume_id,
          target_role, location, work_mode, employment_type, user_id,
          mode, goal_json, budget, max_cost))
    return cur.fetchone()[0]


def create_run(input_summary, resume_id=None, target_role=None,
               location=None, work_mode=None, employment_type=None, user_id=None,
               goal=None):
    """Create an agent run in its OWN transaction (thin wrapper over create_run_tx)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        run_id = create_run_tx(cur, input_summary, resume_id=resume_id,
                               target_role=target_role, location=location,
                               work_mode=work_mode, employment_type=employment_type,
                               user_id=user_id, goal=goal)
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
                -- NULL when no call has a known price (unknown is not $0) ...
                total_cost = (SELECT SUM(cost_usd) FROM llm_calls
                              WHERE run_id = %s AND cost_status = 'priced'),
                -- ... and a PARTIAL total is marked as partial.
                unknown_cost_calls = (SELECT COUNT(*) FROM llm_calls WHERE run_id = %s
                                      AND cost_status = 'unknown')
            WHERE id = %s
        """, (utcnow(), status, redact_secrets(stop_reason), error_code_val,
              run_id, run_id, run_id, run_id))


def _log_llm_attempt(run_id, step_id, operation, prompt, response_text,
                     prompt_tokens, completion_tokens, latency_ms, cost,
                     status, error_message, attempt_number, retry_count, provider_request_id,
                     logical_call_id=None, pricing_version=None, cost_status="priced",
                     cost_upper_bound=None, usage_missing=False, reservation=None,
                     generation=None):
    """One llm_calls row per HTTP attempt. Three counters make retries explainable
    in the Monitor ("why did this run call Gemini 11 times?"):
      logical_call_id  groups every HTTP attempt of ONE logical LLM call;
      attempt_number   which HTTP attempt this is within that logical call;
      run_attempt      which worker execution attempt of the RUN wrote it.

    cost_status is explicit: 'priced' (cost_usd is the estimate from provider
    usage), 'unknown' (usage missing or the request failed after dispatch —
    cost_usd NULL, cost_upper_bound_usd = what was reserved), or 'not_billed' (the
    provider rejected the request before doing work).

    If the attempt holds a cost reservation it is SETTLED IN THE SAME TRANSACTION
    as this insert: the in-flight amount moves into the recorded call exactly once,
    with no window in which it is counted twice or not at all."""
    reservation_id = (reservation or {}).get("reservation_id")
    with get_connection() as conn:
        cur = conn.cursor()
        if reservation_id is not None:
            # Lock order matches the reservation path: run row first.
            cur.execute("SELECT 1 FROM runs WHERE id = %s FOR UPDATE", (run_id,))
            cur.execute("""
                UPDATE llm_cost_reservations r SET status = 'settled', settled_at = NOW()
                FROM (SELECT id, status AS old_status FROM llm_cost_reservations
                      WHERE id = %s FOR UPDATE) o
                WHERE r.id = o.id AND o.old_status IN ('open', 'abandoned')
                RETURNING o.old_status, r.amount_usd
            """, (reservation_id,))
            settled = cur.fetchone()
            if settled and settled[0] == "open" and settled[1] is not None:
                cur.execute("UPDATE runs SET cost_reserved_usd = GREATEST(0, cost_reserved_usd - %s) "
                            "WHERE id = %s", (settled[1], run_id))
        cur.execute("""
            INSERT INTO llm_calls
            (run_id, step_id, model, prompt, response,
             prompt_tokens, completion_tokens, latency_ms, cost_usd, created_at,
             status, error_message, operation_name, attempt_number, retry_count, provider_request_id,
             logical_call_id, pricing_version, cost_status, cost_upper_bound_usd, usage_missing,
             reservation_id, execution_generation, run_attempt)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, (SELECT attempt FROM runs WHERE id = %s))
        """, (
            run_id, step_id, settings.gemini_model, prompt, response_text,
            prompt_tokens, completion_tokens, latency_ms, cost, utcnow(),
            status, error_message, operation, attempt_number, retry_count, provider_request_id,
            logical_call_id, pricing_version, cost_status, cost_upper_bound, bool(usage_missing),
            reservation_id, generation, run_id
        ))


# HTTP-level retry policy for ONE logical LLM call. Kept deliberately small: the
# worker can additionally retry the WHOLE job, so the two layers multiply. Only
# transient failures are retried here (5xx / timeouts / per-minute rate limits); a
# spent quota or a non-transient error surfaces immediately.
LLM_HTTP_MAX_ATTEMPTS = 4
LLM_BACKOFF_BASE = 1.0      # seconds; exponential: 1, 2, 4 ...
LLM_BACKOFF_CAP = 30.0      # never sleep longer than this between HTTP attempts
_TRANSIENT_CODES = {ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_RATE_LIMITED}
_SLEEP_SLICE = 0.5          # retry sleeps wake this often to check run ownership

# Failures where the provider REJECTED the request before doing any work, so
# nothing was billed. Everything else that fails after dispatch (timeout,
# connection reset, 5xx, unexpected SDK error) may have been processed and billed:
# its cost is UNKNOWN, bounded by the reservation.
_NOT_BILLED_CODES = {ErrorCode.LLM_RATE_LIMITED, ErrorCode.LLM_QUOTA_EXHAUSTED,
                     ErrorCode.LLM_NOT_CONFIGURED}
_NOT_BILLED_MARKERS = ("401", "403", "400 ", "400:", "invalid_argument", "permission_denied",
                       "unauthenticated", "api key not valid")


def _failure_not_billed(exc, code):
    if code in _NOT_BILLED_CODES:
        return True
    low = str(exc).lower()
    return any(m in low for m in _NOT_BILLED_MARKERS)


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


_DISPATCH_CHECK_EVERY = 4   # retry sleeps re-check cancel/erasure every 4 slices (2s)


def _interruptible_sleep(seconds, budget):
    """Sleep between HTTP attempts. With a durable budget the sleep is cut into
    _SLEEP_SLICE pieces: run ownership (in-process) is checked before each one,
    and the durable dispatch authorization (generation + not erased + not
    cancelled, budget.authorize_dispatch) every _DISPATCH_CHECK_EVERY slices and
    once more at the end. A worker that lost the run, or whose run was cancelled,
    stops during the backoff instead of sleeping up to 30s first. (The next
    reservation enforces the same rule atomically regardless.)"""
    seconds = max(0.0, float(seconds))
    check = getattr(budget, "check_owner", None)
    authorize = getattr(budget, "authorize_dispatch", None)
    if check is None:
        time.sleep(seconds)
        return
    remaining, n = seconds, 0
    while remaining > 0:
        check()                                  # raises ExecutionLost
        if authorize is not None and n % _DISPATCH_CHECK_EVERY == 0:
            authorize()                          # raises RunCancelled / RunErased
        step = min(_SLEEP_SLICE, remaining)
        time.sleep(step)
        remaining -= step
        n += 1
    check()
    if authorize is not None:
        authorize()


def logged_llm_call(prompt, run_id, step_id, operation="llm_call",
                    max_retries=LLM_HTTP_MAX_ATTEMPTS, budget=None):
    """One LOGICAL model call: up to max_retries HTTP attempts, each one
      1. budget-checked AND reserved immediately before dispatch — with a durable
         budget (budget.reserve_attempt) this is one fenced transaction that proves
         ownership and reserves the request's MAXIMUM cost against the USD cap;
      2. recorded as exactly one llm_calls row whose reservation is settled in the
         same transaction, with an explicit cost_status.
    """
    from pricing import estimate_cost, max_request_cost
    logical_call_id = str(uuid.uuid4())
    last_error = None
    if not settings.gemini_api_key:
        raise ModelNotConfigured("GEMINI_API_KEY is not set — model call skipped")
    remaining = quota_blocked()
    if remaining > 0:
        # Refused locally: no HTTP request and no budget reservation.
        raise QuotaCircuitOpen(
            f"quota exhausted (circuit open for {remaining:.0f}s more) — {operation} skipped")
    prompt_bytes = len((prompt or "").encode("utf-8"))
    reserve = getattr(budget, "reserve_attempt", None)
    generation = getattr(budget, "generation", None)
    # Counted once per LOGICAL call (the prompt is identical across HTTP retries),
    # and only when a reservation will actually be made. A provider-side count is
    # itself an external request: it is authorized by the same durable rule as the
    # reservation BEFORE the prompt leaves the process, and traced.
    in_bound = None
    if reserve is not None:
        in_bound = input_token_bound(
            prompt, authorize=getattr(budget, "authorize_dispatch", None),
            record=_token_count_recorder(run_id, step_id, operation, prompt_bytes))[0]
    for attempt in range(1, max_retries + 1):
        # Upper bound of THIS request's cost, priced at dispatch time.
        projected = max_request_cost(settings.gemini_model, prompt_bytes,
                                     settings.llm_max_output_tokens, input_tokens=in_bound)
        reservation = None
        if reserve is not None:
            reservation = reserve(projected, operation)   # raises BudgetExceeded family /
                                                          # ExecutionLost / RunCancelled;
                                                          # never returns None
        elif budget is not None:
            if not budget.can_spend():
                raise BudgetExceeded(
                    f"LLM budget reached before attempt {attempt} of {operation}")
            budget.spend()
        start = time.time()
        try:
            result = real_llm_once(prompt)
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            last_error = e
            code = classify_exception(e)
            transient = code in _TRANSIENT_CODES   # quota exhaustion is NOT transient
            pt = getattr(e, "prompt_tokens", None)
            ct = getattr(e, "completion_tokens", None)
            if pt is not None and ct is not None:
                cost, pricing_version = estimate_cost(settings.gemini_model, pt, ct)
                cost_status = "priced" if cost is not None else "unknown"
            elif _failure_not_billed(e, code):
                cost, pricing_version, cost_status = None, None, "not_billed"
            else:
                # Failed AFTER dispatch with no usage: the provider may have
                # processed (and billed) it. Unknown, bounded by the reservation.
                cost, pricing_version, cost_status = None, None, "unknown"
            _log_llm_attempt(
                run_id, step_id, operation, prompt, None,
                pt, ct, latency_ms, cost,
                "failed", redact_secrets(e), attempt, attempt - 1, None,
                logical_call_id=logical_call_id, pricing_version=pricing_version,
                cost_status=cost_status,
                cost_upper_bound=projected if cost_status == "unknown" else None,
                usage_missing=cost_status == "unknown", reservation=reservation,
                generation=generation,
            )
            if code == ErrorCode.LLM_QUOTA_EXHAUSTED:
                _open_quota_breaker(settings.gemini_model)
            if attempt == max_retries or not transient:
                raise
            wait = _backoff_seconds(attempt, e)
            log.warning("llm %s: %s — retry %s/%s in %.1fs", operation, code,
                        attempt, max_retries - 1, wait)
            _interruptible_sleep(wait, budget)
            continue
        latency_ms = int((time.time() - start) * 1000)
        prompt_tokens = result["prompt_tokens"]
        completion_tokens = result["completion_tokens"]
        cost, pricing_version = estimate_cost(settings.gemini_model,
                                              prompt_tokens, completion_tokens)
        if result.get("usage_missing"):
            log.warning("llm %s: provider response had no usage metadata — cost recorded "
                        "as UNKNOWN (bounded by the reservation)", operation,
                        extra={"run_id": run_id, "step_id": step_id})
        _log_llm_attempt(
            run_id, step_id, operation, prompt, result["text"],
            prompt_tokens, completion_tokens, latency_ms, cost,
            "success", None, attempt, attempt - 1, result.get("provider_request_id"),
            logical_call_id=logical_call_id, pricing_version=pricing_version,
            cost_status="priced" if cost is not None else "unknown",
            cost_upper_bound=projected if cost is None else None,
            usage_missing=bool(result.get("usage_missing")), reservation=reservation,
            generation=generation,
        )
        return result["text"]
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