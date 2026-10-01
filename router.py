"""
Shared tool library for the agent engine (agent_tools / agent_loop):
resume loading + parsing, per-job evaluation (requirements, scoring, judge,
review flags), human-decision application and the requirements/parse caches.

The legacy fixed-sequence pipeline (autonomous_graph.py) and its search / rank /
advice / dispatch functions were retired: the controller agent is the single
execution engine. Everything here is traced through create_step / logged_*.
"""
from timeutil import utcnow
import hashlib
import json
from datetime import timedelta

from settings import settings
from error_codes import ErrorCode

from parser import parse_resume
from job_parser import extract_requirements
from scorer import calculate_match_score
from schemas import JobDecision
from cache_version import parse_cache_version, reqs_cache_version
from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE
from logging_config import get_logger
from llm import (
    create_step, finish_step, fail_step, logged_llm_call, logged_tool_call,
    record_score, record_context, flag_for_review, get_connection,
    is_cancel_requested, record_judge_signals
)
log = get_logger(__name__)


def is_infrastructure_error(exc):
    """Errors that must NEVER be absorbed by a per-item fallback: lost execution
    ownership and database failures. They propagate so the run fails (or is
    retried) with the right code instead of continuing on silently degraded data."""
    import agent_store
    import run_lock
    if isinstance(exc, (agent_store.ExecutionLost, run_lock.ExecutionLost)):
        return True
    return (type(exc).__module__ or "").startswith(("psycopg2", "psycopg"))


def _hash(text):
    """Stable hash for cache keys (#12) — detects when resume/job text changed."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def resume_content_hash(resume_text):
    """STABLE content identity of a resume: sha256 of the text, independent of any
    parser/schema/model version. The parse cache is keyed by
    (content_hash, cache_version), so erasing a resume can delete EVERY cached parse
    of it — including ones produced by older parser versions (privacy.erase_resume)."""
    return hashlib.sha256((resume_text or "").encode("utf-8")).hexdigest()


def _reqs_cache_key(title, description):
    """Versioned requirements-cache key: TITLE + description + reqs/schema/model
    version. Including the title distinguishes postings that share a description
    but differ by role, and lets a title-borne requirement signal affect the key."""
    return _hash(f"{title}\n{description}|{reqs_cache_version()}")


def _llm_allowed(state):
    """The run's model policy allows a model call AND one could succeed now
    (credentials configured, quota breaker closed, run budget left)."""
    from llm import llm_available
    return (getattr(state, "model_policy", "auto") != "rules_only"
            and llm_available() and not state.budget_exceeded())


def load_resume(state, run_id):
    """Load the resume document from DB (0 LLM calls)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT resume_text, is_deleted FROM resumes WHERE id = %s",
                    (state.resume_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            state.error = f"resume {state.resume_id} not found or empty"
            return
        # R22 defense in depth: an erased resume is never used as input.
        if row[1] or row[0] == "[erased]":
            state.error = f"resume {state.resume_id} was erased"
            return
        state.resume_text = row[0]


def _parse_cache_get(content_hash):
    """Look up a parse of this exact resume text under the CURRENT cache version.
    0 LLM calls on hit. Parses under older versions are never served (stale)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parsed_json FROM parsed_resume_cache "
                    "WHERE content_hash = %s AND cache_version = %s",
                    (content_hash, parse_cache_version()))
        row = cur.fetchone()
        if not row:
            return None
        val = json.loads(row[0])
        # Ignore a poisoned/legacy entry (e.g. a `null` written before parse_resume was
        # hardened) so it's re-parsed instead of flowing downstream as None.
        return val if isinstance(val, dict) else None


def _parse_cache_put(content_hash, parsed):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO parsed_resume_cache (content_hash, cache_version, parsed_json)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash, cache_version) DO NOTHING
        """, (content_hash, parse_cache_version(), json.dumps(parsed)))


def do_parse_resume(state, run_id):
    """Parse the resume — but reuse the cache if we've parsed this exact text (#1)."""
    step_id = create_step(run_id, "parse_resume", len(state.completed_actions))
    try:
        rhash = resume_content_hash(state.resume_text)
        cached = _parse_cache_get(rhash)
        if cached is not None:
            state.parsed_resume = cached
            finish_step(step_id, "success")
            log.info("parsed resume served from cache — 0 LLM calls", extra={"step_id": step_id})
            return
        parsed = None
        if _llm_allowed(state):
            from llm import ModelOutputInvalid, is_degraded_model_error
            try:
                parsed = parse_resume(state.resume_text, run_id, step_id, budget=state)
                # A parse that yields no usable dict must NOT flow downstream as None;
                # never cache a non-dict (it would poison the parse cache).
                if not isinstance(parsed, dict):
                    raise ModelOutputInvalid("resume parse returned no usable data")
            except Exception as llm_err:
                # ONLY a degraded model (not configured, quota/budget/cost stop,
                # provider outage or rate limit, unusable output) falls back to the
                # rules parser. A database error, a TypeError or any other bug in our
                # own code propagates and fails the step VISIBLY instead of being
                # disguised as "rules parser used".
                if not is_degraded_model_error(llm_err):
                    raise
                from error_codes import summarize_error
                log.warning("model resume parse unavailable (%s) — using rules parse",
                            summarize_error(llm_err), extra={"step_id": step_id})
                parsed = None
            if parsed is not None:
                # Outside the fallback try: a cache-write failure is an infrastructure
                # error, never a reason to discard a good model parse silently.
                _parse_cache_put(rhash, parsed)
        if parsed is None:
            # N01: deterministic extraction — zero model calls. Never cached under
            # the model-parse key, so a later model-enabled run still gets a model parse.
            from rule_resume_parser import parse_resume_rules
            parsed = parse_resume_rules(state.resume_text)
            log.info("resume parsed by rules (%d skills) — 0 LLM calls",
                     len(parsed["skills"]), extra={"step_id": step_id})
        state.parsed_resume = parsed
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        if is_infrastructure_error(e):
            raise
        state.error = f"parse failed: {e}"


# Provider fetch status -> run error code. Transient failures come first: if ANY
# provider failed transiently, a retry of the run can still succeed.
_TRANSIENT_PROVIDER = {
    "rate_limited": ErrorCode.JOB_SOURCE_RATE_LIMITED,
    "server_error": ErrorCode.JOB_SOURCE_UNAVAILABLE,
    "network_error": ErrorCode.JOB_SOURCE_UNAVAILABLE,
    "failed": ErrorCode.JOB_SOURCE_UNAVAILABLE,   # unclassified provider-side failure
}
_TERMINAL_PROVIDER = {
    "auth_error": ErrorCode.JOB_SOURCE_AUTH_FAILED,
    "missing_keys": ErrorCode.JOB_SOURCE_AUTH_FAILED,
    "invalid_response": ErrorCode.JOB_SOURCE_INVALID_RESPONSE,
    "http_error": ErrorCode.JOB_SOURCE_INVALID_RESPONSE,
}


def provider_failure_code(provider_status):
    """
    Classify "every live provider failed" into ONE run-level error code:
      rate limit                  -> JOB_SOURCE_RATE_LIMITED   (retryable)
      network / 5xx / unexpected  -> JOB_SOURCE_UNAVAILABLE    (retryable)
      bad credentials / no keys   -> JOB_SOURCE_AUTH_FAILED    (terminal)
      malformed body / other 4xx  -> JOB_SOURCE_INVALID_RESPONSE (terminal)
    A provider that answered with zero jobs is NOT a failure (the run ends as
    no_matches), so this is only consulted when nothing succeeded.
    """
    statuses = list((provider_status or {}).values())
    for st in ("rate_limited",):
        if st in statuses:
            return _TRANSIENT_PROVIDER[st]
    for st in statuses:
        if st in _TRANSIENT_PROVIDER:
            return _TRANSIENT_PROVIDER[st]
    for st in statuses:
        if st in _TERMINAL_PROVIDER:
            return _TERMINAL_PROVIDER[st]
    return ErrorCode.JOB_SOURCE_UNAVAILABLE


# --- requirements cache (#2, #12): a job's requirements don't depend on the
# resume, so extract once per (title+description) and reuse. Each row also records
# PROVENANCE — how it was produced (llm vs rule_based) and under which model/version
# — so a cache hit is traceable and its quality is known. ---

# A rule-based extraction is a DEGRADED result (produced because the LLM was down or
# the budget was spent). It is cached only briefly and is never allowed to shadow a
# later LLM extraction; an LLM row is durable and is never downgraded.
RULE_BASED_CACHE_TTL = timedelta(days=1)


def _reqs_cache_get(desc_hash):
    """
    Return (reqs, provenance) on hit, or (None, None) on miss. provenance is a dict
    {"extraction_method":..., "source_model":...} describing how the cached row was
    produced, so callers can trace/trust it without re-extracting. Expired rows
    (rule-based fallbacks past their TTL) are treated as a miss.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT reqs_json, extraction_method, source_model
            FROM job_reqs_cache
            WHERE desc_hash = %s AND (expires_at IS NULL OR expires_at > NOW())
        """, (desc_hash,))
        row = cur.fetchone()
        if not row:
            return (None, None)
        provenance = {"extraction_method": row[1], "source_model": row[2]}
        return (json.loads(row[0]), provenance)


def _reqs_cache_put(desc_hash, reqs, extraction_method):
    """
    Store a requirements row WITH provenance:
      - extraction_method : 'llm' or 'rule_based'
      - source_model      : the MODEL NAME for LLM extraction, NULL for rule_based.
      - expires_at        : NULL (durable) for LLM rows; now + RULE_BASED_CACHE_TTL
                            for rule-based fallbacks.

    Conflict policy — quality only ever goes UP:
      * an incoming LLM row replaces an existing rule-based row (upgrade);
      * an incoming rule-based row may refresh an existing rule-based row;
      * an existing LLM row is NEVER overwritten by a rule-based one (no downgrade)
        and is kept on an LLM-vs-LLM conflict (first writer wins).
    """
    from cache_version import model_version
    source_model = model_version() if extraction_method == "llm" else None
    expires_at = None if extraction_method == "llm" else utcnow() + RULE_BASED_CACHE_TTL
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO job_reqs_cache
                (desc_hash, reqs_json, cache_version, extraction_method, source_model,
                 expires_at, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (desc_hash) DO UPDATE SET
                reqs_json = EXCLUDED.reqs_json,
                cache_version = EXCLUDED.cache_version,
                extraction_method = EXCLUDED.extraction_method,
                source_model = EXCLUDED.source_model,
                expires_at = EXCLUDED.expires_at,
                created_at = NOW()
            WHERE job_reqs_cache.extraction_method IS DISTINCT FROM 'llm'
        """, (desc_hash, json.dumps(reqs), reqs_cache_version(),
              extraction_method, source_model, expires_at))


def _get_requirements(state, job, run_id, step_id):
    """
    Requirements for one job, cache-aware and quality-aware. Returns
    (requirements, cache_hit, method).

      * durable LLM cache hit            -> use it, 0 LLM calls
      * rule-based cache hit             -> try to UPGRADE via the LLM when budget
                                            allows; on failure keep the cached rules
      * miss                             -> LLM when budget allows, else rules
    """
    from rule_requirements import extract_requirements_rule_based
    dhash = _reqs_cache_key(job["title"], job["description"])
    cached, provenance = _reqs_cache_get(dhash)
    method = (provenance or {}).get("extraction_method")
    if cached is not None and method == "llm":
        log.info("requirements served from cache [llm] — 0 LLM calls", extra={"step_id": step_id})
        return cached, True, "llm"

    if _llm_allowed(state):
        from llm import is_degraded_model_error
        try:
            reqs = extract_requirements(job, run_id, step_id, budget=state)
        except Exception as extract_err:
            # ONLY a degraded model (not configured, quota/budget/cost stop, provider
            # outage, unusable output) takes the rules fallback. A database error,
            # a programming error or lost ownership propagates and fails visibly.
            if not is_degraded_model_error(extract_err):
                raise
            from error_codes import summarize_error
            log.warning("requirements LLM extraction degraded (%s) — rules fallback",
                        summarize_error(extract_err), extra={"step_id": step_id})
        else:
            # Outside the fallback: a cache-write failure is infrastructure, never a
            # reason to throw away a good model extraction.
            _reqs_cache_put(dhash, reqs, "llm")
            if cached is not None:
                log.info("requirements upgraded rule_based -> llm", extra={"step_id": step_id})
            return reqs, False, "llm"

    if cached is not None:
        # Degraded but still valid fallback that hasn't expired — reuse it, don't rewrite.
        log.info("requirements served from cache [%s]", method, extra={"step_id": step_id})
        return cached, True, method or "rule_based"

    reqs = extract_requirements_rule_based(job)
    _reqs_cache_put(dhash, reqs, "rule_based")
    log.info("requirements via rules — 0 LLM calls", extra={"step_id": step_id})
    return reqs, False, "rule_based"


_REAL_DECISIONS = {"Apply", "Maybe", "Skip"}


def _compute_final_decision(score_decision, llm_decision, human_decision=None):
    """
    The AUTHORITATIVE decision that controls downstream ranking/advice:
      human_decision  if a human reviewed (overrides everything)
      else llm_decision  if the judge produced a real Apply/Maybe/Skip
      else score_decision  (deterministic fallback)
    """
    if human_decision in _REAL_DECISIONS:
        return human_decision
    if llm_decision in _REAL_DECISIONS:
        return llm_decision
    return score_decision


def _store_final_decision(step_id, final_decision):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE steps SET final_decision = %s WHERE id = %s",
                    (final_decision, step_id))


def _parse_decision(raw):
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return JobDecision(**json.loads(cleaned)).decision.value


def _step_needs_review(step_id):
    """
    True if the step is flagged for human review for ANY reason (the additive
    needs_human_review flag). This is the source of truth for whether the run pauses
    — score disagreement, prompt injection, hallucination, and evaluation failure all
    set it via flag_for_review, and record_score never clears it.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT needs_human_review FROM steps WHERE id = %s", (step_id,))
        row = cur.fetchone()
        return bool(row and row[0])


def _step_review_reason(step_id):
    """The accumulated review_reason for a step (e.g. 'score_disagreement;
    possible_prompt_injection(job)'), so the reviewer is told WHY the run paused."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT review_reason FROM steps WHERE id = %s", (step_id,))
        row = cur.fetchone()
        return row[0] if row else None


def _evaluation_selection(run_id, step_id, flagged):
    """'flagged' (always evaluated), 'sampled', or None. Sampling is DETERMINISTIC
    per (run, step) so a retried attempt makes the same choice."""
    if flagged:
        return "flagged"
    rate = settings.eval_unflagged_sample_rate
    if rate >= 1.0:
        return "sampled"
    if rate <= 0.0:
        return None
    h = int(hashlib.sha256(f"{run_id}:{step_id}".encode()).hexdigest()[:8], 16)
    return "sampled" if (h / 0xFFFFFFFF) < rate else None


def do_process_job(state, run_id):
    """
    Process ONE job: cancellation check → requirements (structured) →
    requirements (cached) → deterministic score (traced) → Gemini judge ONLY
    if score is in the uncertain middle band (20-80) → evaluator on risky jobs.
    Always advances current_job_index (finally), so the loop can't get stuck.

    KEY: if the judge fails (quota/API), the job KEEPS its deterministic score
    and ranks anyway — a failed judge no longer discards the whole job.
    """
    # Cooperative cancellation: if the user hit Cancel, stop before this job.
    if is_cancel_requested(run_id):
        state.cancelled = True
        return

    job = state.jobs[state.current_job_index]
    step_id = create_step(run_id, job["title"], len(state.completed_actions))
    try:
        # 0. Untrusted job text is scanned EVERY time — not only when the LLM
        #    extractor runs — so an injection attempt is still caught (and pauses the
        #    run for review) when requirements come from the cache.
        #    Policy (prompt_safety.py): HIGH-confidence patterns request human review;
        #    LOW-confidence phrases ("system prompt", "assistant:" — normal in AI and
        #    security job ads) are recorded as a security signal only.
        from prompt_safety import detect_injection, apply_injection_policy
        apply_injection_policy(
            detect_injection(f"{job.get('title', '')}\n{job.get('description', '')}"),
            step_id, source="job", run_id=run_id)

        # 1. requirements — structured, optional-aware, cache- and quality-aware.
        requirements, cache_hit, _method = _get_requirements(state, job, run_id, step_id)

        # 2. deterministic score — traced (0 LLM calls), runs FIRST (#4)
        user_input = {
            "target_role": state.target_role, "location": state.location,
            "work_mode": state.work_mode, "employment_type": state.employment_type
        }
        score_result = logged_tool_call(
            "calculate_match_score",
            lambda p: calculate_match_score(p["parsed"], p["reqs"], p["resume"], p["job"], p["ui"]),
            {"parsed": state.parsed_resume, "reqs": requirements,
             "resume": state.resume_text, "job": job, "ui": user_input},
            run_id, step_id, operation="score")
        score = score_result["score"]

        # 2a. Quality gates that a confident score must NOT bypass (R08, R12):
        #     a required skill the resume only mentions in NEGATED form, or an
        #     extraction that produced no usable requirements at all.
        if score_result.get("negated_skills"):
            flag_for_review(step_id, reason="negated_skill_mention")
        if score_result.get("uncertain_skills"):
            flag_for_review(step_id, reason="uncertain_skill_evidence")
        if score_result.get("insufficient_requirements"):
            flag_for_review(step_id, reason="insufficient_requirements")

        # 2b. Experience discrepancy: the scorer already used the CONSERVATIVE years
        #     (the lower of stated vs. summed). If the questionable stated value would
        #     have produced a DIFFERENT decision, a human must look — the number that
        #     decides this job is exactly the one we can't trust.
        if isinstance((state.parsed_resume or {}).get("experience_discrepancy"), dict):
            stated_result = calculate_match_score(state.parsed_resume, requirements,
                                                  state.resume_text, job, user_input,
                                                  experience_mode="stated")
            if stated_result["decision"] != score_result["decision"]:
                flag_for_review(step_id, reason="experience_discrepancy")

        # 3. Gemini judge ONLY in the uncertain middle band (#5,6,7).
        #    judge_status is the single truth about whether the judge RAN:
        #      ran | invalid_output | unavailable | skipped_budget | not_needed | disabled
        result = None
        judge_status = "not_needed"
        judge_skip_reason = None
        in_uncertain_band = 20 <= score <= 80
        if not in_uncertain_band:
            # Extreme score — judge adds little; skip it (honest, not faked).
            judge_skip_reason = "score_extreme_low" if score < 20 else "score_extreme_high"
            llm_decision = f"skipped ({judge_skip_reason})"
        elif getattr(state, "model_policy", "auto") == "rules_only":
            # The user CHOSE a model-free run: the deterministic score decides, and
            # that is recorded as such (not as an outage requiring review).
            judge_status = "disabled"
            judge_skip_reason = "rules_only"
            llm_decision = "skipped (rules_only)"
        elif not _llm_allowed(state):
            # Uncertain band BY CONSTRUCTION needs a second opinion — none is
            # available, so the deterministic score alone must not decide silently.
            from llm import quota_blocked
            if state.budget_exceeded():
                why = "budget"
            elif quota_blocked():
                why = "quota"
            elif not settings.gemini_api_key:
                why = "not_configured"
            else:
                why = "unavailable"
            judge_status = "skipped_budget" if why == "budget" else "unavailable"
            judge_skip_reason = why
            llm_decision = f"skipped ({why})"
            flag_for_review(step_id, reason=f"judge_unavailable({why})")
        else:
            from agent import build_prompt
            evidence = {"matched_in_resume": score_result["matched_skills"],
                        "missing_from_resume": score_result["missing_skills"]}
            prompt = build_prompt(state.resume_text, state.parsed_resume, job, evidence, requirements)
            try:
                result = logged_llm_call(prompt, run_id, step_id, operation="job_judge", budget=state)
                try:
                    llm_decision = _parse_decision(result)
                    judge_status = "ran"
                except Exception as parse_err:
                    # The call succeeded but the structured output was invalid — an
                    # AI-quality failure, not a judgment. Ask a human. (Logged by
                    # exception TYPE only: the message can quote model output.)
                    judge_status = "invalid_output"
                    judge_skip_reason = "parse_error"
                    llm_decision = "Unknown"
                    result = None          # nothing valid for the evaluator to grade
                    flag_for_review(step_id, reason="judge_invalid_output")
                    log.warning("judge returned invalid structured output (%s)",
                                type(parse_err).__name__,
                                extra={"run_id": run_id, "step_id": step_id})
            except Exception as judge_err:
                from llm import is_degraded_model_error
                if not is_degraded_model_error(judge_err):
                    raise
                # The job KEEPS its deterministic score (it is not discarded), but the
                # decision quality is DEGRADED: an uncertain-band job without its
                # second opinion goes to a human. Other jobs keep processing.
                judge_status = "unavailable"
                judge_skip_reason = "judge_unavailable"
                llm_decision = "skipped (judge_unavailable)"
                flag_for_review(step_id, reason="judge_unavailable")
                from error_codes import summarize_error
                log.warning("judge unavailable (%s) — keeping score, flagged for review",
                            summarize_error(judge_err), extra={"run_id": run_id, "step_id": step_id})

        # record_score sets the score-vs-LLM DISAGREEMENT review flag (additively) as a
        # side effect. Its return value is only that ONE trigger, so it is deliberately
        # NOT used for routing — the authoritative DB flag (_step_needs_review, read
        # below AFTER every trigger incl. the evaluator) is.
        record_score(step_id, score, score_result["decision"],
                     llm_decision, breakdown=score_result["breakdown"],
                     breakdown_max=score_result.get("breakdown_max"))

        # Compute the authoritative decision (human > llm > score). Defaults to the
        # best automated signal now; a human review overrides it later.
        final_decision = _compute_final_decision(score_result["decision"], llm_decision)
        _store_final_decision(step_id, final_decision)

        # Structured AgentOps signals (queryable columns).
        record_judge_signals(step_id, judge_status, judge_skip_reason, cache_hit)

        # --- Evaluator. With evaluate=true, EVERY successful judge result is
        # eligible — not only already-flagged ones (hallucination in an otherwise
        # normal-looking decision is exactly what it exists to catch). Flagged
        # decisions are always evaluated; unflagged ones at
        # EVAL_UNFLAGGED_SAMPLE_RATE (default 1.0 = all), budget permitting.
        eval_selection = None
        if (state.evaluate and result is not None
                and llm_decision in ("Apply", "Maybe", "Skip")):
            eval_selection = _evaluation_selection(run_id, step_id, _step_needs_review(step_id))
        if eval_selection and _llm_allowed(state):
            try:
                from evaluator import evaluate_decision
                from llm import save_evaluation
                eval_result = evaluate_decision(state.resume_text, job, result, run_id, step_id, budget=state)
                save_evaluation(run_id, step_id, eval_result)
                rel = eval_result["relevance_score"]
                faith = eval_result["faithfulness_score"]
                comp = eval_result["completeness_score"]
                if eval_result["hallucination_detected"] or min(rel, faith, comp) <= 2:
                    reason = ("hallucination" if eval_result["hallucination_detected"]
                              else "low_evaluation_scores")
                    flag_for_review(step_id, reason=reason)
                log.info("eval (%s): rel=%s faith=%s complete=%s halluc=%s", eval_selection,
                         rel, faith, comp, eval_result['hallucination_detected'],
                         extra={"run_id": run_id, "step_id": step_id})
            except Exception as eval_err:
                from llm import is_degraded_model_error
                if not is_degraded_model_error(eval_err):
                    raise
                flag_for_review(step_id, reason="evaluation_failed")
                from error_codes import summarize_error
                log.warning("evaluation requested but failed: %s", summarize_error(eval_err),
                            extra={"run_id": run_id, "step_id": step_id})
        elif eval_selection:
            eval_selection = "skipped_budget"

        record_context(step_id, {
            "job_id": job.get("id"), "job_source": job.get("source"),
            "apply_url": job.get("apply_url"),
            "matched_skills": score_result["matched_skills"],
            "missing_skills": score_result["missing_skills"],
            # Whether the judge ACTUALLY ran — not whether the score was in the band.
            "judge_skipped": judge_status != "ran",
            "judge_status": judge_status,
            "judge_skip_reason": judge_skip_reason,
            "score_breakdown": score_result["breakdown"],
            "experience_basis": score_result.get("experience_basis"),
            "candidate_years_used": score_result.get("candidate_years_used"),
            "requirements_method": _method,
            "geo_eligibility": job.get("geo_eligibility"),
            "evaluation": eval_selection or "not_requested",
        })

        # AUTHORITATIVE review decision, AFTER every trigger has run (prompt injection,
        # score disagreement, hallucination, evaluation failure). The graph routes on
        # THIS flag — so a job flagged for prompt injection pauses for a human even when
        # the deterministic score and the LLM judge happen to agree.
        needs_review = _step_needs_review(step_id)
        state.job_results.append({
            "step_id": step_id,
            "job_id": job.get("id"),   # stable link to job_postings.id (not title)
            "title": job["title"], "company": job["company"],
            "score": score, "decision": score_result["decision"],
            "llm_decision": llm_decision, "final_decision": final_decision,
            "needs_review": needs_review, "apply_url": job.get("apply_url")
        })
        state.last_job_needs_review = needs_review
        if needs_review:
            state.last_review_step_id = step_id
            state.last_review_info = {
                "step_id": step_id,
                "job_title": job["title"],
                "company": job.get("company"),
                "score": score,
                "score_decision": score_result["decision"],
                "llm_decision": llm_decision,
                # WHY it paused — score disagreement, prompt injection, hallucination,
                # evaluation failure, invalid judge output (possibly several).
                "review_reason": _step_review_reason(step_id),
            }

        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        if is_infrastructure_error(e):
            raise                # the RUN fails / retries; never "one job failed"
        state.failed_jobs += 1   # count it so the run can report completed_with_errors
        # Job id and exception TYPE only: titles, descriptions and exception text
        # can carry job- or resume-derived content (logging policy, logging_config).
        log.error("job %s evaluation failed (%s)", job.get("id"), type(e).__name__,
                  extra={"run_id": run_id, "step_id": step_id})
    finally:
        # ALWAYS advance to the next job, success or failure. finally runs no
        # matter what — a failing job is skipped, never retried forever.
        state.current_job_index += 1


def apply_human_decision(state, run_id, step_id, decision, comment="",
                         reviewer_user_id=None, reviewer=None):
    """
    Record the human's Apply/Maybe/Skip decision for a reviewed step as the
    AUTHORITATIVE outcome, preserving the agent's original decisions in the
    trace (audit). Called after a LangGraph resume.

    Audit trail: WHO decided (reviewer_user_id + reviewer username), WHEN
    (reviewed_at), WHAT (decision) and WHY (comment). reviewer falls back to
    "unknown" only for legacy queue payloads that predate reviewer identity.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE steps
            SET review_status = %s, reviewed_at = %s, reviewer = %s, reviewer_user_id = %s,
                review_comment = %s, final_decision = %s
            WHERE id = %s
        """, (decision, utcnow(), reviewer or "unknown", reviewer_user_id,
              comment, decision, step_id))
    state.human_decisions[str(step_id)] = {"decision": decision, "comment": comment,
                                           "reviewer_user_id": reviewer_user_id,
                                           "reviewer": reviewer}

    # Make the human decision authoritative in the in-memory results too, so
    # downstream ranking/advice (which read final_decision) use the human's call.
    for r in state.job_results:
        if r.get("step_id") == step_id:
            r["final_decision"] = decision
            break