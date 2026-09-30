"""
Router: executes the action the planner chose, updates state, returns.
Reuses the EXISTING, already-tested tools (parse_resume, search_jobs,
calculate_match_score, etc.) — the autonomous rewrite changes the control
flow, not the tools themselves.

All tool executions pass through logged_tool_call() so the autonomous path
keeps full AgentOps observability. Caching, cooperative cancellation, and
conditional Gemini use also live here.
"""
from timeutil import utcnow
import hashlib
import json
from datetime import timedelta

from settings import settings
from error_codes import ErrorCode

from parser import parse_resume
from job_source import search_jobs
from job_parser import extract_requirements
from scorer import calculate_match_score
from ranker import rank_jobs
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
            log.info("resume parsed by rules (%d skills, years %s) — 0 LLM calls",
                     len(parsed["skills"]),
                     "unknown" if parsed["years_experience"] is None else parsed["years_experience"],
                     extra={"step_id": step_id})
        state.parsed_resume = parsed
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"parse failed: {e}"


def do_search_jobs(state, run_id):
    """
    Fetch LIVE jobs for this search, upsert them (dedup on external_id), then
    search the COMBINED pool (seeded + live). Live fetch is best-effort — if it
    fails (network/API), we simply search the existing pool. 0 LLM calls.
    """
    step_id = create_step(run_id, "search_jobs", len(state.completed_actions))
    try:
        # Refresh the pool with REAL live jobs matching this role + location
        # (Adzuna does real search + location filtering). Best-effort — falls back
        # to the existing pool if the API is unavailable or keys are missing.
        # Refresh the live pool from BOTH live providers, each RUN-SCOPED (so a
        # live_only run actually sees their postings), and track each provider's
        # classified fetch status.
        provider_status = {}
        try:
            from adzuna_jobs import fetch_and_upsert_adzuna
            _, _, provider_status["adzuna"] = fetch_and_upsert_adzuna(
                state.target_role, state.location, run_id=run_id, step_id=step_id)
        except Exception as e:
            log.warning("adzuna fetch skipped (%s) — using existing pool", e)
            provider_status["adzuna"] = "failed"
        try:
            from live_jobs import fetch_and_upsert_remotive
            _, _, provider_status["remotive"] = fetch_and_upsert_remotive(
                state.target_role, state.location, run_id=run_id, step_id=step_id)
        except Exception as e:
            log.warning("remotive fetch skipped (%s) — using existing pool", e)
            provider_status["remotive"] = "failed"

        state.jobs = logged_tool_call(
            "search_jobs",
            lambda p: search_jobs(p["target_role"], p["location"],
                                  p["work_mode"], p["employment_type"],
                                  live_only=p["live_only"], run_id=p["run_id"]),
            {"target_role": state.target_role, "location": state.location,
             "work_mode": state.work_mode, "employment_type": state.employment_type,
             "live_only": state.live_only, "run_id": run_id},
            run_id, step_id, operation="search_jobs")

        # In LIVE-ONLY mode, 0 jobs is only a real "no matches" if at least one live
        # provider FETCHED cleanly (success/empty). If EVERY live provider failed
        # (network/auth/rate-limit/...) and nothing came back, the sources were
        # unavailable — report search_failed, not a misleading no_matches.
        SUCCEEDED = {"success", "empty"}
        if (state.live_only and not state.jobs
                and not any(s in SUCCEEDED for s in provider_status.values())):
            code = provider_failure_code(provider_status)
            raise RuntimeError(f"{code.value}: live job sources unavailable ({provider_status})")

        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"search failed: {e}"


# Provider fetch status -> run error code. Transient failures come first: if ANY
# provider failed transiently, a retry of the run can still succeed.
_TRANSIENT_PROVIDER = {
    "rate_limited": ErrorCode.JOB_SOURCE_RATE_LIMITED,
    "server_error": ErrorCode.JOB_SOURCE_UNAVAILABLE,
    "network_error": ErrorCode.JOB_SOURCE_UNAVAILABLE,
    "failed": ErrorCode.JOB_SOURCE_UNAVAILABLE,   # unexpected error (e.g. DB blip)
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
        log.info("requirements for '%s' served from cache [llm] — 0 LLM calls", job["title"])
        return cached, True, "llm"

    if _llm_allowed(state):
        try:
            reqs = extract_requirements(job, run_id, step_id, budget=state)
            _reqs_cache_put(dhash, reqs, "llm")
            if cached is not None:
                log.info("requirements for '%s' upgraded rule_based -> llm", job["title"])
            return reqs, False, "llm"
        except Exception as extract_err:
            from error_codes import summarize_error
            log.warning("requirements LLM extraction failed (%s)", summarize_error(extract_err))

    if cached is not None:
        # Degraded but still valid fallback that hasn't expired — reuse it, don't rewrite.
        log.info("requirements for '%s' served from cache [%s]", job["title"], method)
        return cached, True, method or "rule_based"

    reqs = extract_requirements_rule_based(job)
    _reqs_cache_put(dhash, reqs, "rule_based")
    log.info("requirements via rules — 0 LLM calls")
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
        from prompt_safety import detect_injection
        if detect_injection(f"{job.get('title', '')}\n{job.get('description', '')}"):
            flag_for_review(step_id, reason="possible_prompt_injection(job)")

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
                    # AI-quality failure, not a judgment. Ask a human.
                    judge_status = "invalid_output"
                    judge_skip_reason = "parse_error"
                    llm_decision = "Unknown"
                    result = None          # nothing valid for the evaluator to grade
                    flag_for_review(step_id, reason="judge_invalid_output")
                    log.warning("judge returned invalid structured output: %s", parse_err,
                                extra={"run_id": run_id, "step_id": step_id})
            except Exception as judge_err:
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
        state.failed_jobs += 1   # count it so the run can report completed_with_errors
        fail_step(step_id, e)
        log.exception("job '%s' failed", job['title'], extra={"step_id": step_id})
    finally:
        # ALWAYS advance to the next job, success or failure. finally runs no
        # matter what — a failing job is skipped, never retried forever.
        state.current_job_index += 1


def do_rank_jobs(state, run_id):
    """Rank scored jobs (traced), then PERSIST the ranked list to run_rankings.
    Sets ranking_done so the loop terminates."""
    step_id = create_step(run_id, "rank_jobs", len(state.completed_actions))
    try:
        state.ranked = logged_tool_call(
            "rank_jobs", lambda r: rank_jobs(r), state.job_results,
            run_id, step_id, operation="rank_jobs")
        _persist_rankings(run_id, state.ranked)
        finish_step(step_id, "success")
    except Exception as e:
        # The ranking IS the run's deliverable: if it was not committed, the run
        # must not report success (R03).
        fail_step(step_id, e)
        state.ranked = state.job_results
        state.error = f"rank failed: {e}"
    finally:
        state.ranking_done = True   # ranking ran (even if empty) — don't loop on it


def _persist_rankings(run_id, ranked):
    """
    Persist the final ranked list as self-contained snapshot rows in run_rankings
    (1-based rank_position). Snapshot fields are stored so the ranking is readable
    later without joining job_postings. Replace-all in ONE transaction, so a retry
    is idempotent. A persistence failure RAISES (R03): the caller decides the run
    outcome; it is never silently reported as a successful ranking.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM run_rankings WHERE run_id = %s", (run_id,))
        for pos, r in enumerate(ranked or [], start=1):
            cur.execute("""
                INSERT INTO run_rankings
                    (run_id, job_id, rank_position, title, company, score,
                     final_decision, apply_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, r.get("job_id"), pos, r.get("title"), r.get("company"),
                  r.get("score"), r.get("final_decision") or r.get("decision"),
                  r.get("apply_url")))


def _advice_profile(parsed_resume, score_result):
    """
    The candidate facts the advice prompt is allowed to use — built from the
    STRUCTURED, grounded parse instead of an arbitrary slice of the raw resume (the
    old resume_text[:3000] silently dropped everything after the first 3000 chars).
    Only GROUNDED skills are included; each list is capped to keep the prompt small.
    """
    pr = parsed_resume or {}
    grounded = pr.get("grounded_skills")
    skills = grounded if isinstance(grounded, list) else (pr.get("skills") or [])
    evidence = [f"{e.get('skill')}: {e.get('evidence')}" for e in (pr.get("skill_evidence") or [])
                if isinstance(e, dict) and e.get("skill") and e.get("evidence")]
    return {
        "skills": list(skills)[:40],
        "years_experience_used": score_result.get("candidate_years_used"),
        "experience": [f"{e.get('title')} at {e.get('company')} ({e.get('years')} yrs)"
                       for e in (pr.get("experience") or []) if isinstance(e, dict)][:10],
        "projects": [f"{p.get('name')}: {', '.join(p.get('tech') or [])}"
                     for p in (pr.get("projects") or []) if isinstance(p, dict)][:10],
        "education": [f"{e.get('degree')}, {e.get('institution')} {e.get('year') or ''}".strip()
                      for e in (pr.get("education") or []) if isinstance(e, dict)][:5],
        "evidence": evidence[:20],
        "matched_requirements": score_result.get("matched_skills") or [],
        "missing_requirements": score_result.get("missing_skills") or [],
        "missing_preferred": score_result.get("missing_preferred") or [],
    }


def _combined_advice(parsed_resume, job, requirements, score_result, run_id, step_id,
                     budget=None):
    """
    ONE Gemini call returning BOTH application strategy and resume-edit advice.
    Used only for top viable jobs. The candidate side of the prompt is the
    structured profile (_advice_profile), not the raw resume text.
    """
    profile = _advice_profile(parsed_resume, score_result)
    prompt = f"""
{HARDENING_PREAMBLE}

You are a career advisor. For the job below, give the candidate BOTH:
1. APPLICATION STRATEGY - how to position themselves for this specific role.
2. RESUME EDITS - concrete, numbered edits to better match this job.
Only rely on the candidate facts given; do not invent experience they don't have.

CANDIDATE SKILLS (grounded in the resume): {wrap_untrusted(profile["skills"], "SKILLS")}
YEARS OF EXPERIENCE USED FOR MATCHING: {profile["years_experience_used"]}
EXPERIENCE: {wrap_untrusted(profile["experience"], "EXPERIENCE")}
PROJECTS: {wrap_untrusted(profile["projects"], "PROJECTS")}
EDUCATION: {wrap_untrusted(profile["education"], "EDUCATION")}
RESUME EVIDENCE SNIPPETS: {wrap_untrusted(profile["evidence"], "EVIDENCE")}

JOB: {wrap_untrusted(job['title'], "JOB_TITLE")} at {wrap_untrusted(job.get('company',''), "COMPANY")}
REQUIRED SKILLS: {wrap_untrusted(requirements.get('required_skills', []), "REQUIRED_SKILLS")}
PREFERRED SKILLS: {wrap_untrusted(requirements.get('preferred_skills', []), "PREFERRED_SKILLS")}
REQUIREMENTS THE CANDIDATE MEETS: {wrap_untrusted(profile["matched_requirements"], "MATCHED")}
REQUIREMENTS THE RESUME IS MISSING: {wrap_untrusted(profile["missing_requirements"], "MISSING_SKILLS")}
PREFERRED SKILLS THE RESUME IS MISSING: {wrap_untrusted(profile["missing_preferred"], "MISSING_PREFERRED")}

Respond in exactly this format:
STRATEGY:
<one paragraph>

RESUME EDITS:
1. <edit>
2. <edit>
3. <edit>
"""
    return logged_llm_call(prompt, run_id, step_id, operation="combined_advice", budget=budget)


def _persist_advice(run_id, job_id, title, advice):
    """Persist one advice text to run_advice, keyed to (run_id, job_id). Re-persisting
    the same (run, job) replaces the prior row. A failure RAISES (R03); the caller
    records it as a partial failure instead of reporting silent success."""
    if not advice:
        return
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM run_advice WHERE run_id = %s AND job_id IS NOT DISTINCT FROM %s",
            (run_id, job_id))
        cur.execute("""
            INSERT INTO run_advice (run_id, job_id, title, advice)
            VALUES (%s, %s, %s, %s)
        """, (run_id, job_id, title, advice.strip()))


def _persist_suggestions(run_id, job_id, resume_id, resume_hash, suggestions):
    """Structured suggestions for the pipeline path (same table as agent mode).
    Replace-per-(run, job); raises on failure (R03)."""
    if job_id is None or resume_id is None:
        return
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM resume_suggestions WHERE run_id = %s AND job_id = %s",
                    (run_id, job_id))
        for pos, sg in enumerate(suggestions, start=1):
            cur.execute("""
                INSERT INTO resume_suggestions (run_id, job_id, resume_id, resume_hash, position,
                    kind, original_text, suggested_text, reason, evidence, method, status,
                    validation_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, job_id, resume_id, resume_hash, pos, sg["kind"], sg.get("original_text"),
                  sg["suggested_text"], sg["reason"], json.dumps(sg.get("evidence") or []),
                  sg["method"], sg["status"], sg.get("validation_notes")))


def do_generate_advice(state, run_id, top_n=2):
    """
    #10: after ranking, generate combined advice for the TOP N viable
    (Apply/Maybe) jobs only - not every job. One combined call each,
    budget-permitting. This is where advice comes back cheaply.
    """
    step_id = create_step(run_id, "generate_advice", len(state.completed_actions))
    try:
        viable = [r for r in (state.ranked or [])
                  if r.get("final_decision", r.get("decision")) in ("Apply", "Maybe")][:top_n]
        for r in viable:
            # Look up the posting by STABLE job_id, falling back to title only when
            # job_id is missing (legacy rows).
            job = None
            if r.get("job_id") is not None:
                job = next((j for j in state.jobs if j.get("id") == r["job_id"]), None)
            if job is None and r.get("job_id") is None:
                same_title = [j for j in state.jobs if j["title"] == r["title"]]
                job = same_title[0] if len(same_title) == 1 else None
            if not job:
                continue
            dhash = _reqs_cache_key(job["title"], job["description"])
            cached_reqs, _prov = _reqs_cache_get(dhash)
            requirements = cached_reqs or {
                "required_skills": [], "required_any_of": [], "preferred_skills": [],
                "min_years_experience": 0, "responsibilities": []
            }
            from skills import normalize_requirements
            requirements = normalize_requirements(requirements)
            sc = calculate_match_score(state.parsed_resume, requirements,
                                       state.resume_text, job, None)
            # Evidence-checked suggestions from rules ALWAYS (no model needed); the
            # model-written strategy text is added only when a model call is allowed.
            from resume_advisor import rule_suggestions, advice_summary
            suggestions = rule_suggestions(state.parsed_resume, state.resume_text, job,
                                           requirements, sc)
            advice = advice_summary(job, suggestions, "rules")
            if _llm_allowed(state):
                try:
                    advice = _combined_advice(state.parsed_resume, job, requirements, sc,
                                              run_id, step_id, budget=state) \
                        + "\n\n" + advice
                except Exception as adv_err:
                    from error_codes import summarize_error
                    log.warning("model advice unavailable (%s) — rules suggestions only",
                                summarize_error(adv_err), extra={"step_id": step_id})
            _persist_advice(run_id, job.get("id"), job["title"], advice)
            _persist_suggestions(run_id, job.get("id"), state.resume_id,
                                 resume_content_hash(state.resume_text or ""), suggestions)
            # METADATA ONLY. The advice text is derived from the resume and the job
            # posting; it lives in run_advice (covered by erasure and retention) and
            # must never leak into application logs, which those controls don't reach.
            log.info("application advice generated",
                     extra={"run_id": run_id, "step_id": step_id})
            log.debug("advice metadata: job_id=%s chars=%d", job.get("id"),
                      len(advice or ""), extra={"run_id": run_id, "step_id": step_id})
        finish_step(step_id, "success")
        state.advice_done = True
    except Exception as e:
        # Advice is optional enrichment: its failure makes the run PARTIAL
        # (completed_with_errors), never a silent success (R03).
        fail_step(step_id, e)
        state.failed_jobs += 1
        state.advice_done = True   # don't loop on advice failure


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


def dispatch(action, state, run_id):
    """Map a planner action to its tool. Mutates state."""
    if action == "load_resume":
        load_resume(state, run_id)
    elif action == "parse_resume":
        do_parse_resume(state, run_id)
    elif action == "search_jobs":
        do_search_jobs(state, run_id)
    elif action == "process_job":
        do_process_job(state, run_id)
    elif action == "rank_jobs":
        do_rank_jobs(state, run_id)
    elif action == "generate_advice":
        do_generate_advice(state, run_id)
    else:
        raise NotImplementedError(f"unknown action '{action}'")
    state.record_action(action)