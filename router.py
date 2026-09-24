"""
Router: executes the action the planner chose, updates state, returns.
Reuses the EXISTING, already-tested tools (parse_resume, search_jobs,
calculate_match_score, etc.) — the autonomous rewrite changes the control
flow, not the tools themselves.

All tool executions pass through logged_tool_call() so the autonomous path
keeps full AgentOps observability. Caching, cooperative cancellation, and
conditional Gemini use also live here.
"""
import hashlib
import json

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


def _resume_cache_key(resume_text):
    """Versioned parse-cache key: text + parser/schema/model version, so a prompt,
    schema, or model change makes old cached parses unreachable (never served stale)."""
    return _hash(f"{resume_text}|{parse_cache_version()}")


def _reqs_cache_key(title, description):
    """Versioned requirements-cache key: TITLE + description + reqs/schema/model
    version. Including the title distinguishes postings that share a description
    but differ by role, and lets a title-borne requirement signal affect the key."""
    return _hash(f"{title}\n{description}|{reqs_cache_version()}")


def load_resume(state, run_id):
    """Load the resume document from DB (0 LLM calls)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT resume_text FROM resumes WHERE id = %s", (state.resume_id,))
        row = cur.fetchone()
        if not row or not row[0]:
            state.error = f"resume {state.resume_id} not found or empty"
            return
        state.resume_text = row[0]


def _parse_cache_get(resume_hash):
    """Look up a previously parsed resume by hash (#1, #12). 0 LLM calls on hit."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT parsed_json FROM parsed_resume_cache WHERE resume_hash = %s", (resume_hash,))
        row = cur.fetchone()
        if not row:
            return None
        val = json.loads(row[0])
        # Ignore a poisoned/legacy entry (e.g. a `null` written before parse_resume was
        # hardened) so it's re-parsed instead of flowing downstream as None.
        return val if isinstance(val, dict) else None


def _parse_cache_put(resume_hash, parsed):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO parsed_resume_cache (resume_hash, parsed_json, cache_version)
            VALUES (%s, %s, %s) ON CONFLICT (resume_hash) DO NOTHING
        """, (resume_hash, json.dumps(parsed), parse_cache_version()))


def do_parse_resume(state, run_id):
    """Parse the resume — but reuse the cache if we've parsed this exact text (#1)."""
    step_id = create_step(run_id, "parse_resume", len(state.completed_actions))
    try:
        rhash = _resume_cache_key(state.resume_text)
        cached = _parse_cache_get(rhash)
        if cached is not None:
            state.parsed_resume = cached
            finish_step(step_id, "success")
            log.info("parsed resume served from cache — 0 LLM calls", extra={"step_id": step_id})
            return
        parsed = parse_resume(state.resume_text, run_id, step_id, budget=state)
        # A parse that yields no usable dict must NOT flow downstream as
        # parsed_resume=None — it would crash scoring/judging on None["..."]. Treat it
        # as a parse failure so route_after_parse ends the run cleanly, and never
        # cache a non-dict (which would poison the parse cache).
        if not isinstance(parsed, dict):
            raise ValueError("resume parse returned no usable data")
        _parse_cache_put(rhash, parsed)
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
            raise RuntimeError(f"live job sources unavailable ({provider_status})")

        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"search failed: {e}"


# --- requirements cache (#2, #12): a job's requirements don't depend on the
# resume, so extract once per (title+description) and reuse. Each row also records
# PROVENANCE — how it was produced (llm vs rule_based) and under which model/version
# — so a cache hit is traceable and its quality is known. ---

def _reqs_cache_get(desc_hash):
    """
    Return (reqs, provenance) on hit, or (None, None) on miss. provenance is a dict
    {"extraction_method":..., "source_model":...} describing how the cached row was
    produced, so callers can trace/trust it without re-extracting.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT reqs_json, extraction_method, source_model
            FROM job_reqs_cache WHERE desc_hash = %s
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
      - source_model      : the MODEL NAME that produced it (e.g. 'gemini-3.6-flash')
                            for LLM extraction, or NULL for rule_based (no model was
                            used). Previously this incorrectly stored the composite
                            cache_version STRING here — a bug; source_model now holds
                            the actual model, from the single source of truth.
    Idempotent (ON CONFLICT DO NOTHING) — first writer wins for a given key.
    """
    from cache_version import model_version
    # rule_based extraction used no model, so its source_model is NULL (not the
    # LLM model), keeping provenance honest.
    source_model = model_version() if extraction_method == "llm" else None
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO job_reqs_cache
                (desc_hash, reqs_json, cache_version, extraction_method, source_model)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (desc_hash) DO NOTHING
        """, (desc_hash, json.dumps(reqs), reqs_cache_version(),
              extraction_method, source_model))


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
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT needs_human_review FROM steps WHERE id = %s", (step_id,))
        row = cur.fetchone()
        return bool(row and row[0])
    finally:
        conn.close()


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
        # 1. requirements FIRST — structured, optional-aware extraction (cached #2).
        dhash = _reqs_cache_key(job["title"], job["description"])
        requirements, provenance = _reqs_cache_get(dhash)
        cache_hit = requirements is not None
        if requirements is None:
            # Extract requirements: LLM when budget allows (best quality), else
            # rule-based fallback (no LLM) so the job still gets requirements and
            # never drops out just because quota ran out. Graceful degradation,
            # same pattern as the judge.
            extraction_method = "rule_based"   # default unless the LLM path succeeds
            if not state.budget_exceeded():
                try:
                    requirements = extract_requirements(job, run_id, step_id, budget=state)
                    extraction_method = "llm"
                except Exception as extract_err:
                    from rule_requirements import extract_requirements_rule_based
                    requirements = extract_requirements_rule_based(job)
                    log.warning("requirements via rules (LLM failed: %s) — 0 LLM calls", extract_err)
            else:
                from rule_requirements import extract_requirements_rule_based
                requirements = extract_requirements_rule_based(job)
                log.info("requirements via rules (budget spent) — 0 LLM calls")
            _reqs_cache_put(dhash, requirements, extraction_method)
        else:
            method = (provenance or {}).get("extraction_method") or "unknown"
            log.info("requirements for '%s' served from cache [%s] — 0 LLM calls", job['title'], method)

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

        # 3. Gemini judge ONLY in the uncertain middle band (#5,6,7).
        result = None
        judge_status = "skipped"
        judge_skip_reason = None
        if not (20 <= score <= 80):
            # Extreme score — judge adds little; skip it (honest, not faked).
            judge_skip_reason = "score_extreme_low" if score < 20 else "score_extreme_high"
            llm_decision = f"skipped ({judge_skip_reason})"
        elif state.budget_exceeded():
            # Middle-band, but quota is spent. Keep the score, skip judge cleanly.
            judge_skip_reason = "budget"
            llm_decision = "skipped (budget)"
        else:
            # Middle-band: judge SHOULD run. But if it fails (quota/API/budget) the
            # job KEEPS its deterministic score instead of being discarded — degrade
            # to "scored, judge unavailable" rather than failing the whole job.
            from agent import build_prompt
            evidence = {"matched_in_resume": score_result["matched_skills"],
                        "missing_from_resume": score_result["missing_skills"]}
            prompt = build_prompt(state.resume_text, state.parsed_resume, job, evidence, requirements)
            try:
                result = logged_llm_call(prompt, run_id, step_id, operation="job_judge", budget=state)
                judge_status = "ran"
                try:
                    llm_decision = _parse_decision(result)
                except Exception:
                    llm_decision = "Unknown"
            except Exception as judge_err:
                judge_skip_reason = "judge_unavailable"
                llm_decision = "skipped (judge_unavailable)"
                log.warning("judge unavailable for '%s' (%s) — keeping score", job['title'], judge_err)

        needs_review = record_score(step_id, score, score_result["decision"],
                                    llm_decision, breakdown=score_result["breakdown"])

        # Compute the authoritative decision (human > llm > score). Defaults to the
        # best automated signal now; a human review overrides it later.
        final_decision = _compute_final_decision(score_result["decision"], llm_decision)
        _store_final_decision(step_id, final_decision)

        record_context(step_id, {
            "job_id": job.get("id"), "job_source": job.get("source"),
            "apply_url": job.get("apply_url"),
            "matched_skills": score_result["matched_skills"],
            "missing_skills": score_result["missing_skills"],
            "judge_skipped": not (20 <= score <= 80),
            "score_breakdown": score_result["breakdown"],
        })

        # Structured AgentOps signals (queryable columns).
        record_judge_signals(step_id, judge_status, judge_skip_reason, cache_hit)

        state.job_results.append({
            "step_id": step_id,
            "job_id": job.get("id"),   # stable link to job_postings.id (not title)
            "title": job["title"], "company": job["company"],
            "score": score, "decision": score_result["decision"],
            "llm_decision": llm_decision, "final_decision": final_decision,
            "needs_review": needs_review, "apply_url": job.get("apply_url")
        })

        # Surface review status to the graph so routing can pause for a human.
        state.last_job_needs_review = bool(needs_review)
        if needs_review:
            state.last_review_step_id = step_id
            state.last_review_info = {
                "step_id": step_id,
                "job_title": job["title"],
                "company": job.get("company"),
                "score": score,
                "score_decision": score_result["decision"],
                "llm_decision": llm_decision,
            }

        # --- Evaluator (#8): ONLY on risky (flagged) jobs where the judge ran
        # (so 'result' exists) and budget allows. Most jobs skip this. ---
        if (state.evaluate and needs_review and result is not None
                and not state.budget_exceeded()
                and llm_decision in ("Apply", "Maybe", "Skip")):
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
                log.info("eval: rel=%s faith=%s complete=%s halluc=%s", rel, faith, comp, eval_result['hallucination_detected'])
            except Exception as eval_err:
                flag_for_review(step_id, reason="evaluation_failed")
                log.warning("evaluation requested but failed: %s", eval_err)

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
        fail_step(step_id, e)
        state.ranked = state.job_results
    finally:
        state.ranking_done = True   # ranking ran (even if empty) — don't loop on it


def _persist_rankings(run_id, ranked):
    """
    Persist the final ranked list as self-contained snapshot rows in run_rankings
    (1-based rank_position). Snapshot fields are stored so the ranking is readable
    later without joining job_postings. Best-effort — a persistence failure never
    breaks the run. Re-persisting a run replaces its previous rows.
    """
    try:
        conn = get_connection()
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
        conn.commit()
        conn.close()
    except Exception as e:
        log.exception("persist rankings failed", extra={"run_id": run_id})


def _combined_advice(resume_text, job, requirements, missing_skills, run_id, step_id, budget=None):
    """
    #9 + #10: ONE Gemini call returning BOTH application strategy and resume-edit
    advice, instead of two separate calls. Used only for top viable jobs.
    """
    prompt = f"""
{HARDENING_PREAMBLE}

You are a career advisor. For the job below, give the candidate BOTH:
1. APPLICATION STRATEGY - how to position themselves for this specific role.
2. RESUME EDITS - concrete, numbered edits to better match this job.

CANDIDATE RESUME:
{wrap_untrusted(resume_text[:3000], "RESUME")}

JOB: {wrap_untrusted(job['title'], "JOB_TITLE")} at {wrap_untrusted(job.get('company',''), "COMPANY")}
REQUIRED SKILLS: {wrap_untrusted(requirements.get('required_skills', []), "REQUIRED_SKILLS")}
PREFERRED SKILLS: {wrap_untrusted(requirements.get('preferred_skills', []), "PREFERRED_SKILLS")}
SKILLS THE RESUME IS MISSING: {wrap_untrusted(missing_skills, "MISSING_SKILLS")}

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
    """Persist one advice text to run_advice, keyed to (run_id, job_id). Best-effort
    — never breaks the run. Re-persisting the same (run, job) replaces the prior row."""
    if not advice:
        return
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM run_advice WHERE run_id = %s AND job_id IS NOT DISTINCT FROM %s",
            (run_id, job_id))
        cur.execute("""
            INSERT INTO run_advice (run_id, job_id, title, advice)
            VALUES (%s, %s, %s, %s)
        """, (run_id, job_id, title, advice.strip()))
        conn.commit()
        conn.close()
    except Exception as e:
        log.exception("persist advice failed", extra={"run_id": run_id})


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
            if state.budget_exceeded():
                log.info("advice skipped — budget reached")
                break
            # Look up the posting by STABLE job_id, falling back to title only when
            # job_id is missing (legacy rows).
            job = None
            if r.get("job_id") is not None:
                job = next((j for j in state.jobs if j.get("id") == r["job_id"]), None)
            if job is None:
                job = next((j for j in state.jobs if j["title"] == r["title"]), None)
            if not job:
                continue
            dhash = _reqs_cache_key(job["title"], job["description"])
            cached_reqs, _prov = _reqs_cache_get(dhash)
            requirements = cached_reqs or {
                "required_skills": [], "required_any_of": [], "preferred_skills": [],
                "min_years_experience": 0, "responsibilities": []
            }
            sc = calculate_match_score(state.parsed_resume, requirements,
                                       state.resume_text, job, None)
            advice = _combined_advice(state.resume_text, job, requirements,
                                      sc["missing_skills"], run_id, step_id,
                                      budget=state)
            _persist_advice(run_id, job.get("id"), job["title"], advice)
            log.info("advice for %s:\n%s", r['title'], advice.strip())
        finish_step(step_id, "success")
        state.advice_done = True
    except Exception as e:
        fail_step(step_id, e)
        state.advice_done = True   # don't loop on advice failure


def apply_human_decision(state, run_id, step_id, decision, comment=""):
    """
    Record the human's Apply/Maybe/Skip decision for a reviewed step as the
    AUTHORITATIVE outcome, preserving the agent's original decisions in the
    trace (audit). Called after a LangGraph resume.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE steps
            SET review_status = %s, reviewed_at = %s, reviewer = %s, review_comment = %s,
                final_decision = %s
            WHERE id = %s
        """, (decision, __import__("datetime").datetime.now(), "human", comment, decision, step_id))
    state.human_decisions[str(step_id)] = {"decision": decision, "comment": comment}

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