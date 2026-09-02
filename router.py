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
from tools import keyword_overlap_tool
from schemas import JobDecision
from llm import (
    create_step, finish_step, fail_step, logged_llm_call, logged_tool_call,
    record_score, record_context, flag_for_review, get_connection,
    is_cancel_requested
)


def _hash(text):
    """Stable hash for cache keys (#12) — detects when resume/job text changed."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_resume(state, run_id):
    """Load the resume document from DB (0 LLM calls)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT resume_text FROM resumes WHERE id = %s", (state.resume_id,))
    row = cur.fetchone()
    conn.close()
    if not row or not row[0]:
        state.error = f"resume {state.resume_id} not found or empty"
        return
    state.resume_text = row[0]


def _parse_cache_get(resume_hash):
    """Look up a previously parsed resume by hash (#1, #12). 0 LLM calls on hit."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT parsed_json FROM parsed_resume_cache WHERE resume_hash = %s", (resume_hash,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _parse_cache_put(resume_hash, parsed):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO parsed_resume_cache (resume_hash, parsed_json)
        VALUES (%s, %s) ON CONFLICT (resume_hash) DO NOTHING
    """, (resume_hash, json.dumps(parsed)))
    conn.commit()
    conn.close()


def do_parse_resume(state, run_id):
    """Parse the resume — but reuse the cache if we've parsed this exact text (#1)."""
    step_id = create_step(run_id, "parse_resume", len(state.completed_actions))
    try:
        rhash = _hash(state.resume_text)
        cached = _parse_cache_get(rhash)
        if cached is not None:
            state.parsed_resume = cached
            finish_step(step_id, "success")
            print("    (parsed resume served from cache — 0 LLM calls)")
            return
        parsed = parse_resume(state.resume_text, run_id, step_id)
        state.llm_calls_made += 1
        _parse_cache_put(rhash, parsed)
        state.parsed_resume = parsed
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"parse failed: {e}"


def do_search_jobs(state, run_id):
    """Search jobs (0 LLM calls). Traced via logged_tool_call for observability."""
    step_id = create_step(run_id, "search_jobs", len(state.completed_actions))
    try:
        state.jobs = logged_tool_call(
            "search_jobs",
            lambda p: search_jobs(p["target_role"], p["location"],
                                  p["work_mode"], p["employment_type"]),
            {"target_role": state.target_role, "location": state.location,
             "work_mode": state.work_mode, "employment_type": state.employment_type},
            run_id, step_id, operation="search_jobs")
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"search failed: {e}"


# --- requirements cache (#2, #12): a job's requirements don't depend on the
# resume, so extract once per job description and reuse. ---

def _reqs_cache_get(desc_hash):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT reqs_json FROM job_reqs_cache WHERE desc_hash = %s", (desc_hash,))
    row = cur.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _reqs_cache_put(desc_hash, reqs):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO job_reqs_cache (desc_hash, reqs_json)
        VALUES (%s, %s) ON CONFLICT (desc_hash) DO NOTHING
    """, (desc_hash, json.dumps(reqs)))
    conn.commit()
    conn.close()


def _parse_decision(raw):
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return JobDecision(**json.loads(cleaned)).decision.value


def do_process_job(state, run_id):
    """
    Process ONE job: cancellation check → keyword overlap (traced) →
    requirements (cached) → deterministic score (traced) → Gemini judge ONLY
    if score is in the uncertain middle band (20-80) → evaluator on risky jobs.
    Always advances current_job_index (finally), so the loop can't get stuck.
    """
    # Cooperative cancellation: if the user hit Cancel, stop before this job.
    if is_cancel_requested(run_id):
        state.cancelled = True
        return

    job = state.jobs[state.current_job_index]
    step_id = create_step(run_id, job["title"], len(state.completed_actions))
    try:
        # 1. keyword overlap — traced (0 LLM calls)
        overlap = logged_tool_call(
            "keyword_overlap_tool", keyword_overlap_tool,
            {"resume": state.resume_text, "job_description": job["description"]},
            run_id, step_id, operation="keyword_overlap")

        # 2. requirements — cached by description hash (#2)
        dhash = _hash(job["description"])
        requirements = _reqs_cache_get(dhash)
        if requirements is None:
            requirements = extract_requirements(job, run_id, step_id)
            state.llm_calls_made += 1
            _reqs_cache_put(dhash, requirements)
        else:
            print(f"    (requirements for '{job['title']}' served from cache — 0 LLM calls)")

        # 3. deterministic score — traced (0 LLM calls), runs FIRST (#4)
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

        # 4. Gemini judge ONLY in the uncertain middle band (#5,6,7).
        #    Extremes skip the judge — recorded honestly, not faked as agreement.
        result = None
        if 20 <= score <= 80 and not state.budget_exceeded():
            from agent import build_prompt
            prompt = build_prompt(state.resume_text, state.parsed_resume, job, overlap, requirements)
            result = logged_llm_call(prompt, run_id, step_id, operation="job_judge")
            state.llm_calls_made += 1
            try:
                llm_decision = _parse_decision(result)
            except Exception:
                llm_decision = "Unknown"
        else:
            reason = "score_extreme_low" if score < 20 else "score_extreme_high"
            llm_decision = f"skipped ({reason})"

        needs_review = record_score(step_id, score, score_result["decision"],
                                    llm_decision, breakdown=score_result["breakdown"])

        record_context(step_id, {
            "job_id": job.get("id"), "job_source": job.get("source"),
            "matched_skills": overlap["matched_in_resume"],
            "missing_skills": overlap["missing_from_resume"],
            "judge_skipped": not (20 <= score <= 80),
            "score_breakdown": score_result["breakdown"],
        })

        state.job_results.append({
            "title": job["title"], "company": job["company"],
            "score": score, "decision": score_result["decision"],
            "llm_decision": llm_decision, "needs_review": needs_review
        })

        # --- Evaluator (#8): ONLY on risky (flagged) jobs where the judge ran
        # (so 'result' exists) and budget allows. Most jobs skip this. ---
        if (state.evaluate and needs_review and result is not None
                and not state.budget_exceeded()
                and llm_decision in ("Apply", "Maybe", "Skip")):
            try:
                from evaluator import evaluate_decision
                from llm import save_evaluation
                eval_result = evaluate_decision(state.resume_text, job, result, run_id, step_id)
                save_evaluation(run_id, step_id, eval_result)
                state.llm_calls_made += 1
                rel = eval_result["relevance_score"]
                faith = eval_result["faithfulness_score"]
                comp = eval_result["completeness_score"]
                if eval_result["hallucination_detected"] or min(rel, faith, comp) <= 2:
                    reason = ("hallucination" if eval_result["hallucination_detected"]
                              else "low_evaluation_scores")
                    flag_for_review(step_id, reason=reason)
                print(f"    eval: rel={rel} faith={faith} complete={comp} "
                      f"halluc={eval_result['hallucination_detected']}")
            except Exception as eval_err:
                flag_for_review(step_id, reason="evaluation_failed")
                state.llm_calls_made += 1
                print(f"    evaluation requested but failed: {eval_err}")

        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        print(f"    job '{job['title']}' failed: {e}")
    finally:
        # ALWAYS advance to the next job, success or failure. finally runs no
        # matter what — a failing job is skipped, never retried forever.
        state.current_job_index += 1


def do_rank_jobs(state, run_id):
    """Rank scored jobs (traced). Sets ranking_done so the loop terminates."""
    step_id = create_step(run_id, "rank_jobs", len(state.completed_actions))
    try:
        state.ranked = logged_tool_call(
            "rank_jobs", lambda r: rank_jobs(r), state.job_results,
            run_id, step_id, operation="rank_jobs")
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.ranked = state.job_results
    finally:
        state.ranking_done = True   # ranking ran (even if empty) — don't loop on it


def _combined_advice(resume_text, job, requirements, missing_skills, run_id, step_id):
    """
    #9 + #10: ONE Gemini call returning BOTH application strategy and resume-edit
    advice, instead of two separate calls. Used only for top viable jobs.
    """
    prompt = f"""
You are a career advisor. For the job below, give the candidate BOTH:
1. APPLICATION STRATEGY - how to position themselves for this specific role.
2. RESUME EDITS - concrete, numbered edits to better match this job.

CANDIDATE RESUME:
{resume_text[:3000]}

JOB: {job['title']} at {job.get('company','')}
REQUIRED SKILLS: {requirements.get('required_skills', [])}
PREFERRED SKILLS: {requirements.get('preferred_skills', [])}
SKILLS THE RESUME IS MISSING: {missing_skills}

Respond in exactly this format:
STRATEGY:
<one paragraph>

RESUME EDITS:
1. <edit>
2. <edit>
3. <edit>
"""
    return logged_llm_call(prompt, run_id, step_id, operation="combined_advice")


def do_generate_advice(state, run_id, top_n=2):
    """
    #10: after ranking, generate combined advice for the TOP N viable
    (Apply/Maybe) jobs only - not every job. One combined call each,
    budget-permitting. This is where advice comes back cheaply.
    """
    step_id = create_step(run_id, "generate_advice", len(state.completed_actions))
    try:
        viable = [r for r in (state.ranked or [])
                  if r.get("decision") in ("Apply", "Maybe")][:top_n]
        for r in viable:
            if state.budget_exceeded():
                print("    (advice skipped - budget reached)")
                break
            job = next((j for j in state.jobs if j["title"] == r["title"]), None)
            if not job:
                continue
            dhash = _hash(job["description"])
            requirements = _reqs_cache_get(dhash) or {}
            overlap = keyword_overlap_tool(
                {"resume": state.resume_text, "job_description": job["description"]}
            )
            advice = _combined_advice(state.resume_text, job, requirements,
                                      overlap["missing_from_resume"], run_id, step_id)
            state.llm_calls_made += 1
            print(f"\n  ADVICE for {r['title']}:\n{advice.strip()}\n")
        finish_step(step_id, "success")
        state.advice_done = True
    except Exception as e:
        fail_step(step_id, e)
        state.advice_done = True   # don't loop on advice failure


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