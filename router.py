"""
Router: executes the action the planner chose, updates state, returns.
Reuses the EXISTING, already-tested tools (parse_resume, search_jobs,
calculate_match_score, etc.) — the autonomous rewrite changes the control
flow, not the tools themselves.

Caching lives here: before spending a Gemini call, check state for a cached
result. This is where autonomy (planner picks the step) and call-reduction
(router skips the call if cached) meet.
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
    record_score, record_context, flag_for_review, get_connection
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
        # cache miss → real LLM parse
        parsed = parse_resume(state.resume_text, run_id, step_id)
        state.llm_calls_made += 1
        _parse_cache_put(rhash, parsed)
        state.parsed_resume = parsed
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"parse failed: {e}"


def do_search_jobs(state, run_id):
    """Search jobs (0 LLM calls — pure DB query)."""
    step_id = create_step(run_id, "search_jobs", len(state.completed_actions))
    try:
        state.jobs = search_jobs(state.target_role, state.location,
                                 state.work_mode, state.employment_type)
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.error = f"search failed: {e}"


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
    else:
        raise NotImplementedError(f"unknown action '{action}'")
    state.record_action(action)

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
    Process ONE job: keyword overlap (free) → requirements (cached) →
    deterministic score (free) → Gemini judge ONLY if score is in the
    uncertain middle band (20-80). Extremes skip the judge (#5,6,7).
    """
    job = state.jobs[state.current_job_index]
    step_id = create_step(run_id, job["title"], len(state.completed_actions))
    try:
        # 1. keyword overlap — 0 LLM calls
        overlap = keyword_overlap_tool(
            {"resume": state.resume_text, "job_description": job["description"]}
        )

        # 2. requirements — cached by description hash (#2)
        dhash = _hash(job["description"])
        requirements = _reqs_cache_get(dhash)
        if requirements is None:
            requirements = extract_requirements(job, run_id, step_id)
            state.llm_calls_made += 1
            _reqs_cache_put(dhash, requirements)
        else:
            print(f"    (requirements for '{job['title']}' served from cache — 0 LLM calls)")

        # 3. deterministic score — 0 LLM calls, runs FIRST (#4)
        user_input = {
            "target_role": state.target_role, "location": state.location,
            "work_mode": state.work_mode, "employment_type": state.employment_type
        }
        score_result = calculate_match_score(state.parsed_resume, requirements,
                                             state.resume_text, job, user_input)
        score = score_result["score"]

        # 4. Gemini judge ONLY in the uncertain middle band (#5,6,7).
        #    Extremes skip the judge — recorded honestly, not faked as agreement.
        if 20 <= score <= 80 and not state.budget_exceeded():
            from agent import build_prompt   # reuse the existing prompt builder
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
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        print(f"    job '{job['title']}' failed: {e}")

    finally:
        # ALWAYS advance to the next job, success or failure. finally runs no
        # matter what happens above — a failing job is skipped, never retried.
        state.current_job_index += 1


def do_rank_jobs(state, run_id):
    step_id = create_step(run_id, "rank_jobs", len(state.completed_actions))
    try:
        state.ranked = rank_jobs(state.job_results)
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        state.ranked = state.job_results
    finally:
        state.ranking_done = True   # ranking ran (even if empty) — don't loop on it