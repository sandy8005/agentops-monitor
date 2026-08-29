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
    """Map a planner action to its tool. Returns nothing; mutates state."""
    if action == "load_resume":
        load_resume(state, run_id)
    elif action == "parse_resume":
        do_parse_resume(state, run_id)
    elif action == "search_jobs":
        do_search_jobs(state, run_id)
    else:
        # process_job, rank_jobs, finish_* handled in step 3
        raise NotImplementedError(f"action '{action}' wired in a later step")
    state.record_action(action)