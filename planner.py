"""
Planner: given AgentState, decide the next action. Pure Python — zero LLM calls.
Routes on WHAT WORK REMAINS, not on budget. Budget is enforced at the actual
Gemini call (logged_llm_call), so free work — cache hits, scoring, ranking,
no-matches — always runs even when quota is exhausted.
"""


def plan_next_action(state):
    if state.error:
        return "fail"
    if state.cancelled:
        return "finish_cancelled"
    if not state.has("resume_text"):
        return "load_resume"
    if not state.has("parsed_resume"):
        return "parse_resume"
    if state.jobs is None:
        return "search_jobs"
    if len(state.jobs) == 0:
        return "finish_no_matches"
    if state.current_job_index < len(state.jobs):
        return "process_job"
    if not state.ranking_done:
        return "rank_jobs"
    if not state.advice_done:
        return "generate_advice"
    return "done"