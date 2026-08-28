"""
Planner: given AgentState, decide the next action. Pure Python — zero LLM calls.
Routing through a known workflow is deterministic logic, not something that
needs the model. The LLM is spent only inside the tools that truly need it.
"""


def plan_next_action(state):
    if state.error:
        return "fail"
    if state.budget_exceeded():          # #20 — stop if we've hit the call ceiling
        return "finish_budget"
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
    if not state.has("ranked"):
        return "rank_jobs"
    return "done"