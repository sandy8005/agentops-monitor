"""
Planner: given AgentState, decide the next action. Pure Python — zero LLM calls.
This single function IS items #13–18 (Python controls routing, stopping, etc.)
and is the natural home for every 'when do we spend a Gemini call?' rule.
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
    if not state.ranking_done:           # use the flag, not has("ranked") —
        return "rank_jobs"               # an empty ranked list must still count as done
    return "done"