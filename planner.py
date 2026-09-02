"""
Planner: given AgentState, decide the next action. Pure Python — zero LLM calls.
This single function IS items #13–18 (Python controls routing, stopping, etc.)
and is the natural home for every 'when do we spend a Gemini call?' rule.
"""


def plan_next_action(state):
    if state.error:
        return "fail"
    if state.cancelled:
        return "finish_cancelled"
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
    if not state.ranking_done:           # flag, not has("ranked") — empty list still counts as done
        return "rank_jobs"
    if not state.advice_done:            # generate combined advice for top viable jobs (#9,#10)
        return "generate_advice"
    return "done"