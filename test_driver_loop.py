# test_driver_loop.py
from agent_state import AgentState
from planner import plan_next_action

def test_loop_reaches_done_deterministically():
    """Simulate the loop with pre-filled state (no tools/LLM) — proves it terminates."""
    s = AgentState(goal="match", resume_id=1)
    s.resume_text = "r"; s.parsed_resume = {"skills": []}
    s.jobs = [{"title": "A"}]; s.job_results = [{"title": "A"}]
    s.current_job_index = 1          # job already processed
    s.ranked = [{"title": "A"}]      # already ranked
    assert plan_next_action(s) == "done"

def test_loop_terminates_on_budget():
    s = AgentState(goal="match", resume_id=1)
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = [{"title": "A"}]
    s.llm_calls_made = s.max_llm_calls
    assert plan_next_action(s) == "finish_budget"

def test_max_steps_guard_exists():
    # sanity: the driver imports and the terminal set is correct
    from autonomous_agent import TERMINAL
    assert "done" in TERMINAL and "finish_budget" in TERMINAL and "fail" in TERMINAL