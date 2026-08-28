# test_planner.py
from agent_state import AgentState
from planner import plan_next_action

def test_planner_walks_workflow():
    s = AgentState(goal="match", resume_id=1)
    assert plan_next_action(s) == "load_resume"
    s.resume_text = "r"
    assert plan_next_action(s) == "parse_resume"
    s.parsed_resume = {"skills": []}
    assert plan_next_action(s) == "search_jobs"
    s.jobs = [{"title": "A"}, {"title": "B"}]
    assert plan_next_action(s) == "process_job"
    s.current_job_index = 2
    assert plan_next_action(s) == "rank_jobs"
    s.ranked = ["A", "B"]
    assert plan_next_action(s) == "done"

def test_planner_stops_on_budget():
    s = AgentState(goal="match", resume_id=1)
    s.resume_text = "r"; s.parsed_resume = {"skills": []}
    s.jobs = [{"title": "A"}]
    s.llm_calls_made = 30
    assert plan_next_action(s) == "finish_budget"

def test_planner_no_matches():
    s = AgentState(goal="match", resume_id=1)
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = []
    assert plan_next_action(s) == "finish_no_matches"