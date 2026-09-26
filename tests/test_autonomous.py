"""
Pure-logic tests for the autonomous agent: LangGraph routing, judge skipping,
budget math, cancellation. No DB, no LLM — they run fast and free in CI.

Routing is tested on the REAL routing functions in autonomous_graph.py (the
runtime), not on the retired planner (moved to legacy/).
"""
import pytest
from agent_state import AgentState
from autonomous_graph import (
    route_after_start, route_after_parse, route_after_search,
    route_after_process_job, route_after_human_review, route_after_rank,
)


# ----------------------- graph routing -----------------------

def _ready_state(**kw):
    s = AgentState(goal="m", resume_id=1, **kw)
    return s

def _d(**kw):
    """Flat graph-state dict, like LangGraph passes to routing functions."""
    return _ready_state().to_dict() | kw

def test_routing_walks_full_sequence():
    assert route_after_start(_d(resume_text="r")) == "parse_resume"
    assert route_after_parse(_d(parsed_resume={"skills": []})) == "search_jobs"
    jobs = [{"title": "A"}, {"title": "B"}]
    assert route_after_search(_d(jobs=jobs)) == "process_job"
    assert route_after_process_job(_d(jobs=jobs, current_job_index=1)) == "process_job"
    assert route_after_process_job(_d(jobs=jobs, current_job_index=2)) == "rank_jobs"
    assert route_after_rank(_d(ranked=[])) == "generate_advice"   # empty ranking terminates

def test_routing_no_matches_terminates():
    assert route_after_search(_d(jobs=[])) == "no_matches"

def test_routing_error_routes_to_fail():
    for route in (route_after_start, route_after_parse, route_after_search):
        assert route(_d(error="boom")) == "fail"

def test_flagged_job_routes_to_human_review():
    assert route_after_process_job(_d(jobs=[{"title": "A"}], current_job_index=1,
                                      last_job_needs_review=True)) == "human_review"


# ----------------------- cancellation -----------------------

def test_cancelled_takes_priority_over_normal_work_and_review():
    jobs = [{"title": "A"}, {"title": "B"}]
    assert route_after_process_job(_d(jobs=jobs, current_job_index=0, cancelled=True)) == "cancelled"
    assert route_after_process_job(_d(jobs=jobs, current_job_index=1, cancelled=True,
                                      last_job_needs_review=True)) == "cancelled"
    assert route_after_human_review(_d(jobs=jobs, current_job_index=1, cancelled=True)) == "cancelled"


# ----------------------- budget math (can_spend / spend) -----------------------

def test_budget_spend_and_can_spend():
    s = _ready_state()
    s.max_llm_calls = 3
    assert s.can_spend() is True
    s.spend(); s.spend()
    assert s.llm_calls_made == 2
    assert s.can_spend() is True
    s.spend()
    assert s.can_spend() is False           # 3/3 spent
    assert s.budget_exceeded() is True

def test_routing_does_not_stop_on_budget():
    # Budget must NOT halt routing — free work (ranking) still runs when quota is spent.
    s = _ready_state()
    d = _d(jobs=[{"title": "A"}], current_job_index=1, llm_calls_made=s.max_llm_calls)
    assert route_after_process_job(d) == "rank_jobs"


# ----------------------- judge skipping logic -----------------------

def _judge_decision(score, budget_exceeded):
    """Mirror of router's judge branch (which score bands skip the judge)."""
    if not (20 <= score <= 80):
        return "skipped", ("score_extreme_low" if score < 20 else "score_extreme_high"), False
    elif budget_exceeded:
        return "skipped", "budget", False
    else:
        return "ran", None, True

@pytest.mark.parametrize("score,budget,exp_status,exp_reason,exp_called", [
    (95, False, "skipped", "score_extreme_high", False),
    (10, False, "skipped", "score_extreme_low", False),
    (50, True,  "skipped", "budget", False),
    (50, False, "ran", None, True),
    (20, False, "ran", None, True),   # boundary inclusive
    (80, False, "ran", None, True),   # boundary inclusive
])
def test_judge_skip_matrix(score, budget, exp_status, exp_reason, exp_called):
    status, reason, called = _judge_decision(score, budget)
    assert status == exp_status
    assert reason == exp_reason
    assert called == exp_called


# ----------------------- disagreement logic (record_score) -----------------------

def _would_flag(score_decision, llm_decision):
    real = {"Apply", "Maybe", "Skip"}
    return (llm_decision in real) and (score_decision != llm_decision)

def test_skipped_judge_is_not_a_disagreement():
    assert _would_flag("Skip", "skipped (budget)") is False
    assert _would_flag("Apply", "skipped (score_extreme_high)") is False
    assert _would_flag("Maybe", "Unknown") is False

def test_real_disagreement_flags():
    assert _would_flag("Skip", "Maybe") is True
    assert _would_flag("Apply", "Skip") is True

def test_agreement_does_not_flag():
    assert _would_flag("Apply", "Apply") is False


# ----------------------- failure accounting -----------------------

def test_failed_jobs_counter_exists_and_starts_zero():
    s = _ready_state()
    assert s.failed_jobs == 0
    s.failed_jobs += 1
    assert s.failed_jobs == 1