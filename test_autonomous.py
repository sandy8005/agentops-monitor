"""
Pure-logic tests for the autonomous agent: planner routing, judge skipping,
budget math, cancellation, and (future) human-review routing. No DB, no LLM —
these mirror the decision logic so they run fast and free in CI.

Run:  pytest test_autonomous.py -v
"""
import pytest
from agent_state import AgentState
from planner import plan_next_action


# ----------------------- planner routing -----------------------

def _ready_state(**kw):
    s = AgentState(goal="m", resume_id=1, **kw)
    return s

def test_planner_walks_full_sequence():
    s = _ready_state()
    assert plan_next_action(s) == "load_resume"
    s.resume_text = "r"
    assert plan_next_action(s) == "parse_resume"
    s.parsed_resume = {"skills": []}
    assert plan_next_action(s) == "search_jobs"
    s.jobs = [{"title": "A"}, {"title": "B"}]
    assert plan_next_action(s) == "process_job"
    s.current_job_index = 2
    assert plan_next_action(s) == "rank_jobs"
    s.ranking_done = True
    s.ranked = [{"title": "A", "decision": "Apply"}]
    assert plan_next_action(s) == "generate_advice"
    s.advice_done = True
    assert plan_next_action(s) == "done"

def test_planner_no_matches_terminates():
    s = _ready_state()
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = []
    assert plan_next_action(s) == "finish_no_matches"

def test_planner_empty_ranked_still_terminates():
    # regression: empty ranked list must NOT loop on rank_jobs
    s = _ready_state()
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = [{"title": "A"}]
    s.current_job_index = 1; s.ranking_done = True; s.ranked = []; s.advice_done = True
    assert plan_next_action(s) == "done"

def test_planner_error_routes_to_fail():
    s = _ready_state()
    s.error = "boom"
    assert plan_next_action(s) == "fail"


# ----------------------- cancellation -----------------------

def test_planner_cancelled_routes_to_finish_cancelled():
    s = _ready_state()
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = [{"title": "A"}]
    s.cancelled = True
    assert plan_next_action(s) == "finish_cancelled"

def test_cancelled_takes_priority_over_normal_work():
    s = _ready_state()
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = [{"title": "A"}]
    s.current_job_index = 0        # work remains
    s.cancelled = True
    assert plan_next_action(s) == "finish_cancelled"   # cancel wins


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

def test_planner_does_not_stop_on_budget():
    # Budget must NOT halt the planner — free work (ranking) still runs when quota is spent.
    s = _ready_state()
    s.resume_text = "r"; s.parsed_resume = {"skills": []}; s.jobs = [{"title": "A"}]
    s.current_job_index = 1        # jobs done
    s.llm_calls_made = s.max_llm_calls   # budget exhausted
    # should route to rank_jobs (free), NOT stop
    assert plan_next_action(s) == "rank_jobs"


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