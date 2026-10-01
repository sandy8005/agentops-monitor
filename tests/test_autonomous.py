"""
Pure-logic tests for the agent engine: graph routing, budget math, disagreement
rules. No DB, no LLM — they run fast and free in CI.

Routing is tested on the REAL routing functions of agent_loop (the only engine;
the fixed-sequence pipeline graph was retired).
"""
import pytest
from agent_state import AgentState
from agent_loop import route_after_setup, route_after_decide, route_after_act, route_after_review


def _ready_state(**kw):
    return AgentState(goal="m", resume_id=1, **kw)


# ----------------------- graph routing -----------------------

def test_setup_failure_goes_straight_to_finalize():
    assert route_after_setup({"stop": {"by": "error", "setup_failed": True}}) == "finalize"
    assert route_after_setup({"stop": None}) == "decide"


def test_decide_stop_finalizes_otherwise_acts():
    assert route_after_decide({"stop": {"by": "limit"}}) == "finalize"
    assert route_after_decide({"stop": None}) == "act"


def test_flagged_job_routes_to_review_before_anything_else():
    assert route_after_act({"review_queue": [{"review_id": "r"}]}) == "review"
    # a limit stop still lets the queued review happen first
    assert route_after_act({"review_queue": [{"review_id": "r"}],
                            "stop": {"by": "limit"}}) == "review"
    assert route_after_act({"review_queue": []}) == "decide"


def test_cancel_takes_priority_over_review():
    assert route_after_act({"review_queue": [{"review_id": "r"}], "cancel_seen": True}) == "finalize"
    assert route_after_act({"review_queue": [{"review_id": "r"}],
                            "stop": {"cancel": True}}) == "finalize"
    assert route_after_review({"review_queue": [{"review_id": "r2"}],
                               "stop": {"cancel": True}}) == "finalize"


def test_review_queue_drains_then_continues():
    assert route_after_review({"review_queue": [{"review_id": "r2"}]}) == "review"
    assert route_after_review({"review_queue": []}) == "decide"
    assert route_after_review({"review_queue": [], "stop": {"by": "user"}}) == "finalize"


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

# ----------------------- disagreement logic (record_score) -----------------------

def _would_flag(score_decision, llm_decision):
    """Exercise the REAL rule in llm.record_score (no DB: flag_for_review and the
    UPDATE are replaced)."""
    import llm
    flagged = []

    class _Cur:
        def execute(self, *a, **k):
            pass

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _Cur()

    orig_conn, orig_flag = llm.get_connection, llm.flag_for_review
    llm.get_connection = lambda: _Conn()
    llm.flag_for_review = lambda step_id, reason=None: flagged.append(reason)
    try:
        return llm.record_score(1, 50.0, score_decision, llm_decision)
    finally:
        llm.get_connection, llm.flag_for_review = orig_conn, orig_flag

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