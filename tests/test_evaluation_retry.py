"""Backlog A: a failed job evaluation is retried (bounded), not silently dropped.

Synthetic data only; the evaluation pipeline (router.do_process_job) and the
store are stubbed, so no database or model is touched.
"""
import pytest

import agent_store
import agent_tools
import router
from agent_controller import rules_decide
from agent_goal import AgentGoal
from agent_tools import (EvaluateArgs, MAX_EVALUATION_ATTEMPTS, ToolContext, ToolRejected,
                         finish_permitted, tool_evaluate_jobs, unevaluated_eligible_ids)


class _Budget:
    def check_owner(self):
        pass


def _ctx(state):
    goal = AgentGoal(target_role="Data Engineer", target_count=1)
    return ToolContext(1, 1, goal, state, _Budget(), 2)


def _state():
    return {"discovered": {"7": {"id": 7, "title": "Data Engineer", "company": "Acme",
                                 "eligible": True}},
            "evaluated": {}, "searches": []}


@pytest.fixture
def stubs(monkeypatch):
    calls = {"n": 0, "fail_until": 1}
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {i: {"id": i, "title": "Data Engineer", "company": "Acme",
                                         "description": "Python and SQL."} for i in ids})
    monkeypatch.setattr(agent_tools, "_existing_evaluation", lambda rid, jid: None)
    monkeypatch.setattr(agent_tools, "qualified_count", lambda ctx: 0)

    def process(ts, run_id):
        calls["n"] += 1
        if calls["n"] <= calls["fail_until"]:
            ts.failed_jobs += 1                     # a per-job failure, absorbed
            return
        ts.job_results.append({"step_id": 100 + calls["n"], "job_id": 7, "title": "Data Engineer",
                               "company": "Acme", "score": 90.0, "decision": "Apply",
                               "llm_decision": "skipped", "final_decision": "Apply",
                               "needs_review": False})
    monkeypatch.setattr(router, "do_process_job", process)
    return calls


def test_one_transient_failure_keeps_the_job_unevaluated(stubs):
    state = _state()
    obs, _, _ = tool_evaluate_jobs(_ctx(state), EvaluateArgs(job_ids=[7]))
    assert obs["failed_job_ids"] == [7]
    entry = state["evaluated"]["7"]
    assert entry["status"] == "failed" and entry["attempts"] == 1 and not entry["terminal"]
    # Still queued: the controller is offered it again and finish is refused.
    assert unevaluated_eligible_ids(state) == [7]
    ok, why = finish_permitted(_ctx(state).goal, state, 0)
    assert not ok and "unevaluated" in why
    assert rules_decide(_ctx(state).goal, state, 0).action == "evaluate_jobs"
    # The retry succeeds and replaces the failure.
    obs, _, progressed = tool_evaluate_jobs(_ctx(state), EvaluateArgs(job_ids=[7]))
    assert obs["evaluated"] == 1 and progressed
    assert state["evaluated"]["7"]["status"] == "ok"
    assert unevaluated_eligible_ids(state) == []


def test_attempt_cap_stops_infinite_retries(stubs):
    stubs["fail_until"] = 99
    state = _state()
    for n in range(1, MAX_EVALUATION_ATTEMPTS + 1):
        tool_evaluate_jobs(_ctx(state), EvaluateArgs(job_ids=[7]))
        assert state["evaluated"]["7"]["attempts"] == n
    entry = state["evaluated"]["7"]
    assert entry["status"] == "failed" and entry["terminal"]
    assert unevaluated_eligible_ids(state) == []
    with pytest.raises(ToolRejected):
        tool_evaluate_jobs(_ctx(state), EvaluateArgs(job_ids=[7]))
    assert stubs["n"] == MAX_EVALUATION_ATTEMPTS


def test_missing_posting_is_a_terminal_failure(stubs, monkeypatch):
    monkeypatch.setattr(agent_store, "load_postings", lambda ids: {})
    state = _state()
    tool_evaluate_jobs(_ctx(state), EvaluateArgs(job_ids=[7]))
    entry = state["evaluated"]["7"]
    assert entry["status"] == "failed" and entry["terminal"] and entry["attempts"] == 1
    assert unevaluated_eligible_ids(state) == []
    assert stubs["n"] == 0


def test_legacy_failed_entry_without_attempts_is_retryable_once():
    state = _state()
    state["evaluated"] = {"7": {"job_id": 7, "status": "failed"}}
    assert unevaluated_eligible_ids(state) == [7]
