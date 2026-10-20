"""Backlog P2: agent_tools._ToolState forwards authorize_dispatch() to the durable
budget, so llm.py's getattr(budget, "authorize_dispatch", None) finds it."""
from types import SimpleNamespace

import pytest

import llm
from agent_tools import _tool_state


class _Budget:
    def __init__(self, cancelled=False):
        self.calls = []
        self.cancelled = cancelled

    def check_owner(self):
        self.calls.append("check_owner")

    def authorize_dispatch(self):
        self.calls.append("authorize_dispatch")
        if self.cancelled:
            raise RuntimeError("cancelled")


def _ctx(budget):
    goal = SimpleNamespace(
        description="find jobs", target_role="Backend Engineer", evaluate_quality=False,
        model_policy="auto",
        constraints=SimpleNamespace(location="", work_mode="", employment_type=""))
    return SimpleNamespace(goal=goal, budget=budget, generation=7,
                           state={"resume_id": 1, "resume_text": "x", "parsed_resume": {}})


def test_tool_state_exposes_authorize_dispatch():
    b = _Budget()
    ts = _tool_state(_ctx(b))
    authorize = getattr(ts, "authorize_dispatch", None)
    assert authorize is not None
    authorize()
    assert b.calls == ["authorize_dispatch"]


def test_llm_backoff_runs_the_dispatch_check_through_tool_state():
    b = _Budget()
    llm._interruptible_sleep(0.0, _tool_state(_ctx(b)))
    assert "authorize_dispatch" in b.calls


def test_dispatch_refusal_propagates_through_tool_state():
    with pytest.raises(RuntimeError):
        llm._interruptible_sleep(0.0, _tool_state(_ctx(_Budget(cancelled=True))))
