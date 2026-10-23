"""Backlog E: a posting with no stated work mode is kept by a remote search but is
never presented as a confirmed remote match. It is labeled
work_mode = "unknown" / work_mode_verified = False through search, evaluation,
ranking and the API, and counts toward a remote goal only when the user allows
it (accept_unknown_work_mode). Synthetic postings only.
"""
import uuid

import pytest

import agent_store
import agent_tools
import llm
import router
from agent_goal import AgentGoal, require_verified_work_mode
from agent_tools import EvaluateArgs, SearchArgs, ToolContext, tool_evaluate_jobs, tool_search_jobs
from job_source import work_mode_ok, work_mode_status

REMOTE = AgentGoal(target_role="Data Engineer", providers=["remotive"],
                   constraints={"work_mode": "remote"})


@pytest.mark.parametrize("job,status", [
    ({"work_mode": "remote"}, "verified"),
    ({"work_mode": "", "location": "Remote - Europe"}, "verified"),
    ({"work_mode": None, "location": "Berlin"}, "unknown"),
    ({"work_mode": "onsite"}, "mismatch"),
    ({"work_mode": "hybrid"}, "mismatch"),
])
def test_work_mode_status(job, status):
    assert work_mode_status(job, "remote") == status
    assert work_mode_ok(job, "remote") == (status != "mismatch")


def test_hybrid_allowed_is_verified_not_unknown():
    assert work_mode_status({"work_mode": "hybrid"}, "remote", allow_hybrid=True) == "verified"


def test_goal_counting_rule():
    assert require_verified_work_mode(REMOTE)
    opt_in = AgentGoal(target_role="Data Engineer",
                       constraints={"work_mode": "remote", "accept_unknown_work_mode": True})
    assert not require_verified_work_mode(opt_in)
    assert not require_verified_work_mode(AgentGoal(target_role="Data Engineer"))


class _Budget:
    def check_owner(self):
        pass


def test_search_labels_unknown_mode_postings(monkeypatch):
    jobs = [{"id": 1, "title": "Data Engineer", "company": "A", "source": "remotive",
             "work_mode": "remote", "work_mode_verified": True},
            {"id": 2, "title": "Data Engineer", "company": "B", "source": "remotive",
             "work_mode": None, "work_mode_verified": False}]
    monkeypatch.setattr(agent_tools, "_collect_candidates", lambda ctx, p, q: jobs)
    monkeypatch.setattr(agent_tools, "_fetch_with_attempt", lambda *a: "success")
    monkeypatch.setattr(agent_store, "search_history", lambda *a: [])
    monkeypatch.setattr(agent_store, "record_search", lambda *a, **k: None)
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 1)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    state = {"searches": [], "discovered": {}}
    obs, _, _ = tool_search_jobs(ToolContext(1, 1, REMOTE, state, _Budget(), 1),
                                 SearchArgs(provider="remotive", query="data engineer"))
    assert state["discovered"]["1"]["work_mode"] == "remote"
    assert state["discovered"]["1"]["work_mode_verified"] is True
    assert state["discovered"]["2"]["work_mode"] == "unknown"
    assert state["discovered"]["2"]["work_mode_verified"] is False
    assert obs["unknown_work_mode"] == 1


def test_evaluation_carries_the_label(monkeypatch):
    seen = []
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {2: {"id": 2, "title": "Data Engineer", "company": "B",
                                         "description": "Python.", "work_mode": None}})
    monkeypatch.setattr(agent_tools, "_existing_evaluation", lambda rid, jid: None)
    monkeypatch.setattr(agent_tools, "qualified_count", lambda ctx: 0)

    def process(ts, run_id):
        seen.append(dict(ts.jobs[0]))
        ts.job_results.append({"step_id": 5, "job_id": 2, "title": "Data Engineer",
                               "company": "B", "score": 90.0, "decision": "Apply",
                               "final_decision": "Apply", "needs_review": False})
    monkeypatch.setattr(router, "do_process_job", process)
    state = {"discovered": {"2": {"id": 2, "title": "Data Engineer", "company": "B",
                                  "eligible": True, "work_mode": "unknown",
                                  "work_mode_verified": False}}, "evaluated": {}}
    tool_evaluate_jobs(ToolContext(1, 1, REMOTE, state, _Budget(), 2), EvaluateArgs(job_ids=[2]))
    assert seen[0]["work_mode_verified"] is False
    assert state["evaluated"]["2"]["work_mode"] == "unknown"
    assert state["evaluated"]["2"]["work_mode_verified"] is False


# ----------------------------------------------------------------- Postgres --

def _db_run():
    from auth import create_user
    from database import get_connection
    uid = create_user("wm_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id, execution_generation",
                    (uid,))
        rid, gen = cur.fetchone()
    return uid, rid, gen


def _step(run_id, job_id, verified):
    import json
    from database import get_connection
    ctx = {"job_id": job_id}
    if verified is not None:
        ctx["work_mode_verified"] = verified
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status, final_decision,"
                    " retrieved_context) VALUES (%s, 'job', 0, 'success', 'Apply', %s)",
                    (run_id, json.dumps(ctx)))


@pytest.mark.db
def test_unknown_mode_matches_are_not_counted_unless_allowed():
    _uid, rid, _gen = _db_run()
    _step(rid, 1, True)
    _step(rid, 2, False)
    _step(rid, 3, None)            # legacy context, no stored posting: fails closed (backlog D)
    assert agent_store.qualified_job_ids(rid, ["Apply"]) == [1, 2, 3]
    assert agent_store.qualified_job_ids(rid, ["Apply"], require_work_mode_verified=True) == [1]


@pytest.mark.db
def test_rankings_and_api_show_the_label():
    import api
    uid, rid, gen = _db_run()
    agent_store.persist_rankings(rid, gen, [
        {"job_id": 1, "title": "Data Engineer", "company": "A", "score": 90,
         "final_decision": "Apply", "work_mode": "remote", "work_mode_verified": True},
        {"job_id": 2, "title": "Data Engineer", "company": "B", "score": 80,
         "final_decision": "Apply", "work_mode": "unknown", "work_mode_verified": False}])
    out = api.get_run_rankings(rid, user={"id": uid})["rankings"]
    assert [(r["work_mode"], r["work_mode_verified"]) for r in out] == \
        [("remote", True), ("unknown", False)]
