"""Round 5, backlog E: every constrained work mode (remote, onsite, hybrid) counts
only VERIFIED matches toward the goal unless the user sets accept_unknown_work_mode.
Unknown-mode postings are still evaluated, shown and labeled. A legacy step context
without the flag counts only when the stored posting states a mode that satisfies
the constraint. Synthetic postings only."""
import json
import uuid

import pytest

import agent_store
from agent_goal import AgentGoal, require_verified_work_mode


def _goal(mode, **c):
    return AgentGoal(target_role="Data Engineer", constraints={"work_mode": mode, **c})


@pytest.mark.parametrize("mode", ["remote", "onsite", "hybrid"])
def test_every_constrained_mode_requires_verification(mode):
    assert require_verified_work_mode(_goal(mode))
    assert not require_verified_work_mode(_goal(mode, accept_unknown_work_mode=True))


def test_no_constraint_requires_nothing():
    assert not require_verified_work_mode(AgentGoal(target_role="Data Engineer"))


def test_verifying_modes_follow_the_constraint():
    assert tuple(require_verified_work_mode(_goal("onsite"))) == ("onsite",)
    assert set(require_verified_work_mode(_goal("onsite", allow_hybrid=True))) == {"onsite", "hybrid"}
    assert tuple(require_verified_work_mode(_goal("hybrid"))) == ("hybrid",)


def test_api_keeps_the_opt_in_for_onsite_and_hybrid():
    import api
    for mode in ("onsite", "hybrid", "remote"):
        body = api.StartRunRequest(resume_id=1, target_role="Data Engineer", work_mode=mode,
                              accept_unknown_work_mode=True)
        goal = api._build_agent_goal(body)
        assert goal.constraints.accept_unknown_work_mode is True
    body = api.StartRunRequest(resume_id=1, target_role="Data Engineer", work_mode="",
                          accept_unknown_work_mode=True)
    assert api._build_agent_goal(body).constraints.accept_unknown_work_mode is False


# ----------------------------------------------------------------- Postgres --

def _db_run():
    from auth import create_user
    from database import get_connection
    uid = create_user("wmc_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id, execution_generation",
                    (uid,))
        return cur.fetchone()[0]


def _posting(work_mode):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO job_postings (title, company, description, location, work_mode,"
                    " source, external_id) VALUES ('Data Engineer', 'Acme', 'Python.', 'Berlin',"
                    " %s, 'adzuna', %s) RETURNING id", (work_mode, "wmc-" + uuid.uuid4().hex))
        return cur.fetchone()[0]


def _step(run_id, job_id, ctx_extra):
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status, final_decision,"
                    " retrieved_context) VALUES (%s, 'job', 0, 'success', 'Apply', %s)",
                    (run_id, json.dumps({"job_id": job_id, **ctx_extra})))


@pytest.mark.db
def test_onsite_goal_counts_only_verified_matches():
    rid = _db_run()
    verified, unknown = _posting("onsite"), _posting(None)
    _step(rid, verified, {"work_mode_verified": True})
    _step(rid, unknown, {"work_mode_verified": False})
    goal = _goal("onsite")
    assert agent_store.qualified_job_ids(
        rid, ["Apply"], require_work_mode_verified=require_verified_work_mode(goal)) == [verified]
    opt_in = _goal("onsite", accept_unknown_work_mode=True)
    assert agent_store.qualified_job_ids(
        rid, ["Apply"], require_work_mode_verified=require_verified_work_mode(opt_in)) == \
        sorted([verified, unknown])


@pytest.mark.db
def test_legacy_context_counts_when_posting_states_the_constrained_mode():
    rid = _db_run()
    onsite, remote, hybrid, none = _posting("onsite"), _posting("remote"), _posting("hybrid"), _posting(None)
    for jid in (onsite, remote, hybrid, none):
        _step(rid, jid, {})                       # legacy context: no flag
    q = agent_store.qualified_job_ids
    assert q(rid, ["Apply"], require_work_mode_verified=require_verified_work_mode(
        _goal("onsite"))) == [onsite]
    assert q(rid, ["Apply"], require_work_mode_verified=require_verified_work_mode(
        _goal("onsite", allow_hybrid=True))) == sorted([onsite, hybrid])
    assert q(rid, ["Apply"], require_work_mode_verified=require_verified_work_mode(
        _goal("remote"))) == [remote]
    # backward-compatible boolean form means a remote constraint
    assert q(rid, ["Apply"], require_work_mode_verified=True) == [remote]
