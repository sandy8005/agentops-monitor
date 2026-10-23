"""Backlog D: advice is generated from the requirements the evaluation used.

The requirements of each evaluation are persisted (evaluation_requirements,
migration 0016) fenced by the execution generation and covered by erasure; advice
reads only that snapshot and fails the item visibly when it is missing — never
falling back to empty placeholder requirements. Synthetic text only.
"""
import uuid

import pytest

import agent_store
import agent_tools
import llm
import resume_advisor
import router
from agent_goal import AgentGoal

GOAL = AgentGoal(target_role="Data Engineer", use_llm_advice=False)
SNAP_REQS = {"required_skills": ["python", "sql"], "required_any_of": [],
             "preferred_skills": [], "min_years_experience": 2, "responsibilities": []}


@pytest.fixture
def advice_env(monkeypatch):
    seen = {"reqs": [], "snapshot_calls": []}
    monkeypatch.setattr(agent_store, "advised_job_ids", lambda rid: [])
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {i: {"id": i, "title": "Data Engineer", "company": "Acme",
                                         "description": "Python and SQL required."} for i in ids})
    monkeypatch.setattr(agent_store, "persist_advice", lambda *a, **k: None)
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 9)
    monkeypatch.setattr(llm, "finish_step", lambda *a, **k: None)
    monkeypatch.setattr(llm, "fail_step", lambda *a, **k: None)
    # The cache must NOT be the source: a miss here used to become empty requirements.
    monkeypatch.setattr(router, "_reqs_cache_get", lambda h: (None, None))

    def build(parsed, text, job, reqs, sc, **k):
        seen["reqs"].append(reqs)
        return [], "rules"
    monkeypatch.setattr(resume_advisor, "build_suggestions", build)
    return seen


def _state():
    return {"evaluated": {"1": {"job_id": 1, "status": "ok", "final_decision": "Apply",
                                "step_id": 41}},
            "resume_text": "Data engineer. Python and SQL.",
            "parsed_resume": {"skills": ["python", "sql"]}}


def test_missing_snapshot_fails_the_item_visibly(advice_env, monkeypatch):
    # Backlog E (round 4) reconstructs a pre-0016 snapshot from the posting; the
    # visible failure remains when that is impossible (posting text erased).
    monkeypatch.setattr(agent_store, "evaluation_requirements", lambda rid, sid: None)
    monkeypatch.setattr(agent_store, "load_postings",
                        lambda ids: {i: {"id": i, "title": "Data Engineer", "company": "Acme",
                                         "description": ""} for i in ids})
    monkeypatch.setattr(agent_store, "record_evaluation_requirements",
                        lambda *a, **k: pytest.fail("nothing to reconstruct from"))
    state = _state()
    ctx = agent_tools.ToolContext(1, 1, GOAL, state, None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["failed_job_ids"] == [1] and obs["advised"] == []
    assert obs["generation"]["1"] == "requirements_snapshot_missing"
    assert advice_env["reqs"] == []            # never built on placeholder requirements
    assert state["output_failures"] == 1


def test_advice_uses_the_evaluation_snapshot(advice_env, monkeypatch):
    asked = []
    monkeypatch.setattr(agent_store, "evaluation_requirements",
                        lambda rid, sid: asked.append((rid, sid)) or (SNAP_REQS, {"m": "llm"}))
    ctx = agent_tools.ToolContext(1, 1, GOAL, _state(), None, 4)
    obs, _sid, _p = agent_tools.execute(ctx, "generate_advice", {"job_ids": [1]})
    assert obs["advised"] == [1] and asked == [(1, 41)]
    assert advice_env["reqs"][0]["required_skills"] == ["python", "sql"]
    assert advice_env["reqs"][0]["min_years_experience"] == 2


# ----------------------------------------------------------------- Postgres --

def _db_run():
    from auth import create_user
    from database import get_connection
    uid = create_user("as_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id", (uid,))
        rid = cur.fetchone()[0]
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status) "
                    "VALUES (%s, 'job', 0, 'running') RETURNING id", (rid,))
        sid = cur.fetchone()[0]
        cur.execute("SELECT execution_generation FROM runs WHERE id = %s", (rid,))
        return rid, sid, cur.fetchone()[0]


@pytest.mark.db
def test_snapshot_round_trip_and_generation_fence():
    rid, sid, gen = _db_run()
    with pytest.raises(agent_store.ExecutionLost):
        agent_store.record_evaluation_requirements(rid, gen + 1, sid, 7, SNAP_REQS, {})
    assert agent_store.evaluation_requirements(rid, sid) is None
    agent_store.record_evaluation_requirements(rid, gen, sid, 7, SNAP_REQS,
                                               {"extraction_method": "rule_based"})
    reqs, prov = agent_store.evaluation_requirements(rid, sid)
    assert reqs["required_skills"] == ["python", "sql"]
    assert prov["extraction_method"] == "rule_based"
    # Idempotent per step: a replay never rewrites the decided-on snapshot.
    agent_store.record_evaluation_requirements(rid, gen, sid, 7, {"required_skills": []}, {})
    assert agent_store.evaluation_requirements(rid, sid)[0]["required_skills"] == ["python", "sql"]


@pytest.mark.db
def test_erasure_blanks_snapshot_and_late_writes():
    import privacy
    from database import get_connection
    rid, sid, gen = _db_run()
    agent_store.record_evaluation_requirements(rid, gen, sid, 7, SNAP_REQS, {})
    with get_connection() as c:
        cur = c.cursor()
        privacy._tombstone_runs(cur, [rid], "retention")
        privacy._scrub_run_payloads(cur, [rid])
    assert agent_store.evaluation_requirements(rid, sid) is None
    # A late write that bypasses the fence is blanked by the trigger.
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status) "
                    "VALUES (%s, 'job', 1, 'running') RETURNING id", (rid,))
        sid2 = cur.fetchone()[0]
        cur.execute("INSERT INTO evaluation_requirements (run_id, step_id, job_id, "
                    "execution_generation, requirements) VALUES (%s, %s, 7, 1, %s)",
                    (rid, sid2, '{"required_skills": ["python"]}'))
        cur.execute("SELECT requirements FROM evaluation_requirements WHERE step_id = %s",
                    (sid2,))
        assert cur.fetchone()[0] is None


@pytest.mark.db
def test_evaluation_writes_the_snapshot_before_success(monkeypatch):
    """The real do_process_job, with model/score stubbed, persists the snapshot
    through the tool state's fenced hook."""
    rid, _sid, gen = _db_run()
    monkeypatch.setattr(router, "_reqs_cache_get", lambda dhash: (None, None))
    monkeypatch.setattr(router, "_llm_allowed", lambda s: False)
    monkeypatch.setattr(router, "_reqs_cache_put", lambda *a: None)
    goal = AgentGoal(target_role="Data Engineer", model_policy="rules_only")

    class _Budget:
        generation = gen

        def check_owner(self):
            pass

        def exhausted(self):
            return True
    ctx = agent_tools.ToolContext(rid, gen, goal, {"resume_text": "Python SQL",
                                                    "parsed_resume": {"skills": ["python"]}},
                                  _Budget(), 2)
    ts = agent_tools._tool_state(ctx)
    from database import get_connection
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO job_postings (title, company, description, source, external_id,"
                    " last_seen_at, created_at) VALUES ('Data Engineer', 'Acme', "
                    "'Python and SQL required.', 'adzuna', %s, NOW(), NOW()) RETURNING id",
                    ("adzuna:as-" + uuid.uuid4().hex[:10],))
        jid = cur.fetchone()[0]
    ts.jobs = [{"id": jid, "title": "Data Engineer", "company": "Acme",
                "description": "Python and SQL required.", "source": "adzuna"}]
    ts.current_job_index = 0
    router.do_process_job(ts, rid)
    assert ts.job_results, "evaluation did not complete"
    step_id = ts.job_results[-1]["step_id"]
    reqs, prov = agent_store.evaluation_requirements(rid, step_id)
    assert "python" in reqs["required_skills"] and prov["extraction_method"] == "rule_based"
