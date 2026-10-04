"""
Round-4 review fixes: cancellation is a PRECONDITION of every external dispatch,
not a flag polled near it.

  * the worker re-reads cancel_requested AFTER begin_execution, so a cancel that
    commits between the worker's (stale) config read and begin_execution stops the
    run before setup — including the real interleaving where the API sees the run
    'queued' after the worker already claimed the QUEUE row (different rows, so
    the two locks do not serialize);
  * reserve_llm_call, begin_search_attempt and authorize_dispatch refuse a
    cancelled run atomically (RunCancelled), so no model request, model retry,
    token count or job-provider search starts after a committed cancel;
  * a refused dispatch ends the run as CANCELLED (never failed, never absorbed by
    a rules fallback);
  * setup is gated by check_limits before the resume parse; the runtime limit
    stops at >=; max_llm_calls=0 needs no price;
  * the release checker rejects reports/, dist/, build/, editor and OS files.
"""
import json
import threading
import time
import uuid

import pytest

import agent_loop
import agent_store
from agent_goal import AgentGoal


# ================================================================ helpers ====

def _user():
    from auth import create_user
    return create_user("cd_" + uuid.uuid4().hex[:10], "password-1234")


def _db():
    from database import get_connection
    return get_connection()


def _one(sql, params=()):
    with _db() as c:
        cur = c.cursor()
        cur.execute(sql, params)
        return cur.fetchone()


def _exec(sql, params=()):
    with _db() as c:
        c.cursor().execute(sql, params)


def _queued_agent_run(uid, claimed=True, **limits):
    """An agent run in 'queued' whose start_run job a worker has ALREADY claimed
    (queue row 'running') — the window the review describes."""
    with _db() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO resumes (name, resume_text, created_at, user_id) "
                    "VALUES ('r', %s, NOW(), %s) RETURNING id",
                    ("Python engineer, 5 years " + uuid.uuid4().hex, uid))
        resume_id = cur.fetchone()[0]
    from llm import create_run
    goal = AgentGoal(target_role="AI Engineer", limits=limits or {})
    run_id = create_run("t", resume_id=resume_id, user_id=uid, goal=goal)
    with _db() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO job_queue (kind, payload, run_id, status, max_attempts, "
                    "enqueued_at) VALUES ('start_run', %s, %s, %s, 3, NOW()) RETURNING id",
                    (json.dumps({"run_id": run_id}), run_id,
                     "running" if claimed else "queued"))
        job_id = cur.fetchone()[0]
    return run_id, job_id


def _no_setup_side_effects(monkeypatch):
    """Record any setup work / checkpointer use: none may happen after a cancel."""
    import router
    calls = []
    monkeypatch.setattr(router, "load_resume", lambda *a, **k: calls.append("load_resume"))
    monkeypatch.setattr(router, "do_parse_resume", lambda *a, **k: calls.append("parse"))

    def factory():
        calls.append("checkpointer")
        raise AssertionError("the graph must not start after a cancel")
    return calls, factory


# ======================================== 1. startup race (db, the review) ====

@pytest.mark.db
def test_cancel_between_stale_config_read_and_begin_execution(monkeypatch):
    """Exactly the review's interleaving: worker claimed the queue row, read
    cancel_requested=false; the API (still seeing 'queued') finds no queued job to
    cancel and sets the flag; THEN begin_execution. The worker must finalize
    'cancelled' before setup — no resume load, no parse, no graph."""
    import api
    uid = _user()
    run_id, _job = _queued_agent_run(uid)
    calls, factory = _no_setup_side_effects(monkeypatch)
    real_load = agent_store.load_run_config
    api_result = {}

    def load_then_user_cancels(rid):
        cfg = real_load(rid)                       # stale: cancel_requested False
        assert cfg["cancel_requested"] is False
        api_result.update(api.cancel_run(rid, user={"id": uid}, _csrf=None))
        return cfg
    monkeypatch.setattr(agent_store, "load_run_config", load_then_user_cancels)

    out = agent_loop.run_agent_loop(run_id, checkpointer_factory=factory)

    # The API could not cancel the (already claimed) job — it fell back to the flag.
    assert api_result == {"run_id": run_id, "cancel_requested": True}
    assert out == {"cancelled": True}
    assert calls == []
    status, code = _one("SELECT status, error_code FROM runs WHERE id = %s", (run_id,))
    assert (status, code) == ("cancelled", "cancelled")
    assert _one("SELECT COUNT(*) FROM llm_cost_reservations WHERE run_id = %s",
                (run_id,))[0] == 0


@pytest.mark.db
def test_api_run_lock_blocks_begin_execution_and_worker_sees_the_cancel(monkeypatch):
    """Real concurrency: the API holds the RUN row lock (status still 'queued')
    while the worker calls begin_execution. begin_execution must wait for that
    lock, and the post-begin_execution re-read must see the committed cancel."""
    uid = _user()
    run_id, _job = _queued_agent_run(uid)
    calls, factory = _no_setup_side_effects(monkeypatch)

    api_conn = _db()
    cur = api_conn.cursor()
    cur.execute("SELECT status FROM runs WHERE id = %s FOR UPDATE", (run_id,))
    assert cur.fetchone()[0] == "queued"

    result = {}
    worker = threading.Thread(
        target=lambda: result.update(
            out=agent_loop.run_agent_loop(run_id, checkpointer_factory=factory)))
    worker.start()
    # Wait until the worker is BLOCKED on our row lock inside begin_execution.
    deadline = time.time() + 10
    while time.time() < deadline:
        waiting = _one("SELECT COUNT(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                       "AND query ILIKE '%%execution_generation = execution_generation + 1%%'")[0]
        if waiting:
            break
        time.sleep(0.05)
    else:
        api_conn.rollback()
        api_conn.close()
        worker.join(10)
        pytest.fail("worker never reached begin_execution")

    # The API's 'queued' branch: no queued job to cancel (claimed) -> set the flag.
    cur.execute("UPDATE job_queue SET status = 'cancelled' "
                "WHERE run_id = %s AND status = 'queued'", (run_id,))
    assert cur.rowcount == 0
    cur.execute("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
    api_conn.commit()
    api_conn.close()

    worker.join(15)
    assert not worker.is_alive()
    assert result["out"] == {"cancelled": True}
    assert calls == []
    assert _one("SELECT status FROM runs WHERE id = %s", (run_id,))[0] == "cancelled"


@pytest.mark.db
def test_resume_job_rereads_cancel_after_begin_execution(monkeypatch):
    """resume_agent_loop had the same stale read: cancel must win there too."""
    uid = _user()
    run_id, _job = _queued_agent_run(uid)
    real_load = agent_store.load_run_config

    def load_then_cancel(rid):
        cfg = real_load(rid)
        _exec("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (rid,))
        return cfg
    monkeypatch.setattr(agent_store, "load_run_config", load_then_cancel)

    class _Snap:
        values, next = {}, ()

    class _Graph:
        def get_state(self, config):
            return _Snap()

    class _CP:
        def __enter__(self):
            return object()

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(agent_loop, "build_agent_graph", lambda cp: _Graph())
    monkeypatch.setattr(agent_loop, "pending_interrupt_value", lambda g, c: None)
    out = agent_loop.resume_agent_loop(run_id, {"review_id": "x"}, checkpointer_factory=_CP)
    assert out == {"cancelled": True}
    assert _one("SELECT status FROM runs WHERE id = %s", (run_id,))[0] == "cancelled"


# =============================== 2. every dispatch requires "not cancelled" ===

@pytest.mark.db
def test_llm_reservation_refuses_a_cancelled_run():
    uid = _user()
    run_id, _ = _queued_agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    assert agent_store.reserve_llm_call(run_id, gen, 10, projected_usd=0.001,
                                        max_cost_usd=1.0)
    _exec("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
    before = _one("SELECT llm_calls_reserved, cost_reserved_usd FROM runs WHERE id = %s",
                  (run_id,))
    with pytest.raises(agent_store.RunCancelled):
        agent_store.reserve_llm_call(run_id, gen, 10, projected_usd=0.001, max_cost_usd=1.0)
    assert _one("SELECT llm_calls_reserved, cost_reserved_usd FROM runs WHERE id = %s",
                (run_id,)) == before
    assert _one("SELECT COUNT(*) FROM llm_cost_reservations WHERE run_id = %s",
                (run_id,))[0] == 1


@pytest.mark.db
def test_search_attempt_refuses_a_cancelled_run():
    uid = _user()
    run_id, _ = _queued_agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    _exec("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
    with pytest.raises(agent_store.RunCancelled):
        agent_store.begin_search_attempt(run_id, gen, 1, "adzuna", "ai engineer")
    assert agent_store.search_attempts(run_id, "adzuna", "ai engineer") == []


@pytest.mark.db
def test_authorize_dispatch_rule_order():
    """Erased beats lost beats cancelled; an owned, live run passes."""
    uid = _user()
    run_id, _ = _queued_agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    agent_store.authorize_dispatch(run_id, gen)                      # allowed
    _exec("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
    with pytest.raises(agent_store.RunCancelled):
        agent_store.authorize_dispatch(run_id, gen)
    with pytest.raises(agent_store.ExecutionLost) as lost:
        agent_store.authorize_dispatch(run_id, gen + 1)
    assert not isinstance(lost.value, agent_store.RunErased)


@pytest.mark.db
def test_model_retry_after_cancel_is_never_dispatched(monkeypatch):
    """Transient failure -> user cancels -> retry backoff: the backoff stops on the
    cancel and no second HTTP attempt is reserved or sent."""
    import llm
    uid = _user()
    run_id, _ = _queued_agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    budget = agent_loop.DurableBudget(run_id, 10, 0.0, generation=gen)
    monkeypatch.setattr(agent_loop.run_lock, "check_owner", lambda rid: None)
    monkeypatch.setattr(llm, "_backoff_seconds", lambda attempt, exc=None: 0.2)
    sent = []

    def flaky(prompt):
        sent.append(prompt)
        _exec("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
        raise TimeoutError("timed out")                 # transient -> would retry
    monkeypatch.setattr(llm, "real_llm_once", flaky)
    with pytest.raises(agent_store.RunCancelled):
        llm.logged_llm_call("prompt", run_id, None, operation="t", budget=budget)
    assert len(sent) == 1
    assert _one("SELECT llm_calls_reserved FROM runs WHERE id = %s", (run_id,))[0] == 1


@pytest.mark.db
def test_reservation_refuses_retry_even_without_the_sleep_check(monkeypatch):
    """Defence in depth: with the backoff check disabled, the RESERVATION still
    refuses the retry — the invariant is enforced centrally."""
    import llm
    uid = _user()
    run_id, _ = _queued_agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    budget = agent_loop.DurableBudget(run_id, 10, 0.0, generation=gen)
    monkeypatch.setattr(agent_loop.run_lock, "check_owner", lambda rid: None)
    monkeypatch.setattr(llm, "_interruptible_sleep", lambda s, b: None)
    sent = []

    def flaky(prompt):
        sent.append(prompt)
        _exec("UPDATE runs SET cancel_requested = TRUE WHERE id = %s", (run_id,))
        raise TimeoutError("timed out")
    monkeypatch.setattr(llm, "real_llm_once", flaky)
    with pytest.raises(agent_store.RunCancelled):
        llm.logged_llm_call("prompt", run_id, None, operation="t", budget=budget)
    assert len(sent) == 1


@pytest.mark.db
def test_refused_dispatch_finalizes_cancelled_not_failed():
    uid = _user()
    run_id, _ = _queued_agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    out = agent_loop._fail(run_id, gen, agent_store.RunCancelled("refused"))
    assert out == {"cancelled": True}
    assert _one("SELECT status, error_code FROM runs WHERE id = %s", (run_id,)) == \
        ("cancelled", "cancelled")


# ==================================================== 3. pure: propagation ====

def test_run_cancelled_is_never_absorbed_by_a_fallback():
    from llm import is_degraded_model_error
    from router import is_infrastructure_error, must_propagate
    from error_codes import classify_exception, ErrorCode
    e = agent_store.RunCancelled("refused")
    assert not is_degraded_model_error(e)       # no "use the rules parser instead"
    assert is_infrastructure_error(e) and must_propagate(e)
    assert classify_exception(e) == ErrorCode.CANCELLED
    assert not isinstance(e, agent_store.ExecutionLost)   # must finalize, not abandon


@pytest.fixture
def pure_ctx(monkeypatch):
    monkeypatch.setattr(agent_loop.run_lock, "check_owner", lambda rid: None)
    monkeypatch.setattr(agent_store, "is_cancel_requested", lambda rid: False)
    monkeypatch.setattr(agent_loop, "price_known", lambda *a: True)
    usage = {"active": 0.0}
    monkeypatch.setattr(agent_store, "run_usage", lambda rid: {
        "elapsed_seconds": usage["active"], "active_runtime_seconds": usage["active"],
        "llm_calls_reserved": 0, "llm_call_budget": None, "known_cost_usd": 0.0,
        "unknown_cost_calls": 0, "unknown_cost_bound_usd": 0.0, "unbounded_unknown": 0,
        "reserved_open_usd": 0.0, "committed_usd": 0.0})
    goal = AgentGoal(target_role="AI Engineer", limits={"max_runtime_seconds": 600})
    state = {"run_id": 7, "goal": goal.model_dump(), "resume_id": 1}
    config = {"configurable": {"generation": 3}}
    return goal, state, config, usage


def test_setup_cancelled_at_the_parse_reservation_stops_as_cancel(monkeypatch, pure_ctx):
    import router
    goal, state, config, _ = pure_ctx
    monkeypatch.setattr(router, "load_resume", lambda s, rid: setattr(s, "resume_text", "x"))

    def parse(s, rid):
        raise agent_store.RunCancelled("refused at reservation")
    monkeypatch.setattr(router, "do_parse_resume", parse)
    out = agent_loop.node_setup(state, config)
    assert out["stop"]["cancel"] is True and out["cancel_seen"] is True
    assert not out["stop"].get("setup_failed")
    assert agent_loop.route_after_setup(out) == "finalize"


def test_setup_checks_limits_before_the_resume_parse(monkeypatch, pure_ctx):
    """Runtime exhausted (e.g. an earlier attempt used it up): setup stops before
    loading or parsing the resume — which may call the model."""
    import router
    goal, state, config, usage = pure_ctx
    usage["active"] = 600.0
    called = []
    monkeypatch.setattr(router, "load_resume", lambda *a: called.append("load"))
    monkeypatch.setattr(router, "do_parse_resume", lambda *a: called.append("parse"))
    out = agent_loop.node_setup(state, config)
    assert called == []
    assert "runtime limit" in out["stop"]["reason"]


def test_setup_stops_before_parse_when_cancelled(monkeypatch, pure_ctx):
    import router
    goal, state, config, _ = pure_ctx
    monkeypatch.setattr(agent_store, "is_cancel_requested", lambda rid: True)
    called = []
    monkeypatch.setattr(router, "load_resume", lambda *a: called.append("load"))
    out = agent_loop.node_setup(state, config)
    assert called == [] and out["stop"]["cancel"] is True


def test_runtime_limit_is_reached_at_equality(pure_ctx):
    goal, _state, _config, usage = pure_ctx
    usage["active"] = 599.9
    assert agent_loop.check_limits(1, goal, {"iteration": 1}) is None
    usage["active"] = 600.0
    stop = agent_loop.check_limits(1, goal, {"iteration": 1})
    assert stop and "runtime limit" in stop["reason"]


def test_decide_cancelled_at_controller_reservation(monkeypatch, pure_ctx):
    import llm
    goal, state, config, _ = pure_ctx
    goal = AgentGoal(target_role="AI Engineer", use_llm_controller=True)
    state = {**state, "goal": goal.model_dump(), "iteration": 1, "controller_mode": "llm"}
    monkeypatch.setattr(agent_store, "get_action", lambda rid, i: None)
    monkeypatch.setattr(agent_loop, "qualified_count", lambda ctx: 0)
    monkeypatch.setattr(agent_loop.DurableBudget, "exhausted", lambda self: False)
    monkeypatch.setattr(llm, "quota_blocked", lambda: 0)
    failed = []
    monkeypatch.setattr(llm, "create_step", lambda *a, **k: 11)
    monkeypatch.setattr(llm, "fail_step", lambda sid, e: failed.append(sid))

    def refused(*a, **k):
        raise agent_store.RunCancelled("refused")
    monkeypatch.setattr(agent_loop, "llm_decide", refused)
    recorded = []
    monkeypatch.setattr(agent_store, "record_proposed_action",
                        lambda *a, **k: recorded.append(a))
    out = agent_loop.node_decide(state, config)
    assert out["stop"]["cancel"] is True and failed == [11] and recorded == []


def test_act_cancelled_inside_a_tool_keeps_work_and_routes_to_finalize(monkeypatch, pure_ctx):
    goal, state, config, _ = pure_ctx
    state = {**state, "iteration": 2, "ranked": True,
             "pending": {"action": "evaluate_jobs", "arguments": {"job_ids": [1, 2]},
                         "decided_by": "rules", "replayed": False},
             "evaluated": {"1": {"status": "ok", "job_id": 1}}, "discovered": {},
             "searches": [], "observations": []}
    outcomes = []
    monkeypatch.setattr(agent_store, "begin_action_attempt", lambda *a, **k: None)
    monkeypatch.setattr(agent_store, "record_action_outcome",
                        lambda rid, g, i, status, obs=None, **k: outcomes.append((status, obs)))
    monkeypatch.setattr(agent_store, "set_controller_mode", lambda *a, **k: None)
    monkeypatch.setattr(agent_loop, "_progress", lambda *a: {})

    def tool(ctx, action, args):
        ctx.state["evaluated"]["2"] = {"status": "ok", "job_id": 2}    # job 2 done...
        raise agent_store.RunCancelled("refused")                      # ...job 3 refused
    monkeypatch.setattr(agent_loop, "execute", tool)
    out = agent_loop.node_act(state, config)
    assert outcomes == [("rejected", {"stopped_during_execution": "cancelled by user"})]
    assert out["stop"]["cancel"] is True and out["cancel_seen"] is True
    assert set(out["evaluated"]) == {"1", "2"} and out["ranked"] is False
    assert agent_loop.route_after_act(out) == "finalize"


@pytest.mark.parametrize("calls,expect_stop", [(0, False), (5, True)])
def test_cost_preflight_ignores_unknown_price_when_no_call_is_allowed(monkeypatch, calls,
                                                                      expect_stop):
    from settings import settings
    monkeypatch.setattr(agent_loop, "price_known", lambda *a: False)
    monkeypatch.setattr(settings, "gemini_api_key", "k")
    goal = AgentGoal(target_role="AI Engineer",
                     limits={"max_llm_calls": calls, "max_cost_usd": 1.0})
    assert bool(agent_loop.cost_preflight(goal)) is expect_stop


# ====================================================== 4. release checker ====

def _checker():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "check_release.py")
    spec = importlib.util.spec_from_file_location("check_release", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("path,kind", [
    ("reports/pure.xml", "forbidden directory"),
    ("dist/agentops-monitor-abc.zip", "forbidden directory"),
    ("build/lib/x.py", "forbidden directory"),
    ("htmlcov/index.html", "forbidden directory"),
    (".idea/workspace.xml", "forbidden directory"),
    (".vscode/settings.json", "forbidden directory"),
    ("agentops.egg-info/PKG-INFO", "forbidden directory"),
    ("old-release.zip", "forbidden file type"),
    (".coverage", "forbidden local/OS file"),
    (".coverage.host.123", "forbidden local/OS file"),
    (".DS_Store", "forbidden local/OS file"),
    ("static/Thumbs.db", "forbidden local/OS file"),
    (".env", "secret file"),
    (".env.local", "secret file"),
])
def test_release_check_rejects_generated_local_and_os_files(path, kind):
    cr = _checker()
    # A root-level sibling, so the single-top-folder normalization keeps the path.
    found = cr.problems({path: b"x", "api.py": b"x = 1\n"})
    assert any(p.startswith(kind) and p.endswith(path) for p in found), found


def test_release_check_rejects_the_october_4_upload_shape():
    """The reviewed a1.zip: populated .env, 51 .pyc files across migrations/,
    scripts/ and tests/, and JUnit reports — every one must be reported."""
    cr = _checker()
    upload = {"a1/api.py": b"x = 1\n", "a1/.env": b"GEMINI_API_KEY=not-real\n",
              "a1/reports/pure.xml": b"<testsuites/>", "a1/reports/db.xml": b"<testsuites/>"}
    for d in ("migrations", "scripts", "tests"):
        upload[f"a1/{d}/__pycache__/m.cpython-311.pyc"] = b"\x00"
    found = cr.problems(upload)
    assert "secret file in release: .env" in found
    assert sum("__pycache__" in p for p in found) == 3
    assert sum(p.startswith("forbidden directory in release: reports/") for p in found) == 2