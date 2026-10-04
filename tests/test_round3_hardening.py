"""
Regression tests for the third production review:

  * import smoke (candidate_queries / title_variants and every top-level module)
  * monotonic erasure (tombstones + database guards)                       [db]
  * cost-bound configuration (bytes/token <= 1.0, provider token counting,
    both-or-neither price overrides, price-change horizon)
  * agent-only migration (no 'pool', 'agent' default, explicit retirement)  [db]
  * durable external-search attempts                                        [db]
  * cost-ledger constraints                                                  [db]
  * structured country / region resolution
"""
import importlib
import json
import os
import uuid
from datetime import datetime, timezone

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


# ================================================================ imports ====

def _top_level_modules():
    return sorted(f[:-3] for f in os.listdir(ROOT)
                  if f.endswith(".py") and f not in ("conftest.py",))


@pytest.mark.parametrize("name", _top_level_modules())
def test_every_top_level_module_imports(name):
    importlib.import_module(name)


def test_controller_query_helpers_are_importable_and_work():
    from agent_controller import candidate_queries, title_variants
    from agent_goal import AgentGoal
    import agent_tools                       # imports candidate_queries at module load
    assert agent_tools.candidate_queries is candidate_queries
    assert title_variants("Junior AI Engineer")[0] == "junior machine learning engineer"
    g = AgentGoal(target_role="AI Engineer", alternative_titles=["ML Engineer"])
    qs = candidate_queries(g)
    assert qs[0] == "ai engineer" and "ml engineer" in qs and len(qs) == len(set(qs))


# ============================================================ cost bounds ====

def _settings_with(monkeypatch, **env):
    import settings as settings_mod
    for k in ("LLM_RESERVE_BYTES_PER_TOKEN", "LLM_INPUT_PRICE_PER_MILLION",
              "LLM_OUTPUT_PRICE_PER_MILLION", "LLM_INPUT_TOKEN_BOUND"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return settings_mod.Settings()


@pytest.mark.parametrize("value", ["1.01", "2", "4.0", "nan", "inf", "0.1"])
def test_reserve_bytes_per_token_rejects_under_reserving_values(monkeypatch, value):
    from settings import SettingsError
    with pytest.raises(SettingsError):
        _settings_with(monkeypatch, LLM_RESERVE_BYTES_PER_TOKEN=value)


@pytest.mark.parametrize("value", ["1.0", "0.5", "0.25"])
def test_reserve_bytes_per_token_accepts_conservative_values(monkeypatch, value):
    s = _settings_with(monkeypatch, LLM_RESERVE_BYTES_PER_TOKEN=value)
    assert s.llm_reserve_bytes_per_token == float(value)


@pytest.mark.parametrize("env", [{"LLM_INPUT_PRICE_PER_MILLION": "1"},
                                 {"LLM_OUTPUT_PRICE_PER_MILLION": "1"}])
def test_price_overrides_are_both_or_neither(monkeypatch, env):
    from settings import SettingsError
    with pytest.raises(SettingsError, match="both or neither"):
        _settings_with(monkeypatch, **env)
    s = _settings_with(monkeypatch, LLM_INPUT_PRICE_PER_MILLION="1",
                       LLM_OUTPUT_PRICE_PER_MILLION="2")
    assert (s.llm_input_price_per_million, s.llm_output_price_per_million) == (1.0, 2.0)


def test_half_override_at_runtime_is_unknown_price(monkeypatch):
    from settings import settings
    import pricing
    monkeypatch.setattr(settings, "llm_input_price_per_million", 1.0)
    monkeypatch.setattr(settings, "llm_output_price_per_million", None)
    assert pricing.rates_for("gemini-3.6-flash") == (None, None, None)
    assert pricing.price_known("gemini-3.6-flash") is False


def test_byte_bound_ignores_an_under_reserving_runtime_value(monkeypatch):
    from settings import settings
    import pricing
    monkeypatch.setattr(settings, "llm_reserve_bytes_per_token", 4.0)   # bypassed validation
    assert pricing.byte_token_bound(1000) == 1001


def test_reservation_uses_highest_rate_across_a_price_change():
    from pricing import max_request_cost
    before = max_request_cost("gemini-3.6-flash", 1_000_000, 0, at=_utc(2026, 12, 1))
    straddle = max_request_cost("gemini-3.6-flash", 1_000_000, 0,
                                at=_utc(2026, 12, 31, 23, 55))
    assert straddle > before and straddle >= 1.5      # Jan-2027 rate, not the intro rate


def test_provider_count_tightens_but_never_exceeds_the_byte_bound(monkeypatch):
    import llm
    from settings import settings
    monkeypatch.setattr(settings, "llm_input_token_bound", "provider")
    prompt = "x" * 10_000
    monkeypatch.setattr(llm, "provider_token_count", lambda p: 2_500)
    bound, how = llm.input_token_bound(prompt)
    assert how == "provider" and 2_500 < bound < 10_001
    # An implausible count (more tokens than bytes) is not trusted.
    monkeypatch.setattr(llm, "provider_token_count", lambda p: 50_000)
    assert llm.input_token_bound(prompt) == (10_001, "bytes_fallback")


@pytest.mark.parametrize("exc", [TimeoutError("slow"), RuntimeError("500"), ValueError("x")])
def test_token_count_failure_fails_closed_to_the_byte_bound(monkeypatch, exc):
    import llm
    from settings import settings
    monkeypatch.setattr(settings, "llm_input_token_bound", "provider")

    def boom(prompt):
        raise exc
    monkeypatch.setattr(llm, "provider_token_count", boom)
    assert llm.input_token_bound("é" * 100) == (201, "bytes_fallback")


def test_logged_llm_call_reserves_with_the_tightened_bound(monkeypatch):
    import llm
    from settings import settings
    monkeypatch.setattr(settings, "llm_input_token_bound", "provider")
    monkeypatch.setattr(llm, "provider_token_count", lambda p: 100)
    seen = []

    class Budget:
        generation = 1

        def reserve_attempt(self, projected, operation=None):
            seen.append(projected)
            raise llm.CostLimitReached("stop here")
    with pytest.raises(llm.CostLimitReached):
        llm.logged_llm_call("y" * 50_000, run_id=1, step_id=None, budget=Budget())
    from pricing import max_request_cost
    assert seen[0] == max_request_cost(settings.gemini_model, 50_000,
                                       settings.llm_max_output_tokens, input_tokens=126)
    assert seen[0] < max_request_cost(settings.gemini_model, 50_000,
                                      settings.llm_max_output_tokens)


# =================================================================== geo =====

@pytest.mark.parametrize("posting,candidate,expected", [
    ("Germany only", "Paris, France", "ineligible"),      # was 'eligible' (both "europe")
    ("Germany, France", "Lyon, France", "eligible"),
    ("Germany only", "Berlin", "unknown"),                 # city not resolvable
    ("EU only", "London, UK", "ineligible"),
    ("EU only", "Madrid, Spain", "eligible"),
    ("USA Only", "Austin, TX", "eligible"),
    ("USA Only", "San Jose, CA", "eligible"),
    ("Canada", "San Jose, CA", "ineligible"),
    ("Latin America", "Mexico City, Mexico", "eligible"),
    ("North America", "Mexico", "eligible"),
    ("Latin America", "United States", "ineligible"),
    ("Americas", "Toronto, Canada", "eligible"),
    ("South Africa", "Kenya", "ineligible"),               # not swallowed by "africa"
    ("Africa", "South Africa", "eligible"),
    ("Anywhere in the World", "Kenya", "eligible"),
    ("Anywhere in the EU", "Austin, TX", "ineligible"),     # not "anywhere"
    ("Worldwide except US", "Austin, TX", "unknown"),      # exclusion: never guessed
    ("Europe", "Georgia", "unknown"),                      # ambiguous name
    ("APAC", "Sydney, Australia", "eligible"),
    ("Remote", "Germany", "unknown"),
    ("Europe", "APAC", "ineligible"),
    ("Europe", "EMEA", "unknown"),
    ("EMEA", "Dubai, UAE", "eligible"),
])
def test_structured_geo_eligibility(posting, candidate, expected):
    from job_source import geo_eligibility
    assert geo_eligibility({"location": posting}, candidate) == expected


def test_geo_resolve_is_structured():
    from geo import resolve
    s = resolve("UK, Germany or anywhere in the EU")
    assert s.countries == {"gb", "de"} and s.regions == {"eu"} and not s.worldwide
    assert resolve("Latin America").countries == frozenset()


# ======================================================== agent-only goal ====

def test_pool_is_not_a_provider():
    from agent_goal import AgentGoal
    with pytest.raises(Exception):
        AgentGoal(target_role="AI Engineer", providers=["pool"])
    with pytest.raises(Exception):
        AgentGoal(target_role="AI Engineer", providers=["adzuna", "pool"])


def test_create_run_refuses_non_agent_mode():
    from llm import create_run_tx
    with pytest.raises(ValueError, match="retired"):
        create_run_tx(None, "x", mode="pipeline")


# ================================================================ DB tests ===

def _db():
    from database import get_connection
    return get_connection()


def _user():
    from auth import create_user
    return create_user("r3_" + uuid.uuid4().hex[:10], "password-1234")


def _resume(uid, text=None):
    with _db() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO resumes (name, resume_text, created_at, user_id) "
                    "VALUES ('r', %s, NOW(), %s) RETURNING id",
                    (text or ("resume " + uuid.uuid4().hex), uid))
        return cur.fetchone()[0]


def _run(uid, resume_id=None, status="success", goal=None):
    with _db() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, resume_id, goal_json, "
                    "ended_at) VALUES (%s, 't', %s, %s, %s, NOW()) RETURNING id",
                    (status, uid, resume_id, None if goal is None else json.dumps(goal)))
        return cur.fetchone()[0]


def _one(sql, params=()):
    with _db() as c:
        cur = c.cursor()
        cur.execute(sql, params)
        return cur.fetchone()


def _exec(sql, params=()):
    with _db() as c:
        c.cursor().execute(sql, params)


# ---------------------------------------------------------------- erasure ----

@pytest.mark.db
def test_late_writes_after_resume_erasure_are_neutralized():
    """A stale worker finishing AFTER the erase cannot re-persist anything."""
    from privacy import erase_resume
    from router import resume_content_hash
    uid = _user()
    text = "Private resume " + uuid.uuid4().hex
    rid = _resume(uid, text)
    run_id = _run(uid, rid)
    gen_before = _one("SELECT execution_generation FROM runs WHERE id = %s", (run_id,))[0]
    assert erase_resume(rid, uid)
    assert _one("SELECT reason FROM erasure_tombstones WHERE run_id = %s", (run_id,)) == \
        ("resume_erased",)
    assert _one("SELECT execution_generation FROM runs WHERE id = %s", (run_id,))[0] > gen_before

    # --- the stale worker's late writes ---
    _exec("INSERT INTO llm_calls (run_id, prompt, response, prompt_tokens, completion_tokens, "
          "cost_usd, cost_status, status) VALUES (%s, 'PRIVATE', 'PRIVATE', 10, 5, 0.001, "
          "'priced', 'success')", (run_id,))
    assert _one("SELECT prompt, response, prompt_tokens, cost_usd::float FROM llm_calls "
                "WHERE run_id = %s", (run_id,)) == (None, None, 10, 0.001)   # metrics kept
    _exec("INSERT INTO tool_calls (run_id, tool_name, input_json, output_json) "
          "VALUES (%s, 't', '{\"q\": \"PRIVATE\"}', '{\"r\": \"PRIVATE\"}')", (run_id,))
    assert _one("SELECT input_json, output_json FROM tool_calls WHERE run_id = %s",
                (run_id,)) == (None, None)
    _exec("INSERT INTO steps (run_id, step_name, step_order, status, retrieved_context) "
          "VALUES (%s, 's', 1, 'success', '{\"job_id\": 1, \"x\": \"PRIVATE\"}')", (run_id,))
    assert _one("SELECT retrieved_context FROM steps WHERE run_id = %s", (run_id,)) == (None,)
    _exec("INSERT INTO run_advice (run_id, title, advice) VALUES (%s, 'j', 'PRIVATE')",
          (run_id,))
    assert _one("SELECT advice FROM run_advice WHERE run_id = %s", (run_id,)) == ("[erased]",)
    _exec("INSERT INTO resume_suggestions (position, resume_id, job_id, run_id, status, "
          "resume_hash, suggested_text, kind, method, reason, evidence) VALUES "
          "(0, %s, 1, %s, 'validated', 'h', 'PRIVATE', 'rewrite', 'rules', 'r', '[]')",
          (rid, run_id))
    assert _one("SELECT count(*) FROM resume_suggestions WHERE run_id = %s", (run_id,)) == (0,)
    _exec("UPDATE runs SET pending_review = '{\"q\": \"PRIVATE\"}', stop_reason = 'PRIVATE' "
          "WHERE id = %s", (run_id,))
    assert _one("SELECT pending_review, stop_reason FROM runs WHERE id = %s",
                (run_id,)) == (None, "[erased]")
    # parse cache cannot be re-populated for the erased text
    _exec("INSERT INTO parsed_resume_cache (content_hash, cache_version, parsed_json) "
          "VALUES (%s, 'v', '{}') ON CONFLICT DO NOTHING", (resume_content_hash(text),))
    assert _one("SELECT count(*) FROM parsed_resume_cache WHERE content_hash = %s",
                (resume_content_hash(text),)) == (0,)


@pytest.mark.db
def test_erasure_is_monotonic():
    import psycopg2
    from privacy import erase_resume
    uid = _user()
    rid = _resume(uid)
    run_id = _run(uid, rid)
    erase_resume(rid, uid)
    with pytest.raises(psycopg2.Error):                     # cannot un-delete
        _exec("UPDATE resumes SET is_deleted = FALSE WHERE id = %s", (rid,))
    with pytest.raises(psycopg2.Error):                     # cannot rewrite the text
        _exec("UPDATE resumes SET resume_text = 'back' WHERE id = %s", (rid,))
    with pytest.raises(psycopg2.Error):                     # tombstones are append-only
        _exec("DELETE FROM erasure_tombstones WHERE run_id = %s", (run_id,))
    with pytest.raises(psycopg2.Error):
        _exec("UPDATE erasure_tombstones SET reason = 'retention' WHERE run_id = %s", (run_id,))
    assert erase_resume(rid, uid)                           # idempotent


@pytest.mark.db
def test_stale_generation_cannot_execute_or_reserve_after_erasure():
    import agent_store
    from privacy import erase_resume
    uid = _user()
    rid = _resume(uid)
    run_id = _run(uid, rid)
    gen = _one("SELECT execution_generation FROM runs WHERE id = %s", (run_id,))[0]
    erase_resume(rid, uid)
    with pytest.raises(agent_store.RunErased):
        agent_store.record_proposed_action(run_id, gen, 1, "finish", {}, "r", "rules")
    with pytest.raises(agent_store.RunErased):
        agent_store.reserve_llm_call(run_id, gen + 1, 10)
    with pytest.raises(agent_store.RunErased):
        agent_store.begin_execution(run_id, new_attempt=True)
    import worker
    assert worker._job_is_stale({"run_id": run_id}) is True


@pytest.mark.db
def test_deleted_run_checkpoints_cannot_be_recreated():
    from checkpointing import setup_schema
    from privacy import delete_run
    setup_schema()
    uid = _user()
    run_id = _run(uid)
    thread = str(run_id)
    _exec("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, "
          "metadata) VALUES (%s, '', 'c1', '{}', '{}')", (thread,))
    assert delete_run(run_id, uid)
    assert _one("SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread,)) == (0,)
    # A stale worker flushes its checkpoint after the delete: dropped by the guard.
    _exec("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, "
          "metadata) VALUES (%s, '', 'c2', '{}', '{}')", (thread,))
    _exec("INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, "
          "channel, blob) VALUES (%s, '', 'c2', 't', 0, 'x', 'PRIVATE')", (thread,))
    assert _one("SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread,)) == (0,)
    assert _one("SELECT count(*) FROM checkpoint_writes WHERE thread_id = %s", (thread,)) == (0,)
    assert _one("SELECT reason FROM erasure_tombstones WHERE run_id = %s", (run_id,)) == \
        ("run_deleted",)
    # A different (live) thread is untouched.
    other = _run(uid)
    _exec("INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, "
          "metadata) VALUES (%s, '', 'c1', '{}', '{}')", (str(other),))
    assert _one("SELECT count(*) FROM checkpoints WHERE thread_id = %s", (str(other),)) == (1,)


@pytest.mark.db
def test_retention_purge_tombstones_runs():
    from privacy import purge_expired_traces
    uid = _user()
    run_id = _run(uid)
    _exec("UPDATE runs SET ended_at = NOW() - INTERVAL '90 days' WHERE id = %s", (run_id,))
    purge_expired_traces(30)
    assert _one("SELECT reason FROM erasure_tombstones WHERE run_id = %s", (run_id,)) == \
        ("retention",)


@pytest.mark.db
def test_all_guards_installed():
    from checkpointing import setup_schema
    from erasure_guards import missing_guards
    setup_schema()
    with _db() as c:
        assert missing_guards(c.cursor()) == []


# ------------------------------------------------------- agent-only (db) ----

@pytest.mark.db
def test_helper_default_is_agent_and_goal_is_stored_atomically():
    from agent_goal import AgentGoal
    from llm import create_run
    uid = _user()
    g = AgentGoal(target_role="Data Engineer", limits={"max_llm_calls": 7, "max_cost_usd": 0.3})
    rid = create_run("x", user_id=uid, goal=g)
    mode, goal, budget, cap = _one("SELECT mode, goal_json, llm_call_budget, max_cost_usd "
                                   "FROM runs WHERE id = %s", (rid,))
    assert (mode, goal["target_role"], budget, cap) == ("agent", "Data Engineer", 7, 0.3)
    assert _one("SELECT column_default FROM information_schema.columns "
                "WHERE table_name = 'runs' AND column_name = 'mode'")[0].startswith("'agent'")


@pytest.mark.db
def test_migration_retires_legacy_goal_records_explicitly():
    """Re-running migration 0013 on rows that look like legacy records retires them:
    a queued run whose goal names the removed 'pool' provider is closed, never run."""
    import importlib.util
    uid = _user()
    queued = _run(uid, status="queued", goal={"target_role": "x", "providers": ["adzuna", "pool"]})
    finished = _run(uid, status="success", goal={"target_role": "x", "providers": ["pool"]})
    fine = _run(uid, status="queued", goal={"target_role": "x", "providers": ["remotive"]})
    spec = importlib.util.spec_from_file_location(
        "m13", os.path.join(ROOT, "migrations", "0013_agent_only.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with _db() as c:
        m.upgrade(c.cursor())
    assert _one("SELECT status, error_code, goal_retired_reason FROM runs WHERE id = %s",
                (queued,)) == ("failed", "engine_retired", "retired_provider")
    assert _one("SELECT status, goal_retired_reason FROM runs WHERE id = %s",
                (finished,)) == ("success", "retired_provider")      # history kept
    assert _one("SELECT status, goal_retired_at FROM runs WHERE id = %s",
                (fine,)) == ("queued", None)
    _exec("UPDATE runs SET status = 'cancelled' WHERE id = %s", (fine,))


@pytest.mark.db
def test_worker_never_executes_a_retired_goal(monkeypatch):
    import agent_loop
    import worker
    uid = _user()
    rid = _run(uid, status="queued", goal={"target_role": "x", "providers": ["pool"]})
    monkeypatch.setattr(agent_loop, "run_agent_loop", lambda *a, **k: pytest.fail("executed"))
    worker._run_job({"kind": "start_run", "payload": {"run_id": rid}, "run_id": rid})
    assert _one("SELECT status, error_code, goal_retired_reason FROM runs WHERE id = %s",
                (rid,)) == ("failed", "engine_retired", "retired_provider")


# --------------------------------------------------- search attempts (db) ----

def _agent_run(uid):
    from agent_goal import AgentGoal
    from llm import create_run
    return create_run("x", user_id=uid, goal=AgentGoal(target_role="AI Engineer"))


@pytest.mark.db
def test_search_attempt_is_durable_before_dispatch_and_abandoned_on_crash():
    import agent_store
    uid = _user()
    run_id = _agent_run(uid)
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    aid = agent_store.begin_search_attempt(run_id, gen, 1, "adzuna", "ai engineer")
    # (worker dies here, mid-request)
    gen2 = agent_store.begin_execution(run_id, new_attempt=True)
    rows = agent_store.search_attempts(run_id, "adzuna", "ai engineer")
    assert [(r["id"], r["status"], r["provider_detail"]) for r in rows] == \
        [(aid, "abandoned", "outcome_unknown")]
    with pytest.raises(agent_store.ExecutionLost):            # fenced
        agent_store.begin_search_attempt(run_id, gen, 1, "adzuna", "ai engineer")
    a2 = agent_store.begin_search_attempt(run_id, gen2, 1, "adzuna", "ai engineer")
    agent_store.finish_search_attempt(run_id, gen2, a2, "succeeded", "success")
    assert [r["status"] for r in agent_store.search_attempts(run_id, "adzuna", "ai engineer")] \
        == ["abandoned", "succeeded"]


@pytest.mark.db
def test_provider_request_cap_survives_crash_loops(monkeypatch):
    import adzuna_jobs
    import agent_store
    import agent_tools
    from agent_goal import AgentGoal
    uid = _user()
    run_id = _agent_run(uid)
    calls = []
    monkeypatch.setattr(adzuna_jobs, "fetch_and_upsert_adzuna",
                        lambda *a, **k: calls.append(1) or (0, 0, "server_error"))
    g = AgentGoal(target_role="AI Engineer", providers=["adzuna"])

    class A:
        provider, query = "adzuna", "AI Engineer"
    for _ in range(agent_tools.MAX_PROVIDER_REQUESTS_PER_SEARCH + 2):
        gen = agent_store.begin_execution(run_id, new_attempt=True)
        ctx = agent_tools.ToolContext(run_id, gen, g, {"searches": []}, None, 1)
        agent_tools._fetch_with_attempt(ctx, A, "ai engineer", None, {"mode": "fetch"})
    assert len(calls) == agent_tools.MAX_PROVIDER_REQUESTS_PER_SEARCH
    ctx = agent_tools.ToolContext(run_id, gen, g, {"searches": []}, None, 1)
    assert agent_tools._fetch_with_attempt(ctx, A, "ai engineer", None, {}) == "retry_limit"


# ------------------------------------------------------ cost ledger (db) ----

@pytest.mark.db
@pytest.mark.parametrize("cols,vals", [
    ("cost_status, cost_usd", "'unknown', 0.01"),          # unknown must have NULL cost
    ("cost_status, cost_usd", "'not_billed', 0.01"),       # not billed cannot cost money
    ("cost_status, cost_usd", "'priced', -0.01"),          # negative cost
    ("cost_status, prompt_tokens", "'unknown', -5"),       # negative tokens
    ("cost_status, cost_upper_bound_usd", "'unknown', -1"),
])
def test_llm_call_ledger_constraints(cols, vals):
    import psycopg2
    run_id = _agent_run(_user())
    with pytest.raises(psycopg2.errors.CheckViolation):
        _exec(f"INSERT INTO llm_calls (run_id, {cols}) VALUES (%s, {vals})", (run_id,))


@pytest.mark.db
def test_reservation_settles_exactly_one_call_and_ledger_reconciles():
    import psycopg2
    import agent_store
    from llm import _log_llm_attempt
    run_id = _agent_run(_user())
    gen = agent_store.begin_execution(run_id, new_attempt=True)
    res = agent_store.reserve_llm_call(run_id, gen, 5, projected_usd=0.01, max_cost_usd=1.0)
    _log_llm_attempt(run_id, None, "op", None, "ok", 10, 5, 1, 0.0000004, "success", None, 1, 0,
                     None, cost_status="priced", reservation=res, generation=gen)
    # Six-decimal storage used to round this to 0.
    assert _one("SELECT cost_usd::float FROM llm_calls WHERE run_id = %s", (run_id,)) == \
        (0.0000004,)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _exec("INSERT INTO llm_calls (run_id, reservation_id, cost_status) "
              "VALUES (%s, %s, 'unknown')", (run_id, res["reservation_id"]))
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        _exec("INSERT INTO llm_calls (run_id, reservation_id, cost_status) "
              "VALUES (%s, 987654321, 'unknown')", (run_id,))
    with _db() as c:
        cur = c.cursor()
        cur.execute("SELECT problem FROM cost_ledger_violations(%s)", (run_id,))
        assert cur.fetchall() == []
    with pytest.raises(psycopg2.errors.CheckViolation):     # status/settled_at agree
        _exec("UPDATE llm_cost_reservations SET settled_at = NULL WHERE id = %s",
              (res["reservation_id"],))
    with pytest.raises(psycopg2.errors.CheckViolation):
        _exec("UPDATE runs SET max_cost_usd = -1 WHERE id = %s", (run_id,))


# ======================================================= release hygiene ====

def _check_release():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_release", os.path.join(ROOT, "scripts", "check_release.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _shipped_required_files(m):
    """{path: bytes} of the REQUIRED files as they are in this repository."""
    out = {}
    for r in m.REQUIRED:
        with open(os.path.join(ROOT, *r.split("/")), "rb") as fh:
            out[r] = fh.read()
    return out


def test_release_check_rejects_artefacts_and_requires_shipped_files():
    m = _check_release()
    good = dict(_shipped_required_files(m), **{"api.py": b"x = 1\n", "tests/test_x.py": b""})
    assert m.problems(good) == []
    assert m.problems({"a1/" + p: d for p, d in good.items()}) == []  # archive root folder
    junk = ["__pycache__/api.cpython-312.pyc", "tests/x.pyc", "resume.pdf", ".env",
            ".env.prod", ".pytest_cache/v/x"]
    bad = m.problems(dict(good, **{p: b"junk" for p in junk}))
    assert len(bad) == 6
    no_ci = {p: d for p, d in good.items() if p != ".github/workflows/ci.yml"}
    assert "missing required file: .github/workflows/ci.yml" in m.problems(no_ci)
    # A bare path list cannot prove content: it is NOT a pass.
    assert any("cannot read content" in p for p in m.problems(list(good)))


def test_repository_ships_the_hygiene_files():
    for f in (".env.example", ".gitignore", os.path.join(".github", "workflows", "ci.yml")):
        assert os.path.isfile(os.path.join(ROOT, f)), f
    example = open(os.path.join(ROOT, ".env.example"), encoding="utf-8").read()
    import re
    settings_src = open(os.path.join(ROOT, "settings.py"), encoding="utf-8").read()
    names = set(re.findall(r'r\.(?:str|int|float|bool)\("([A-Z_]+)"', settings_src))
    names.discard("REDACT_SENSITIVE")                               # deprecated alias
    missing = [n for n in sorted(names) if n not in example]
    assert missing == [], f".env.example does not document {missing}"
    gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read().split()