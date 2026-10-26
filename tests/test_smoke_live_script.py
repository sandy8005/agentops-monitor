"""Backlog F (3): scripts/smoke_live.py is a live operator check, never a pytest
test. These guards keep it that way and prove it refuses to start (exit 2, no
database, no network) without a real key or with an out-of-range cap."""
import configparser
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "smoke_live.py")


def _load():
    spec = importlib.util.spec_from_file_location("smoke_live_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_is_outside_pytest_testpaths():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(ROOT, "pytest.ini"))
    assert cfg["pytest"]["testpaths"].split() == ["tests"]
    assert not os.path.basename(SCRIPT).startswith("test_")


@pytest.fixture
def no_side_effects(monkeypatch):
    import database
    import llm
    monkeypatch.setattr(database, "get_connection",
                        lambda *a, **k: pytest.fail("must not touch the database"))
    monkeypatch.setattr(llm, "get_client", lambda: pytest.fail("must not call the model"))


@pytest.mark.parametrize("key", ["", "   ", "test-key-not-real"])
def test_without_a_real_key_exits_2_before_any_request(monkeypatch, no_side_effects, key, capsys):
    from settings import settings
    monkeypatch.setattr(settings, "gemini_api_key", key, raising=False)
    assert _load().main([]) == 2
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--max-cost", "0.5"], ["--max-cost", "0"],
                                  ["--max-llm-calls", "0"]])
def test_cap_is_bounded(no_side_effects, argv):
    assert _load().main(argv) == 2


# --- Round 5, backlog D: check_run passes only a clean outcome (fake records) ---

def _good(**over):
    rec = {"status": "success", "error_code": None, "search_exhausted": False,
           "llm_budget_spent": False, "llm_calls": 4, "llm_failed_unrecovered": 0,
           "tool_failed_unrecovered": 0, "failed_steps_unrecovered": 0, "action_errors": 0,
           "judge_invalid_output": 0, "rules_fallbacks": 0, "rules_fallback_reasons": {},
           "provider_errors": 0, "provider_errors_unrecovered": 0,
           "open_reservations": 0, "committed_usd": 0.01, "known_cost_usd": 0.01,
           "unknown_cost_calls": 0, "violations": 0}
    rec.update(over)
    return rec


def _failed(rec, cap=0.05):
    return [what for ok, what in _load().check_run(rec, cap) if not ok]


@pytest.mark.parametrize("over", [
    {},
    {"status": "partial_success"},
    {"status": "no_matches", "error_code": "no_matches", "search_exhausted": True},
    # budget spent, recorded on each fallback: rules expected (round 6, backlog C)
    {"rules_fallbacks": 3, "rules_fallback_reasons": {"llm_budget_spent": 3},
     "llm_budget_spent": True},
])
def test_clean_runs_pass(over):
    assert _failed(_good(**over)) == []


@pytest.mark.parametrize("over,check", [
    ({"status": "completed_with_errors"}, "outcome"),
    ({"status": "failed", "error_code": "internal_error"}, "outcome"),
    ({"status": "cancelled", "error_code": "cancelled"}, "outcome"),
    ({"status": "success", "error_code": "cost_unknown"}, "outcome"),
    ({"status": "no_matches", "error_code": "limit_reached", "search_exhausted": False}, "outcome"),
    ({"status": "no_matches", "error_code": "no_matches", "search_exhausted": False}, "outcome"),
    ({"status": "no_matches", "error_code": "no_matches", "search_exhausted": True,
      "provider_errors": 1}, "outcome"),
    ({"llm_calls": 0}, "model was called"),
    ({"llm_failed_unrecovered": 1}, "failed LLM call"),
    ({"tool_failed_unrecovered": 2}, "failed tool call"),
    ({"failed_steps_unrecovered": 1}, "failed step"),
    ({"action_errors": 1}, "error-coded agent action"),
    ({"judge_invalid_output": 1}, "invalid_output"),
    ({"rules_fallbacks": 1}, "rules fallback"),
    ({"open_reservations": 1}, "open cost reservation"),
    ({"committed_usd": 0.0500001}, "within the"),
    ({"violations": 2}, "cost_ledger_violations"),
    ({"violations": None}, "cost_ledger_violations"),     # unknown count fails closed
    ({"committed_usd": None}, "within the"),
])
def test_each_failure_is_reported(over, check):
    failed = _failed(_good(**over))
    assert len(failed) == 1 and check in failed[0], failed


def test_report_prints_metadata_only_lines(capsys):
    n = _load().report(_good(status="completed_with_errors", llm_failed_unrecovered=1), 0.05)
    out = capsys.readouterr().out
    assert n == 2
    assert out.count("[FAIL]") == 2 and out.count("[ok]") == 10
    assert all(len(line) < 200 for line in out.splitlines())


@pytest.mark.db
def test_collect_run_record_reads_a_real_run():
    import uuid
    from auth import create_user
    from database import get_connection
    uid = create_user("smk_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode, error_code, goal_progress) "
                    "VALUES ('completed_with_errors', 't', %s, 'agent', NULL, "
                    "'{\"search_exhausted\": true}') RETURNING id", (uid,))
        rid = cur.fetchone()[0]
        cur.execute("INSERT INTO steps (run_id, step_name, step_order, status, retrieved_context) "
                    "VALUES (%s, 'job', 0, 'failed', '{\"job_id\": 5}')", (rid,))
    mod = _load()
    rec = mod.collect_run_record(rid)
    assert rec["status"] == "completed_with_errors" and rec["search_exhausted"] is True
    assert rec["llm_calls"] == 0 and rec["failed_steps_unrecovered"] == 1
    assert rec["violations"] == 0 and rec["open_reservations"] == 0
    failed = [w for ok, w in mod.check_run(rec, 0.05) if not ok]
    assert any("outcome" in w for w in failed) and any("failed step" in w for w in failed)
