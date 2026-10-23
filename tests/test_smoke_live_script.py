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
