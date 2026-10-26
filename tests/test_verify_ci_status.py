"""Round 5, backlog F: scripts/verify_ci_status.py is an operator check (network),
never run by pytest. Its pure summarize() is tested with fake GitHub payloads, and
main() is tested with the network call stubbed. No network access."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "verify_ci_status.py")
SHA = "a" * 40


def _load():
    spec = importlib.util.spec_from_file_location("verify_ci_status_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(n, status, conclusion, sha=SHA, attempt=1):
    return {"id": 100 + n, "name": "CI", "run_number": n, "run_attempt": attempt,
            "head_sha": sha, "event": "push", "head_branch": "main", "status": status,
            "conclusion": conclusion, "html_url": f"https://github.com/x/y/actions/runs/{n}"}


@pytest.mark.parametrize("runs,code", [
    ([_run(1, "completed", "success")], 0),
    ([_run(1, "completed", "failure")], 1),
    ([_run(1, "completed", "cancelled")], 1),
    ([_run(1, "in_progress", None)], 3),
    ([_run(1, "queued", None)], 3),
    ([], 4),
    ([_run(1, "completed", "success", sha="b" * 40)], 4),          # another commit
    ([_run(1, "completed", "success"), _run(2, "completed", "failure")], 1),   # latest wins
    ([_run(2, "completed", "failure"), _run(2, "completed", "success", attempt=2)], 0),
])
def test_summarize(runs, code):
    got, lines = _load().summarize({"workflow_runs": runs}, SHA)
    assert got == code
    assert lines


def test_main_without_network(monkeypatch, capsys):
    mod = _load()
    monkeypatch.setattr(mod, "fetch_runs",
                        lambda *a, **k: {"workflow_runs": [_run(7, "completed", "success")]})
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token-value")
    assert mod.main(["--sha", SHA]) == 0
    out = capsys.readouterr().out
    assert "conclusion=success" in out and "secret-token-value" not in out


def test_network_error_exits_2(monkeypatch):
    import urllib.error
    mod = _load()

    def boom(*a, **k):
        raise urllib.error.URLError("blocked")
    monkeypatch.setattr(mod, "fetch_runs", boom)
    assert mod.main(["--sha", SHA]) == 2


def test_bad_sha_exits_2():
    assert _load().main(["--sha", "abc"]) == 2
