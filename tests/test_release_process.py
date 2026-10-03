"""
Release-process guarantees (pure, no database unless marked):

  * scripts/check_release.py checks CONTENT, not just paths: empty required files,
    an empty / invalid / incomplete CI workflow, steps allowed to fail, and secret
    values in .env.example are all release failures;
  * the CI workflow shipped in this repository passes that check;
  * the exact archive from the previous upload (populated .env, bytecode, empty
    ci.yml, no .env.example) is rejected for every one of those reasons;
  * db_pg.py and migrate.py run the same entrypoint (migrations + checkpoint schema);
  * scripts/ci_summary.py rejects failures, skips, overlaps and missed tests.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CI = ".github/workflows/ci.yml"


def _load(rel):
    path = os.path.join(ROOT, *rel.split("/"))
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cr():
    return _load("scripts/check_release.py")


def _read(rel):
    with open(os.path.join(ROOT, *rel.split("/")), "rb") as fh:
        return fh.read()


@pytest.fixture
def good(cr):
    files = {r: _read(r) for r in cr.REQUIRED}
    files["api.py"] = b"x = 1\n"
    return files


# ------------------------------------------------------------ check_release ----

def test_repository_release_files_pass(cr, good):
    assert cr.problems(good) == []


def test_shipped_ci_workflow_is_real(cr):
    text = _read(CI).decode("utf-8")
    assert len(text.strip()) > 500
    assert cr.ci_problems(text) == []


@pytest.mark.parametrize("required", [
    ".env.example", ".gitignore", CI, "requirements.txt", "requirements-dev.txt",
    "migrate.py", "README.md"])
@pytest.mark.parametrize("blank", [b"", b"   \r\n\n\t"])
def test_empty_required_file_is_rejected(cr, good, required, blank):
    assert f"required file is empty: {required}" in cr.problems(dict(good, **{required: blank}))


def test_previous_upload_is_rejected_for_every_reason(cr, good):
    """The a1.zip that was sent for review: real .env, bytecode, 0-byte ci.yml, no
    .env.example. Each must be reported."""
    upload = {"a1/" + p: d for p, d in good.items() if p != ".env.example"}
    upload["a1/" + CI] = b""
    upload["a1/.env"] = b"GEMINI_API_KEY=not-a-real-key\n"
    upload["a1/__pycache__/api.cpython-312.pyc"] = b"\x00"
    upload["a1/tests/__pycache__/test_x.cpython-312.pyc"] = b"\x00"
    found = cr.problems(upload)
    assert "secret file in release: .env" in found
    assert "missing required file: .env.example" in found
    assert f"required file is empty: {CI}" in found
    assert sum("forbidden directory" in p for p in found) == 2


def test_release_check_reads_zip_archives(cr, good, tmp_path):
    import zipfile
    z = tmp_path / "release.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p, d in dict(good, **{CI: b""}).items():               # 0-byte ci.yml
            zf.writestr("agentops-monitor/" + p, d)
    assert cr.main(["check_release.py", str(z)]) == 1
    with zipfile.ZipFile(z, "w") as zf:
        for p, d in good.items():
            zf.writestr("agentops-monitor/" + p, d)
    assert cr.main(["check_release.py", str(z)]) == 0


def test_ci_workflow_garbage_is_rejected(cr):
    assert cr.ci_problems("") == [f"{CI} is empty"]
    assert any("not valid YAML" in p for p in cr.ci_problems("jobs: [unclosed"))
    assert any("not a workflow" in p for p in cr.ci_problems("- just\n- a list\n"))
    assert any("defines no jobs" in p for p in cr.ci_problems("name: x\non: push\n"))


@pytest.mark.parametrize("needle", [
    "git archive", "scripts/check_release.py", "-m compileall",
    'pytest -m "not db"', 'pytest -m "db and not e2e"', "pytest -m e2e",
    "scripts/ci_summary.py"])
def test_ci_workflow_missing_a_stage_is_rejected(cr, needle):
    text = _read(CI).decode("utf-8")
    assert needle in text
    broken = text.replace(needle, "echo skipped")
    assert cr.ci_problems(broken) != []


def test_ci_workflow_must_migrate_twice_and_start_postgres(cr):
    text = _read(CI).decode("utf-8")
    once = text.replace("python migrate.py | tee", "echo | tee")
    assert any("migrations twice" in p for p in cr.ci_problems(once))
    no_pg = text.replace("image: postgres:16", "image: redis:7")
    assert any("no PostgreSQL service" in p for p in cr.ci_problems(no_pg))


def test_ci_workflow_cannot_be_allowed_to_fail(cr):
    text = _read(CI).decode("utf-8")
    soft = text.replace("python scripts/ci_summary.py reports/pure.xml",
                        "python scripts/ci_summary.py reports/pure.xml || true #")
    assert any("|| true" in p for p in cr.ci_problems(soft))
    soft = text.replace("      - name: Byte-compile every module\n",
                        "      - name: Byte-compile every module\n"
                        "        continue-on-error: true\n")
    assert any("continue-on-error" in p for p in cr.ci_problems(soft))


def test_env_example_secrets_must_be_blank(cr):
    assert cr.env_example_problems(_read(".env.example").decode("utf-8")) == []
    for line in ("GEMINI_API_KEY=AIzaSyExample", "DB_PASSWORD=hunter2",
                 "SESSION_SECRET='abc'", "export ADZUNA_APP_KEY=k", "ADZUNA_APP_ID=1"):
        assert cr.env_example_problems(line + "\n"), line
    # numeric limits whose names contain TOKEN are not secrets
    assert cr.env_example_problems("LLM_MAX_OUTPUT_TOKENS=8192\n"
                                   "LLM_RESERVE_BYTES_PER_TOKEN=1.0\n"
                                   "DB_HOST=localhost\n") == []


def test_gitignore_keeps_secrets_and_bytecode_out():
    gi = _read(".gitignore").decode("utf-8").split()
    for pattern in (".env", ".env.*", "!.env.example", "__pycache__/", "*.py[cod]", "dist/"):
        assert pattern in gi, pattern


# ------------------------------------------------------- db_pg == migrate ----

def test_db_pg_runs_the_same_entrypoint_as_migrate():
    import db_pg
    import migrate
    assert db_pg.main is migrate.main
    src = _read("db_pg.py").decode("utf-8")
    assert "sys.exit(main())" in src
    assert "migrate()" not in src.replace("migrate.py", "")  # no private shortcut


def test_migrate_main_dispatch(monkeypatch):
    import migrate
    calls = []
    monkeypatch.setattr(migrate, "migrate", lambda target=None: calls.append(("m", target)))
    monkeypatch.setattr(migrate, "setup_checkpointer", lambda: calls.append(("cp",)))
    monkeypatch.setattr(migrate, "status", lambda: calls.append(("status",)))
    assert migrate.main([]) == 0 and calls == [("m", None), ("cp",)]
    calls.clear()
    assert migrate.main(["--skip-checkpointer"]) == 0 and calls == [("m", None)]
    calls.clear()
    assert migrate.main(["--status"]) == 0 and calls == [("status",)]
    calls.clear()
    assert migrate.main(["--to", "0014", "--skip-checkpointer"]) == 0
    assert calls == [("m", "0014")]
    calls.clear()
    assert migrate.main(["--to=0003"]) == 0 and calls == [("m", "0003"), ("cp",)]
    calls.clear()
    assert migrate.main(["--bogus"]) == 2
    assert migrate.main(["--to", "14"]) == 2
    assert calls == []


@pytest.mark.db
def test_db_pg_cli_installs_checkpoint_schema_and_guards():
    """`python db_pg.py` on a migrated database: no-op migrations PLUS the LangGraph
    checkpoint schema and its erasure guards — exactly like `python migrate.py`."""
    import psycopg2
    from erasure_guards import missing_guards
    from settings import settings
    for script in ("db_pg.py", "migrate.py"):
        res = subprocess.run([sys.executable, script], cwd=ROOT,
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr
        assert "LangGraph checkpoint schema is up to date" in res.stdout, script
        status = subprocess.run([sys.executable, script, "--status"], cwd=ROOT,
                                capture_output=True, text=True)
        assert status.returncode == 0 and "PENDING" not in status.stdout
    conn = psycopg2.connect(**settings.db_kwargs())
    try:
        cur = conn.cursor()
        assert missing_guards(cur) == []
        cur.execute("SELECT count(*) FROM pg_tables WHERE tablename = 'checkpoints'")
        assert cur.fetchone()[0] == 1
    finally:
        conn.close()


# ------------------------------------------------------------- ci_summary ----

def _junit(path, cases):
    body = "".join(
        f'<testcase classname="{c}" name="{n}">{extra}</testcase>' for c, n, extra in cases)
    path.write_text(f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return str(path)


def test_ci_summary(tmp_path, monkeypatch, capsys):
    cs = _load("scripts/ci_summary.py")
    monkeypatch.setattr(cs, "collected_count", lambda: 3)
    a = _junit(tmp_path / "a.xml", [("t.m", "one", ""), ("t.m", "two", "")])
    b = _junit(tmp_path / "b.xml", [("t.n", "three", "")])
    assert cs.main([a, b]) == 0
    assert cs.main([a]) == 1                                     # a test fell through
    dup = _junit(tmp_path / "d.xml", [("t.m", "one", ""), ("t.n", "three", "")])
    assert cs.main([a, dup]) == 1                                # ran twice
    skip = _junit(tmp_path / "s.xml", [("t.n", "three", '<skipped message="x"/>')])
    assert cs.main([a, skip]) == 1                               # skipped is not green
    fail = _junit(tmp_path / "f.xml", [("t.n", "three", '<failure message="x"/>')])
    assert cs.main([a, fail]) == 1
    capsys.readouterr()