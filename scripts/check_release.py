"""
Release hygiene check. Fails (exit 1) if a release tree or archive contains build
artefacts, caches, secrets or personal documents, is missing a file every release
must ship, ships one of those files EMPTY, or ships a CI workflow that does not
actually run the release pipeline.

    python scripts/check_release.py                 # the git-tracked tree (index)
    python scripts/check_release.py release.zip     # an archive
    python scripts/check_release.py path/to/dir     # a directory

Content checks (not just "the path exists"):
  * every REQUIRED file has non-blank content;
  * .github/workflows/ci.yml parses as YAML, defines `on` and at least one job with
    `runs-on` and `steps`, starts a PostgreSQL service, and its `run:` steps include
    every stage of the documented pipeline (release check of a `git archive`,
    byte-compilation, the pure tests, two migration runs, the DB tests, the E2E
    test, the zero-skip summary); nothing in it may be allowed to fail
    (`continue-on-error`, `|| true`);
  * .env.example assigns NO value to any secret-looking variable.

Build releases with scripts/make_release.py (git archive HEAD + this check), never by
compressing a working directory.
"""
import os
import re
import subprocess
import sys
import zipfile

REQUIRED = (".env.example", ".gitignore", ".github/workflows/ci.yml", "requirements.txt",
            "requirements-dev.txt", "migrate.py", "README.md")
CI_PATH = ".github/workflows/ci.yml"
# Anything generated, local or personal. JUnit reports (reports/) carry machine
# metadata (host name, timestamps); dist/ and build/ hold build output (including
# earlier release archives); editor folders hold local settings.
FORBIDDEN_DIRS = ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "venv", ".venv",
                  "reports", "dist", "build", "htmlcov", ".idea", ".vscode", "uploads",
                  ".tox", ".nox", "__MACOSX")
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".pdf", ".docx", ".zip")
FORBIDDEN_NAMES = (".env", ".coverage", ".DS_Store", "Thumbs.db", "desktop.ini")
SECRET_NAMES = (".env",)

# Variables in .env.example whose name looks like a credential must be blank.
# TOKEN only in credential names: LLM_MAX_OUTPUT_TOKENS, LLM_RESERVE_BYTES_PER_TOKEN
# etc. are numeric limits, not secrets.
SECRET_NAME_RE = re.compile(r"(PASSWORD|SECRET|API_KEY|APP_KEY|APP_ID|ACCESS_KEY|"
                            r"PRIVATE_KEY|CREDENTIAL|"
                            r"(^|_)(ACCESS|AUTH|API|BEARER|REFRESH|SESSION)_TOKEN$|^TOKEN$)",
                            re.IGNORECASE)

# (description, regex over the concatenated `run:` scripts of ci.yml)
CI_REQUIRED_STEPS = (
    ("build the archive with `git archive`", r"git\s+archive\b"),
    ("run scripts/check_release.py", r"scripts/check_release\.py"),
    ("byte-compile the sources (python -m compileall)", r"-m\s+compileall\b"),
    ('run the pure tests (pytest -m "not db")', r"pytest\b[^\n]*-m\s+[\"']not db[\"']"),
    ("run the DB tests (pytest -m db...)", r"pytest\b[^\n]*-m\s+[\"']?db\b"),
    ("run the end-to-end test (pytest -m e2e)", r"pytest\b[^\n]*-m\s+[\"']?e2e\b"),
    ("verify the test summary (scripts/ci_summary.py)", r"scripts/ci_summary\.py"),
)
CI_FORBIDDEN = (
    ("a step that can fail silently (`|| true`)", r"\|\|\s*true\b"),
    ("a step that can fail silently (`|| :`)", r"\|\|\s*:\s*$"),
)


def _normalize(files):
    """{path: content} with "./" and a single top-level archive folder removed;
    directory entries dropped."""
    items = {}
    for p, data in files.items():
        if not p or p.endswith("/"):
            continue
        items[p[2:] if p.startswith("./") else p] = data
    tops = {p.split("/", 1)[0] for p in items}
    if len(tops) == 1 and items and all("/" in p for p in items):
        prefix = next(iter(tops)) + "/"
        items = {p[len(prefix):]: d for p, d in items.items()}
    return items


def ci_problems(text):
    """Problems with the CI workflow's CONTENT (empty file, bad YAML, a missing
    stage). Returns [] for a workflow that runs the whole release pipeline."""
    if not text.strip():
        return [f"{CI_PATH} is empty"]
    try:
        import yaml
    except ImportError:
        return [f"cannot validate {CI_PATH}: PyYAML is not installed "
                "(pip install -r requirements-dev.txt)"]
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return [f"{CI_PATH} is not valid YAML: {str(e).splitlines()[0]}"]
    if not isinstance(doc, dict):
        return [f"{CI_PATH} is not a workflow (top level is not a mapping)"]
    out = []
    # YAML 1.1 reads a bare `on:` key as boolean True.
    if "on" not in doc and True not in doc:
        out.append(f"{CI_PATH} has no `on:` trigger")
    jobs = doc.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        return out + [f"{CI_PATH} defines no jobs"]
    runs, has_postgres = [], False
    for name, job in jobs.items():
        if not isinstance(job, dict):
            out.append(f"{CI_PATH}: job {name!r} is not a mapping")
            continue
        if "runs-on" not in job:
            out.append(f"{CI_PATH}: job {name!r} has no runs-on")
        if job.get("continue-on-error"):
            out.append(f"{CI_PATH}: job {name!r} sets continue-on-error")
        steps = job.get("steps")
        if not isinstance(steps, list) or not steps:
            out.append(f"{CI_PATH}: job {name!r} has no steps")
            continue
        for svc in (job.get("services") or {}).values():
            if isinstance(svc, dict) and str(svc.get("image", "")).startswith("postgres"):
                has_postgres = True
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("continue-on-error"):
                out.append(f"{CI_PATH}: step {step.get('name', '?')!r} sets "
                           "continue-on-error")
            if isinstance(step.get("run"), str):
                runs.append(step["run"])
    script = "\n".join(runs)
    if not has_postgres:
        out.append(f"{CI_PATH} starts no PostgreSQL service")
    for what, rx in CI_REQUIRED_STEPS:
        if not re.search(rx, script):
            out.append(f"{CI_PATH} does not {what}")
    if len(re.findall(r"\b(?:migrate|db_pg)\.py\b(?!\s+--status)", script)) < 2:
        out.append(f"{CI_PATH} does not run the migrations twice "
                   "(empty database, then the no-op re-run)")
    for what, rx in CI_FORBIDDEN:
        if re.search(rx, script, re.MULTILINE):
            out.append(f"{CI_PATH} contains {what}")
    return out


def env_example_problems(text):
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if SECRET_NAME_RE.search(key) and value:
            out.append(f".env.example line {n}: {key} has a value — secrets must be blank")
    return out


def problems(files):
    """Problems in a release. `files` maps relative POSIX path -> content (bytes).
    A list of paths is also accepted; their content is then treated as unknown and
    the content checks report it (a release check that cannot see content is not
    a pass)."""
    if not isinstance(files, dict):
        files = {p: None for p in files}
    content = _normalize(files)
    out = []
    for p in content:
        parts = p.split("/")
        name = parts[-1]
        if any(d in FORBIDDEN_DIRS or d.endswith(".egg-info") for d in parts[:-1]):
            out.append(f"forbidden directory in release: {p}")
        elif name in SECRET_NAMES or (name.startswith(".env.") and name != ".env.example"):
            out.append(f"secret file in release: {p}")
        elif p.endswith(FORBIDDEN_SUFFIXES):
            out.append(f"forbidden file type in release: {p}")
        elif name in FORBIDDEN_NAMES or name.startswith(".coverage."):
            out.append(f"forbidden local/OS file in release: {p}")
    for r in REQUIRED:
        if r not in content:
            out.append(f"missing required file: {r}")
            continue
        data = content[r]
        if data is None:
            out.append(f"cannot read content of required file: {r}")
            continue
        text = data.decode("utf-8", errors="replace")
        if not text.strip():
            out.append(f"required file is empty: {r}")
        elif r == CI_PATH:
            out += ci_problems(text)
        elif r == ".env.example":
            out += env_example_problems(text)
    return out


def _read_target(target):
    """{relative path: bytes} for the git index, a zip archive, or a directory."""
    if target is None:
        res = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
        paths = [p for p in res.stdout.decode("utf-8").split("\0") if p]
        out = {}
        for p in paths:
            blob = subprocess.run(["git", "show", f":{p}"], capture_output=True)
            out[p] = blob.stdout if blob.returncode == 0 else None
        return out
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z:
            return {i.filename: (None if i.is_dir() else z.read(i))
                    for i in z.infolist() if not i.is_dir()}
    if os.path.isdir(target):
        out = {}
        for root, _dirs, files in os.walk(target):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, target).replace(os.sep, "/")
                with open(full, "rb") as fh:
                    out[rel] = fh.read()
        return out
    raise SystemExit(f"not a zip archive or directory: {target}")


def main(argv):
    target = argv[1] if len(argv) > 1 else None
    found = problems(_read_target(target))
    for p in found:
        print("RELEASE CHECK FAILED:", p)
    if not found:
        print("release check passed")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))