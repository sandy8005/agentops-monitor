"""
Print the GitHub Actions status of the latest CI workflow run for one commit, via
the public GitHub REST API. Stdlib only; an operator check, not a test (it lives
outside tests/, so CI never collects it and pytest never needs the network).

    python scripts/verify_ci_status.py                  # HEAD of the local checkout
    python scripts/verify_ci_status.py --sha 3f7a0b2... # a specific (full) commit
    GITHUB_TOKEN=... python scripts/verify_ci_status.py # higher rate limit / private repo

Exit status:
  0  the latest run for the commit completed with conclusion "success"
  1  it completed with any other conclusion (failure, cancelled, timed_out, ...)
  3  it is still queued / in progress
  4  no workflow run exists for that commit (not pushed, or CI not triggered)
  2  the check could not be made (network, HTTP error, bad arguments)

Output is metadata only: workflow name, run id, event, branch, status, conclusion,
timestamps and the run's URL. The token is sent only to api.github.com and never
printed.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPO = "sandy8005/agentops-monitor"
DEFAULT_WORKFLOW = "ci.yml"
API = "https://api.github.com"


def _head_sha():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


def fetch_runs(repo, workflow, sha, token=None, timeout=20):
    """The decoded JSON of the workflow-runs listing filtered to one commit."""
    q = urllib.parse.urlencode({"head_sha": sha, "per_page": 20})
    url = f"{API}/repos/{repo}/actions/workflows/{urllib.parse.quote(workflow)}/runs?{q}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "agentops-monitor-verify-ci-status",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def summarize(payload, sha):
    """(exit_code, [lines]) for a workflow-runs payload. Pure, so it is unit-tested
    without the network (tests/test_verify_ci_status.py)."""
    runs = [r for r in (payload or {}).get("workflow_runs") or []
            if isinstance(r, dict) and r.get("head_sha") == sha]
    if not runs:
        return 4, [f"verify_ci_status: no workflow run found for {sha}"]
    latest = max(runs, key=lambda r: (r.get("run_number") or 0, r.get("run_attempt") or 0))
    status, conclusion = latest.get("status"), latest.get("conclusion")
    lines = [
        f"workflow={latest.get('name')} run_id={latest.get('id')} "
        f"run_number={latest.get('run_number')} attempt={latest.get('run_attempt')}",
        f"event={latest.get('event')} branch={latest.get('head_branch')} sha={sha}",
        f"status={status} conclusion={conclusion}",
        f"created={latest.get('created_at')} updated={latest.get('updated_at')}",
        f"url={latest.get('html_url')}",
    ]
    if status != "completed":
        return 3, lines + ["verify_ci_status: NOT FINISHED"]
    if conclusion == "success":
        return 0, lines + ["verify_ci_status: GREEN"]
    return 1, lines + [f"verify_ci_status: NOT GREEN ({conclusion})"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="owner/name")
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="workflow file name or id")
    ap.add_argument("--sha", default=None, help="full commit SHA (default: local HEAD)")
    args = ap.parse_args(argv)
    try:
        sha = args.sha or _head_sha()
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"verify_ci_status: cannot read HEAD ({type(e).__name__})", file=sys.stderr)
        return 2
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha.lower()):
        print("verify_ci_status: --sha must be a full 40-character commit SHA", file=sys.stderr)
        return 2
    token = (os.environ.get("GITHUB_TOKEN") or "").strip() or None
    try:
        payload = fetch_runs(args.repo, args.workflow, sha.lower(), token)
    except urllib.error.HTTPError as e:
        print(f"verify_ci_status: GitHub API HTTP {e.code}", file=sys.stderr)
        return 2
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"verify_ci_status: GitHub API unreachable ({type(e).__name__})", file=sys.stderr)
        return 2
    code, lines = summarize(payload, sha.lower())
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
