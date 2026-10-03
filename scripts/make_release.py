"""
Build the release archive the ONLY supported way, then check it.

    python scripts/make_release.py                 # -> dist/agentops-monitor-<sha>.zip
    python scripts/make_release.py -o out.zip

1. Refuses to run if the working tree has uncommitted changes to tracked files:
   `git archive HEAD` ships the last COMMIT, so a dirty tree means the archive is
   not what you just tested.
2. Builds the archive with `git archive --format=zip HEAD` — tracked files only, so
   .env, __pycache__/, *.pyc, venv/, uploads and resumes can never get in.
3. Runs scripts/check_release.py on that exact file; on failure the archive is
   deleted and the exit status is 1.
4. Prints the archive's SHA-256 so the file you send can be matched to this run.

Never zip the project folder with Explorer / VS Code / `Compress-Archive`.
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-o", "--output", help="archive path (default dist/agentops-monitor-<sha>.zip)")
    args = ap.parse_args(argv)

    head = _git("rev-parse", "--short=12", "HEAD")
    if head.returncode != 0:
        print("not a git repository with a commit — commit your work first")
        return 1
    sha = head.stdout.strip()

    dirty = _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        print("uncommitted changes to tracked files — commit or stash them first, "
              "otherwise the archive would not contain what you tested:")
        print(dirty)
        return 1

    out = args.output or os.path.join(ROOT, "dist", f"agentops-monitor-{sha}.zip")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    res = _git("archive", "--format=zip", "--prefix=agentops-monitor/", "-o",
               os.path.abspath(out), "HEAD")
    if res.returncode != 0:
        print("git archive failed:", res.stderr.strip())
        return 1

    check = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_release.py"),
                            os.path.abspath(out)], cwd=ROOT)
    if check.returncode != 0:
        os.remove(out)
        print("release check FAILED — archive deleted")
        return 1

    with open(out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print(f"release archive: {out}")
    print(f"commit:          {sha}")
    print(f"sha256:          {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())