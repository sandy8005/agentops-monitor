"""
Proof that the WHOLE suite ran green — not just "each pytest command exited 0".

CI runs the suite in three disjoint slices (pure, DB, E2E), each writing a JUnit
report. This script fails (exit 1) unless, across all reports:

  * no test failed or errored;
  * no test was SKIPPED (a skipped test proves nothing — e.g. a missing optional
    dependency silently turning authorization tests into skips);
  * no test ran twice (the slices really are disjoint);
  * the number of tests that ran equals the number pytest collects for the whole
    suite (no test fell between the slices' -m filters).

    python scripts/ci_summary.py reports/pure.xml reports/db.xml reports/e2e.xml
"""
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collected_count():
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                         cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", res.stdout)
    if res.returncode != 0 or not m:
        raise SystemExit("could not collect the test suite:\n" + res.stdout + res.stderr)
    return int(m.group(1))


def summarize(paths):
    seen, failed, skipped = {}, [], []
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            tid = f"{case.get('classname')}::{case.get('name')}"
            seen.setdefault(tid, []).append(path)
            if case.find("failure") is not None or case.find("error") is not None:
                failed.append(tid)
            elif case.find("skipped") is not None:
                skipped.append(tid)
    duplicated = sorted(t for t, where in seen.items() if len(where) > 1)
    return seen, failed, skipped, duplicated


def main(argv):
    if not argv:
        print("usage: python scripts/ci_summary.py REPORT.xml [REPORT.xml ...]")
        return 2
    seen, failed, skipped, duplicated = summarize(argv)
    expected = collected_count()
    problems = []
    problems += [f"failed: {t}" for t in failed]
    problems += [f"skipped: {t}" for t in skipped]
    problems += [f"ran in more than one slice: {t}" for t in duplicated]
    if len(seen) != expected:
        problems.append(f"{len(seen)} distinct tests ran, but the suite collects "
                        f"{expected} — some tests are outside every CI slice")
    for p in problems:
        print("TEST SUMMARY FAILED:", p)
    print(f"{len(seen)} tests ran / {expected} collected; "
          f"{len(failed)} failed, {len(skipped)} skipped")
    if not problems:
        print(f"all {expected} tests passed, none skipped")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))