"""
How much does the RULE-BASED requirements fallback change outcomes?

For every posting that has a durable LLM requirements extraction in the cache,
this re-extracts the requirements with rule_requirements (zero model calls) and
compares the two:

  * requirement level: required-skill precision / recall of the rules extraction
    against the LLM one, and the min_years difference;
  * decision level (optional, --resume-id N): the deterministic score and the
    Apply / Maybe / Skip decision for that resume under each extraction —
    agreement rate and the confusion matrix (llm decision -> rules decision).

It reads the database only (no provider calls, no writes). Run it on a copy of
production data before trusting rules-only / degraded runs:

    python scripts/eval/fallback_agreement.py --limit 500 --resume-id 12

This measures FALLBACK DRIFT relative to the LLM extraction. It is not a quality
benchmark: neither extraction is ground truth, and the Apply >= 75 / Maybe >= 55
thresholds are heuristics until calibrated against human labels (see README,
"Scoring is heuristic").
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import get_connection                     # noqa: E402
from router import _reqs_cache_get, _reqs_cache_key       # noqa: E402
from rule_requirements import extract_requirements_rule_based   # noqa: E402
from skills import normalize_requirements               # noqa: E402


def _skills(reqs):
    return {str(s).lower() for s in (reqs or {}).get("required_skills") or []}


def _load_resume(resume_id):
    from rule_resume_parser import parse_resume_rules
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT resume_text FROM resumes WHERE id = %s AND is_deleted = FALSE",
                    (resume_id,))
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"resume {resume_id} not found")
    return row[0], parse_resume_rules(row[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--resume-id", type=int, default=None)
    args = ap.parse_args()

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, title, description FROM job_postings "
                    "WHERE source IN ('adzuna', 'remotive') ORDER BY id DESC LIMIT %s",
                    (args.limit,))
        postings = cur.fetchall()

    resume = _load_resume(args.resume_id) if args.resume_id else None
    compared, tp, fp, fn, years_diff = 0, 0, 0, 0, []
    decisions = Counter()
    for pid, title, desc in postings:
        llm_reqs, prov = _reqs_cache_get(_reqs_cache_key(title, desc))
        if not llm_reqs or (prov or {}).get("extraction_method") != "llm":
            continue
        rules = extract_requirements_rule_based({"title": title, "description": desc})
        a, b = _skills(llm_reqs), _skills(rules)
        tp, fp, fn = tp + len(a & b), fp + len(b - a), fn + len(a - b)
        years_diff.append(abs(float(rules.get("min_years_experience") or 0)
                              - float(llm_reqs.get("min_years_experience") or 0)))
        compared += 1
        if resume:
            from scorer import calculate_match_score
            text, parsed = resume
            job = {"id": pid, "title": title, "description": desc}
            d_llm = calculate_match_score(parsed, normalize_requirements(llm_reqs), text, job)["decision"]
            d_rules = calculate_match_score(parsed, normalize_requirements(rules), text, job)["decision"]
            decisions[(d_llm, d_rules)] += 1

    if not compared:
        print("no postings with a cached LLM extraction — nothing to compare")
        return
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"postings compared:          {compared}")
    print(f"required-skill precision:   {precision:.3f}  (rules skills also found by the LLM)")
    print(f"required-skill recall:      {recall:.3f}  (LLM skills also found by rules)")
    print(f"mean |min_years| difference: {sum(years_diff) / len(years_diff):.2f}")
    if decisions:
        total = sum(decisions.values())
        agree = sum(n for (x, y), n in decisions.items() if x == y)
        print(f"decision agreement:         {agree / total:.3f}  ({agree}/{total})")
        print("confusion (llm -> rules):")
        for (x, y), n in sorted(decisions.items()):
            print(f"  {x:>5} -> {y:<5} {n}")
        false_apply = sum(n for (x, y), n in decisions.items() if y == "Apply" and x != "Apply")
        false_skip = sum(n for (x, y), n in decisions.items() if y == "Skip" and x != "Skip")
        print(f"rules-only false Apply:     {false_apply}   rules-only false Skip: {false_skip}")


if __name__ == "__main__":
    main()