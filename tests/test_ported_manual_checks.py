"""
Asserting replacements for the old scripts/manual/ print-scripts.

Those scripts printed values for a human to eyeball, could not be run from their
own directory (no sys.path setup), and 13 of 24 were dead: they imported modules
that no longer exist (jobs, hallucination, input_handler), read a private
sample_resume.pdf, queried a hard-coded run id, or called search_jobs without the
run_id it now requires. Several of their comments also described behaviour that
has since changed (e.g. experience reconciliation keeps the stated value and FLAGS
the discrepancy). The checks that still describe real behaviour are kept here as
assertions; the rest are covered by tests/test_parser_universal.py,
tests/test_search_fields.py, tests/test_live_search.py and the agent E2E test.
"""
import re

import pydantic
import pytest

from scorer import calculate_match_score


def test_any_of_group_satisfied_by_one_member_counts_fully():
    parsed = {"skills": ["python", "django"], "years_experience": 3,
              "projects": [{"name": "web app", "tech": ["python", "django"]}], "education": []}
    reqs = {"required_skills": ["python"], "required_any_of": [["flask", "django"]],
            "preferred_skills": [], "min_years_experience": 2, "responsibilities": []}
    r = calculate_match_score(parsed, reqs, "python django developer", None, None)
    # python + (flask OR django) are both satisfied: required and projects are full,
    # i.e. equal to the same score with no alternative group at all.
    plain = calculate_match_score(parsed, {**reqs, "required_skills": ["python", "django"],
                                           "required_any_of": []},
                                  "python django developer", None, None)
    assert r["breakdown"]["required"] == plain["breakdown"]["required"]
    assert r["breakdown"]["projects"] == plain["breakdown"]["projects"]
    assert r["missing_skills"] == []


def test_preferred_skills_are_not_reported_as_missing_requirements():
    parsed = {"skills": ["python", "flask"], "years_experience": 3,
              "projects": [{"name": "a", "tech": ["python"]}], "education": []}
    reqs = {"required_skills": ["python", "flask"], "required_any_of": [],
            "preferred_skills": ["docker"], "min_years_experience": 2, "responsibilities": []}
    r = calculate_match_score(parsed, reqs, "python flask dev", None, None)
    assert r["missing_skills"] == []
    assert r["missing_preferred"] == ["docker"]


def test_empty_requirements_are_insufficient_not_a_match():
    parsed = {"skills": ["python"], "years_experience": 3,
              "projects": [{"name": "x", "tech": ["python"]}], "education": []}
    empty = {"required_skills": [], "preferred_skills": [], "min_years_experience": 0,
             "responsibilities": []}
    r = calculate_match_score(parsed, empty, "python developer resume")
    assert r["insufficient_requirements"] is True
    assert r["decision"] != "Apply"
    assert r["breakdown"]["required"] == 0


@pytest.mark.parametrize("skill,text,expected", [
    ("Go", "Experienced Django developer", False),
    ("Go", "I write Go and Rust", True),
    ("R", "React and Redux expert", False),
    ("Java", "JavaScript and TypeScript", False),
    ("Python", "Senior Python engineer", True),
    ("machine learning", "did machine learning research", True),
    ("c++", "strong c++ background", True),
])
def test_skill_match_is_whole_token(skill, text, expected):
    from scorer import _skill_present
    low = text.lower()
    assert _skill_present(skill, low, set(re.findall(r"[a-z0-9\+\#\.]+", low))) is expected


def test_experience_discrepancy_is_flagged():
    from parser import _reconcile_experience
    out = _reconcile_experience({"years_experience": 4.0, "experience": [
        {"title": "A", "company": "X", "years": 1.0},
        {"title": "B", "company": "Y", "years": 0.5}]})
    assert out["years_experience_summed"] == 1.5
    assert out["years_experience_stated"] == 4.0
    assert out["experience_discrepancy"] is not None
    agree = _reconcile_experience({"years_experience": 3.0, "experience": [
        {"title": "A", "company": "X", "years": 2.0},
        {"title": "B", "company": "Y", "years": 1.0}]})
    assert agree["years_experience"] == 3.0 and agree["experience_discrepancy"] is None


def test_month_durations_become_years():
    from parser import _normalize_experience
    assert _normalize_experience([{"title": "Dev", "company": "X", "months": "24 months"}])[0][
        "years"] == 2.0
    assert _normalize_experience([{"title": "Dev", "company": "Y", "months": 6}])[0][
        "years"] == 0.5


def test_strict_numeric_validation():
    from schemas import Evaluation, _strict_float
    assert (_strict_float(2), _strict_float("2 years"), _strict_float("24 months")) == \
        (2.0, 2.0, 2.0)
    for bad in ("about two", "several", "N/A"):
        with pytest.raises(ValueError):
            _strict_float(bad, "years")
    with pytest.raises(pydantic.ValidationError):
        Evaluation(relevance_score=99, faithfulness_score=-2, completeness_score=5,
                   hallucination_detected=False, hallucinated_claims=[], notes="x")


def test_judge_decision_parsing():
    from agent import parse_decision
    assert parse_decision('{"decision": "Apply", "reason": "strong match"}') == "Apply"
    # The word "apply" inside the reason must not change a Skip.
    assert parse_decision('{"decision": "Skip", "reason": "do not apply here"}') == "Skip"
    assert parse_decision('```json\n{"decision": "Maybe", "reason": "partial"}\n```') == "Maybe"
    with pytest.raises(Exception):
        parse_decision('{"decision": "Probably", "reason": "x"}')


def test_ranker_orders_by_decision_then_score():
    from ranker import rank_jobs
    ranked = rank_jobs([
        {"title": "b", "score": 90.0, "decision": "Maybe", "final_decision": "Maybe"},
        {"title": "a", "score": 70.0, "decision": "Apply", "final_decision": "Apply"},
        {"title": "c", "score": 95.0, "decision": "Skip", "final_decision": "Skip"},
    ])
    assert [r["title"] for r in ranked][0] == "a"
    assert [r["title"] for r in ranked][-1] == "c"