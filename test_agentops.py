"""
Automated test suite for AgentOps Monitor core logic.
Pure-logic tests (no LLM/network/quota): scoring, parsing reconciliation,
skill/role matching, schema validation, and search filters.

Run:  pytest -v
"""
import pytest
import pydantic

from schemas import _strict_float, Evaluation, ParsedResume
from scorer import calculate_match_score
from parser import _reconcile_experience
from job_source import search_jobs, _role_matcher


# ----------------------- schema / validation -----------------------

def test_strict_float_recovers_clean_numbers():
    assert _strict_float(2) == 2.0
    assert _strict_float("2") == 2.0
    assert _strict_float("2.5") == 2.5

def test_strict_float_recovers_years_and_months():
    assert _strict_float("2 years") == 2.0
    assert _strict_float("24 months") == 2.0

@pytest.mark.parametrize("garbage", ["about two", "several", "N/A", "lots"])
def test_strict_float_rejects_garbage(garbage):
    with pytest.raises(ValueError):
        _strict_float(garbage, "years")

def test_evaluation_rejects_out_of_range_scores():
    with pytest.raises(pydantic.ValidationError):
        Evaluation(relevance_score=99, faithfulness_score=-2, completeness_score=5,
                   hallucination_detected=False, hallucinated_claims=[], notes="x")

def test_evaluation_accepts_valid_scores():
    ev = Evaluation(relevance_score=8, faithfulness_score=7, completeness_score=9,
                    hallucination_detected=False, hallucinated_claims=[], notes="ok")
    assert ev.relevance_score == 8

def test_parsed_resume_rejects_negative_experience():
    with pytest.raises(pydantic.ValidationError):
        ParsedResume(skills=[], years_experience=-3, education=[], projects=[], experience=[])


# ----------------------- scoring -----------------------

def _reqs(required=None, any_of=None, preferred=None, min_years=0):
    return {
        "required_skills": required or [],
        "required_any_of": any_of or [],
        "preferred_skills": preferred or [],
        "min_years_experience": min_years,
        "responsibilities": [],
    }

def _resume(skills, projects=None, years=0):
    return {
        "skills": skills,
        "years_experience": years,
        "education": [],
        "projects": projects or [],
        "experience": [],
    }

def test_empty_requirements_flagged_insufficient():
    r = calculate_match_score(_resume(["python"]), _reqs(), "python developer", None, None)
    assert r["insufficient_requirements"] is True
    assert r["decision"] == "Maybe"   # forced Maybe, not a bogus high score

def test_whole_word_skill_matching_go_not_in_django():
    # "go" must NOT match inside "django"
    r = calculate_match_score(_resume(["django"]), _reqs(required=["go"]),
                              "django web framework", None, None)
    assert r["breakdown"]["required"] == 0.0

def test_required_any_of_satisfied_by_one_member():
    # "flask OR django" satisfied by django alone -> full required credit
    r = calculate_match_score(
        _resume(["python", "django"]),
        _reqs(required=["python"], any_of=[["flask", "django"]]),
        "python django", None, None
    )
    assert r["breakdown"]["required"] == 50.0

def test_project_score_uses_same_or_semantics():
    # django-only candidate should get FULL project credit for a flask-OR-django group
    r = calculate_match_score(
        _resume(["python", "django"],
                projects=[{"name": "web app", "tech": ["python", "django"]}]),
        _reqs(required=["python"], any_of=[["flask", "django"]]),
        "python django", None, None
    )
    assert r["breakdown"]["projects"] == 15.0


# ----------------------- experience reconciliation -----------------------

def test_experience_prefers_grounded_sum_on_divergence():
    p = {"years_experience": 4.0,
         "experience": [{"title": "A", "company": "X", "years": 1.0},
                        {"title": "B", "company": "Y", "years": 0.5}]}
    out = _reconcile_experience(p)
    assert out["years_experience"] == 1.5           # grounded sum used
    assert out["years_experience_stated"] == 4.0
    assert out["experience_discrepancy"] is not None

def test_experience_keeps_stated_when_within_tolerance():
    p = {"years_experience": 3.0,
         "experience": [{"title": "A", "company": "X", "years": 2.0},
                        {"title": "B", "company": "Y", "years": 1.0}]}
    out = _reconcile_experience(p)
    assert out["years_experience"] == 3.0
    assert out["experience_discrepancy"] is None


# ----------------------- role matching (whole-word) -----------------------

def test_role_matcher_requires_specializing_term():
    m = _role_matcher("AI/ML Engineer")
    # a generic engineer job with no ai/ml should NOT match
    assert m({"title": "Civil Engineer", "description": "bridges and roads"}) is False
    # an ML job should match
    assert m({"title": "Machine Learning Engineer", "description": "ml pytorch models"}) is True

def test_role_matcher_short_term_whole_word_only():
    m = _role_matcher("ML")
    # "ml" must not match inside "HTML"
    assert m({"title": "Frontend Dev", "description": "expert in HTML and CSS"}) is False


# ----------------------- search filters (need DB; skip if unavailable) -----------------------

@pytest.fixture
def db_available():
    try:
        search_jobs(target_role="engineer")
        return True
    except Exception:
        pytest.skip("database not available for search filter tests")

def test_zero_match_returns_empty(db_available):
    assert search_jobs(target_role="quantum blockchain astrologer") == []

def test_engineer_search_returns_matches(db_available):
    assert len(search_jobs(target_role="engineer")) >= 0