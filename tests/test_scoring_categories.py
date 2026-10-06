"""
A scoring category participates only when the POSTING gives it something to
score. A section the posting omits must never cost the candidate its weight.

  required   — applies iff required skills or any-of groups exist
  preferred  — applies iff preferred skills exist
  projects   — applies iff there are REQUIRED tech units (and the candidate has
               projects, or no professional experience to stand in for them)
  experience — applies iff an explicit minimum (min_years > 0) exists
  insufficient_requirements — nothing at all to score (no skills of any kind and
               no experience minimum)
"""
import pytest

from scorer import calculate_match_score


def _reqs(required=None, any_of=None, preferred=None, min_years=0):
    return {"required_skills": required or [], "required_any_of": any_of or [],
            "preferred_skills": preferred or [], "min_years_experience": min_years,
            "responsibilities": []}


def _resume(skills, years=5, projects=None, experience=None):
    return {"skills": skills, "years_experience": years, "education": [],
            "projects": projects or [], "experience": experience or []}


PY_RESUME_TEXT = "Python developer with 5 years of experience. Built an analytics tool in Python."
PY_RESUME = _resume(["Python"], projects=[{"name": "analytics tool", "tech": ["Python"]}])


def test_preferred_only_perfect_match_is_not_penalized():
    """The reported case: no required section, Python preferred, no minimum.
    It used to score 35 / Skip; a candidate meeting everything the job lists
    must score 100."""
    r = calculate_match_score(PY_RESUME, _reqs(preferred=["Python"]), PY_RESUME_TEXT)
    assert r["score"] == 100.0
    assert r["decision"] == "Apply"
    assert r["scored_categories"] == ["preferred"]
    assert r["insufficient_requirements"] is False
    assert r["breakdown_max"]["required"] == 0.0
    assert r["breakdown_max"]["projects"] == 0.0


def test_preferred_only_missing_skill_scores_low():
    r = calculate_match_score(PY_RESUME, _reqs(preferred=["Rust"]), PY_RESUME_TEXT)
    assert r["score"] == 0.0
    assert r["decision"] == "Skip"


@pytest.mark.parametrize("years,expected_score,expected_decision", [
    (5, 100.0, "Maybe"),     # meets the minimum: Maybe (thin basis, no auto-Apply)
    (2, 40.0, "Skip"),       # 2 of 5 years: under-qualified
])
def test_experience_only_requirement_determines_the_score(years, expected_score,
                                                          expected_decision):
    r = calculate_match_score(_resume(["Python"], years=years), _reqs(min_years=5),
                              f"Engineer with {years} years of experience.")
    assert r["insufficient_requirements"] is False
    assert r["experience_only_requirements"] is True
    assert r["scored_categories"] == ["experience"]
    assert r["score"] == expected_score
    assert r["decision"] == expected_decision


def test_experience_only_with_unknown_candidate_years_is_maybe():
    p = {**_resume(["Python"]), "years_experience": None, "experience_unknown": True}
    r = calculate_match_score(p, _reqs(min_years=3), "Engineer.")
    assert r["scored_categories"] == []
    assert r["decision"] == "Maybe"


def test_no_explicit_requirements_is_flagged_insufficient():
    r = calculate_match_score(PY_RESUME, _reqs(), PY_RESUME_TEXT)
    assert r["insufficient_requirements"] is True
    assert r["experience_only_requirements"] is False
    assert r["scored_categories"] == []
    assert r["decision"] == "Maybe"


def test_required_only_normalizes_over_required_and_projects():
    """No preferred, no minimum: required (50) + projects (15) -> /65."""
    r = calculate_match_score(PY_RESUME, _reqs(required=["Python", "Rust"]), PY_RESUME_TEXT)
    assert r["scored_categories"] == ["required", "projects"]
    assert r["breakdown_max"]["required"] == 76.9
    assert r["breakdown_max"]["projects"] == 23.1
    assert r["score"] == 50.0                     # half of each applicable category
    assert r["missing_skills"] == ["rust"]


def test_required_only_with_experience_standing_in_for_projects():
    """No projects section but real work history: projects absent, required alone."""
    p = _resume(["Python"], projects=[], experience=[{"title": "Dev", "years": 5}])
    r = calculate_match_score(p, _reqs(required=["Python"]), PY_RESUME_TEXT)
    assert r["scored_categories"] == ["required"]
    assert r["score"] == 100.0


def test_required_plus_preferred_plus_minimum_keeps_the_base_weights():
    """Every category present: the original 50/20/15/15 weights, unchanged."""
    r = calculate_match_score(PY_RESUME, _reqs(required=["Python"], preferred=["Docker"],
                                               min_years=3), PY_RESUME_TEXT)
    assert r["scored_categories"] == ["required", "preferred", "projects", "experience"]
    assert r["breakdown_max"] == {"required": 50.0, "preferred": 20.0,
                                  "projects": 15.0, "experience": 15.0}
    assert r["score"] == 80.0                     # everything except Docker
    assert r["decision"] == "Apply"


def test_required_plus_preferred_without_minimum():
    r = calculate_match_score(PY_RESUME, _reqs(required=["Python"], preferred=["Docker"]),
                              PY_RESUME_TEXT)
    assert r["scored_categories"] == ["required", "preferred", "projects"]
    assert r["breakdown_max"]["experience"] == 0.0
    # (50 + 0 + 15) / 85
    assert r["score"] == 76.5


def test_any_of_only_group_behaves_as_one_required_unit():
    r = calculate_match_score(
        _resume(["Django"], projects=[{"name": "site", "tech": ["Django"]}]),
        _reqs(any_of=[["Flask", "Django"]]), "Built a site with Django. 5 years.")
    assert r["scored_categories"] == ["required", "projects"]
    assert r["matched_skills"] == ["flask / django"]
    assert r["score"] == 100.0


def test_experience_shortfall_still_scales_against_an_explicit_minimum():
    r = calculate_match_score(_resume(["Python"], years=2, projects=[
        {"name": "x", "tech": ["Python"]}]), _reqs(required=["Python"], min_years=4),
        "Python developer, 2 years. Built x in Python.")
    assert r["breakdown_max"]["experience"] == 18.8      # 15 / 80
    assert r["breakdown"]["experience"] == 9.4           # half of it