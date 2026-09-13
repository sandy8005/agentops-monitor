"""
Regression tests for optional-category score normalization in scorer.py.

The bug: when a job listed NO preferred skills, the 20-pt "preferred" bucket
scored 0 but its weight still counted toward 100, so a perfect candidate was
capped below 100 (and could be pushed from Apply to Maybe) purely because an
OPTIONAL category was absent. The fix drops an absent optional category's weight
and renormalizes the present categories proportionally.

Pure logic — imports only scorer, no DB, no LLM.  Run:  pytest test_scoring_normalization.py -v
"""
from scorer import calculate_match_score


def _parsed(years, tech):
    return {"projects": [{"tech": tech}], "years_experience": years}


RESUME = "python sql django aws docker kubernetes"


def test_full_match_scores_100_without_preferred():
    # Perfect required + projects + experience, job listed NO preferred skills.
    reqs = {"required_skills": ["python", "sql"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 3}
    r = calculate_match_score(_parsed(5, ["python", "sql"]), reqs, RESUME)
    assert r["score"] == 100.0
    assert r["decision"] == "Apply"
    assert r["breakdown"]["preferred"] == 0.0   # shown as 0 (absent), not counted


def test_full_match_scores_100_with_preferred():
    # Same candidate, job DID list preferred skills the candidate fully has.
    reqs = {"required_skills": ["python", "sql"], "required_any_of": [],
            "preferred_skills": ["aws", "docker"], "min_years_experience": 3}
    r = calculate_match_score(_parsed(5, ["python", "sql"]), reqs, RESUME)
    assert r["score"] == 100.0
    assert r["breakdown"]["preferred"] == 20.0


def test_absent_preferred_does_not_lower_score():
    # The two scenarios above must yield the SAME total — absence of the optional
    # category is not a penalty.
    base = {"required_skills": ["python", "sql"], "required_any_of": [],
            "min_years_experience": 3}
    no_pref = calculate_match_score(_parsed(5, ["python", "sql"]),
                                    {**base, "preferred_skills": []}, RESUME)
    with_pref = calculate_match_score(_parsed(5, ["python", "sql"]),
                                      {**base, "preferred_skills": ["aws", "docker"]}, RESUME)
    assert no_pref["score"] == with_pref["score"] == 100.0


def test_present_preferred_miss_still_costs():
    # When the job DOES list preferred skills the candidate lacks, that's a real
    # miss and must still reduce the score (redistribution only applies when the
    # category is absent, not when it's present-but-unmet).
    reqs = {"required_skills": ["python", "sql"], "required_any_of": [],
            "preferred_skills": ["rust", "go"], "min_years_experience": 3}
    r = calculate_match_score(_parsed(5, ["python", "sql"]), reqs, RESUME)
    assert r["score"] < 100.0
    assert r["breakdown"]["preferred"] == 0.0


def test_breakdown_sums_to_total():
    # Whatever the shape, the reported parts sum to the total (no drift past 100).
    reqs = {"required_skills": ["python", "sql", "aws"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 4}
    r = calculate_match_score(_parsed(2, ["python"]), reqs, RESUME)
    assert abs(sum(r["breakdown"].values()) - r["score"]) <= 0.1
    assert r["score"] <= 100.0


def test_relative_weights_preserved_when_preferred_absent():
    # With preferred (20) dropped, the remaining 50/15/15 renormalize to 100 while
    # keeping their ratios: required should be 62.5, projects/experience 18.75 each
    # at full satisfaction.
    reqs = {"required_skills": ["python", "sql"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 3}
    r = calculate_match_score(_parsed(5, ["python", "sql"]), reqs, RESUME)
    assert r["breakdown"]["required"] == 62.5
    assert r["breakdown"]["projects"] == 18.8    # 18.75 rounded
    assert r["breakdown"]["experience"] == 18.8