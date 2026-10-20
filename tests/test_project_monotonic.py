"""Backlog P2: adding an unrelated, truthful project must never lower the score.
Synthetic resume text only."""
import itertools

import pytest

from scorer import calculate_match_score

RESUME = ("Backend developer. Skills: Python, Django, PostgreSQL, Docker.\n"
          "Recipe Blog: a static site built with Hugo.\n"
          "Inventory API: REST service built with Python and Django.")
REQS = {"required_skills": ["Python", "Django", "Docker"], "required_any_of": [],
        "preferred_skills": ["PostgreSQL"], "min_years_experience": 3,
        "responsibilities": []}
UNRELATED = {"name": "Recipe Blog", "tech": ["Hugo"]}
RELATED = {"name": "Inventory API", "tech": ["Python", "Django"]}


def _parsed(projects, years=4, experience=True):
    return {"skills": ["Python", "Django", "PostgreSQL", "Docker"], "years_experience": years,
            "education": [], "projects": projects,
            "experience": [{"title": "Backend Developer", "company": "Acme", "years": years}]
            if experience else []}


def test_unrelated_project_does_not_lower_the_score():
    without = calculate_match_score(_parsed([]), REQS, RESUME)
    with_extra = calculate_match_score(_parsed([UNRELATED]), REQS, RESUME)
    assert with_extra["score"] >= without["score"]
    assert with_extra["decision"] == without["decision"]
    assert with_extra["projects_counted"] is False      # it would only have hurt


def test_relevant_project_can_still_raise_the_score():
    base = calculate_match_score(_parsed([]), REQS, RESUME)
    strong = calculate_match_score(_parsed([{"name": "Inventory API",
                                             "tech": ["Python", "Django", "Docker"]}]),
                                   REQS, RESUME)
    assert strong["projects_counted"] is True
    assert strong["score"] >= base["score"]


@pytest.mark.parametrize("years,experience", [(4, True), (1, True), (0, False)])
def test_adding_any_project_is_monotonic(years, experience):
    pool = [UNRELATED, RELATED, {"name": "CLI Tool", "tech": ["Docker"]}]
    for n in range(len(pool)):
        for base in itertools.combinations(pool, n):
            for extra in pool:
                if extra in base:
                    continue
                before = calculate_match_score(_parsed(list(base), years, experience),
                                               REQS, RESUME)["score"]
                after = calculate_match_score(_parsed(list(base) + [extra], years, experience),
                                              REQS, RESUME)["score"]
                assert after >= before, (base, extra)
