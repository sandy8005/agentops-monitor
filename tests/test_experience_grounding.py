"""Backlog P1: the model's stated years_experience is grounded against the resume.
Synthetic resume text only."""
import datetime
import json

import parser as parsermod
from parser import employment_years_from_dates, ground_years_experience
from scorer import calculate_match_score, effective_years

TODAY = datetime.date(2026, 6, 15)


def test_date_ranges_are_unioned_not_summed():
    text = ("EXPERIENCE\n"
            "Backend Engineer, Acme  Jan 2020 - Dec 2021\n"
            "Contract Developer, Beta  Jun 2021 - Mar 2022\n")   # overlaps Acme
    # Jan 2020 .. Mar 2022 inclusive = 27 months
    assert employment_years_from_dates(text, today=TODAY) == 2.25


def test_present_and_numeric_and_year_only_forms():
    assert employment_years_from_dates("Engineer, X  03/2024 - Present", today=TODAY) == round(28 / 12, 2)
    assert employment_years_from_dates("Engineer, X  2022 – 2023", today=TODAY) == 2.0
    assert employment_years_from_dates("Engineer at X since forever", today=TODAY) is None


def test_education_dates_are_not_employment():
    text = ("EDUCATION\nState University  2014 - 2018\n"
            "EXPERIENCE\nAnalyst, Gamma  Jan 2023 - Dec 2023\n"
            "B.Tech Computer Science 2010 - 2014\n")
    assert employment_years_from_dates(text, today=TODAY) == 1.0


def test_inflated_stated_years_are_clamped_and_recorded():
    text = "EXPERIENCE\nDeveloper, Acme  Jan 2022 - Dec 2023\n"
    out = ground_years_experience({"years_experience": 8.0}, text, today=TODAY)
    assert out["years_experience"] == 2.0
    assert out["ungrounded_experience"] == {"stated": 8.0, "supported": 2.0,
                                            "reason": "exceeds_resume_evidence"}
    assert effective_years(out) == (2.0, "resume_evidence_cap")


def test_supported_stated_years_are_kept():
    text = "Software engineer with 6 years of professional experience.\nAcme 2023 - 2024"
    out = ground_years_experience({"years_experience": 6.0}, text, today=TODAY)
    assert out["years_experience"] == 6.0 and out["ungrounded_experience"] is None
    # within the rounding tolerance of the date ranges
    out = ground_years_experience({"years_experience": 2.5}, "Acme 2023 - 2024", today=TODAY)
    assert out["years_experience"] == 2.5 and out["ungrounded_experience"] is None


def test_no_evidence_makes_experience_unknown_not_zero():
    out = ground_years_experience({"years_experience": 7.0}, "Python developer. Built APIs.",
                                  today=TODAY)
    assert out["years_experience"] is None and out["experience_unknown"] is True
    assert out["ungrounded_experience"]["reason"] == "no_resume_evidence"
    assert effective_years(out) == (None, "unknown")


def test_negated_statement_is_not_evidence():
    out = ground_years_experience({"years_experience": 5.0},
                                  "I have no 5 years of experience claim to make.", today=TODAY)
    assert out["years_experience"] is None


def test_zero_stated_is_never_flagged():
    out = ground_years_experience({"years_experience": 0.0}, "student", today=TODAY)
    assert out["years_experience"] == 0.0 and out["ungrounded_experience"] is None


def test_inflated_years_no_longer_flip_the_decision(monkeypatch):
    """End to end: the model claims 9 years against a 5-year minimum; the resume
    supports 2. The experience category must use 2, not 9."""
    resume = "EXPERIENCE\nPython Developer, Acme  Jan 2022 - Dec 2023\nSkills: Python"
    model = {"skills": ["Python"], "years_experience": 9, "education": [], "projects": [],
             "experience": [{"title": "Python Developer", "company": "Acme", "years": 9}],
             "skill_evidence": []}
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: json.dumps(model))
    parsed = parsermod.parse_resume(resume, run_id=1, step_id=1)
    assert parsed["years_experience"] == 2.0
    assert parsed["ungrounded_experience"]["stated"] == 9.0
    reqs = {"required_skills": ["Python"], "required_any_of": [], "preferred_skills": [],
            "min_years_experience": 5, "responsibilities": []}
    r = calculate_match_score(parsed, reqs, resume)
    assert r["candidate_years_used"] == 2.0
    assert r["experience_basis"] == "resume_evidence_cap"
    assert r["breakdown"]["experience"] < r["breakdown_max"]["experience"]
