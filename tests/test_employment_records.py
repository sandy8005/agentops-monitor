"""Backlog A / D / G: professional experience comes only from VERIFIED employment
records (a date range under an experience/employment heading with a title or
company). Project, academic and volunteer timelines are kept separate and never
count as employment. Synthetic resume text only."""
import datetime
import json

import parser as parsermod
from parser import (employment_records, employment_years_from_dates,
                    ground_years_experience, section_heading)
from scorer import calculate_match_score

TODAY = datetime.date(2026, 6, 15)

REQS_PY_5Y = {"required_skills": ["Python"], "required_any_of": [], "preferred_skills": [],
              "min_years_experience": 5, "responsibilities": []}


def _stub(monkeypatch, model):
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: json.dumps(model))


# ------------------------------------------------------------------ item A ----

def test_project_dates_are_not_employment_counterexample(monkeypatch):
    """The reported counterexample: a projects-only resume extracted 6.0 years and
    scored 100 / Apply against 'Python required, 5 years minimum'."""
    resume = "Projects\nAI Resume Analyzer | Jan 2019 - Dec 2024\nBuilt with Python"
    assert employment_years_from_dates(resume, today=TODAY) is None
    _stub(monkeypatch, {"skills": ["Python"], "years_experience": 6, "education": [],
                        "projects": [{"name": "AI Resume Analyzer", "tech": ["Python"],
                                      "evidence": "Built with Python"}],
                        "experience": [], "skill_evidence": []})
    parsed = parsermod.parse_resume(resume, run_id=1, step_id=1)
    assert parsed["years_experience"] != 6.0
    assert parsed["employment_records"] == []
    # the project timeline is recorded separately, never as employment
    assert parsed["years_experience_evidence"]["date_ranges"] is None
    assert parsed["years_experience_evidence"]["non_employment_ranges"]["project"] == 6.0
    r = calculate_match_score(parsed, REQS_PY_5Y, resume)
    assert r["decision"] != "Apply"
    assert r["candidate_years_used"] != 6.0


def test_volunteer_and_academic_headings_are_separate_timelines():
    text = ("VOLUNTEER EXPERIENCE\nMentor, Code Club  2015 - 2020\n"
            "Research Experience\nResearch Assistant, Lab  2012 - 2014\n"
            "Academic Projects\nCompiler  2011 - 2012\n"
            "Work Experience\nAnalyst, Gamma  Jan 2023 - Dec 2023\n")
    assert employment_years_from_dates(text, today=TODAY) == 1.0
    recs = employment_records(text, today=TODAY)
    assert [r["header"] for r in recs] == ["Analyst, Gamma"]


def test_unsectioned_dates_are_not_verified_employment():
    # no experience/employment heading: nothing is verified
    assert employment_years_from_dates("Engineer, X  03/2024 - Present", today=TODAY) is None


def test_record_needs_a_title_or_company():
    text = "EXPERIENCE\nJan 2020 - Dec 2021\n"
    assert employment_records(text, today=TODAY) == []
    # header on the line above the date line
    text = "EXPERIENCE\nBackend Engineer, Acme\nJan 2020 - Dec 2021\n- Built APIs\n"
    recs = employment_records(text, today=TODAY)
    assert len(recs) == 1 and recs[0]["header"] == "Backend Engineer, Acme"
    assert recs[0]["months"] == 24


def test_section_heading_kinds():
    assert section_heading("Work Experience") == "employment"
    assert section_heading("## PROFESSIONAL EXPERIENCE:") == "employment"
    assert section_heading("Experience & Internships") == "employment"
    assert section_heading("Personal Projects") == "project"
    assert section_heading("Volunteer Experience") == "volunteer"
    assert section_heading("Education") == "academic"
    assert section_heading("Technical Skills") == "other"
    assert section_heading("Built a Python service") is None


# ------------------------------------------------------------------ item D ----

def test_fabricated_experience_entry_no_longer_changes_weights_scorer():
    """Same resume and job: no experience entry gave 76.9, a fabricated entry 100.0."""
    resume = "Skills: Python, Docker"
    reqs = {"required_skills": ["Python", "Docker"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 0, "responsibilities": []}
    base = {"skills": ["Python", "Docker"], "years_experience": 0, "projects": [],
            "education": [], "employment_records": []}
    without = calculate_match_score({**base, "experience": []}, reqs, resume)
    fabricated = calculate_match_score(
        {**base, "experience": [{"title": "Staff Engineer", "company": "Initech", "years": 4}]},
        reqs, resume)
    assert without["score"] == 76.9
    assert fabricated["score"] == 76.9


def test_fabricated_experience_entry_is_dropped_by_parser(monkeypatch):
    resume = "Skills: Python, Docker"
    _stub(monkeypatch, {"skills": ["Python", "Docker"], "years_experience": 0,
                        "education": [], "projects": [],
                        "experience": [{"title": "Staff Engineer", "company": "Initech",
                                        "years": 4}],
                        "skill_evidence": []})
    parsed = parsermod.parse_resume(resume, run_id=1, step_id=1)
    assert parsed["experience"] == []
    assert parsed["ungrounded_experience_entries"] == [
        {"title": "Staff Engineer", "company": "Initech", "years": 4.0}]
    reqs = {"required_skills": ["Python", "Docker"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 0, "responsibilities": []}
    assert calculate_match_score(parsed, reqs, resume)["score"] == 76.9


def test_supported_experience_entry_is_kept(monkeypatch):
    resume = ("Skills: Python, Docker\nExperience\nData Engineer, Initech  Jan 2022 - Dec 2023\n"
              "- Python and Docker pipelines\n")
    _stub(monkeypatch, {"skills": ["Python", "Docker"], "years_experience": 2,
                        "education": [], "projects": [],
                        "experience": [{"title": "Data Engineer", "company": "Initech",
                                        "years": 2}],
                        "skill_evidence": []})
    parsed = parsermod.parse_resume(resume, run_id=1, step_id=1)
    assert parsed["experience"] == [{"title": "Data Engineer", "company": "Initech",
                                     "years": 2.0}]
    assert parsed["ungrounded_experience_entries"] == []
    reqs = {"required_skills": ["Python", "Docker"], "required_any_of": [],
            "preferred_skills": [], "min_years_experience": 0, "responsibilities": []}
    assert calculate_match_score(parsed, reqs, resume)["score"] == 100.0


def test_entry_named_only_in_projects_is_not_employment(monkeypatch):
    resume = "Projects\nAI Resume Analyzer | 2019 - 2024\nBuilt with Python"
    _stub(monkeypatch, {"skills": ["Python"], "years_experience": 0, "education": [],
                        "projects": [],
                        "experience": [{"title": "AI Resume Analyzer", "company": "",
                                        "years": 6}],
                        "skill_evidence": []})
    parsed = parsermod.parse_resume(resume, run_id=1, step_id=1)
    assert parsed["experience"] == []


# ------------------------------------------------------------------ item G ----

def test_zero_stated_is_corrected_from_verified_records():
    text = "EXPERIENCE\nDeveloper, Acme  Jan 2022 - Dec 2023\n"
    out = ground_years_experience({"years_experience": 0.0}, text, today=TODAY)
    assert out["years_experience"] == 2.0
    assert out["experience_corrected"] == {"stated": 0.0, "corrected_to": 2.0,
                                           "reason": "verified_employment_records"}
    assert out["ungrounded_experience"] is None


def test_missing_stated_is_corrected_but_never_above_evidence():
    text = "EXPERIENCE\nDeveloper, Acme  Jan 2022 - Dec 2023\n"
    out = ground_years_experience({"years_experience": None}, text, today=TODAY)
    assert out["years_experience"] == 2.0


def test_zero_stated_without_records_stays_zero():
    text = "Projects\nTool  2019 - 2024\n"
    out = ground_years_experience({"years_experience": 0.0}, text, today=TODAY)
    assert out["years_experience"] == 0.0 and out["experience_corrected"] is None


def test_zero_stated_correction_end_to_end(monkeypatch):
    resume = "Experience\nPython Developer, Acme  Jan 2018 - Dec 2023\nSkills: Python"
    _stub(monkeypatch, {"skills": ["Python"], "years_experience": 0, "education": [],
                        "projects": [],
                        "experience": [{"title": "Python Developer", "company": "Acme",
                                        "years": 0}],
                        "skill_evidence": []})
    parsed = parsermod.parse_resume(resume, run_id=1, step_id=1)
    assert parsed["years_experience"] == 6.0      # Jan 2018 .. Dec 2023 (before today)
    r = calculate_match_score(parsed, REQS_PY_5Y, resume)
    assert r["candidate_years_used"] == 6.0
    assert r["breakdown"]["experience"] == r["breakdown_max"]["experience"]
