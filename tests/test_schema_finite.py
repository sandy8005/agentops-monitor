"""Backlog F: non-finite and absurd numbers are rejected before they reach the
scorer or the database. JSON's 1e999 parses to +inf; "inf"/"nan" strings and
float('nan') must fail too. Years fields are bounded at MAX_YEARS (60)."""
import json
import math

import pytest
from pydantic import ValidationError

import parser as parsermod
from llm import ModelOutputInvalid
from schemas import MAX_YEARS, Experience, JobRequirements, ParsedResume, _strict_float

NON_FINITE = [json.loads("1e999"), -json.loads("1e999"), float("nan"),
              "inf", "-inf", "Infinity", "nan", "NaN"]


def _resume(**kw):
    base = {"skills": [], "years_experience": 1, "education": [], "projects": [],
            "experience": []}
    base.update(kw)
    return base


def _reqs(**kw):
    base = {"required_skills": [], "preferred_skills": [], "min_years_experience": 1,
            "responsibilities": []}
    base.update(kw)
    return base


def test_json_1e999_is_infinity():
    assert math.isinf(json.loads("1e999"))


@pytest.mark.parametrize("bad", NON_FINITE)
def test_strict_float_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        _strict_float(bad, "years")


@pytest.mark.parametrize("bad", NON_FINITE)
def test_parsed_resume_years_rejects_non_finite(bad):
    with pytest.raises(ValidationError):
        ParsedResume(**_resume(years_experience=bad))


@pytest.mark.parametrize("bad", NON_FINITE)
def test_experience_years_rejects_non_finite(bad):
    with pytest.raises(ValidationError):
        Experience(title="Dev", company="Acme", years=bad)


@pytest.mark.parametrize("bad", NON_FINITE)
def test_job_requirements_min_years_rejects_non_finite(bad):
    with pytest.raises(ValidationError):
        JobRequirements(**_reqs(min_years_experience=bad))


def test_years_upper_bound():
    assert MAX_YEARS == 60
    assert ParsedResume(**_resume(years_experience=60)).years_experience == 60.0
    with pytest.raises(ValidationError):
        ParsedResume(**_resume(years_experience=61))
    with pytest.raises(ValidationError):
        Experience(title="Dev", company="Acme", years="732 months")    # 61 years
    assert Experience(title="Dev", company="Acme", years="720 months").years == 60.0
    with pytest.raises(ValidationError):
        JobRequirements(**_reqs(min_years_experience=99))


def test_parse_resume_with_1e999_is_unusable_model_output(monkeypatch):
    raw = ('{"skills": [], "years_experience": 1e999, "education": [], "projects": [], '
           '"experience": []}')
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: raw)
    with pytest.raises(ModelOutputInvalid):
        parsermod.parse_resume("Skills: Python", run_id=1, step_id=1)


def test_parse_resume_with_infinite_months_is_unusable(monkeypatch):
    raw = ('{"skills": [], "years_experience": 1, "education": [], "projects": [], '
           '"experience": [{"title": "Dev", "company": "Acme", "months": 1e999}]}')
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: raw)
    with pytest.raises(ModelOutputInvalid):
        parsermod.parse_resume("Experience\nDev, Acme 2020 - 2021", run_id=1, step_id=1)


def test_extract_requirements_with_1e999_is_unusable_model_output(monkeypatch):
    import job_parser
    raw = ('{"required_skills": [], "required_any_of": [], "preferred_skills": [], '
           '"min_years_experience": 1e999, "responsibilities": []}')
    monkeypatch.setattr(job_parser, "logged_llm_call", lambda *a, **k: raw)
    with pytest.raises(ModelOutputInvalid):
        job_parser.extract_requirements({"title": "Dev", "description": "Python"},
                                        run_id=1, step_id=1)


def test_parse_resume_with_huge_integer_months_is_unusable(monkeypatch):
    raw = ('{"skills": [], "years_experience": 1, "education": [], "projects": [], '
           '"experience": [{"title": "Dev", "company": "Acme", "months": 1' + "0" * 400 + '}]}')
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: raw)
    with pytest.raises(ModelOutputInvalid):
        parsermod.parse_resume("Experience\nDev, Acme 2020 - 2021", run_id=1, step_id=1)
