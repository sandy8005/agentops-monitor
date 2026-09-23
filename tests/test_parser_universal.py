"""
Assertion-based tests for the universal resume normalizer. Replaces the old
print-style test that imported the removed `_to_float`. Tests the functions that
exist now: _normalize_experience (incl. correct months->years), _normalize_education.
"""
from parser import _normalize_experience, _normalize_education


def test_normalize_experience_years_field():
    r = _normalize_experience([{"title": "Dev", "company": "X", "years": 3}])
    assert r[0]["years"] == 3.0
    assert r[0]["title"] == "Dev"

def test_normalize_experience_months_single_conversion():
    # "24 months" must be 2.0 years, NOT 0.17 (the double-conversion bug)
    r = _normalize_experience([{"title": "Dev", "company": "X", "months": "24 months"}])
    assert r[0]["years"] == 2.0
    r2 = _normalize_experience([{"title": "Dev", "company": "Y", "months": 6}])
    assert r2[0]["years"] == 0.5

def test_normalize_experience_alt_keys():
    # role/position instead of title, employer instead of company
    r = _normalize_experience([{"role": "Engineer", "employer": "Acme", "years": 2}])
    assert r[0]["title"] == "Engineer"
    assert r[0]["company"] == "Acme"

def test_normalize_experience_skips_non_dicts():
    r = _normalize_experience(["garbage", {"title": "Dev", "company": "X", "years": 1}])
    assert len(r) == 1
    assert r[0]["title"] == "Dev"

def test_normalize_education_alt_keys():
    r = _normalize_education([{"qualification": "BSc", "school": "MIT", "graduation_year": 2020}])
    assert r[0]["degree"] == "BSc"
    assert r[0]["institution"] == "MIT"
    assert r[0]["year"] == "2020"

def test_normalize_education_empty():
    assert _normalize_education([]) == []
    assert _normalize_education(None) == []

# --- Full parse_resume() with a mocked LLM -----------------------------------
# The regression guard the suite was missing. These call the WHOLE parse_resume,
# so a gutted body (no json.loads / no return -> None) is caught instead of only
# the _normalize_* helpers. Uses a mocked LLM response — no network, no DB.
import pytest


def test_parse_resume_end_to_end_with_mocked_llm(monkeypatch):
    import parser as parsermod
    mock_json = (
        '```json\n'
        '{"skills": ["Python", "SQL"], "years_experience": 5,\n'
        ' "education": [{"degree": "BS CS", "institution": "MIT", "year": "2019"}],\n'
        ' "projects": [{"name": "AgentOps", "tech": ["Python", "FastAPI"]}],\n'
        ' "experience": [{"title": "Engineer", "company": "Acme", "years": 3},\n'
        '                {"title": "Senior Eng", "company": "Beta", "months": 24}]}\n'
        '```'
    )
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: mock_json)
    result = parsermod.parse_resume("resume text", run_id=1, step_id=1)
    assert isinstance(result, dict)                        # NOT None — the regression
    assert result["skills"] == ["Python", "SQL"]
    assert result["projects"][0]["tech"] == ["Python", "FastAPI"]
    assert result["experience"][1]["years"] == 2.0        # 24 months -> 2 years, once
    assert result["years_experience_summed"] == 5.0       # reconcile ran


def test_parse_resume_raises_on_unparseable_output(monkeypatch):
    # Unparseable LLM output must RAISE (run fails cleanly, nothing cached),
    # never return None (which poisoned the parse cache).
    import parser as parsermod
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: "not json at all")
    with pytest.raises(Exception):
        parsermod.parse_resume("resume text", run_id=1, step_id=1)