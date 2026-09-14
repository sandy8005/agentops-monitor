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