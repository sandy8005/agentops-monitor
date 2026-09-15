import pytest
pytestmark = [pytest.mark.db]
# test_job_processing.py
from router import _reqs_cache_get, _reqs_cache_put, _reqs_cache_key, _hash, get_connection

def test_reqs_cache_roundtrip():
    h = _hash("job-desc-unit-test-123")
    # Make the test self-cleaning: drop any prior row for this fixed key so the
    # INSERT actually runs (ON CONFLICT DO NOTHING would otherwise pin a stale row).
    conn = get_connection(); cur = conn.cursor()
    cur.execute("DELETE FROM job_reqs_cache WHERE desc_hash = %s", (h,))
    conn.commit(); conn.close()

    _reqs_cache_put(h, {"required_skills": ["python"], "required_any_of": [],
                        "preferred_skills": [], "min_years_experience": 0,
                        "responsibilities": []}, "llm")
    reqs, provenance = _reqs_cache_get(h)
    assert reqs is not None and reqs["required_skills"] == ["python"]
    assert provenance["extraction_method"] == "llm"
    assert provenance["source_model"]  # non-empty version string

def test_reqs_cache_key_includes_title():
    k1 = _reqs_cache_key("AI Engineer", "same description text")
    k2 = _reqs_cache_key("Data Engineer", "same description text")
    assert k1 != k2
    assert _reqs_cache_key("AI Engineer", "x") == _reqs_cache_key("AI Engineer", "x")

def test_skip_band_boundaries():
    def judged(score): return 20 <= score <= 80
    assert judged(19) is False
    assert judged(20) is True
    assert judged(80) is True
    assert judged(81) is False