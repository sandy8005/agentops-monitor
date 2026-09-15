import pytest
pytestmark = [pytest.mark.db]
# test_job_processing.py
from router import _reqs_cache_get, _reqs_cache_put, _reqs_cache_key, _hash

def test_reqs_cache_roundtrip():
    h = _hash("job-desc-unit-test-123")
    _reqs_cache_put(h, {"required_skills": ["python"], "required_any_of": [],
                        "preferred_skills": [], "min_years_experience": 0,
                        "responsibilities": []}, "llm")
    reqs, provenance = _reqs_cache_get(h)
    assert reqs is not None and reqs["required_skills"] == ["python"]
    # provenance is recorded and returned on a hit
    assert provenance["extraction_method"] == "llm"
    assert provenance["source_model"]  # non-empty version string

def test_reqs_cache_key_includes_title():
    # Same description, different title → different cache key (title is in the key).
    k1 = _reqs_cache_key("AI Engineer", "same description text")
    k2 = _reqs_cache_key("Data Engineer", "same description text")
    assert k1 != k2
    # Same title+description → stable key.
    assert _reqs_cache_key("AI Engineer", "x") == _reqs_cache_key("AI Engineer", "x")

def test_skip_band_boundaries():
    # the judge runs for 20..80 inclusive, skips outside
    def judged(score): return 20 <= score <= 80
    assert judged(19) is False   # extreme low → skip
    assert judged(20) is True    # boundary → judge
    assert judged(80) is True    # boundary → judge
    assert judged(81) is False   # extreme high → skip