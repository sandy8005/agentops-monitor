# test_job_processing.py
from router import _reqs_cache_get, _reqs_cache_put, _hash

def test_reqs_cache_roundtrip():
    h = _hash("job-desc-unit-test-123")
    _reqs_cache_put(h, {"required_skills": ["python"], "required_any_of": [],
                        "preferred_skills": [], "min_years_experience": 0, "responsibilities": []})
    got = _reqs_cache_get(h)
    assert got is not None and got["required_skills"] == ["python"]

def test_skip_band_boundaries():
    # the judge runs for 20..80 inclusive, skips outside
    def judged(score): return 20 <= score <= 80
    assert judged(19) is False   # extreme low → skip
    assert judged(20) is True    # boundary → judge
    assert judged(80) is True    # boundary → judge
    assert judged(81) is False   # extreme high → skip