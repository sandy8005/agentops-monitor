# test_router_cache.py
from router import _hash, _parse_cache_get, _parse_cache_put

def test_hash_is_stable_and_changes_with_text():
    assert _hash("abc") == _hash("abc")          # same text → same key
    assert _hash("abc") != _hash("abd")          # changed text → different key

def test_parse_cache_roundtrip():
    h = _hash("unit-test-resume-text-xyz")
    _parse_cache_put(h, {"skills": ["python"], "years_experience": 2})
    got = _parse_cache_get(h)
    assert got is not None
    assert got["skills"] == ["python"]