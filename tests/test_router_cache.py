# test_router_cache.py
import pytest

from router import _hash, _parse_cache_get, _parse_cache_put

def test_hash_is_stable_and_changes_with_text():
    ...                                   # unchanged

@pytest.mark.db
def test_parse_cache_roundtrip():
    ...                                   # unchanged