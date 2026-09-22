"""
Tests for run-level error-code classification (error_codes.py). Pure logic.

Run:  pytest tests/test_error_codes.py -v
"""
from error_codes import ErrorCode, classify_exception, ALL_CODES


def test_quota_exhaustion_maps_to_quota_code():
    assert classify_exception(Exception("429 RESOURCE_EXHAUSTED")) == ErrorCode.LLM_QUOTA_EXHAUSTED
    assert classify_exception(Exception("You exceeded your quota RESOURCE_EXHAUSTED")) == ErrorCode.LLM_QUOTA_EXHAUSTED


def test_unavailable_and_timeout_map_to_unavailable():
    assert classify_exception(Exception("503 UNAVAILABLE")) == ErrorCode.LLM_UNAVAILABLE
    assert classify_exception(Exception("The read operation timed out")) == ErrorCode.LLM_UNAVAILABLE
    assert classify_exception(Exception("connection timeout")) == ErrorCode.LLM_UNAVAILABLE


def test_budget_maps_to_budget_code():
    assert classify_exception(Exception("LLM budget reached before attempt 4")) == ErrorCode.BUDGET_EXCEEDED


def test_unknown_maps_to_internal():
    assert classify_exception(Exception("something odd broke")) == ErrorCode.INTERNAL


def test_all_codes_are_stable_strings():
    for c in ALL_CODES:
        assert isinstance(c, str) and c
    # "No error" is represented by None / SQL NULL, not an ErrorCode member.
    assert not hasattr(ErrorCode, "NONE")