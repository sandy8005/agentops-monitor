"""
Tests for run-level error-code classification (error_codes.py). Pure logic.

Run:  pytest tests/test_error_codes.py -v
"""
from error_codes import ErrorCode, classify_exception, ALL_CODES


def test_plain_429_is_a_transient_rate_limit():
    # A bare 429 is a per-minute rate limit: transient, worth retrying.
    assert classify_exception(Exception("429 RESOURCE_EXHAUSTED")) == ErrorCode.LLM_RATE_LIMITED
    assert classify_exception(Exception(
        "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel")) \
        == ErrorCode.LLM_RATE_LIMITED


def test_daily_or_zero_quota_is_quota_exhausted():
    # A spent daily/project quota won't recover on a short backoff: distinct, terminal.
    assert classify_exception(Exception(
        "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")) \
        == ErrorCode.LLM_QUOTA_EXHAUSTED
    assert classify_exception(Exception("429 RESOURCE_EXHAUSTED ... limit: 0")) \
        == ErrorCode.LLM_QUOTA_EXHAUSTED


def test_provider_retry_delay_is_extracted():
    from error_codes import provider_retry_after
    e = Exception("429 RESOURCE_EXHAUSTED {'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
                  "'retryDelay': '31s'}")
    assert provider_retry_after(e) == 31.0
    assert provider_retry_after(Exception("503 UNAVAILABLE")) is None


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