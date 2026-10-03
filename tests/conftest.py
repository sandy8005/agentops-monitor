"""
Test-suite fixtures.

The /login endpoint is rate-limited (5/minute) and slowapi keys the limit on the
client IP. Under Starlette's TestClient every request comes from the same address
("testclient"), so a suite that logs in more than a few times shares ONE bucket and
starts getting 429s — which has nothing to do with what the tests assert (data
ownership / CSRF). This autouse fixture clears the limiter before each test so every
test starts with a fresh allowance. The real 5/minute limit stays in force in the
app; we're only isolating tests from each other, not disabling the protection.
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    try:
        import api
        api.limiter.reset()
    except Exception:
        # slowapi not installed, or app not importable for a pure-logic test —
        # nothing to reset, so let the test run as-is.
        pass
    yield

@pytest.fixture(autouse=True)
def _reset_quota_breaker():
    """The LLM quota circuit breaker is process-global. A test that trips it
    (e.g. an exhausted-quota test) must not make later, unrelated tests take their
    rules fallback. Reset before and after every test."""
    try:
        import llm
        llm.reset_quota_breaker()
    except Exception:
        pass
    yield
    try:
        import llm
        llm.reset_quota_breaker()
    except Exception:
        pass

@pytest.fixture(autouse=True)
def _deterministic_model_settings(monkeypatch):
    """Tests never depend on the developer's .env for MODEL settings: a fake API key
    (no test makes a real provider request — they monkeypatch real_llm_once) and
    the default pricing, so pass/fail is identical on a laptop and in CI. Tests
    that need "no key configured" override it themselves."""
    from settings import settings
    monkeypatch.setattr(settings, "gemini_api_key", "test-key-not-real", raising=False)
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.6-flash", raising=False)
    monkeypatch.setattr(settings, "llm_input_price_per_million", None, raising=False)
    monkeypatch.setattr(settings, "llm_output_price_per_million", None, raising=False)
    # No network in tests: the reservation uses the proven byte bound unless a
    # test opts into (and fakes) provider token counting.
    monkeypatch.setattr(settings, "llm_input_token_bound", "bytes", raising=False)
    monkeypatch.setattr(settings, "llm_reserve_bytes_per_token", 1.0, raising=False)
    yield