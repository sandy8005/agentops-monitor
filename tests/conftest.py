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