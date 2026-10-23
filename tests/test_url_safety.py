"""Backlog A: provider URLs are untrusted data. Canonicalization must never raise,
and an invalid URL is treated as "no URL" (dedupe falls back to the fingerprint;
no apply_url is stored or returned). Synthetic postings only."""
import pytest

from job_source import _canonical_url, _dedupe_key, _dedupe_jobs, _fingerprint, safe_http_url

BAD_URLS = [
    "http://x.com:99999/",          # port out of range (urlsplit .port raises)
    "http://x.com:0/",              # port 0 is not a usable port
    "http://x.com:abc/",            # non-numeric port
    "http://[::1",                  # unterminated IPv6 bracket
    "http://[not-ipv6]/a",          # bracket host that is not IPv6
    "https://exa mple.com/a b",     # whitespace in host
    "https://example.com/a\nb",     # control character
    "javascript:alert(1)",          # non-http scheme
    "javascript://example.com/%0aalert(1)",
    "ftp://example.com/job",
    "//example.com/job",            # scheme-relative
    "https://",                     # empty host
    "https:///path",
    "   ",
    "",
]


@pytest.mark.parametrize("u", BAD_URLS)
def test_invalid_url_is_no_url(u):
    assert safe_http_url(u) is None
    # never raises, and dedupe identity falls back to the content fingerprint
    job = {"apply_url": u, "title": "Data Engineer", "company": "Acme", "location": "Remote"}
    assert _canonical_url(job) is None
    assert _dedupe_key(job) == ("content",) + _fingerprint(job)


@pytest.mark.parametrize("u", [None, 123, 4.5, ["https://a.com"], {"u": 1}, b"https://a.com"])
def test_non_string_url_is_no_url(u):
    assert safe_http_url(u) is None
    assert _canonical_url({"apply_url": u}) is None


@pytest.mark.parametrize("u", [
    "https://example.com/jobs/1",
    "http://example.com:8080/jobs?id=2",
    "https://[2001:db8::1]/jobs/3",
    "  https://ats.example.org/Job/9  ",
])
def test_valid_urls_kept(u):
    assert safe_http_url(u) == u.strip()
    assert _canonical_url({"apply_url": u}) is not None


def test_bad_apply_url_falls_through_to_valid_url_field():
    job = {"apply_url": "http://x.com:99999/", "url": "https://example.com/j/1"}
    assert _canonical_url(job) == "https://example.com/j/1"


def test_one_bad_posting_does_not_stop_the_others():
    jobs = [
        {"title": "ML Engineer", "company": "A", "location": "Remote",
         "apply_url": "http://x.com:99999/", "source": "remotive"},
        {"title": "Data Engineer", "company": "B", "location": "Remote",
         "apply_url": "https://exa mple.com/a b", "source": "remotive"},
        {"title": "AI Engineer", "company": "C", "location": "Remote",
         "apply_url": "https://c.example.com/1", "source": "remotive"},
        {"title": "AI Engineer", "company": "C", "location": "Remote",
         "apply_url": "https://c.example.com/1?utm_source=x", "source": "adzuna"},
    ]
    out = _dedupe_jobs(jobs)
    assert sorted(j["company"] for j in out) == ["A", "B", "C"]


def test_api_never_returns_invalid_apply_url():
    import api
    assert api._safe_url("http://x.com:99999/") is None
    assert api._safe_url("https://exa mple.com/a b") is None
    assert api._safe_url("javascript:alert(1)") is None
    assert api._safe_url("https://example.com/j") == "https://example.com/j"


class _Cur:
    def __init__(self):
        self.params = []

    def execute(self, sql, params):
        self.params.append(params)

    def fetchone(self):
        return (1, True)


class _Conn:
    def __init__(self):
        self.cur = _Cur()

    def cursor(self):
        return self.cur


def test_invalid_apply_url_is_stored_as_null():
    from job_persistence import upsert_postings
    conn = _Conn()
    upsert_postings([{"title": "T", "company": "C", "description": "d", "location": "Remote",
                      "work_mode": "remote", "employment_type": None, "source": "remotive",
                      "external_id": "x1", "apply_url": "http://x.com:99999/"}], conn=conn)
    assert conn.cur.params[0][-1] is None
