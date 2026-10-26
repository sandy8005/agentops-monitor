"""Round 5, backlog C: deceptive apply URLs must not pass validation, and the
registered domain of an accepted apply URL is returned so the dashboard can show
the real destination next to the link. Synthetic URLs only."""
import pytest

from job_source import _canonical_url, apply_url_domain, safe_http_url

DECEPTIVE = [
    "https://apply.example.com@evil.com/",       # userinfo: real host is evil.com
    "https://user:pass@evil.com/x",              # credentials
    "https://@evil.com/x",                       # empty userinfo
    "https://evil.com\\@apply.example.com/",     # backslash (browsers read it as /)
    "https://apply.example.com\\evil.com/",
    "https://evil.com/a\\b",
    "http://127.0.0.1/jobs",                     # IPv4 literal
    "http://10.0.0.5:8080/jobs",
    "https://[2001:db8::1]/jobs/3",              # IPv6 literal
    "http://[::1]/",
    "http://2130706433/",                        # integer IPv4 form
    "http://0x7f.1/",                            # hex IPv4 form
    "http://127.1/",                             # short IPv4 form
    "http://localhost/jobs",
    "http://localhost:8000/jobs",
    "http://app.localhost/jobs",
    "http://printer.local/jobs",
    "http://LOCALHOST./jobs",
    "http://intranet/jobs",                      # no dot
    "https://xn--zz-invalid-.com/",              # invalid punycode label
    "https://exa_mple.com/",                     # underscore is not a hostname char
]


@pytest.mark.parametrize("u", DECEPTIVE)
def test_deceptive_url_rejected(u):
    assert safe_http_url(u) is None
    assert _canonical_url({"apply_url": u}) is None
    assert apply_url_domain(u) is None


def test_api_and_persistence_use_the_same_rule():
    import api
    for u in DECEPTIVE:
        assert api._safe_url(u) is None


@pytest.mark.parametrize("u,domain", [
    ("https://jobs.example.com/apply/1", "example.com"),
    ("https://boards.greenhouse.io/acme/jobs/1", "greenhouse.io"),
    ("https://careers.acme.co.uk/j/2", "acme.co.uk"),
    ("https://example.com:8443/j", "example.com"),
    ("https://Apply.Example.COM/j", "example.com"),
])
def test_registered_domain_shown(u, domain):
    assert safe_http_url(u) is not None
    assert apply_url_domain(u) == domain


def test_idn_host_is_normalized_and_visible():
    # A Cyrillic "а" lookalike of apple.com: accepted as a URL but normalized to
    # punycode, so the visible domain is not "apple.com".
    u = "https://jobs.аpple.com/1"
    out = safe_http_url(u)
    assert out is not None and out.isascii()
    assert "xn--" in out
    dom = apply_url_domain(u)
    assert dom.startswith("xn--") and dom != "apple.com"


def test_punycode_host_domain_is_shown_as_punycode():
    assert apply_url_domain("https://xn--pple-43d.com/jobs") == "xn--pple-43d.com"


def test_rankings_api_returns_apply_domain():
    import api
    row = api._apply_link("https://jobs.example.com/apply/1")
    assert row == {"apply_url": "https://jobs.example.com/apply/1", "apply_domain": "example.com"}
    assert api._apply_link("https://apply.example.com@evil.com/") == {"apply_url": None, "apply_domain": None}


# ----------------------------------------------------------------- Postgres --

@pytest.mark.db
def test_rankings_store_and_return_only_safe_links_with_domain():
    import uuid
    import agent_store
    import api
    from auth import create_user
    from database import get_connection
    uid = create_user("url_" + uuid.uuid4().hex[:10], "password-1234")
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("INSERT INTO runs (status, input_summary, user_id, mode) "
                    "VALUES ('running', 't', %s, 'agent') RETURNING id, execution_generation",
                    (uid,))
        rid, gen = cur.fetchone()
    agent_store.persist_rankings(rid, gen, [
        {"job_id": 1, "title": "Data Engineer", "company": "A", "score": 90,
         "final_decision": "Apply", "apply_url": "https://apply.example.com@evil.com/"},
        {"job_id": 2, "title": "ML Engineer", "company": "B", "score": 80,
         "final_decision": "Apply", "apply_url": "https://jobs.example.com/apply/2"}])
    with get_connection() as c:
        cur = c.cursor()
        cur.execute("SELECT apply_url FROM run_rankings WHERE run_id = %s ORDER BY rank_position",
                    (rid,))
        assert [r[0] for r in cur.fetchall()] == [None, "https://jobs.example.com/apply/2"]
    out = api.get_run_rankings(rid, user={"id": uid})["rankings"]
    assert (out[0]["apply_url"], out[0]["apply_domain"]) == (None, None)
    assert (out[1]["apply_url"], out[1]["apply_domain"]) == ("https://jobs.example.com/apply/2",
                                                             "example.com")
