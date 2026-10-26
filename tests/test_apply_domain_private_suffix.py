"""Round 6, backlog D: the apply-link domain shows the shared-host TENANT.

apply_url_domain("https://company.github.io/jobs") used to be "github.io", so
every GitHub Pages / Heroku / Netlify site looked like the same domain. Hosts
under a suffix from the PRIVATE section of the Public Suffix List
(https://publicsuffix.org/list/public_suffix_list.dat, "===BEGIN PRIVATE
DOMAINS===") are registered one label below that suffix. The list is a curated,
vendored subset (job_source._PRIVATE_SUFFIXES): nothing is fetched at runtime or
in tests.

ATS platforms (jobs.lever.co, boards.greenhouse.io, *.myworkdayjobs.com) are not
on the PSL; their apply links show the PLATFORM domain (lever.co), which is the
real destination. The browser shows the API's apply_domain instead of
recomputing it, so server and browser can never disagree. Synthetic URLs only."""
import os
import re

import pytest

import job_source
from job_source import apply_url_domain

APP_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "static", "app.js")


@pytest.mark.parametrize("u,domain", [
    # the reproduced counterexamples
    ("https://company.github.io/jobs", "company.github.io"),
    ("https://acme.herokuapp.com/careers", "acme.herokuapp.com"),
    # other private-section shared hosts
    ("https://acme.netlify.app/", "acme.netlify.app"),
    ("https://acme-jobs.vercel.app/apply", "acme-jobs.vercel.app"),
    ("https://acme.pages.dev/", "acme.pages.dev"),
    ("https://acme.web.app/", "acme.web.app"),
    ("https://acme.firebaseapp.com/", "acme.firebaseapp.com"),
    ("https://acme.azurewebsites.net/jobs", "acme.azurewebsites.net"),
    ("https://d111111abcdef8.cloudfront.net/x", "d111111abcdef8.cloudfront.net"),
    ("https://acmejobs.blogspot.com/", "acmejobs.blogspot.com"),
    ("https://acme.gitlab.io/careers", "acme.gitlab.io"),
    ("https://www.company.github.io/jobs", "company.github.io"),   # deeper host
    ("https://Company.GitHub.IO/jobs", "company.github.io"),
    # the suffix itself has no tenant: shown as is
    ("https://github.io/", "github.io"),
    # ATS platforms show the platform domain (documented choice)
    ("https://jobs.lever.co/acme/123", "lever.co"),
    ("https://boards.greenhouse.io/acme/jobs/1", "greenhouse.io"),
    ("https://acme.wd5.myworkdayjobs.com/en-US/x", "myworkdayjobs.com"),
    # unchanged ICANN behavior
    ("https://jobs.example.com/apply/1", "example.com"),
    ("https://careers.acme.co.uk/j/2", "acme.co.uk"),
])
def test_registrable_domain(u, domain):
    assert apply_url_domain(u) == domain


@pytest.mark.parametrize("u,domain", [
    ("https://apple.com.evil.com/jobs", "evil.com"),
    ("https://company.github.io.evil.com/jobs", "evil.com"),
    ("https://github.io.evil.com/", "evil.com"),
    ("https://evilgithub.io/", "evilgithub.io"),           # not a label boundary
    ("https://acme.herokuapp.com.attacker.net/", "attacker.net"),
])
def test_anti_lookalike_keeps_the_real_destination(u, domain):
    assert apply_url_domain(u) == domain


def test_private_suffix_list_is_curated_and_documented():
    sfx = job_source._PRIVATE_SUFFIXES
    for must in ("github.io", "herokuapp.com", "netlify.app", "vercel.app", "pages.dev",
                 "web.app", "firebaseapp.com", "azurewebsites.net", "cloudfront.net",
                 "blogspot.com"):
        assert must in sfx
    assert all(s == s.lower() and "." in s and not s.startswith(".") for s in sfx)
    import inspect
    src = inspect.getsource(job_source)
    assert "publicsuffix.org" in src and "PRIVATE" in src


def test_browser_shows_the_api_domain_instead_of_recomputing():
    js = open(APP_JS, encoding="utf-8").read()
    assert "function linkDomain" not in js and "SECOND_LEVEL_LABELS" not in js
    assert re.search(r"escapeHtml\(r\.apply_domain\)", js)
