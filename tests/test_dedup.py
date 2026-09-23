"""
Deduplication tests for job_source._dedupe_jobs.

Identity is URL-first (canonical apply/source URL), with the title+company+location
content fingerprint used only as a conservative fallback for URL-less postings — so
two genuinely distinct requisitions that share a title/company/location are never
silently merged. Pure functions, no DB needed.
"""
from job_source import _dedupe_jobs, _canonical_url, _dedupe_key


def test_distinct_requisitions_with_different_urls_are_kept():
    # Same title/company/location, DIFFERENT apply URLs = two real jobs, not one.
    out = _dedupe_jobs([
        {"title": "AI Engineer", "company": "Google", "location": "NYC",
         "apply_url": "https://g.co/jobs/ads-123", "source": "adzuna"},
        {"title": "AI Engineer", "company": "Google", "location": "NYC",
         "apply_url": "https://g.co/jobs/search-456", "source": "adzuna"},
    ])
    assert len(out) == 2


def test_same_canonical_url_is_merged_across_providers():
    # Same underlying URL (differing only in tracking params / fragment) = one job.
    out = _dedupe_jobs([
        {"title": "AI Engineer", "company": "Wipro", "location": "TX",
         "apply_url": "https://ats.co/job/9?utm_source=adzuna", "source": "adzuna"},
        {"title": "AI Engineer", "company": "Wipro", "location": "TX",
         "apply_url": "https://ats.co/job/9?utm_source=remotive#apply", "source": "remotive"},
    ])
    assert len(out) == 1


def test_urlless_postings_fall_back_to_content_fingerprint():
    out = _dedupe_jobs([
        {"title": "Data Scientist", "company": "Acme", "location": "Remote", "source": "seed"},
        {"title": "Data Scientist", "company": "Acme", "location": "Remote", "source": "csv"},
    ])
    assert len(out) == 1


def test_url_and_urlless_same_content_merge_and_salvage():
    # One row has a URL, the other (same content) doesn't = the SAME job seen twice.
    # Only one distinct URL is present, so they merge and the URL is salvaged.
    out = _dedupe_jobs([
        {"title": "ML Eng", "company": "Beta", "location": "SF", "source": "adzuna"},
        {"title": "ML Eng", "company": "Beta", "location": "SF",
         "apply_url": "https://x.co/1", "source": "seed"},
    ])
    assert len(out) == 1
    assert out[0]["source"] == "adzuna"                 # higher rank kept
    assert out[0]["apply_url"] == "https://x.co/1"       # salvaged from the loser


def test_distinct_urls_in_same_content_group_stay_separate():
    # The reviewer's core case: same title/company/location, TWO real requisitions
    # (two apply URLs) -> both survive, neither is silently discarded.
    out = _dedupe_jobs([
        {"title": "AI Engineer", "company": "Google", "location": "NYC",
         "apply_url": "https://g.co/ads", "source": "adzuna"},
        {"title": "AI Engineer", "company": "Google", "location": "NYC",
         "apply_url": "https://g.co/search", "source": "adzuna"},
    ])
    assert len(out) == 2


def test_canonical_url_normalizes_tracking_and_case():
    a = _canonical_url({"apply_url": "HTTPS://ATS.co/Job/9?utm=x#frag"})
    b = _canonical_url({"apply_url": "https://ats.co/Job/9/"})
    assert a == b == "https://ats.co/job/9"


def test_higher_rank_source_wins_and_fields_merge():
    # Adzuna outranks seed; the kept row backfills a missing apply_url from the loser.
    out = _dedupe_jobs([
        {"title": "SRE", "company": "Acme", "location": "Remote", "source": "seed",
         "apply_url": "https://acme.co/sre"},
        {"title": "SRE", "company": "Acme", "location": "Remote", "source": "adzuna",
         "apply_url": "https://acme.co/sre"},   # same URL -> same job
    ])
    assert len(out) == 1
    assert out[0]["source"] == "adzuna"          # higher rank kept


def test_dedupe_key_prefers_url_over_content():
    with_url = _dedupe_key({"title": "x", "company": "y", "location": "z",
                            "apply_url": "https://a.co/1"})
    without = _dedupe_key({"title": "x", "company": "y", "location": "z"})
    assert with_url[0] == "url"
    assert without[0] == "content"