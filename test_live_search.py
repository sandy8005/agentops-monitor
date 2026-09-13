# ============ 2. Cross-source deduplication ============

def test_dedupe_collapses_same_title_company_across_sources():
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas", "source": "seed"},
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas", "source": "scraped"},  # dup
        {"title": "Data Engineer", "company": "Acme", "location": "Texas", "source": "seed"},
    ]
    out = _dedupe_jobs(jobs)
    titles = [(j["title"], j["company"]) for j in out]
    assert titles.count(("AI Engineer", "Wipro")) == 1   # collapsed to one
    assert len(out) == 2

def test_dedupe_collapses_across_providers_despite_different_external_ids():
    # THE redesign case: same posting on Adzuna and Remotive, each with its OWN
    # provider-namespaced external_id. The old key kept them apart; the content
    # fingerprint collapses them. Higher source-rank (adzuna) wins.
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas",
         "source": "live", "external_id": "remotive:456"},
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas",
         "source": "adzuna", "external_id": "adzuna:123"},
    ]
    out = _dedupe_jobs(jobs)
    assert len(out) == 1
    assert out[0]["source"] == "adzuna"          # higher rank wins

def test_dedupe_normalizes_seniority_and_company_suffix():
    # "Senior AI Engineer @ Wipro Inc." and "AI Engineer @ Wipro" are the same job.
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "Senior AI Engineer", "company": "Wipro Inc.", "location": "Texas", "source": "live"},
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas", "source": "adzuna"},
    ]
    assert len(_dedupe_jobs(jobs)) == 1

def test_dedupe_merges_fields_from_loser():
    # Winner (adzuna) lacks an apply_url; loser (live) has one — it's salvaged.
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas",
         "source": "adzuna", "apply_url": None},
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas",
         "source": "live", "apply_url": "https://apply.example/xyz"},
    ]
    out = _dedupe_jobs(jobs)
    assert len(out) == 1
    assert out[0]["source"] == "adzuna"
    assert out[0]["apply_url"] == "https://apply.example/xyz"   # merged from loser

def test_dedupe_keeps_distinct_jobs():
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "AI Engineer", "company": "Wipro"},
        {"title": "AI Engineer", "company": "Google"},   # same title, DIFFERENT company — distinct
    ]
    assert len(_dedupe_jobs(jobs)) == 2

def test_dedupe_keeps_different_locations_distinct():
    # Location stays in the fingerprint: same role/company, two cities = two jobs.
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas", "source": "adzuna"},
        {"title": "AI Engineer", "company": "Wipro", "location": "California", "source": "adzuna"},
    ]
    assert len(_dedupe_jobs(jobs)) == 2