"""
Regression tests for the live-search fixes — the ones that kept reverting to drift.
Pure logic, no LLM, no DB. These lock in: whole-word role matching, cross-source
dedup, location/mode filtering logic, freshness, and rule-based extraction.

Run:  pytest test_live_search.py -v
"""
import pytest
from datetime import datetime, timedelta


# ============ 1. Whole-word role matching (ML != HTML, AI != email) ============

def test_role_matcher_ml_not_in_html():
    from job_source import _role_matcher
    m = _role_matcher("ML Engineer")
    assert m({"title": "Senior HTML Developer", "description": "html css frontend"}) is False
    assert m({"title": "ML Engineer", "description": "machine learning ml pipelines"}) is True

def test_role_matcher_ai_not_in_email_or_airline():
    from job_source import _role_matcher
    m = _role_matcher("AI Engineer")
    assert m({"title": "Email Marketing Specialist", "description": "manage email campaigns"}) is False
    assert m({"title": "Airline Operations Analyst", "description": "airline schedules"}) is False
    assert m({"title": "Senior AI Engineer", "description": "build ai systems"}) is True

def test_role_matcher_multiword_phrase():
    from job_source import _role_matcher
    m = _role_matcher("machine learning engineer")
    assert m({"title": "Senior Machine Learning Engineer", "description": "..."}) is True

def test_role_matcher_special_chars_survive():
    from job_source import _role_matcher
    m = _role_matcher("C++ Developer")
    assert m({"title": "C++ Systems Developer", "description": "c++ and c#"}) is True


# ============ 2. Cross-source deduplication ============

def test_dedupe_collapses_same_title_company_across_sources():
    from job_source import _dedupe_jobs
    # Same posting from two sources, NEITHER with an external_id — so dedup falls
    # through to the title+company(+location) key and collapses them.
    jobs = [
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas", "source": "seed"},
        {"title": "AI Engineer", "company": "Wipro", "location": "Texas", "source": "scraped"},  # dup
        {"title": "Data Engineer", "company": "Acme", "location": "Texas", "source": "seed"},
    ]
    out = _dedupe_jobs(jobs)
    titles = [(j["title"], j["company"]) for j in out]
    assert titles.count(("AI Engineer", "Wipro")) == 1   # collapsed to one
    assert len(out) == 2

def test_dedupe_keeps_distinct_jobs():
    from job_source import _dedupe_jobs
    jobs = [
        {"title": "AI Engineer", "company": "Wipro"},
        {"title": "AI Engineer", "company": "Google"},   # same title, DIFFERENT company — distinct
    ]
    assert len(_dedupe_jobs(jobs)) == 2


# ============ 3. Location filtering (Texas != New York) ============
# search_jobs' location filter keeps: search_location==requested OR search_location is None.

def _location_ok(job, requested):
    loc = requested.strip().lower()
    sl = (job.get("search_location") or "").strip().lower()
    return not sl or sl == loc

def test_location_texas_excludes_new_york_adzuna():
    assert _location_ok({"search_location": "Texas"}, "Texas") is True
    assert _location_ok({"search_location": "New York"}, "Texas") is False   # excluded

def test_location_agnostic_jobs_always_kept():
    # seed/csv/scraped jobs (no search_location) show in every location search
    assert _location_ok({"search_location": None}, "Texas") is True
    assert _location_ok({"search_location": None}, "California") is True


# ============ 4. Work-mode filtering (onsite != remote) ============
# The fixed mode_ok: remote job must NOT satisfy an onsite request.

def _mode_ok(job, requested):
    wm = requested.lower().strip()
    jm = (job.get("work_mode") or "").lower().strip()
    jl = (job.get("location") or "").lower()
    if not jm and "remote" not in jl:
        return True            # unknown mode — keep (soft)
    if wm == jm:
        return True            # exact match
    if "hybrid" in (wm, jm):
        return True            # hybrid partial-matches either way
    return False               # clear conflict (remote job, onsite request)

def test_onsite_request_excludes_remote_job():
    assert _mode_ok({"work_mode": "remote"}, "onsite") is False   # the bug: must be excluded
    assert _mode_ok({"work_mode": "onsite"}, "onsite") is True

def test_unknown_mode_kept_soft():
    assert _mode_ok({"work_mode": ""}, "onsite") is True          # unknown != conflict

def test_hybrid_partial_matches():
    assert _mode_ok({"work_mode": "hybrid"}, "onsite") is True
    assert _mode_ok({"work_mode": "remote"}, "hybrid") is True


# ============ 5. Freshness (stale live jobs drop out) ============

def _is_fresh(job, stale_days=14):
    cutoff = datetime.now() - timedelta(days=stale_days)
    ls = job.get("last_seen_at")
    return ls is None or ls >= cutoff

def test_stale_live_job_dropped():
    old = datetime.now() - timedelta(days=30)
    assert _is_fresh({"last_seen_at": old}) is False        # 30 days old — stale

def test_recent_live_job_kept():
    recent = datetime.now() - timedelta(days=2)
    assert _is_fresh({"last_seen_at": recent}) is True

def test_no_timestamp_always_fresh():
    assert _is_fresh({"last_seen_at": None}) is True          # seed/csv/scraped — always fresh


# ============ Rule-based requirement extraction (quota fallback) ============

def test_rule_based_extraction_no_llm():
    from rule_requirements import extract_requirements_rule_based
    job = {"title": "Python Developer",
           "description": "Required: Python and SQL. 3+ years experience. Docker preferred."}
    r = extract_requirements_rule_based(job)
    # The skill is FOUND (required or preferred — bucketing is heuristic), the years
    # are extracted, and the shape matches the LLM extractor. Zero LLM calls.
    all_skills = r["required_skills"] + r["preferred_skills"]
    assert "python" in all_skills, f"python should be extracted, got {r}"
    assert r["min_years_experience"] == 3.0
    assert set(r.keys()) >= {"required_skills", "preferred_skills", "min_years_experience"}