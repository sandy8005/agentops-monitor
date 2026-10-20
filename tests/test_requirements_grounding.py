"""Backlog P1: model-extracted JD requirements are grounded against the posting.
Synthetic job text only."""
import json

import job_parser
from job_parser import ground_requirements
from rule_requirements import extract_requirements_rule_based
from scorer import calculate_match_score

JOB = {"title": "Backend Engineer",
       "description": "We need Python and K8s experience. Flask or FastAPI. "
                      "Nice to have: PostgreSQL."}


def _reqs(**kw):
    base = {"required_skills": [], "required_any_of": [], "preferred_skills": [],
            "min_years_experience": 0, "responsibilities": []}
    base.update(kw)
    return base


def test_invented_required_skill_is_dropped_and_recorded():
    out = ground_requirements(_reqs(required_skills=["Python", "Kubernetes", "Scala"]), JOB)
    assert out["required_skills"] == ["Python", "Kubernetes"]      # K8s alias grounds it
    assert out["ungrounded_requirements"] == [{"field": "required_skills", "skill": "Scala"}]


def test_whole_word_matching_go_not_in_django():
    job = {"title": "Dev", "description": "Django services."}
    out = ground_requirements(_reqs(required_skills=["Go", "Django"]), job)
    assert out["required_skills"] == ["Django"]


def test_any_of_members_and_preferred_are_grounded():
    out = ground_requirements(_reqs(required_any_of=[["Flask", "Django"], ["Rust", "Elixir"]],
                                    preferred_skills=["PostgreSQL", "Redis"]), JOB)
    assert out["required_any_of"] == [["Flask"]]                   # empty group dropped
    assert out["preferred_skills"] == ["PostgreSQL"]
    fields = {(d["field"], d["skill"]) for d in out["ungrounded_requirements"]}
    assert fields == {("required_any_of", "Django"), ("required_any_of", "Rust"),
                      ("required_any_of", "Elixir"), ("preferred_skills", "Redis")}


def test_title_counts_as_posting_text():
    out = ground_requirements(_reqs(required_skills=["Python"]),
                              {"title": "Python Developer", "description": "Build APIs."})
    assert out["required_skills"] == ["Python"] and out["ungrounded_requirements"] == []


def test_extract_requirements_grounds_model_output(monkeypatch):
    model = _reqs(required_skills=["Python", "Scala"], min_years_experience=2)
    monkeypatch.setattr(job_parser, "logged_llm_call", lambda *a, **k: json.dumps(model))
    out = job_parser.extract_requirements(JOB, run_id=1, step_id=1)
    assert out["required_skills"] == ["Python"]
    assert out["min_years_experience"] == 2.0
    assert out["ungrounded_requirements"] == [{"field": "required_skills", "skill": "Scala"}]


def test_invented_requirement_no_longer_causes_false_skip(monkeypatch):
    resume = "Skills: Python, Kubernetes, Flask."
    parsed = {"skills": ["Python", "Kubernetes", "Flask"], "years_experience": 3,
              "education": [], "projects": [], "experience": [{"title": "Dev"}]}
    model = _reqs(required_skills=["Python", "Kubernetes", "Scala", "Haskell", "Erlang"])
    monkeypatch.setattr(job_parser, "logged_llm_call", lambda *a, **k: json.dumps(model))
    grounded = job_parser.extract_requirements(JOB, run_id=1, step_id=1)
    assert calculate_match_score(parsed, model, resume)["decision"] == "Skip"
    assert calculate_match_score(parsed, grounded, resume)["decision"] == "Apply"


def test_rules_fallback_has_same_shape():
    out = extract_requirements_rule_based(JOB)
    assert out["ungrounded_requirements"] == []
