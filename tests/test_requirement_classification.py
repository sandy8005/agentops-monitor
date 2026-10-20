"""Backlog B: requirements are grounded by MEANING, not only by mention.

Each extracted skill is classified from the posting's own wording around its
mentions (required / preferred / negated / context-only). A negated requirement,
or one whose label the posting contradicts, becomes UNCERTAIN: recorded, flagged
for review, and kept out of the deterministic score. Synthetic job text only."""
import json

import job_parser
from job_parser import ground_requirements
from rule_requirements import extract_requirements_rule_based
from scorer import calculate_match_score


def _reqs(**kw):
    base = {"required_skills": [], "required_any_of": [], "preferred_skills": [],
            "min_years_experience": 0, "responsibilities": []}
    base.update(kw)
    return base


def _job(desc, title="Data Engineer"):
    return {"title": title, "description": desc}


def _uncertain(out):
    return {(u["field"], u["skill"], u["reason"]) for u in out["uncertain_requirements"]}


def test_negated_requirement_counterexample():
    out = ground_requirements(_reqs(required_skills=["Python", "Scala"]),
                              _job("Python is not required. Scala is mandatory."))
    assert out["required_skills"] == ["Scala"]
    assert _uncertain(out) == {("required_skills", "Python", "negated")}


def test_nice_to_have_labeled_mandatory_counterexample():
    out = ground_requirements(_reqs(required_skills=["Python", "SQL"]),
                              _job("SQL is required. Python is nice to have."))
    assert out["required_skills"] == ["SQL"]
    assert _uncertain(out) == {("required_skills", "Python", "classified_preferred")}


def test_correct_labels_from_section_headings_are_kept():
    desc = "Requirements:\n- Python\n- Docker\nNice to have:\n- Go\n"
    out = ground_requirements(_reqs(required_skills=["Python", "Docker"],
                                    preferred_skills=["Go"]), _job(desc))
    assert out["required_skills"] == ["Python", "Docker"]
    assert out["preferred_skills"] == ["Go"]
    assert out["uncertain_requirements"] == []
    # the same posting with Go mislabeled as required
    out = ground_requirements(_reqs(required_skills=["Python", "Go"]), _job(desc))
    assert out["required_skills"] == ["Python"]
    assert _uncertain(out) == {("required_skills", "Go", "classified_preferred")}


def test_clause_level_signals():
    job = _job("Python is required, Go is a plus.")
    out = ground_requirements(_reqs(required_skills=["Python"], preferred_skills=["Go"]), job)
    assert out["required_skills"] == ["Python"] and out["preferred_skills"] == ["Go"]
    out = ground_requirements(_reqs(required_skills=["Go"], preferred_skills=["Python"]), job)
    assert out["required_skills"] == [] and out["preferred_skills"] == []
    assert _uncertain(out) == {("required_skills", "Go", "classified_preferred"),
                               ("preferred_skills", "Python", "classified_required")}


def test_experience_with_x_is_a_plus_is_preferred():
    out = ground_requirements(_reqs(preferred_skills=["Kafka"]),
                              _job("Experience with Kafka is a plus."))
    assert out["preferred_skills"] == ["Kafka"] and out["uncertain_requirements"] == []


def test_context_only_mention_keeps_the_label_and_records_it():
    out = ground_requirements(_reqs(required_skills=["Python"]),
                              _job("Our platform team uses Python daily."))
    assert out["required_skills"] == ["Python"]
    assert out["uncertain_requirements"] == []
    assert {"field": "required_skills", "skill": "Python",
            "classification": "context_only"} in out["requirement_classification"]


def test_negation_before_the_skill():
    out = ground_requirements(_reqs(required_skills=["Kubernetes", "Python"]),
                              _job("Strong Python skills needed. No prior Kubernetes "
                                   "experience required; we will teach you."))
    assert out["required_skills"] == ["Python"]
    assert _uncertain(out) == {("required_skills", "Kubernetes", "negated")}


def test_any_of_group_marked_optional_becomes_uncertain():
    out = ground_requirements(_reqs(required_any_of=[["Flask", "Django"]]),
                              _job("Flask or Django is a plus."))
    assert out["required_any_of"] == []
    assert _uncertain(out) == {("required_any_of", "Flask", "classified_preferred"),
                               ("required_any_of", "Django", "classified_preferred")}


def test_uncertain_requirement_does_not_feed_the_score(monkeypatch):
    job = _job("Python is not required. Scala is mandatory.")
    model = _reqs(required_skills=["Python", "Scala"])
    monkeypatch.setattr(job_parser, "logged_llm_call", lambda *a, **k: json.dumps(model))
    grounded = job_parser.extract_requirements(job, run_id=1, step_id=1)
    resume = "Skills: Scala"
    parsed = {"skills": ["Scala"], "years_experience": 3, "education": [], "projects": [],
              "experience": []}
    assert calculate_match_score(parsed, model, resume)["score"] < 75
    assert calculate_match_score(parsed, grounded, resume)["score"] == 100.0
    assert _uncertain(grounded) == {("required_skills", "Python", "negated")}


def test_rules_fallback_drops_negated_requirement():
    out = extract_requirements_rule_based(_job("Python is not required. Scala is mandatory."))
    assert "python" not in out["required_skills"]
    assert "scala" in out["required_skills"]
    assert {(u["skill"], u["reason"]) for u in out["uncertain_requirements"]} == {
        ("python", "negated")}


def test_incidental_negation_words_do_not_negate_a_requirement():
    job = _job("Run zero-downtime deploys with Kubernetes (required). "
               "Not required: Java.")
    out = ground_requirements(_reqs(required_skills=["Kubernetes", "Java"]), job)
    assert out["required_skills"] == ["Kubernetes"]
    assert _uncertain(out) == {("required_skills", "Java", "negated")}
