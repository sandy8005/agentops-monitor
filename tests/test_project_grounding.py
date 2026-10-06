"""
Project technology feeds the scorer (project relevance), so a technology the
model attributes to a project must be supported by THAT project's resume text —
the same deterministic anti-hallucination rule skills already follow. Education
entries the resume does not contain are not stored as facts either.
"""
import json

import parser as parsermod
from parser import ground_education, ground_projects
from scorer import calculate_match_score

RESUME = """Jane Doe — Data Engineer, 4 years of experience.
Skills: Python, SQL, AWS, Kubernetes

Projects
Analytics Project — reporting dashboard built with Python and Pandas.
  - deployed on a single VM (no Kubernetes)

Billing Service: REST API in Go, currently learning Rust for a rewrite.

Education
BSc Computer Science, University of Leeds, 2018
"""


def test_tech_only_mentioned_elsewhere_is_not_project_evidence():
    out = ground_projects([{"name": "Analytics Project",
                            "tech": ["Python", "AWS", "Kubernetes"], "evidence": ""}], RESUME)
    # AWS is a real resume skill, but nothing says this project used it; the
    # project's own text NEGATES Kubernetes.
    assert out["projects"] == [{"name": "Analytics Project", "tech": ["Python"]}]
    assert out["ungrounded_project_claims"] == [
        {"project": "Analytics Project", "tech": ["AWS", "Kubernetes"],
         "dropped_project": False}]


def test_aspirational_mention_in_project_text_grounds_nothing():
    out = ground_projects([{"name": "Billing Service", "tech": ["Go", "Rust"],
                            "evidence": ""}], RESUME)
    assert out["projects"][0]["tech"] == ["Go"]


def test_verified_evidence_quote_grounds_a_paraphrased_project_name():
    out = ground_projects([{"name": "Reporting Dashboard", "tech": ["Pandas"],
                            "evidence": "reporting dashboard built with Python and Pandas"}],
                          RESUME)
    assert out["projects"] == [{"name": "Reporting Dashboard", "tech": ["Pandas"]}]


def test_fabricated_evidence_quote_does_not_ground():
    out = ground_projects([{"name": "Data Lake", "tech": ["Spark"],
                            "evidence": "large-scale pipelines in Spark"}], RESUME)
    assert out["projects"] == []
    assert out["ungrounded_project_claims"][0]["dropped_project"] is True


def test_hallucinated_project_is_dropped_entirely():
    out = ground_projects([{"name": "Quantum Trading Bot", "tech": ["Python"],
                            "evidence": ""}], RESUME)
    assert out["projects"] == []
    assert out["ungrounded_project_claims"] == [
        {"project": "Quantum Trading Bot", "tech": ["Python"], "dropped_project": True}]


def test_education_grounding():
    out = ground_education([
        {"degree": "BSc Computer Science", "institution": "University of Leeds", "year": "2018"},
        {"degree": "PhD Physics", "institution": "MIT", "year": "2021"},
        {"degree": "BSc Computer Science", "institution": "University of Leeds", "year": "2017"},
    ], RESUME)
    assert out["education"][0] == {"degree": "BSc Computer Science",
                                   "institution": "University of Leeds", "year": "2018"}
    assert out["education"][1]["year"] == ""              # year not in the resume
    assert out["ungrounded_education"] == [
        {"degree": "PhD Physics", "institution": "MIT", "year": "2021"}]


def test_parse_resume_stores_only_grounded_project_tech(monkeypatch):
    model = {"skills": ["Python", "SQL", "AWS", "Kubernetes"], "years_experience": 4,
             "education": [{"degree": "PhD Physics", "institution": "MIT", "year": "2021"}],
             "projects": [{"name": "Analytics Project",
                           "tech": ["Python", "AWS", "Kubernetes"],
                           "evidence": "reporting dashboard built with Python and Pandas"}],
             "experience": [{"title": "Data Engineer", "company": "X", "years": 4}],
             "skill_evidence": []}
    monkeypatch.setattr(parsermod, "logged_llm_call", lambda *a, **k: json.dumps(model))
    parsed = parsermod.parse_resume(RESUME, run_id=1, step_id=1)
    assert parsed["projects"] == [{"name": "Analytics Project", "tech": ["Python"]}]
    assert parsed["education"] == []
    assert parsed["ungrounded_project_claims"][0]["tech"] == ["AWS", "Kubernetes"]
    assert parsed["ungrounded_education"][0]["institution"] == "MIT"


def test_hallucinated_project_tech_no_longer_earns_project_points():
    """End to end into the scorer: AWS required, the model claimed the project
    used AWS, the resume does not say so -> no project credit for AWS."""
    hallucinated = [{"name": "Analytics Project", "tech": ["Python", "AWS"], "evidence": ""}]
    grounded = ground_projects(hallucinated, RESUME)["projects"]
    reqs = {"required_skills": ["AWS"], "required_any_of": [], "preferred_skills": [],
            "min_years_experience": 0, "responsibilities": []}
    base = {"skills": ["AWS"], "years_experience": 4, "education": [], "experience": []}
    before = calculate_match_score({**base, "projects": hallucinated}, reqs, RESUME)
    after = calculate_match_score({**base, "projects": grounded}, reqs, RESUME)
    assert before["breakdown"]["projects"] > 0
    assert after["breakdown"]["projects"] == 0.0