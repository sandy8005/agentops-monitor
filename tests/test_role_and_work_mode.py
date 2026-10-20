"""
Local job validation does not trust providers to obey the query.

Role: a query with both a specialization ("AI") and a role family ("Engineer")
needs evidence for BOTH — the specialization in the title or description, the
family in the TITLE. Work mode: a fixed constraint matched strictly; hybrid is
accepted for a remote/onsite search only with the explicit allow_hybrid opt-in.
"""
import pytest

from agent_goal import AgentGoal
from job_source import _role_matcher, work_mode_ok


def _job(title, description=""):
    return {"title": title, "description": description}


# ------------------------------------------------------------- role family ----

@pytest.mark.parametrize("title,description", [
    ("AI Product Manager", "Own the roadmap for our AI platform."),
    ("Technical Recruiter — AI Engineering", "Hire AI engineers for our team."),
    ("Sales Executive for AI Solutions", "Sell AI products to enterprises."),
    ("Product Manager", "You will collaborate with AI engineers daily."),
    ("Engineering Manager", "Lead a team building AI systems."),
])
def test_ai_engineer_rejects_unrelated_job_families(title, description):
    assert _role_matcher("AI Engineer")(_job(title, description)) is False


@pytest.mark.parametrize("title,description", [
    ("AI Engineer", ""),
    ("Senior AI Engineer", "LLM applications."),
    ("Software Engineer, AI Platform", ""),
    ("Machine Learning Engineer", "Build artificial intelligence features."),
    ("AI Developer", "Python, LLMs."),                 # engineer == developer family
    ("AI Engineers (multiple openings)", ""),           # plural
    ("Backend Engineer", "Our product is an AI assistant."),  # spec in description
])
def test_ai_engineer_accepts_real_engineering_roles(title, description):
    assert _role_matcher("AI Engineer")(_job(title, description)) is True


def test_security_analyst_requires_analyst_in_title():
    m = _role_matcher("Security Analyst")
    assert m(_job("SOC Security Analyst")) is True
    assert m(_job("Security Engineer", "work with our analysts")) is False
    assert m(_job("Account Manager", "sell security products to analysts")) is False


def test_specialization_still_required():
    m = _role_matcher("ML Engineer")
    assert m(_job("Civil Engineer", "bridges and roads")) is False


def test_specialization_only_query_keeps_single_dimension_rule():
    m = _role_matcher("Machine Learning")
    assert m(_job("Research Scientist", "machine learning models")) is True


def test_generic_only_query_unchanged():
    m = _role_matcher("Engineer")
    assert m(_job("Civil Engineer")) is True


def test_seniority_word_is_not_a_family_requirement():
    m = _role_matcher("Senior AI Engineer")
    assert m(_job("AI Engineer", "")) is True       # 'senior' is not a job family


# --------------------------------------------------------------- work mode ----

@pytest.mark.parametrize("requested,job_mode,expected", [
    ("remote", "remote", True),
    ("remote", "hybrid", False),
    ("remote", "onsite", False),
    ("onsite", "hybrid", False),
    ("hybrid", "hybrid", True),
    ("hybrid", "remote", False),
    ("hybrid", "onsite", False),
])
def test_work_mode_is_strict_by_default(requested, job_mode, expected):
    assert work_mode_ok({"work_mode": job_mode}, requested) is expected


@pytest.mark.parametrize("requested,job_mode,expected", [
    ("remote", "hybrid", True),
    ("onsite", "hybrid", True),
    ("remote", "onsite", False),
    ("hybrid", "remote", False),       # never widens a hybrid search
])
def test_allow_hybrid_is_an_explicit_one_way_relaxation(requested, job_mode, expected):
    assert work_mode_ok({"work_mode": job_mode}, requested, allow_hybrid=True) is expected


def test_unknown_job_mode_is_kept_and_remote_location_counts_as_remote():
    assert work_mode_ok({"work_mode": ""}, "onsite") is True
    assert work_mode_ok({"work_mode": "", "location": "Remote (US)"}, "onsite") is False


def test_goal_defaults_to_strict_and_old_goals_still_load():
    g = AgentGoal(target_role="AI Engineer", constraints={"work_mode": "remote"})
    assert g.constraints.allow_hybrid is False
    # a goal stored before allow_hybrid existed
    old = AgentGoal(**{"target_role": "AI Engineer", "constraints": {"work_mode": "onsite"}})
    assert old.constraints.allow_hybrid is False

# ------------------------------------------- scientist / researcher family ----
# Backlog P2: "ML Scientist" matched unrelated scientist jobs (and missed real ones).

@pytest.mark.parametrize("title,description,expected", [
    ("Machine Learning Scientist", "PyTorch research.", True),
    ("Research Scientist, ML", "Train large models.", True),
    ("ML Researcher", "Publish papers.", True),                 # researcher == scientist
    ("Soil Scientist", "Field sampling of soils.", False),      # no specialization
    ("Marketing Manager", "Partner with our ML scientists.", False),  # no family in title
    ("ML Engineer", "Ship ML services.", False),                # different family
])
def test_ml_scientist_needs_specialization_and_scientist_family(title, description, expected):
    assert _role_matcher("ML Scientist")(_job(title, description)) is expected


def test_researcher_query_accepts_scientist_titles():
    m = _role_matcher("AI Researcher")
    assert m(_job("AI Research Scientist", "")) is True
    assert m(_job("Clinical Researcher", "wet-lab assays")) is False


def test_bare_scientist_query_keeps_single_dimension_rule():
    assert _role_matcher("Scientist")(_job("Soil Scientist")) is True
