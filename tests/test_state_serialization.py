"""
Round-trip test for AgentState serialization — the foundation for LangGraph
checkpointing. from_dict(to_dict(s)) must reproduce s EXACTLY, or state would
silently drop across a checkpoint pause/resume.
"""
from agent_state import AgentState


def _fully_populated():
    s = AgentState(goal="match", resume_id=7, target_role="AI engineer",
                   location="USA", work_mode="remote", employment_type="full-time",
                   evaluate=True)
    # simulate mid-run state
    s.resume_text = "resume body text"
    s.parsed_resume = {"skills": ["python", "flask"], "years_experience": 3,
                       "projects": [{"name": "app", "tech": ["python"]}]}
    s.jobs = [{"id": 1, "title": "A", "description": "..."},
              {"id": 2, "title": "B", "description": "..."}]
    s.current_job_index = 1
    s.job_results = [{"title": "A", "score": 74.0, "decision": "Maybe",
                      "llm_decision": "Apply", "needs_review": True}]
    s.ranked = [{"title": "A", "score": 74.0}]
    s.ranking_done = True
    s.advice_done = False
    s.cancelled = False
    s.failed_jobs = 2
    s.requirements_cache = {"abc": {"required_skills": ["python"]}}
    s.llm_calls_made = 5
    s.max_llm_calls = 30
    s.completed_actions = ["load_resume", "parse_resume", "search_jobs", "process_job"]
    s.done = False
    s.error = None
    return s


def test_round_trip_is_lossless():
    s = _fully_populated()
    restored = AgentState.from_dict(s.to_dict())
    assert restored == s, "round-trip must reproduce state exactly"


def test_round_trip_preserves_every_field():
    s = _fully_populated()
    d = s.to_dict()
    restored = AgentState.from_dict(d)
    for f in AgentState._FIELDS:
        assert getattr(restored, f) == getattr(s, f), f"field '{f}' did not survive round-trip"


def test_to_dict_is_json_serializable():
    import json
    s = _fully_populated()
    # must serialize to JSON without error (checkpointer requires this)
    dumped = json.dumps(s.to_dict())
    reloaded = json.loads(dumped)
    restored = AgentState.from_dict(reloaded)
    assert restored == s, "state must survive a JSON round-trip (checkpointer uses serialization)"


def test_fresh_state_round_trips():
    s = AgentState(goal="m", resume_id=1)
    assert AgentState.from_dict(s.to_dict()) == s


def test_fields_list_matches_init_attributes():
    # guard: if someone adds an __init__ field but forgets _FIELDS, catch it.
    s = AgentState(goal="m", resume_id=1)
    init_attrs = {k for k in vars(s).keys()}
    # requirements_cache and completed_actions etc. are all in _FIELDS;
    # every serializable instance attribute should be listed.
    missing = init_attrs - set(AgentState._FIELDS)
    assert not missing, f"instance attributes missing from _FIELDS (would drop on checkpoint): {missing}"