"""Round 6, backlog A: a conditional "for the senior/lead/manager track" clause
skips only the "N years" mention it belongs to, not every mention in its
sentence. "3 years Python required, 5 years for the manager track." used to give
0 (the whole sentence was skipped), and job_parser._ground_min_years zeroed a
correct model 3 with it. Synthetic posting text only."""
import pytest

from job_parser import _ground_min_years
from rule_requirements import _years_from_text

COUNTEREXAMPLE = "3 years Python required, 5 years for the manager track."

CASES = [
    # --- the reproduced counterexample and same-sentence mixes ---
    (COUNTEREXAMPLE, 3.0),
    ("3 years Python required; 5 years for the manager track.", 3.0),
    ("3 years Python required and 5 years for the senior role.", 3.0),
    ("3 years Python required while 5 years for the lead level.", 3.0),
    ("3 years Python required, but 5 years for the senior track.", 3.0),
    ("3+ years of Python required, and 6+ years for the principal level.", 3.0),
    ("4+ years of experience required, for the senior level 6+ years.", 4.0),
    ("3 years of experience with Python, Django, for the manager track 5 years.", 3.0),
    # --- conditional first ---
    ("5 years for the manager track, 3 years Python required.", 3.0),
    ("5 years for the senior role and 3 years of Python required.", 3.0),
    ("For the lead track 7 years, while 4 years of Python is required.", 4.0),
    ("7 years for the staff level but 2 years of SQL is mandatory.", 2.0),
    # --- as two sentences (unchanged) ---
    ("3 years Python required. 5 years for the manager track.", 3.0),
    # --- controls: the conditional still removes its own mention ---
    ("5 years required for the manager track.", 0.0),
    ("Must have 7 years experience for the manager track.", 0.0),
    ("For the senior track, 5+ years of experience.", 0.0),
    ("5+ years of experience in Python and Django, for the lead role.", 0.0),
    ("If applying for the senior level, 6 years required.", 0.0),
    ("5 years for the manager track and 7 years for the senior role.", 0.0),
    # ambiguous: a mention-less conditional clause between two mentions applies to
    # both (conservative, the old sentence-level behavior)
    ("3 years of Python, for the senior role, 5 years of Go.", 0.0),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_conditional_is_scoped_to_its_clause(text, expected):
    assert _years_from_text(text.lower()) == expected


@pytest.mark.parametrize("text", [c for c, v in CASES if v == 3.0])
def test_model_correct_minimum_is_grounded(text):
    out = _ground_min_years(3, {"title": "Backend Engineer", "description": text})
    assert out == {"min_years_experience": 3.0, "ungrounded_min_years": None}


def test_conditional_figure_is_not_grounded():
    out = _ground_min_years(5, {"title": "Backend Engineer", "description": COUNTEREXAMPLE})
    assert out == {"min_years_experience": 0.0,
                   "ungrounded_min_years": {"stated": 5.0, "reason": "not_in_posting"}}


def test_reqs_cache_version_bumped():
    import cache_version
    assert int(cache_version.REQS_VERSION) >= 9
