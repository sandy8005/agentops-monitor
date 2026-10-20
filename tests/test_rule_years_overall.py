"""Backlog P3: the rules fallback prefers an OVERALL/total/professional years
mention over a smaller skill-specific one. Synthetic posting text only."""
import pytest

from rule_requirements import _years_from_text


@pytest.mark.parametrize("text,years", [
    ("3 years Python, 5 years overall.", 5.0),
    ("3 years Python, 5+ years of professional experience.", 5.0),
    ("3 years Kubernetes; total of 6 years experience.", 6.0),
    ("3 years Python and 5 years of industry experience.", 5.0),   # cue stays with the 5
    ("Total 6 years experience and 3 years Python.", 6.0),
    ("3 years Python, 5 years industry experience, 4 years total.", 4.0),  # min of overall
])
def test_overall_mention_wins(text, years):
    assert _years_from_text(text.lower()) == years


@pytest.mark.parametrize("text,years", [
    ("3 years Python. 5 years experience in ML.", 3.0),         # no overall cue: smallest
    ("2-4 years experience required.", 2.0),
    ("5 years overall experience preferred. 2 years Python.", 2.0),  # preferred skipped
    ("Founded 12 years ago. No experience requirement.", 0.0),
])
def test_without_overall_cue_the_current_rule_holds(text, years):
    assert _years_from_text(text.lower()) == years
