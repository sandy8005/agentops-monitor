"""Backlog C: mixed required/preferred "N years" mentions are judged per clause.

Synthetic posting text only.
"""
import pytest

from rule_requirements import _years_from_text, extract_requirements_rule_based


def _years(text):
    return _years_from_text(text.lower())


def test_counterexample_required_then_preferred_in_one_sentence():
    assert _years("3+ years Python experience required, while 5+ years Java "
                  "experience preferred.") == 3


@pytest.mark.parametrize("text,expected", [
    ("5+ years Java experience preferred, but 3+ years Python experience required.", 3),
    ("5+ years of Java preferred and 3+ years of Python required.", 3),
    ("Ideally 6 years in fintech; 2 years of SQL is mandatory.", 2),
])
def test_preferred_first_ordering(text, expected):
    assert _years(text) == expected


def test_required_stated_by_section_heading():
    text = ("About the role\nWe build data tools.\n"
            "Requirements:\n- 4+ years of Python experience\n"
            "Nice to have:\n- 7+ years with Kafka\n")
    assert _years(text) == 4


def test_required_heading_keeps_unmarked_clause_beside_a_preferred_one():
    text = "Requirements:\n- 3 years of Python, 5 years of Java preferred\n"
    assert _years(text) == 3


def test_preferred_section_heading_alone_is_not_a_minimum():
    text = "Nice to have:\n- 5+ years with Kafka\n"
    assert _years(text) == 0


@pytest.mark.parametrize("text", [
    "5 years of experience is preferred.",
    "Bonus points if you have experience with Kafka and 5 years in fintech.",
    "3 years of Python, 5 years of Java preferred.",     # unmarked clause: conservative
])
def test_preferred_sentences_still_excluded(text):
    assert _years(text) == 0


def test_not_required_years_are_not_a_minimum():
    assert _years("5 years of Go is not required.") == 0


def test_conditional_and_company_age_unchanged():
    assert _years("5 years required for the manager track.") == 0
    assert _years("We have been in business for 20 years.") == 0


def test_rules_extractor_uses_the_clause_rule():
    job = {"title": "Backend Engineer",
           "description": "3+ years Python experience required, while 5+ years Java "
                          "experience preferred."}
    assert extract_requirements_rule_based(job)["min_years_experience"] == 3
