"""Backlog B: rule-based years extraction must read common phrasings: spelled-out
numbers (with or without a parenthesized digit), "N or more years", and a range
followed by a soft "ideally/preferably M" (the minimum stays N). Controls
(preferred, negated, conditional, company age, "years ago") keep their results.
Synthetic posting text only."""
import pytest

from rule_requirements import _years_from_text, years_requirement_mentions
from job_parser import _ground_min_years

CASES = [
    # --- the three reproduced counterexamples ---
    ("At least three (3) years of experience.", 3.0),
    ("You have 3 or more years of experience", 3.0),
    ("We need someone with 3-5 years of experience, ideally 5.", 3.0),
    # --- spelled-out numbers ---
    ("Three years of experience required.", 3.0),
    ("Minimum of five (5) years of professional experience.", 5.0),
    ("At least two years of experience with Python.", 2.0),
    ("one year of experience required.", 1.0),
    ("Ten+ years of experience building systems.", 10.0),
    ("twelve (12) years of experience.", 12.0),
    ("3 (three) years of experience.", 3.0),
    ("two to four years of experience", 2.0),
    ("Requirements: three years of experience", 3.0),
    # --- "or more" ---
    ("4 or more years of software engineering experience.", 4.0),
    ("Three or more years of experience with AWS.", 3.0),
    # --- soft upper value after a range / open minimum ---
    ("3 to 5 years of experience, ideally 5 in a SaaS company.", 3.0),
    ("3-5 years of experience, ideally 5 years in fintech.", 3.0),
    ("Requires 2-4 years of experience, preferably 4.", 2.0),
    ("2-3 yrs experience, ideally 3+.", 2.0),
    ("We need 3-5 years of experience (ideally 5).", 3.0),
    # --- unchanged positives ---
    ("5+ years of experience required.", 5.0),
    ("At least 1 year of experience.", 1.0),
    ("We need someone with 3-5 years of experience.", 3.0),
    ("3+ years Python experience required, while 5+ years Java experience preferred.", 3.0),
    ("We need 3-5 years of experience; ideally 7 years.", 3.0),
    # --- controls: must stay 0 ---
    ("5 years of experience preferred.", 0.0),
    ("Five years of experience is a plus.", 0.0),
    ("Ideally 5 years of experience.", 0.0),
    ("Ideally five (5) years of experience.", 0.0),
    ("5 or more years preferred.", 0.0),
    ("3 years of experience is not required.", 0.0),
    ("Experience is not required.", 0.0),
    ("Must have 7 years experience for the manager track.", 0.0),
    ("Our company was founded 20 years ago.", 0.0),
    ("We have been in business for over 15 years.", 0.0),
    ("We have been in business for over fifteen years.", 0.0),
    ("2 years ago we launched our product.", 0.0),
    ("Two years ago we launched our product.", 0.0),
    ("This is a two-year contract.", 0.0),
    ("We grew 3x over the last three years.", 0.0),
    ("Someone with years of experience in our domain.", 0.0),
    ("We are one of the top companies, everyone has years of experience.", 0.0),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_years_phrasings(text, expected):
    assert _years_from_text(text.lower()) == expected


def test_table_is_large_enough():
    assert len(CASES) >= 25


@pytest.mark.parametrize("text", [
    "At least three (3) years of experience.",
    "You have 3 or more years of experience",
    "We need someone with 3-5 years of experience, ideally 5.",
])
def test_llm_correct_minimum_is_grounded(text):
    # job_parser reuses the mentions: the model's correct 3 is no longer zeroed.
    out = _ground_min_years(3, {"title": "Data Engineer", "description": text})
    assert out == {"min_years_experience": 3.0, "ungrounded_min_years": None}


def test_soft_upper_value_lowers_to_minimum():
    out = _ground_min_years(5, {"title": "Data Engineer",
                                "description": "We need someone with 3-5 years of experience, ideally 5."})
    assert out["min_years_experience"] == 3.0
    assert out["ungrounded_min_years"]["reason"] == "range_upper_bound"


def test_parenthesized_number_is_one_mention():
    assert years_requirement_mentions("at least three (3) years of experience.") == [(3.0, 3.0, False)]
