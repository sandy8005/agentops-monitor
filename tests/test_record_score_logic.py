"""
Test the disagreement logic in isolation: a skipped/unknown judge must NOT
count as a score-vs-LLM disagreement, but a real differing judgment must.
This is pure logic — it mirrors what record_score computes, no DB needed.
"""

REAL = {"Apply", "Maybe", "Skip"}

def would_flag(score_decision, llm_decision):
    has_real = llm_decision in REAL
    return has_real and (score_decision != llm_decision)

def test_skipped_judge_is_not_a_disagreement():
    assert would_flag("Skip", "skipped (score_extreme_low)") is False
    assert would_flag("Apply", "skipped (score_extreme_high)") is False

def test_unknown_judge_is_not_a_disagreement():
    assert would_flag("Maybe", "Unknown") is False

def test_real_disagreement_still_flags():
    assert would_flag("Skip", "Maybe") is True
    assert would_flag("Apply", "Skip") is True

def test_real_agreement_does_not_flag():
    assert would_flag("Apply", "Apply") is False
    assert would_flag("Skip", "Skip") is False