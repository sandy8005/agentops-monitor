"""
Final ranking of scored jobs.

The ranking honours the AUTHORITATIVE decision bucket first and the score second:

    Apply  >  Maybe  >  Skip          (bucket, from final_decision)
    higher score first                (inside each bucket)

final_decision is human > LLM judge > deterministic score (see
router._compute_final_decision), so a job a human rejected ("Skip") can never be
ranked above a job that was approved ("Apply"), however high its raw score is.
"""

DECISION_ORDER = {"Apply": 2, "Maybe": 1, "Skip": 0}
_UNKNOWN_BUCKET = -1   # no usable decision at all -> always last


def _bucket(result):
    """The authoritative decision for one result: final_decision, falling back to
    the deterministic decision for legacy rows that predate final_decision."""
    decision = result.get("final_decision") or result.get("decision")
    return DECISION_ORDER.get(decision, _UNKNOWN_BUCKET)


def _score(result):
    score = result.get("score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def rank_jobs(results):
    """
    results: list of dicts, each like
      {"title": ..., "company": ..., "score": ..., "decision": ...,
       "llm_decision": ..., "final_decision": ..., "needs_review": ...}

    Returns a NEW list sorted by (decision bucket, score), best first. The sort is
    stable, so jobs with the same bucket and score keep their processing order.
    """
    return sorted(results or [], key=lambda r: (_bucket(r), _score(r)), reverse=True)