def rank_jobs(results):
    """
    results: list of dicts, each like
      {"title": ..., "company": ..., "score": ..., "decision": ...,
       "llm_decision": ..., "needs_review": ...}
    Returns the same list sorted by score, highest first.
    """
    return sorted(results, key=lambda r: r["score"], reverse=True)