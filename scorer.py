def _location_score(job_location, job_work_mode, user_location, user_work_mode):
    """
    Award 0-5 points for location fit. Generous when data is missing —
    absence of location data is not evidence of a bad match.
    """
    jl = (job_location or "").lower()
    jm = (job_work_mode or "").lower()
    ul = (user_location or "").lower()
    um = (user_work_mode or "").lower()

    # Remote job satisfies any location preference
    if "remote" in jl or "remote" in jm or "anywhere" in jl:
        return 5.0

    # No location data on the job — can't verify, don't penalize
    if not jl:
        return 3.0

    # Job location mentions the user's location (state or city)
    if ul and ul in jl:
        return 5.0

    # User wants remote/hybrid and the job location is unspecified
    if um in ("remote", "hybrid") and not jl:
        return 4.0

    # Job specifies a location that doesn't match
    return 1.0


def calculate_match_score(parsed_resume, requirements, resume_text, job=None, user_input=None):
    resume_text_lower = resume_text.lower()
    project_tech = [t.lower() for p in parsed_resume["projects"] for t in p["tech"]]
    candidate_years = parsed_resume["years_experience"]

    required = [s.lower() for s in requirements["required_skills"]]
    preferred = [s.lower() for s in requirements["preferred_skills"]]
    min_years = requirements["min_years_experience"]

    # If the extractor found essentially no requirements, we can't score this
    # job meaningfully. Do NOT award full marks for empty categories — that
    # would read "unknown job" as "perfect match". Flag it for review instead.
    insufficient = (len(required) == 0 and len(preferred) == 0)

    breakdown = {}

    # Required skills — 50 pts. No data => 0 (not full marks).
    if required:
        met = [s for s in required if s in resume_text_lower]
        breakdown["required"] = round(50 * len(met) / len(required), 1)
    else:
        breakdown["required"] = 0.0

    # Preferred skills — 20 pts.
    if preferred:
        met = [s for s in preferred if s in resume_text_lower]
        breakdown["preferred"] = round(20 * len(met) / len(preferred), 1)
    else:
        breakdown["preferred"] = 0.0

    # Project relevance — 15 pts: required skills used in the candidate's projects.
    if required:
        relevant = [s for s in required if s in project_tech]
        breakdown["projects"] = round(15 * len(relevant) / len(required), 1)
    else:
        breakdown["projects"] = 0.0

    # Experience — 10 pts: meets the minimum?
    if candidate_years >= min_years:
        breakdown["experience"] = 10.0
    elif min_years > 0:
        breakdown["experience"] = round(10 * candidate_years / min_years, 1)
    else:
        breakdown["experience"] = 10.0

    # Location — 5 pts: real comparison, generous on missing data.
    if job is not None and user_input is not None:
        breakdown["location"] = _location_score(
            job.get("location"), job.get("work_mode"),
            user_input.get("location"), user_input.get("work_mode")
        )
    else:
        breakdown["location"] = 3.0

    total = round(sum(breakdown.values()), 1)

    # Decision. If we couldn't extract requirements, the score is untrustworthy
    # — force a Maybe (never a confident Apply/Skip) and let the caller flag it.
    if insufficient:
        decision = "Maybe"
    elif total >= 75:
        decision = "Apply"
    elif total >= 55:
        decision = "Maybe"
    else:
        decision = "Skip"

    return {
        "score": total,
        "decision": decision,
        "breakdown": breakdown,
        "insufficient_requirements": insufficient
    }