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

    breakdown = {}

    # Required skills — 50 pts, checked against full resume text
    if required:
        met = [s for s in required if s in resume_text_lower]
        breakdown["required"] = round(50 * len(met) / len(required), 1)
    else:
        breakdown["required"] = 50.0

    # Preferred skills — 20 pts
    if preferred:
        met = [s for s in preferred if s in resume_text_lower]
        breakdown["preferred"] = round(20 * len(met) / len(preferred), 1)
    else:
        breakdown["preferred"] = 20.0

    # Project relevance — 15 pts: projects use any required skills?
    if required:
        relevant = [s for s in required if s in project_tech]
        breakdown["projects"] = round(15 * len(relevant) / len(required), 1)
    else:
        breakdown["projects"] = 15.0

    # Experience — 10 pts: meets minimum?
    if candidate_years >= min_years:
        breakdown["experience"] = 10.0
    elif min_years > 0:
        breakdown["experience"] = round(10 * candidate_years / min_years, 1)
    else:
        breakdown["experience"] = 10.0

    # Location — 5 pts, now a real comparison (falls back gracefully)
    if job is not None and user_input is not None:
        breakdown["location"] = _location_score(
            job.get("location"), job.get("work_mode"),
            user_input.get("location"), user_input.get("work_mode")
        )
    else:
        breakdown["location"] = 3.0   # neutral default if job/user not provided

    total = round(sum(breakdown.values()), 1)

    if total >= 75:
        decision = "Apply"
    elif total >= 55:
        decision = "Maybe"
    else:
        decision = "Skip"

    return {"score": total, "decision": decision, "breakdown": breakdown}