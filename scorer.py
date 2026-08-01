def calculate_match_score(parsed_resume, requirements, resume_text):
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

    # Experience — 10 pts
    if candidate_years >= min_years:
        breakdown["experience"] = 10.0
    elif min_years > 0:
        breakdown["experience"] = round(10 * candidate_years / min_years, 1)
    else:
        breakdown["experience"] = 10.0

    # Location — 5 pts: stubbed (no location data in jobs yet)
    breakdown["location"] = 5.0

    total = round(sum(breakdown.values()), 1)

    if total >= 75:
        decision = "Apply"
    elif total >= 55:
        decision = "Maybe"
    else:
        decision = "Skip"

    return {"score": total, "decision": decision, "breakdown": breakdown}