import re


def _skill_present(skill, resume_text_lower, resume_tokens):
    """
    Whole-word skill match. Avoids 'go' matching inside 'django'.
    Multi-word skills (e.g. 'machine learning') fall back to phrase search.
    """
    s = skill.lower().strip()
    if not s:
        return False
    if " " in s:
        return s in resume_text_lower
    return s in resume_tokens


def _location_score(job_location, job_work_mode, user_location, user_work_mode):
    """Award 0-5 points for location fit. Generous when data is missing."""
    jl = (job_location or "").lower()
    jm = (job_work_mode or "").lower()
    ul = (user_location or "").lower()
    um = (user_work_mode or "").lower()

    if "remote" in jl or "remote" in jm or "anywhere" in jl:
        return 5.0
    if not jl:
        return 3.0
    if ul and ul in jl:
        return 5.0
    if um in ("remote", "hybrid") and not jl:
        return 4.0
    return 1.0


def calculate_match_score(parsed_resume, requirements, resume_text, job=None, user_input=None):
    resume_text_lower = resume_text.lower()
    resume_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", resume_text_lower))

    project_tech = [t.lower() for p in parsed_resume["projects"] for t in p["tech"]]
    project_tech_set = set(project_tech)
    candidate_years = parsed_resume["years_experience"]

    required = [s.lower() for s in requirements["required_skills"]]
    any_of_groups = [[s.lower() for s in group]
                     for group in requirements.get("required_any_of", [])]
    preferred = [s.lower() for s in requirements["preferred_skills"]]
    min_years = requirements["min_years_experience"]

    # "No requirements at all" => can't score meaningfully.
    insufficient = (len(required) == 0 and len(any_of_groups) == 0 and len(preferred) == 0)

    breakdown = {}

    # Required section — 50 pts. Covers flat required skills + any-of groups.
    # A group counts as one item, satisfied if ANY of its alternatives is present.
    total_required_items = len(required) + len(any_of_groups)
    if total_required_items > 0:
        met = 0
        for s in required:
            if _skill_present(s, resume_text_lower, resume_tokens):
                met += 1
        for group in any_of_groups:
            if any(_skill_present(s, resume_text_lower, resume_tokens) for s in group):
                met += 1
        breakdown["required"] = round(50 * met / total_required_items, 1)
    else:
        breakdown["required"] = 0.0

    # Preferred skills — 20 pts.
    if preferred:
        met = [s for s in preferred if _skill_present(s, resume_text_lower, resume_tokens)]
        breakdown["preferred"] = round(20 * len(met) / len(preferred), 1)
    else:
        breakdown["preferred"] = 0.0

    # Project relevance — 15 pts: required skills (flat + any-of members) in projects.
    all_required_flat = list(required)
    for group in any_of_groups:
        all_required_flat.extend(group)
    if all_required_flat:
        relevant = [s for s in set(all_required_flat) if s in project_tech_set]
        # normalise against distinct required items so it stays 0-15
        denom = len(set(all_required_flat))
        breakdown["projects"] = round(15 * len(relevant) / denom, 1)
    else:
        breakdown["projects"] = 0.0

    # Experience — 10 pts.
    if candidate_years >= min_years:
        breakdown["experience"] = 10.0
    elif min_years > 0:
        breakdown["experience"] = round(10 * candidate_years / min_years, 1)
    else:
        breakdown["experience"] = 10.0

    # Location — 5 pts.
    if job is not None and user_input is not None:
        breakdown["location"] = _location_score(
            job.get("location"), job.get("work_mode"),
            user_input.get("location"), user_input.get("work_mode")
        )
    else:
        breakdown["location"] = 3.0

    total = round(sum(breakdown.values()), 1)

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