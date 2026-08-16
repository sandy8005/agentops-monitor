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
        # multi-word skill: phrase match against the full text
        return s in resume_text_lower
    # single-word skill: must appear as a whole token, not a substring
    return s in resume_tokens


def _location_score(job_location, job_work_mode, user_location, user_work_mode):
    """
    Award 0-5 points for location fit. Generous when data is missing —
    absence of location data is not evidence of a bad match.
    """
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
    # Tokenize into whole words for exact skill matching (respects word
    # boundaries so 'go' does not match inside 'django'). The character class
    # keeps c++, c#, node.js intact as single tokens.
    resume_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", resume_text_lower))

    project_tech = [t.lower() for p in parsed_resume["projects"] for t in p["tech"]]
    project_tech_set = set(project_tech)
    candidate_years = parsed_resume["years_experience"]

    required = [s.lower() for s in requirements["required_skills"]]
    preferred = [s.lower() for s in requirements["preferred_skills"]]
    min_years = requirements["min_years_experience"]

    # If the extractor found essentially no requirements, we can't score this
    # job meaningfully. Do NOT award full marks for empty categories.
    insufficient = (len(required) == 0 and len(preferred) == 0)

    breakdown = {}

    # Required skills — 50 pts. Whole-word matched. No data => 0.
    if required:
        met = [s for s in required if _skill_present(s, resume_text_lower, resume_tokens)]
        breakdown["required"] = round(50 * len(met) / len(required), 1)
    else:
        breakdown["required"] = 0.0

    # Preferred skills — 20 pts.
    if preferred:
        met = [s for s in preferred if _skill_present(s, resume_text_lower, resume_tokens)]
        breakdown["preferred"] = round(20 * len(met) / len(preferred), 1)
    else:
        breakdown["preferred"] = 0.0

    # Project relevance — 15 pts: required skills that appear in project tech.
    if required:
        relevant = [s for s in required if s in project_tech_set]
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