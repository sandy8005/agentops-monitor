import re


def _skill_present(skill, resume_text_lower, resume_tokens):
    """Whole-word skill match. Avoids 'go' matching inside 'django'."""
    s = skill.lower().strip()
    if not s:
        return False
    if " " in s:
        return s in resume_text_lower
    return s in resume_tokens


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

    insufficient = (len(required) == 0 and len(any_of_groups) == 0 and len(preferred) == 0)

    breakdown = {}

    # Required section — 50 pts. Flat required skills + any-of groups.
    # A group counts as one item, satisfied if ANY of its alternatives is present.
    # While scoring the required section, capture which required skills are
    # present vs missing — this is the ACCURATE, whole-word, requirements-based
    # evidence (computed against the STRUCTURED required list, so an "optional"
    # skill that extraction placed in preferred is never counted as missing).
    matched_required = []
    missing_required = []
    total_required_items = len(required) + len(any_of_groups)
    if total_required_items > 0:
        met = 0
        for s in required:
            if _skill_present(s, resume_text_lower, resume_tokens):
                met += 1
                matched_required.append(s)
            else:
                missing_required.append(s)
        for group in any_of_groups:
            if any(_skill_present(s, resume_text_lower, resume_tokens) for s in group):
                met += 1
                matched_required.append(" / ".join(group))
            else:
                missing_required.append(" / ".join(group))
        breakdown["required"] = round(50 * met / total_required_items, 1)
    else:
        breakdown["required"] = 0.0

    # Preferred skills — 20 pts.
    matched_preferred = []
    missing_preferred = []
    if preferred:
        for s in preferred:
            if _skill_present(s, resume_text_lower, resume_tokens):
                matched_preferred.append(s)
            else:
                missing_preferred.append(s)
        breakdown["preferred"] = round(20 * len(matched_preferred) / len(preferred), 1)
    else:
        breakdown["preferred"] = 0.0

    # Project relevance — 15 pts. Uses the SAME group semantics as the required
    # section: each flat required skill is one unit, and each any-of GROUP is one
    # unit satisfied if ANY member appears in the projects. This keeps the project
    # score consistent with the required score — a Django-only candidate gets full
    # credit for a "Flask OR Django" group in BOTH sections, not half in projects.
    def _in_projects(skill):
        s = skill.lower().strip()
        if " " in s:
            return any(s in t for t in project_tech_set)
        return s in project_tech_set

    total_project_units = len(required) + len(any_of_groups)
    if total_project_units > 0:
        units_met = 0
        for s in required:
            if _in_projects(s):
                units_met += 1
        for group in any_of_groups:
            if any(_in_projects(s) for s in group):
                units_met += 1
        breakdown["projects"] = round(15 * units_met / total_project_units, 1)
    else:
        breakdown["projects"] = 0.0

    # Experience — 15 pts.
    if candidate_years >= min_years:
        breakdown["experience"] = 15.0
    elif min_years > 0:
        breakdown["experience"] = round(15 * candidate_years / min_years, 1)
    else:
        breakdown["experience"] = 15.0

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
        "insufficient_requirements": insufficient,
        # Accurate, requirements-based evidence (whole-word, optional-aware).
        # 'missing_skills' is REQUIRED-only — so an optional skill is never a gap.
        "matched_skills": matched_required,
        "missing_skills": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
    }