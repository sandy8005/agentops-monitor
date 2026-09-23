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

    parsed_resume = parsed_resume or {}
    projects = parsed_resume.get("projects") or []
    project_tech = [t.lower() for p in projects if isinstance(p, dict) for t in (p.get("tech") or [])]
    project_tech_set = set(project_tech)
    candidate_years = parsed_resume.get("years_experience") or 0

    required = [s.lower() for s in requirements["required_skills"]]
    any_of_groups = [[s.lower() for s in group]
                     for group in requirements.get("required_any_of", [])]
    preferred = [s.lower() for s in requirements["preferred_skills"]]
    min_years = requirements["min_years_experience"]

    insufficient = (len(required) == 0 and len(any_of_groups) == 0 and len(preferred) == 0)

    # --- Scoring model ---------------------------------------------------------
    # Each category contributes a FRACTION (0..1) of how well it's satisfied, and
    # carries a base weight. The final score is the weighted average over only the
    # categories that APPLY, times 100. "Preferred" is optional: when the job listed
    # NO preferred skills there is nothing to score, so its weight is dropped and the
    # remaining categories are renormalized proportionally (their relative weights
    # are preserved). A perfect candidate therefore scores 100 whether or not the
    # posting happened to list preferred skills. Required/projects/experience always
    # apply (projects scores 0 if the required tech simply isn't in the resume's
    # projects — that's a real signal, not an absent category).
    BASE_WEIGHTS = {"required": 50.0, "preferred": 20.0, "projects": 15.0, "experience": 15.0}
    fractions = {}       # category -> satisfied fraction in [0, 1]
    applicable = {}      # category -> weight, only for categories that apply

    # Required — flat required skills + any-of groups (each group is one item).
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
        fractions["required"] = met / total_required_items
    else:
        fractions["required"] = 0.0
    applicable["required"] = BASE_WEIGHTS["required"]   # required always applies

    # Preferred — OPTIONAL. Applies only if the job actually listed preferred skills.
    matched_preferred = []
    missing_preferred = []
    if preferred:
        for s in preferred:
            if _skill_present(s, resume_text_lower, resume_tokens):
                matched_preferred.append(s)
            else:
                missing_preferred.append(s)
        fractions["preferred"] = len(matched_preferred) / len(preferred)
        applicable["preferred"] = BASE_WEIGHTS["preferred"]
    # else: no preferred skills to score → category ABSENT; its weight is not added
    # to `applicable`, so it's redistributed across the present categories below.

    # Project relevance — same group semantics as required (each flat skill / any-of
    # group is one unit, satisfied if it appears in the candidate's project tech).
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
        fractions["projects"] = units_met / total_project_units
    else:
        fractions["projects"] = 0.0
    applicable["projects"] = BASE_WEIGHTS["projects"]   # projects always applies

    # Experience.
    if candidate_years >= min_years:
        fractions["experience"] = 1.0
    elif min_years > 0:
        fractions["experience"] = candidate_years / min_years
    else:
        fractions["experience"] = 1.0
    applicable["experience"] = BASE_WEIGHTS["experience"]   # experience always applies

    # --- Normalize over applicable weight and build the reported breakdown --------
    # Renormalize the applicable weights so they still sum to 100 (this is where an
    # absent optional category's weight is proportionally redistributed). The
    # breakdown reports each present category's EARNED points on the normalized
    # scale, rounded for display; the TOTAL is summed from the UNROUNDED earned
    # points so per-bucket rounding can never push it past 100.
    applicable_total = sum(applicable.values()) or 1.0
    breakdown = {}
    exact_total = 0.0
    for cat in ("required", "preferred", "projects", "experience"):
        if cat in applicable:
            norm_weight = applicable[cat] * 100.0 / applicable_total
            earned = fractions[cat] * norm_weight
            exact_total += earned
            breakdown[cat] = round(earned, 1)
        else:
            breakdown[cat] = 0.0   # absent optional category — shown as 0 for clarity

    total = round(exact_total, 1)

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