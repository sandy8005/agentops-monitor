from skills import (canonical_skill, canonical_set, normalize_requirements,
                    skill_in_text, text_index)


def _skill_present(skill, resume_text_lower, resume_tokens):
    """Whole-word, ALIAS-AWARE skill match ('go' never matches inside 'django';
    'Kubernetes' matches 'K8s'). See skills.py for the canonical vocabulary."""
    if not (skill or "").strip():
        return False
    return skill_in_text(skill, resume_text_lower, resume_tokens)


def effective_years(parsed_resume, mode="conservative"):
    """
    Candidate years used for the experience category, and the basis for it.

    The parser keeps the LLM's STATED total but flags experience_discrepancy when
    it diverges from the sum of itemized roles by more than a year. A questionable
    total must not silently drive the deterministic score, so in "conservative" mode
    (the default) the LOWER of stated vs. summed is used whenever a discrepancy was
    flagged. mode="stated" reproduces the raw stated value (used by the router to
    check whether the discrepancy actually changes the decision).

    Returns (years, basis) with basis in {"stated", "conservative_min"}.
    """
    parsed_resume = parsed_resume or {}
    stated = parsed_resume.get("years_experience") or 0
    disc = parsed_resume.get("experience_discrepancy")
    if mode == "conservative" and isinstance(disc, dict):
        summed = disc.get("summed")
        try:
            summed = float(summed)
        except (TypeError, ValueError):
            summed = None
        if summed is not None and summed < stated:
            return summed, "conservative_min"
    return stated, "stated"


def calculate_match_score(parsed_resume, requirements, resume_text, job=None, user_input=None,
                          experience_mode="conservative"):
    resume_text_lower, resume_tokens = text_index(resume_text)

    parsed_resume = parsed_resume or {}
    projects = parsed_resume.get("projects") or []
    project_tech_set = canonical_set(
        t for p in projects if isinstance(p, dict) for t in (p.get("tech") or []))
    candidate_years, experience_basis = effective_years(parsed_resume, experience_mode)

    # Canonicalize + DEDUPLICATE first: "K8s"/"Kubernetes" is one requirement, and a
    # flat skill that also appears in an any-of group is not counted twice.
    requirements = normalize_requirements(requirements)
    required = list(requirements["required_skills"])
    any_of_groups = [list(g) for g in requirements.get("required_any_of", [])]
    preferred = list(requirements["preferred_skills"])
    min_years = requirements.get("min_years_experience") or 0

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
    #
    # Projects are EVIDENCE of applied skill, and professional experience is at least
    # as strong evidence. A candidate with real work history but no separate
    # "Projects" section must not lose a fixed 15% for the missing section, so in
    # that case the projects category is treated as ABSENT and its weight is
    # redistributed (same mechanism as an absent "preferred" category). A candidate
    # with neither projects nor experience still scores 0 here — that is a real gap.
    has_professional_experience = bool(
        (candidate_years or 0) > 0 or (parsed_resume.get("experience") or []))
    projects_category_applies = bool(projects) or not has_professional_experience
    def _in_projects(skill):
        c = canonical_skill(skill)
        if not c:
            return False
        if c in project_tech_set:
            return True
        # multi-word canonical skill mentioned inside a longer tech label
        return " " in c and any(c in t for t in project_tech_set)

    total_project_units = len(required) + len(any_of_groups)
    if not projects_category_applies:
        pass   # absent: experience stands in as the evidence; weight redistributed
    elif total_project_units > 0:
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
    if projects_category_applies:
        applicable["projects"] = BASE_WEIGHTS["projects"]

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
    breakdown_max = {}   # the REAL maximum for each category after renormalization
    exact_total = 0.0
    for cat in ("required", "preferred", "projects", "experience"):
        if cat in applicable:
            norm_weight = applicable[cat] * 100.0 / applicable_total
            earned = fractions[cat] * norm_weight
            exact_total += earned
            breakdown[cat] = round(earned, 1)
            breakdown_max[cat] = round(norm_weight, 1)
        else:
            breakdown[cat] = 0.0       # absent optional category — shown as 0 for clarity
            breakdown_max[cat] = 0.0   # ... and it was worth nothing (not counted)

    total = round(exact_total, 1)

    # Decision thresholds (75 = Apply, 55 = Maybe) are HAND-SELECTED for this
    # proof of concept. They are not calibrated against a labeled evaluation set and
    # must not be described as validated until one exists.
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
        "breakdown_max": breakdown_max,
        "projects_counted": projects_category_applies,
        "candidate_years_used": candidate_years,
        "experience_basis": experience_basis,
        "insufficient_requirements": insufficient,
        # Accurate, requirements-based evidence (whole-word, optional-aware).
        # 'missing_skills' is REQUIRED-only — so an optional skill is never a gap.
        "matched_skills": matched_required,
        "missing_skills": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
    }