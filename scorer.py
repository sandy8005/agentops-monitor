from skills import (canonical_skill, canonical_set, normalize_requirements,
                    skill_in_text, text_index, skill_mention_states)


def _skill_present(skill, resume_text_lower, resume_tokens, raw_text=None,
                   surface_forms=None, negated_out=None, uncertain_out=None):
    """Whole-word, ALIAS-AWARE, NEGATION-AWARE skill match ('go' never matches inside
    'django'; 'Kubernetes' matches 'K8s'). Only an AFFIRMATIVE mention counts:
    'No Python experience', 'Python experience: none' (negated) and 'currently
    learning Python' (uncertain) are not evidence. Skills that are mentioned but only
    in those forms are recorded so the router routes the job to review (R08/N05)."""
    if not (skill or "").strip():
        return False
    forms = (surface_forms or {}).get(canonical_skill(skill))
    if not skill_in_text(skill, resume_text_lower, resume_tokens, forms):
        return False
    if raw_text is None:
        return True
    st = skill_mention_states(skill, raw_text, forms)
    if st["affirmative"]:
        return True
    if st["negated"] and negated_out is not None and skill not in negated_out:
        negated_out.append(skill)
    elif st["uncertain"] and uncertain_out is not None and skill not in uncertain_out:
        uncertain_out.append(skill)
    return False


def effective_years(parsed_resume, mode="conservative"):
    """
    Candidate years used for the experience category, and the basis for it.

    The parser keeps the LLM's STATED total but flags experience_discrepancy when
    it diverges from the sum of itemized roles by more than a year. A questionable
    total must not silently drive the deterministic score, so in "conservative" mode
    (the default) the LOWER of stated vs. summed is used whenever a discrepancy was
    flagged. mode="stated" reproduces the raw stated value (used by the router to
    check whether the discrepancy actually changes the decision).

    Returns (years, basis) with basis in {"stated", "conservative_min", "unknown"}.
    """
    parsed_resume = parsed_resume or {}
    if parsed_resume.get("years_experience") is None and parsed_resume.get("experience_unknown"):
        return None, "unknown"          # rules parse found no explicit statement
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
    surface = requirements.get("_surface_forms") or {}
    negated = []
    uncertain = []
    raw = resume_text or ""

    def present(skill):
        return _skill_present(skill, resume_text_lower, resume_tokens, raw, surface,
                              negated, uncertain)

    required = list(requirements["required_skills"])
    any_of_groups = [list(g) for g in requirements.get("required_any_of", [])]
    preferred = list(requirements["preferred_skills"])
    min_years = requirements.get("min_years_experience") or 0

    # --- Which categories APPLY -------------------------------------------------
    # A category participates only when the POSTING contains something for it to
    # score. An absent category contributes nothing and its weight is redistributed
    # proportionally over the categories that do apply. A posting that omits a
    # section must never cost the candidate that section's points:
    #
    #   required   — applies iff the job lists required skills or any-of groups.
    #   preferred  — applies iff the job lists preferred skills.
    #   projects   — applies iff there are REQUIRED technology units to look for in
    #                the candidate's projects (and see the experience-stands-in rule
    #                below). Preferred-only postings have no required tech units, so
    #                there is nothing for project evidence to corroborate.
    #   experience — applies iff the job states an explicit minimum (min_years > 0).
    #                With no stated minimum there is no requirement to satisfy, so
    #                general experience earns no free points (it used to add a
    #                constant 15% to every such job, which inflated weak matches).
    #
    # insufficient_requirements: the posting gave us NOTHING to score — no required
    # skills, no any-of groups, no preferred skills and no explicit experience
    # minimum. The router routes such a job to human review.
    total_required_items = len(required) + len(any_of_groups)
    has_skill_requirements = bool(total_required_items or preferred)
    has_experience_requirement = min_years > 0
    insufficient = not (has_skill_requirements or has_experience_requirement)
    # Only a years figure, no skill requirement at all: one numeric fact cannot
    # justify an automatic Apply (the decision is capped below), but it DOES decide
    # between Skip (under-qualified) and Maybe.
    experience_only = has_experience_requirement and not has_skill_requirements

    BASE_WEIGHTS = {"required": 50.0, "preferred": 20.0, "projects": 15.0, "experience": 15.0}
    fractions = {}       # category -> satisfied fraction in [0, 1]
    applicable = {}      # category -> weight, only for categories that apply

    # Required — flat required skills + any-of groups (each group is one item).
    matched_required = []
    missing_required = []
    if total_required_items > 0:
        met = 0
        for s in required:
            if present(s):
                met += 1
                matched_required.append(s)
            else:
                missing_required.append(s)
        for group in any_of_groups:
            if any(present(s) for s in group):
                met += 1
                matched_required.append(" / ".join(group))
            else:
                missing_required.append(" / ".join(group))
        fractions["required"] = met / total_required_items
        applicable["required"] = BASE_WEIGHTS["required"]

    # Preferred — applies only if the job actually listed preferred skills.
    matched_preferred = []
    missing_preferred = []
    if preferred:
        for s in preferred:
            if present(s):
                matched_preferred.append(s)
            else:
                missing_preferred.append(s)
        fractions["preferred"] = len(matched_preferred) / len(preferred)
        applicable["preferred"] = BASE_WEIGHTS["preferred"]

    # Project relevance — same group semantics as required (each flat skill / any-of
    # group is one unit, satisfied if it appears in the candidate's project tech).
    #
    # Projects are EVIDENCE of applied skill, and professional experience is at least
    # as strong evidence. A candidate with real work history but no separate
    # "Projects" section must not lose a fixed 15% for the missing section, so in
    # that case the projects category is treated as ABSENT and its weight is
    # redistributed. A candidate with neither projects nor experience still scores
    # 0 here when there are required tech units — that is a real gap.
    has_professional_experience = bool(
        (candidate_years or 0) > 0 or (parsed_resume.get("experience") or []))

    def _in_projects(skill):
        c = canonical_skill(skill)
        if not c:
            return False
        if c in project_tech_set:
            return True
        # multi-word canonical skill mentioned inside a longer tech label
        return " " in c and any(c in t for t in project_tech_set)

    projects_category_applies = (total_required_items > 0
                                 and (bool(projects) or not has_professional_experience))
    if projects_category_applies:
        units_met = 0
        for s in required:
            if _in_projects(s):
                units_met += 1
        for group in any_of_groups:
            if any(_in_projects(s) for s in group):
                units_met += 1
        fractions["projects"] = units_met / total_required_items
        applicable["projects"] = BASE_WEIGHTS["projects"]

    # Experience — only against an EXPLICIT minimum. UNKNOWN candidate experience
    # (rules-only parse with no explicit statement) is not scored as zero and not
    # assumed sufficient: the category is absent and the decision is capped below.
    experience_unknown = candidate_years is None
    if has_experience_requirement and not experience_unknown:
        fractions["experience"] = min(1.0, candidate_years / min_years)
        applicable["experience"] = BASE_WEIGHTS["experience"]

    # --- Normalize over applicable weight and build the reported breakdown --------
    # Renormalize the applicable weights so they still sum to 100. The breakdown
    # reports each present category's EARNED points on the normalized scale, rounded
    # for display; the TOTAL is summed from the UNROUNDED earned points so
    # per-bucket rounding can never push it past 100.
    applicable_total = sum(applicable.values())
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
            breakdown[cat] = 0.0       # absent category — shown as 0 for clarity
            breakdown_max[cat] = 0.0   # ... and it was worth nothing (not counted)

    total = round(exact_total, 1)

    # Decision thresholds (75 = Apply, 55 = Maybe) are HAND-SELECTED for this
    # proof of concept. They are not calibrated against a labeled evaluation set and
    # must not be described as validated until one exists.
    if insufficient or not applicable:
        # Nothing to score (or the only requirement is a years figure we cannot
        # check because the candidate's experience is unknown): never Skip/Apply
        # on an empty basis.
        decision = "Maybe"
    elif total >= 75:
        decision = "Apply"
    elif total >= 55:
        decision = "Maybe"
    else:
        decision = "Skip"
    if decision == "Apply" and (experience_only
                                or (experience_unknown and has_experience_requirement)):
        decision = "Maybe"              # one unverifiable / thin basis: no auto-Apply

    return {
        "score": total,
        "decision": decision,
        "breakdown": breakdown,
        "breakdown_max": breakdown_max,
        "projects_counted": projects_category_applies,
        # The categories that actually contributed to the score.
        "scored_categories": [c for c in ("required", "preferred", "projects", "experience")
                              if c in applicable],
        "candidate_years_used": candidate_years,
        "experience_basis": experience_basis,
        "insufficient_requirements": insufficient,
        "experience_only_requirements": experience_only,
        "experience_unknown": experience_unknown,
        # Accurate, requirements-based evidence (whole-word, optional-aware).
        # 'missing_skills' is REQUIRED-only — so an optional skill is never a gap.
        "matched_skills": matched_required,
        "missing_skills": missing_required,
        "matched_preferred": matched_preferred,
        "missing_preferred": missing_preferred,
        # Skills the resume mentions ONLY in negated form ("no Python experience").
        # They are scored as absent; the router forces a review when any appear.
        "negated_skills": negated,
        # Mentioned only aspirationally ("currently learning X") — also absent.
        "uncertain_skills": uncertain,
    }