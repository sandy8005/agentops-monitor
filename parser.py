import json
import re
from llm import logged_llm_call, ModelOutputInvalid
from pydantic import ValidationError
from schemas import ParsedResume, _strict_float
from logging_config import get_logger

log = get_logger(__name__)


def _raw_number(v, default=0.0):
    """
    Extract just the numeric value from 24, "24", or "24 months", WITHOUT unit
    conversion. Used for the months field, where the unit is already known from
    the field name — so we convert months->years exactly ONCE at the call site,
    never twice (the old bug: "24 months" -> 2.0 -> /12 -> 0.17).
    """
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    import re
    m = re.search(r"\d+(\.\d+)?", str(v))
    return float(m.group()) if m else default


def _to_float_or_default(v, default=0.0):
    """
    For fields where a MISSING value has a sensible default (like a per-entry
    duration the LLM omitted): coerce recoverable values, fall back to default
    only for None/absent — but still FAIL on genuine garbage like 'about two',
    so bad LLM output surfaces instead of silently becoming 0.
    """
    if v is None:
        return default
    return _strict_float(v, field_name="years")


def _normalize_education(edu_list):
    """Accept whatever the LLM returned and coerce each entry to {degree, institution, year}."""
    out = []
    for e in (edu_list or []):
        if not isinstance(e, dict):
            continue
        out.append({
            "degree": str(e.get("degree") or e.get("qualification") or e.get("name") or ""),
            "institution": str(e.get("institution") or e.get("school") or e.get("university") or e.get("college") or ""),
            "year": str(e.get("year") or e.get("graduation_year") or e.get("end_year") or e.get("completed") or "")
        })
    return out


def _normalize_projects(proj_list):
    """Coerce each project to {name, tech}. Accepts tech as list or comma string."""
    out = []
    for p in (proj_list or []):
        if not isinstance(p, dict):
            continue
        tech = p.get("tech") or p.get("technologies") or p.get("stack") or []
        if isinstance(tech, str):
            tech = [t.strip() for t in tech.split(",") if t.strip()]
        elif not isinstance(tech, list):
            tech = []
        out.append({
            "name": str(p.get("name") or p.get("title") or p.get("project") or ""),
            "tech": [str(t) for t in tech]
        })
    return out


def _normalize_experience(exp_list):
    """
    Coerce each experience entry to {title, company, years}.
    Handles alternative keys (role/position, months, duration) universally.
    Garbage durations fail loudly (via _strict_float) rather than becoming 0.
    """
    out = []
    for x in (exp_list or []):
        if not isinstance(x, dict):
            continue
        title = x.get("title") or x.get("role") or x.get("position") or x.get("job_title") or ""
        company = x.get("company") or x.get("employer") or x.get("organization") or ""

        if x.get("years") is not None:
            years = _to_float_or_default(x.get("years"))
        elif x.get("months") is not None:
            # months field: unit is known, take RAW number, convert to years ONCE.
            years = round(_raw_number(x.get("months")) / 12.0, 2)
        elif x.get("duration") is not None:
            years = _to_float_or_default(x.get("duration"))
        else:
            years = 0.0

        out.append({
            "title": str(title),
            "company": str(company),
            "years": years
        })
    return out


def _reconcile_experience(parsed_dict):
    """
    Cross-check the LLM's stated years_experience against the sum of itemized role
    durations and FLAG a large divergence — but do NOT overwrite the stated total.
    A naive sum is not ground truth: it double-counts concurrent/overlapping roles
    (accurate reconciliation needs date-range unioning) and under-counts when only
    some roles are itemized. So we keep the model's stated total and surface the
    discrepancy for a human / downstream awareness instead of silently replacing it.
    """
    stated = parsed_dict.get("years_experience", 0.0) or 0.0
    summed = round(sum(e.get("years", 0.0) or 0.0 for e in parsed_dict.get("experience", [])), 2)

    parsed_dict["years_experience_stated"] = stated
    parsed_dict["years_experience_summed"] = summed

    TOLERANCE_YEARS = 1.0
    if summed > 0 and abs(stated - summed) > TOLERANCE_YEARS:
        # Divergence > 1yr: FLAG it, but keep the stated total (the sum is unreliable
        # without date-range unioning). Do not silently replace years_experience.
        parsed_dict["experience_discrepancy"] = {
            "stated": stated,
            "summed": summed,
            "note": ("stated total and summed itemized-role durations diverge by "
                     ">1 year; the sum may double-count concurrent roles or miss "
                     "un-itemized ones — keeping the stated total, flagged for review."),
        }
    else:
        parsed_dict["experience_discrepancy"] = None

    # years_experience is left as STATED (not replaced).
    return parsed_dict


def _norm_ws(text):
    return " ".join(str(text or "").lower().split())


def ground_skills(skills, skill_evidence, resume_text):
    """
    Make the parser's skill claims TRACEABLE to the resume text, deterministically.

    The LLM is asked for a verbatim evidence snippet per skill. Here we keep only
    snippets that really occur in the resume (whitespace/case-normalized), and mark a
    skill as grounded if it has verified evidence OR the skill term itself appears in
    the resume as a whole word/phrase. Anything else is an UNGROUNDED claim — a
    candidate extraction error or hallucination — surfaced for monitoring and kept
    out of the judge's "candidate skills".

    Returns {"skill_evidence": [...verified...], "grounded_skills": [...],
             "ungrounded_skills": [...]}.
    """
    from skills import affirmative_skill_in_text
    resume_norm = _norm_ws(resume_text)
    resume_tokens = set(re.findall(r"[a-z0-9\+\#\.]+", resume_norm))
    verified = {}
    for item in (skill_evidence or []):
        if not isinstance(item, dict):
            continue
        skill = str(item.get("skill") or "").strip()
        ev = str(item.get("evidence") or "").strip()
        # The snippet must (a) occur verbatim in the resume AND (b) itself
        # AFFIRMATIVELY mention the claimed skill. A real quote about something else
        # ("Python engineer" offered as evidence for "Kubernetes") grounds nothing (R10).
        if skill and ev and _norm_ws(ev) in resume_norm \
                and affirmative_skill_in_text(skill, ev):
            verified.setdefault(skill.lower(), {"skill": skill, "evidence": ev})

    from skills import skill_in_text, text_index
    idx_low, idx_tokens = text_index(resume_text)

    def _literal(skill):
        # Alias-aware ("K8s" in the resume grounds a parsed "Kubernetes"), using the
        # SAME canonical vocabulary the scorer uses — and only AFFIRMATIVE mentions:
        # "no Kubernetes experience" does not ground Kubernetes.
        if not _norm_ws(skill):
            return False
        return (skill_in_text(skill, idx_low, idx_tokens)
                and affirmative_skill_in_text(skill, resume_text))

    grounded, ungrounded = [], []
    for sk in skills:
        (grounded if (sk.lower() in verified or _literal(sk)) else ungrounded).append(sk)
    return {"skill_evidence": list(verified.values()),
            "grounded_skills": grounded, "ungrounded_skills": ungrounded}


def parse_resume(resume_text, run_id, step_id, budget=None):
    from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE, detect_injection
    # The resume is the most attacker-controlled input (a candidate uploads it).
    # Injection-looking patterns are recorded as a SECURITY WARNING, not a review
    # request: the graph deliberately continues after parse_resume (there is no
    # human-review route here), so writing needs_human_review would make the Monitor
    # claim "human review required" for a review that will never happen. The
    # structural defense (delimiting + hardening) below is what protects the prompt.
    try:
        from prompt_safety import apply_injection_policy
        apply_injection_policy(detect_injection(resume_text), step_id, source="resume",
                               run_id=run_id, review_allowed=False)
    except Exception:
        # Observability must not break parsing; log the failure TYPE only.
        log.warning("could not record resume security signal", extra={"step_id": step_id})
    prompt = f"""
{HARDENING_PREAMBLE}

Extract structured information from this resume.

RESUME:
{wrap_untrusted(resume_text, "RESUME")}

Return ONLY valid JSON, no markdown fences, no explanation, in EXACTLY this shape
and using EXACTLY these key names:
{{
  "skills": ["skill1", "skill2"],
  "years_experience": <number>,
  "education": [
    {{"degree": "...", "institution": "...", "year": "YYYY"}}
  ],
  "projects": [
    {{"name": "...", "tech": ["..."]}}
  ],
  "experience": [
    {{"title": "...", "company": "...", "years": <number>}}
  ],
  "skill_evidence": [
    {{"skill": "FastAPI", "evidence": "built a FastAPI backend serving 2k req/s"}}
  ]
}}

RULES:
- education "year": a STRING like "2025".
- experience "title": the job title (key must be "title", not "role").
- experience "years": total years in that role as a NUMBER (key must be "years", not "months").
- "years_experience": total professional years as a number.
- "skill_evidence": for each skill, a SHORT phrase COPIED VERBATIM from the resume
  that shows it. Never paraphrase. Omit a skill here if no such text exists.
- If a value is unknown, use an empty string "" or 0 — never omit a key.
"""
    raw = logged_llm_call(prompt, run_id, step_id, operation="parse_resume", budget=budget)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    # --- Parse model output -----------------------------------------------
    # Fail LOUD on unparseable output rather than returning None: a None return
    # would let the router cache json.dumps(None) == "null" and poison the parse
    # cache; raising surfaces the failure so the run fails cleanly and nothing bad
    # is cached.
    # Every "the model's output is unusable" condition raises ModelOutputInvalid, a
    # degraded-MODEL signal the router may fall back on. Errors in OUR code below
    # (reconciliation, grounding) are deliberately NOT wrapped — they must surface.
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # The model occasionally wraps the object in prose despite instructions;
        # salvage the outermost {...} before giving up.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ModelOutputInvalid(
                f"parse_resume: LLM output was not valid JSON (step_id={step_id})"
            )
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ModelOutputInvalid(
                f"parse_resume: LLM output was not valid JSON (step_id={step_id})") from e

    if not isinstance(parsed, dict):
        raise ModelOutputInvalid(
            f"parse_resume: expected a JSON object, got "
            f"{type(parsed).__name__} (step_id={step_id})"
        )

    # --- Normalize each field into the canonical shape --------------------
    skills = parsed.get("skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        skills = [str(s).strip() for s in skills if str(s).strip()]
    else:
        skills = []

    try:
        # _strict_float raises ValueError on garbage durations ("about two") — that
        # is unusable MODEL output, not a bug in this module.
        normalized = {
            "skills": skills,
            "years_experience": _to_float_or_default(parsed.get("years_experience")),
            "education": _normalize_education(parsed.get("education")),
            "projects": _normalize_projects(parsed.get("projects")),
            "experience": _normalize_experience(parsed.get("experience")),
        }
    except ValueError as e:
        raise ModelOutputInvalid(f"parse_resume: unusable value in model output: {e} "
                                 f"(step_id={step_id})") from e

    # --- Schema validation (core fields) ----------------------------------
    try:
        validated = ParsedResume.model_validate(normalized).model_dump()
    except ValidationError as e:
        raise ModelOutputInvalid(
            f"parse_resume: model output failed schema validation ({e.error_count()} "
            f"error(s), step_id={step_id})") from e

    # --- Cross-check stated vs summed experience (adds audit metadata) ----
    validated = _reconcile_experience(validated)

    # --- Deterministic grounding of the parsed skills ----------------------
    validated.update(ground_skills(skills, parsed.get("skill_evidence"), resume_text))

    return validated