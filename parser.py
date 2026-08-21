import json
import re
from llm import logged_llm_call
from schemas import ParsedResume, _strict_float


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
            years = round(_to_float_or_default(x.get("months")) / 12.0, 2)
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
    Cross-check the LLM's stated years_experience against the sum of individual
    experience durations. The two are independent sources for the same fact; if
    they diverge meaningfully, prefer the GROUNDED sum (itemized per-role
    durations resist hallucination better than a free-floating total), and record
    the discrepancy so it's VISIBLE rather than silently resolved.
    """
    stated = parsed_dict.get("years_experience", 0.0) or 0.0
    summed = round(sum(e.get("years", 0.0) or 0.0 for e in parsed_dict.get("experience", [])), 2)

    parsed_dict["years_experience_stated"] = stated
    parsed_dict["years_experience_summed"] = summed

    TOLERANCE_YEARS = 1.0
    if summed > 0 and abs(stated - summed) > TOLERANCE_YEARS:
        # Meaningful divergence: trust the grounded sum, flag the discrepancy.
        parsed_dict["years_experience"] = summed
        parsed_dict["experience_discrepancy"] = {
            "stated": stated,
            "summed": summed,
            "used": summed,
            "note": ("LLM-stated total diverged from the sum of itemized role "
                     "durations by more than 1 year; using the grounded sum.")
        }
    else:
        # They agree, or there's no itemized data to check against — keep stated.
        parsed_dict["experience_discrepancy"] = None

    return parsed_dict


def parse_resume(resume_text, run_id, step_id):
    prompt = f"""
Extract structured information from this resume.

RESUME:
{resume_text}

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
  ]
}}

RULES:
- education "year": a STRING like "2025".
- experience "title": the job title (key must be "title", not "role").
- experience "years": total years in that role as a NUMBER (key must be "years", not "months").
- "years_experience": total professional years as a number.
- If a value is unknown, use an empty string "" or 0 — never omit a key.
"""
    raw = logged_llm_call(prompt, run_id, step_id)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"resume parse: model did not return valid JSON ({e})")

    normalized = {
        "skills": [str(s) for s in (data.get("skills") or [])],
        "years_experience": _to_float_or_default(data.get("years_experience")),
        "education": _normalize_education(data.get("education")),
        "projects": _normalize_projects(data.get("projects")),
        "experience": _normalize_experience(data.get("experience")),
    }

    parsed = ParsedResume(**normalized).model_dump()
    parsed = _reconcile_experience(parsed)
    return parsed