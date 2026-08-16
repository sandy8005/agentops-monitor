import json
from llm import logged_llm_call
from schemas import ParsedResume


def _to_float(v, default=0.0):
    """Coerce anything numeric-ish to float; fall back to default."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        # pull the first number out of strings like "2 years", "24 months", "1.5"
        import re
        m = re.search(r"\d+(\.\d+)?", str(v))
        return float(m.group()) if m else default
    except (ValueError, TypeError):
        return default


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
    """
    out = []
    for x in (exp_list or []):
        if not isinstance(x, dict):
            continue
        title = x.get("title") or x.get("role") or x.get("position") or x.get("job_title") or ""
        company = x.get("company") or x.get("employer") or x.get("organization") or ""

        # years: prefer an explicit years field; else convert months; else parse duration
        if x.get("years") is not None:
            years = _to_float(x.get("years"))
        elif x.get("months") is not None:
            years = round(_to_float(x.get("months")) / 12.0, 2)
        elif x.get("duration") is not None:
            years = _to_float(x.get("duration"))
        else:
            years = 0.0

        out.append({
            "title": str(title),
            "company": str(company),
            "years": years
        })
    return out


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

    # Parse JSON defensively — if the model wrapped or malformed it, fail with a clear message.
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"resume parse: model did not return valid JSON ({e})")

    # Normalize EVERYTHING before validation, so schema differences never crash the run.
    normalized = {
        "skills": [str(s) for s in (data.get("skills") or [])],
        "years_experience": _to_float(data.get("years_experience")),
        "education": _normalize_education(data.get("education")),
        "projects": _normalize_projects(data.get("projects")),
        "experience": _normalize_experience(data.get("experience")),
    }

    return ParsedResume(**normalized).model_dump()