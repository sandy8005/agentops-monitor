import json
from llm import logged_llm_call
from schemas import JobRequirements


def extract_requirements(job, run_id, step_id):
    prompt = f"""
Extract structured requirements from this job posting.

JOB TITLE: {job['title']}
JOB DESCRIPTION:
{job['description']}

RULES for skills:
- Each skill must be a SINGLE atomic technology, tool, or language (e.g. "Python", "Docker", "Kubernetes", "React").
- Do NOT use phrases, sentences, or responsibilities as skills (e.g. NOT "Building ML systems at scale", NOT "Full Stack Development").
- Do NOT combine skills with "or" or "and". Split "Flask or Django" into two separate entries: "Flask", "Django".
- Distinguish REQUIRED skills (must-have, required, minimum) from PREFERRED skills (a plus, bonus, nice to have, preferred).
- Put duties and tasks in "responsibilities", NOT in skills.

Return ONLY valid JSON, no markdown fences, no explanation, in exactly this shape:
{{
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill3"],
  "min_years_experience": <number>,
  "responsibilities": ["short phrase", "short phrase"]
}}
"""
    raw = logged_llm_call(prompt, run_id, step_id)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return JobRequirements(**data).model_dump()