import json
from llm import logged_llm_call
from schemas import ParsedResume


def parse_resume(resume_text, run_id, step_id):
    prompt = f"""
Extract structured information from this resume.

RESUME:
{resume_text}

Return ONLY valid JSON, no markdown fences, no explanation, in exactly this shape:
{{
  "skills": ["skill1", "skill2"],
  "years_experience": <number>,
  "education": [{{"degree": "...", "institution": "...", "year": <number>}}],
  "projects": [{{"name": "...", "tech": ["..."], "description": "one sentence"}}],
  "experience": [{{"role": "...", "company": "...", "months": <number>}}]
}}
"""
    raw = logged_llm_call(prompt, run_id, step_id)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return ParsedResume(**data).model_dump()