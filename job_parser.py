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
- required_skills: mandatory skills the candidate MUST have (all of them).
- required_any_of: groups of ALTERNATIVE skills where the candidate needs AT LEAST ONE per group. If the posting says "Flask or Django", output ["Flask", "Django"] as ONE group here. Do NOT split alternatives into separate required_skills — that would wrongly demand all of them.
- preferred_skills: skills that are a plus, bonus, nice to have, or preferred (not mandatory).
- Put duties and tasks in "responsibilities", NOT in skills.

Return ONLY valid JSON, no markdown fences, no explanation, in exactly this shape:
{{
  "required_skills": ["skill1", "skill2"],
  "required_any_of": [["AlternativeA", "AlternativeB"]],
  "preferred_skills": ["skill3"],
  "min_years_experience": <number>,
  "responsibilities": ["short phrase", "short phrase"]
}}

If there are no "or" alternatives in the posting, return an empty list for required_any_of: [].
"""
    raw = logged_llm_call(prompt, run_id, step_id, operation="extract_requirements")
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return JobRequirements(**data).model_dump()