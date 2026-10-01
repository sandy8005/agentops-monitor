import json
from pydantic import ValidationError
from llm import logged_llm_call, ModelOutputInvalid
from schemas import JobRequirements
from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE


def extract_requirements(job, run_id, step_id, budget=None):
    # A job posting is untrusted input (a poisoned listing can carry injected
    # instructions aimed at hijacking this extraction). Injection scanning for this input happens ONCE per item, where the policy
    # can act on it: job text in router.do_process_job (every evaluation, cache hit
    # or not), resume text in parser.parse_resume. The structural defense below
    # (hardening preamble + fenced data) is what protects this prompt.
    prompt = f"""
{HARDENING_PREAMBLE}

Extract structured requirements from this job posting.

JOB TITLE:
{wrap_untrusted(job['title'], "JOB_TITLE")}
JOB DESCRIPTION:
{wrap_untrusted(job['description'], "JOB_DESCRIPTION")}

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
    raw = logged_llm_call(prompt, run_id, step_id, operation="extract_requirements", budget=budget)
    cleaned = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    # Unusable MODEL output is a degraded-model condition (ModelOutputInvalid), so
    # the caller takes its rules fallback; it is never confused with a bug.
    try:
        return JobRequirements(**json.loads(cleaned)).model_dump()
    except (json.JSONDecodeError, ValidationError, TypeError) as e:
        raise ModelOutputInvalid(f"extract_requirements: unusable model output "
                                 f"({type(e).__name__}, step_id={step_id})") from None