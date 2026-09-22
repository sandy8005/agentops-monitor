from llm import logged_llm_call
from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE


def application_strategy(resume_text, job, requirements, missing_skills, run_id, step_id):
    prompt = f"""
{HARDENING_PREAMBLE}

You are a career coach. The candidate is considering applying to this job.

CANDIDATE RESUME:
{wrap_untrusted(resume_text, "RESUME")}

JOB TITLE: {wrap_untrusted(job['title'], "JOB_TITLE")}
COMPANY: {wrap_untrusted(job['company'], "COMPANY")}
REQUIRED SKILLS: {wrap_untrusted(requirements['required_skills'], "REQUIRED_SKILLS")}
PREFERRED SKILLS: {wrap_untrusted(requirements['preferred_skills'], "PREFERRED_SKILLS")}
SKILLS THE CANDIDATE IS MISSING: {wrap_untrusted(missing_skills, "MISSING_SKILLS")}

Give a short, practical application strategy in 3-4 sentences:
- What strengths should the candidate emphasize for this specific role?
- How should they address or downplay the missing skills?
- One concrete tip to stand out.

Write plain prose, no headers, no bullet points.
"""
    return logged_llm_call(prompt, run_id, step_id, operation="application_strategy")


def resume_edit_advice(resume_text, job, requirements, missing_skills, run_id, step_id):
    prompt = f"""
{HARDENING_PREAMBLE}

You are a resume editor. Suggest concrete edits to tailor this resume for this specific job.

CANDIDATE RESUME:
{wrap_untrusted(resume_text, "RESUME")}

JOB TITLE: {wrap_untrusted(job['title'], "JOB_TITLE")}
REQUIRED SKILLS: {wrap_untrusted(requirements['required_skills'], "REQUIRED_SKILLS")}
PREFERRED SKILLS: {wrap_untrusted(requirements['preferred_skills'], "PREFERRED_SKILLS")}
SKILLS THE CANDIDATE IS MISSING: {wrap_untrusted(missing_skills, "MISSING_SKILLS")}

Give 3-4 specific, actionable resume edits for THIS job:
- Which existing experience or projects to emphasize or reword to match the job's language.
- Which real skills the candidate has but should surface more prominently.
- Do NOT invent skills or experience the candidate doesn't have.

Number each suggestion. Keep each to one sentence.
"""
    return logged_llm_call(prompt, run_id, step_id, operation="resume_edit")