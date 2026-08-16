from llm import logged_llm_call


def application_strategy(resume_text, job, requirements, missing_skills, run_id, step_id):
    prompt = f"""
You are a career coach. The candidate is considering applying to this job.

CANDIDATE RESUME:
{resume_text}

JOB TITLE: {job['title']}
COMPANY: {job['company']}
REQUIRED SKILLS: {requirements['required_skills']}
PREFERRED SKILLS: {requirements['preferred_skills']}
SKILLS THE CANDIDATE IS MISSING: {missing_skills}

Give a short, practical application strategy in 3-4 sentences:
- What strengths should the candidate emphasize for this specific role?
- How should they address or downplay the missing skills?
- One concrete tip to stand out.

Write plain prose, no headers, no bullet points.
"""
    return logged_llm_call(prompt, run_id, step_id, operation="application_strategy")


def resume_edit_advice(resume_text, job, requirements, missing_skills, run_id, step_id):
    prompt = f"""
You are a resume editor. Suggest concrete edits to tailor this resume for this specific job.

CANDIDATE RESUME:
{resume_text}

JOB TITLE: {job['title']}
REQUIRED SKILLS: {requirements['required_skills']}
PREFERRED SKILLS: {requirements['preferred_skills']}
SKILLS THE CANDIDATE IS MISSING: {missing_skills}

Give 3-4 specific, actionable resume edits for THIS job:
- Which existing experience or projects to emphasize or reword to match the job's language.
- Which real skills the candidate has but should surface more prominently.
- Do NOT invent skills or experience the candidate doesn't have.

Number each suggestion. Keep each to one sentence.
"""
    return logged_llm_call(prompt, run_id, step_id, operation="resume_edit")