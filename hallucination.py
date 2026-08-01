def check_hallucination_risk(llm_response, requirements, resume_text):
    """
    Check whether the LLM's reasoning credits the candidate with any
    required/preferred skill that does NOT actually appear in the resume.
    Deterministic — no LLM call.
    """
    resume_lower = resume_text.lower()
    response_lower = llm_response.lower()

    all_job_skills = requirements["required_skills"] + requirements["preferred_skills"]

    # skills the LLM mentioned in its reasoning
    mentioned = [s for s in all_job_skills if s.lower() in response_lower]

    # of those, which are NOT actually in the resume?
    hallucinated = [s for s in mentioned if s.lower() not in resume_lower]

    risk = "HIGH" if hallucinated else "LOW"

    return {
        "risk": risk,
        "mentioned_skills": mentioned,
        "hallucinated_skills": hallucinated
    }