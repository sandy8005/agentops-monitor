def keyword_overlap_tool(data):
    resume_text = data["resume"].lower()
    job_description = data["job_description"].lower()

    SKILLS = [
        "Python", "Flask", "Django", "FastAPI", "PostgreSQL", "SQL",
        "Docker", "Git", "AWS", "Kubernetes", "Terraform", "React",
        "PyTorch", "NLP", "Redis"
    ]

    required = [s for s in SKILLS if s.lower() in job_description]
    matched = [s for s in required if s.lower() in resume_text]
    missing = [s for s in required if s not in matched]

    return {
        "required_skills_in_job": required,
        "matched_in_resume": matched,
        "missing_from_resume": missing
    }