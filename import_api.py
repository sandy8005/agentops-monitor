# When building each job from the Remotive API response, map job_type:
def map_remotive_type(job_type):
    jt = (job_type or "").lower().replace("_", "-")
    if "intern" in jt:
        return "internship"
    if "part-time" in jt:
        return "part-time"
    if "contract" in jt or "freelance" in jt:
        return "contract"
    return "full-time"

# then in the INSERT (add employment_type column + value):
#   emp_type = map_remotive_type(job.get("job_type"))
#   INSERT INTO job_postings (title, company, description, location, work_mode, employment_type, source)
#   VALUES (%s, %s, %s, %s, %s, %s, 'api')
#   ... with emp_type as the employment_type value