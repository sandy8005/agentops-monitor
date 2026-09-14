# test_parser_anyof.py
from job_parser import extract_requirements
from llm import create_run, create_step

job = {
    "title": "Backend Engineer",
    "description": "Required: Python and 3+ years experience. "
                   "You must know Flask or Django. Bonus: Docker."
}
run_id = create_run("test parser any_of")
step_id = create_step(run_id, "extract", 1)
reqs = extract_requirements(job, run_id, step_id)
print("required_skills:", reqs["required_skills"])
print("required_any_of:", reqs["required_any_of"])
print("preferred_skills:", reqs["preferred_skills"])
# expect: python in required_skills, ["Flask","Django"] as a group in required_any_of, Docker in preferred