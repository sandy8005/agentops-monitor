# test_job_parser.py
from jobs import JOBS
from llm import create_run, create_step, finish_step, finish_run
from job_parser import extract_requirements
import json

run_id = create_run("job requirement extraction test")

for job in JOBS:   # just the first job for now
    step_id = create_step(run_id, f"extract_{job['title']}", 0)
    reqs = extract_requirements(job, run_id, step_id)
    finish_step(step_id)
    print(f"\n{job['title']}:")
    print(json.dumps(reqs, indent=2))

finish_run(run_id)