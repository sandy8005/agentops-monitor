# test_parser.py
from pdf_reader import read_resume_file
from llm import create_run, create_step, finish_step, finish_run
from parser import parse_resume
import json

resume_text = read_resume_file("SANDEEP_BARIGE_Resume.pdf")

run_id = create_run("resume parsing test")
step_id = create_step(run_id, "parse_resume", 0)
parsed = parse_resume(resume_text, run_id, step_id)
finish_step(step_id)
finish_run(run_id)

print(json.dumps(parsed, indent=2))