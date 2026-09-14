"""
Legacy monolithic agent, retired.

The old run_agent() loop and its CLI entrypoint were replaced by the autonomous
planner/router and, now, the LangGraph runner (autonomous_graph.run_agent_graph,
which /runs dispatches). Those dead paths, plus load_resume_from_db (superseded by
router.load_resume) and their now-unused imports, have been removed.

What remains are the TWO helpers still used by the live pipeline:
  - build_prompt():   the job-judge prompt, imported by router.do_process_job.
  - parse_decision(): parse the judge's JSON verdict; used by test_decision_parse.
Keep this module small and focused on those two.
"""
import json

from schemas import JobDecision


def build_prompt(resume_text, parsed, job, overlap, requirements):
    prompt = f"""
You are a hiring assistant. Compare the candidate below against the job posting.

CANDIDATE SKILLS: {parsed['skills']}
YEARS OF EXPERIENCE: {parsed['years_experience']}
EDUCATION: {[e['degree'] for e in parsed['education']]}
PROJECTS: {[{'name': p['name'], 'tech': p['tech']} for p in parsed['projects']]}

JOB TITLE: {job['title']}
COMPANY: {job['company']}
REQUIRED SKILLS: {requirements['required_skills']}
REQUIRED (ANY OF EACH GROUP): {requirements['required_any_of']}
PREFERRED SKILLS: {requirements['preferred_skills']}
MINIMUM YEARS EXPERIENCE: {requirements['min_years_experience']}

A keyword check found these required skills mentioned in the resume: {overlap['matched_in_resume']}
And these NOT mentioned at all: {overlap['missing_from_resume']}

IMPORTANT — the keyword check only confirms whether a term APPEARS in the resume
text. It does NOT confirm real or deep experience. A skill listed in a flat
skills section is weaker evidence than one demonstrated through projects, job
responsibilities, or years of use. When you decide, weigh HOW the resume actually
uses each skill (in real projects and roles, with duration and depth) rather than
mere presence. Treat a bare mention with appropriate caution.

Based on the match between the candidate and this job, respond with ONLY valid JSON
(no markdown fences, no extra text) in exactly this shape:
{{
  "decision": "Apply",
  "reason": "one sentence explaining why"
}}
The "decision" field MUST be exactly one of: "Apply", "Maybe", or "Skip".
"""
    return prompt


def parse_decision(raw):
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    parsed = JobDecision(**json.loads(cleaned))
    return parsed.decision.value