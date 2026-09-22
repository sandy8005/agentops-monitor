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
from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE


def build_prompt(resume_text, parsed, job, overlap, requirements):
    # Everything interpolated below is derived from untrusted input: the parsed
    # resume fields (candidate-uploaded), the job posting (title/company), and the
    # requirements (LLM-extracted from the untrusted description). This is the
    # LIVE job-judge prompt that decides Apply/Maybe/Skip, so each attacker-
    # controllable value is fenced and the hardening preamble is prepended.
    prompt = f"""
{HARDENING_PREAMBLE}

You are a hiring assistant. Compare the candidate below against the job posting.

CANDIDATE SKILLS: {wrap_untrusted(parsed['skills'], "CANDIDATE_SKILLS")}
YEARS OF EXPERIENCE: {wrap_untrusted(parsed['years_experience'], "YEARS_EXPERIENCE")}
EDUCATION: {wrap_untrusted([e['degree'] for e in parsed['education']], "EDUCATION")}
PROJECTS: {wrap_untrusted([{'name': p['name'], 'tech': p['tech']} for p in parsed['projects']], "PROJECTS")}

JOB TITLE: {wrap_untrusted(job['title'], "JOB_TITLE")}
COMPANY: {wrap_untrusted(job['company'], "COMPANY")}
REQUIRED SKILLS: {wrap_untrusted(requirements['required_skills'], "REQUIRED_SKILLS")}
REQUIRED (ANY OF EACH GROUP): {wrap_untrusted(requirements['required_any_of'], "REQUIRED_ANY_OF")}
PREFERRED SKILLS: {wrap_untrusted(requirements['preferred_skills'], "PREFERRED_SKILLS")}
MINIMUM YEARS EXPERIENCE: {wrap_untrusted(requirements['min_years_experience'], "MIN_YEARS_EXPERIENCE")}

A keyword check found these required skills mentioned in the resume: {wrap_untrusted(overlap['matched_in_resume'], "MATCHED_SKILLS")}
And these NOT mentioned at all: {wrap_untrusted(overlap['missing_from_resume'], "MISSING_SKILLS")}

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