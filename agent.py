import time
from input_handler import receive_user_input
from pdf_reader import read_resume_file
from parser import parse_resume
from job_parser import extract_requirements
from scorer import calculate_match_score
from ranker import rank_jobs
from job_source import search_jobs
from advisor import application_strategy, resume_edit_advice
from llm import (
    create_run, create_step, logged_llm_call, logged_tool_call,
    finish_step, finish_run, fail_step, record_score, record_context,
    quota_available
)
from tools import keyword_overlap_tool


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
PREFERRED SKILLS: {requirements['preferred_skills']}
MINIMUM YEARS EXPERIENCE: {requirements['min_years_experience']}

A keyword check found these required skills missing from the resume: {overlap['missing_from_resume']}
And these present: {overlap['matched_in_resume']}

Based on the match between the candidate and this job, respond in this exact format:

Decision: <Apply / Maybe / Skip>
Reason: <one sentence explaining why>
"""
    return prompt


def run_agent():
    # Pre-check: don't waste time if the daily quota is already spent
    if not quota_available():
        print("⚠ API quota exhausted — skipping run. Try again after reset.")
        return

    run_id = create_run("job search run")
    failures = 0

    # Step 0: receive_user_input
    step_id = create_step(run_id, "receive_user_input", 0)
    try:
        user_input = logged_tool_call(
            "receive_user_input", receive_user_input,
            "user_input.json", run_id, step_id
        )
        if user_input is None:
            raise RuntimeError("receive_user_input returned no data")
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        finish_run(run_id, "failed")
        print(f"Input failed: {e}")
        return

    print(f"Target role: {user_input['target_role']}")
    print(f"Location: {user_input['location']}  |  Mode: {user_input['work_mode']}")

    # Step 1: read_resume_file
    step_id = create_step(run_id, "read_resume_file", 1)
    try:
        resume_text = logged_tool_call(
            "read_resume_file", read_resume_file,
            user_input["resume_file"], run_id, step_id
        )
        if not resume_text:
            raise RuntimeError("read_resume_file returned no text")
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        finish_run(run_id, "failed")
        print(f"PDF read failed: {e}")
        return

    print(f"Resume read: {len(resume_text)} characters")

    # Step 2: parse_resume
    step_id = create_step(run_id, "parse_resume", 2)
    try:
        parsed = parse_resume(resume_text, run_id, step_id)
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        finish_run(run_id, "failed")
        print(f"Resume parsing failed: {e}")
        return

    print(f"Parsed: {len(parsed['skills'])} skills, {len(parsed['projects'])} projects")

    # Step 3: search_jobs
    step_id = create_step(run_id, "search_jobs", 3)
    try:
        jobs = logged_tool_call(
            "search_jobs",
            lambda _: search_jobs(user_input["target_role"], user_input["location"]),
            "query", run_id, step_id
        )
        if not jobs:
            raise RuntimeError("search_jobs returned no jobs")
        finish_step(step_id, "success")
    except Exception as e:
        fail_step(step_id, e)
        finish_run(run_id, "failed")
        print(f"Job search failed: {e}")
        return

    print(f"Found {len(jobs)} jobs")
    print("=" * 60)

    results = []

    # Steps 4..N: evaluate each job
    for index, job in enumerate(jobs, start=4):
        time.sleep(15)
        step_id = create_step(run_id, job["title"], index)

        try:
            overlap = logged_tool_call(
                "keyword_overlap_tool", keyword_overlap_tool,
                {"resume": resume_text, "job_description": job["description"]},
                run_id, step_id
            )
            if overlap is None:
                raise RuntimeError("keyword_overlap_tool returned no result")

            requirements = extract_requirements(job, run_id, step_id)

            prompt = build_prompt(resume_text, parsed, job, overlap, requirements)
            result = logged_llm_call(prompt, run_id, step_id)

            # extract the LLM's decision word
            llm_decision = "Unknown"
            for line in result.splitlines():
                if line.lower().startswith("decision:"):
                    llm_decision = line.split(":", 1)[1].strip()
                    break

            # deterministic score
            score_result = calculate_match_score(parsed, requirements, resume_text)

            # record both + flag disagreement
            needs_review = record_score(
                step_id, score_result["score"],
                score_result["decision"], llm_decision
            )

            # record retrieved context (the evidence behind this verdict)
            record_context(step_id, {
                "required_skills": requirements["required_skills"],
                "preferred_skills": requirements["preferred_skills"],
                "min_years_experience": requirements["min_years_experience"],
                "matched_skills": overlap["matched_in_resume"],
                "missing_skills": overlap["missing_from_resume"],
                "candidate_years": parsed["years_experience"]
            })

            results.append({
                "title": job["title"],
                "company": job["company"],
                "score": score_result["score"],
                "decision": score_result["decision"],
                "llm_decision": llm_decision,
                "needs_review": needs_review
            })

            flag = "  ** REVIEW **" if needs_review else ""
            print(f"{job['title']}: LLM={llm_decision}  Score={score_result['score']}({score_result['decision']}){flag}")

            # Steps 10 & 11: only for viable jobs (Apply / Maybe)
            if llm_decision in ("Apply", "Maybe"):
                strategy = application_strategy(
                    resume_text, job, requirements,
                    overlap["missing_from_resume"], run_id, step_id
                )
                edits = resume_edit_advice(
                    resume_text, job, requirements,
                    overlap["missing_from_resume"], run_id, step_id
                )
                print(f"\n  STRATEGY: {strategy.strip()}")
                print(f"\n  RESUME EDITS: {edits.strip()}\n")

            # only mark the step complete once ALL its work (including advice) is done
            finish_step(step_id, "success")

        except Exception as e:
            failures += 1
            fail_step(step_id, e)
            print(f"{job['title']} ({job['company']}): FAILED — {e}")

        print("-" * 60)

    # Step N+1: rank jobs — now a monitored step
    rank_step_id = create_step(run_id, "rank_jobs", len(jobs) + 4)
    try:
        ranked = logged_tool_call(
            "rank_jobs", lambda _: rank_jobs(results),
            "results", run_id, rank_step_id
        )
        if ranked is None:
            ranked = []
        finish_step(rank_step_id, "success")
    except Exception as e:
        fail_step(rank_step_id, e)
        ranked = results  # fall back to unranked so we still print something

    overall_status = "success" if failures == 0 else "completed_with_errors"
    finish_run(run_id, overall_status)
    print(f"Run {run_id} complete. {failures} failure(s).")

    # display ranked results
    print("\n" + "=" * 60)
    print("RANKED JOBS (best match first):")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        flag = "  ** REVIEW **" if r["needs_review"] else ""
        print(f"{i}. {r['title']} ({r['company']})")
        print(f"   Score: {r['score']}/100 ({r['decision']})  |  LLM: {r['llm_decision']}{flag}")


if __name__ == "__main__":
    run_agent()