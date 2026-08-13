import time
import sys
from input_handler import receive_user_input
from pdf_reader import read_resume_file
from parser import parse_resume
from job_parser import extract_requirements
from scorer import calculate_match_score
from ranker import rank_jobs
from job_source import search_jobs
from advisor import application_strategy, resume_edit_advice
from evaluator import evaluate_decision
from llm import (
    create_run, create_step, logged_llm_call, logged_tool_call,
    finish_step, finish_run, fail_step, record_score, record_context,
    save_evaluation, quota_available, get_connection
)
from tools import keyword_overlap_tool


def normalize_decision(raw):
    """Map an LLM decision to Apply / Maybe / Skip / Unknown, guarding against negation."""
    text = raw.lower().strip()
    negated = any(neg in text for neg in ["not ", "n't", "do not", "don't", "avoid", "shouldn't"])
    words = text.replace(".", " ").replace(",", " ").split()
    if "skip" in words:
        return "Skip"
    if "maybe" in words:
        return "Maybe"
    if "apply" in words:
        return "Skip" if negated else "Apply"
    return "Unknown"


def load_resume_from_db(resume_id):
    """Load a stored resume + its search config from the resumes table."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT resume_text, target_role, location, work_mode, employment_type
        FROM resumes WHERE id = %s
    """, (resume_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"Resume {resume_id} not found")
    return {
        "resume_text": row[0],
        "target_role": row[1],
        "location": row[2] or "",
        "work_mode": row[3] or "",
        "employment_type": row[4] or ""
    }


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


def run_agent(run_id=None, evaluate=False, resume_id=None):
    if not quota_available():
        print("⚠ API quota exhausted — skipping run. Try again after reset.")
        if run_id is not None:
            finish_run(run_id, "failed")
        return

    if run_id is None:
        run_id = create_run("job search run")
    failures = 0

    # --- Load inputs: from DB (uploaded resume) or user_input.json (CLI) ---
    if resume_id is not None:
        step_id = create_step(run_id, "load_resume_from_db", 0)
        try:
            db_resume = load_resume_from_db(resume_id)
            user_input = {
                "resume_file": None,
                "target_role": db_resume["target_role"],
                "location": db_resume["location"],
                "work_mode": db_resume["work_mode"],
                "employment_type": db_resume["employment_type"]
            }
            resume_text = db_resume["resume_text"]
            if not resume_text:
                raise RuntimeError("stored resume has no text")
            finish_step(step_id, "success")
        except Exception as e:
            fail_step(step_id, e)
            finish_run(run_id, "failed")
            print(f"Resume load failed: {e}")
            return
    else:
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

    print(f"Target role: {user_input['target_role']}")
    print(f"Location: {user_input['location']}  |  Mode: {user_input['work_mode']}")
    print(f"Resume: {len(resume_text)} characters")

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
            lambda _: search_jobs(
                user_input["target_role"], user_input["location"], user_input.get("work_mode")
            ),
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
    if evaluate:
        print("(evaluation enabled — LLM-as-judge will grade each decision)")
    print("=" * 60)

    results = []

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

            llm_decision = "Unknown"
            for line in result.splitlines():
                if line.lower().startswith("decision:"):
                    llm_decision = normalize_decision(line.split(":", 1)[1])
                    break

            score_result = calculate_match_score(parsed, requirements, resume_text, job, user_input)

            needs_review = record_score(
                step_id, score_result["score"],
                score_result["decision"], llm_decision
            )

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

            if evaluate:
                try:
                    eval_result = evaluate_decision(resume_text, job, result, run_id, step_id)
                    save_evaluation(run_id, step_id, eval_result)
                    print(f"    eval: rel={eval_result['relevance_score']} "
                          f"faith={eval_result['faithfulness_score']} "
                          f"complete={eval_result['completeness_score']} "
                          f"halluc={eval_result['hallucination_detected']}")
                except Exception as eval_err:
                    print(f"    (evaluation skipped: {eval_err})")

            finish_step(step_id, "success")

        except Exception as e:
            failures += 1
            fail_step(step_id, e)
            print(f"{job['title']} ({job['company']}): FAILED — {e}")

        print("-" * 60)

    rank_step_id = create_step(run_id, "rank_jobs", len(jobs) + 4)
    try:
        ranked = rank_jobs(results)
        if ranked is None:
            raise RuntimeError("rank_jobs returned None")
        logged_tool_call("rank_jobs", lambda _: ranked, "results", run_id, rank_step_id)
        finish_step(rank_step_id, "success")
    except Exception as e:
        failures += 1
        fail_step(rank_step_id, e)
        ranked = results
        print(f"Ranking failed: {e}")

    overall_status = "success" if failures == 0 else "completed_with_errors"
    finish_run(run_id, overall_status)
    print(f"Run {run_id} complete. {failures} failure(s).")

    print("\n" + "=" * 60)
    print("RANKED JOBS (best match first):")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        flag = "  ** REVIEW **" if r["needs_review"] else ""
        print(f"{i}. {r['title']} ({r['company']})")
        print(f"   Score: {r['score']}/100 ({r['decision']})  |  LLM: {r['llm_decision']}{flag}")


if __name__ == "__main__":
    evaluate = "--evaluate" in sys.argv
    run_agent(evaluate=evaluate)