import json
from llm import logged_llm_call
from schemas import Evaluation
from prompt_safety import wrap_untrusted, HARDENING_PREAMBLE, detect_injection


def evaluate_decision(resume_text, job, agent_response, run_id, step_id, budget=None):
    # The evaluator is the component that is supposed to CATCH hallucinations and
    # quality problems, so it must not itself be hijacked. Its inputs are
    # untrusted: resume_text (candidate-uploaded) and the job posting. Log any
    # injection-looking patterns, then fence every untrusted block below.
    flags = detect_injection(f"{resume_text}\n{job.get('title', '')}\n{job.get('description', '')}")
    if flags:
        try:
            from llm import flag_for_review
            flag_for_review(step_id, reason="possible_prompt_injection(evaluator_input)")
        except Exception:
            pass
        print(f"    [prompt-safety] injection-like patterns in evaluator input: {flags}")

    prompt = f"""
{HARDENING_PREAMBLE}

You are an evaluation judge. Grade the AI agent's job recommendation below.

CANDIDATE RESUME:
{wrap_untrusted(resume_text, "RESUME")}

JOB TITLE:
{wrap_untrusted(job['title'], "JOB_TITLE")}
JOB DESCRIPTION:
{wrap_untrusted(job['description'], "JOB_DESCRIPTION")}

THE AGENT'S RECOMMENDATION:
{wrap_untrusted(agent_response, "AGENT_RECOMMENDATION")}

Evaluate the recommendation on these criteria:
- relevance_score (0-10): Does the recommendation directly address THIS candidate and THIS job?
- faithfulness_score (0-10): Is the reasoning grounded in what the resume actually says, with no invented facts?
- completeness_score (0-10): Does the recommendation fully address the decision — a clear verdict AND a reason that covers the key required skills and experience, rather than only part of the picture?
- hallucination_detected (true/false): Does the reasoning claim the candidate HAS a skill or experience that is NOT in the resume? (Saying the candidate LACKS a skill is NOT a hallucination — that is correct reasoning.)
- hallucinated_claims: list any specific skills/facts the agent falsely claimed the candidate has (empty list if none).
- notes: one sentence explaining your evaluation.

Return ONLY valid JSON, no markdown fences, no explanation, in exactly this shape:
{{
  "relevance_score": <0-10>,
  "faithfulness_score": <0-10>,
  "completeness_score": <0-10>,
  "hallucination_detected": <true/false>,
  "hallucinated_claims": ["..."],
  "notes": "one sentence"
}}
"""
    raw = logged_llm_call(prompt, run_id, step_id, budget=budget)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return Evaluation(**data).model_dump()