import json
from llm import logged_llm_call
from schemas import Evaluation


def evaluate_decision(resume_text, job, agent_response, run_id, step_id):
    prompt = f"""
You are an evaluation judge. Grade the AI agent's job recommendation below.

CANDIDATE RESUME:
{resume_text}

JOB TITLE: {job['title']}
JOB DESCRIPTION:
{job['description']}

THE AGENT'S RECOMMENDATION:
{agent_response}

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
    raw = logged_llm_call(prompt, run_id, step_id, operation="evaluation")
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return Evaluation(**data).model_dump()