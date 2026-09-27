"""
Resume edit suggestions without a paid model. Rules always; Gemini wording is
optional and every model suggestion passes validate_rewrite(). Valid JSON is not
proof of truthful advice.
"""
import json
import re
from typing import List

from pydantic import BaseModel, Field, ValidationError

from skills import (AMBIGUOUS_VARIANTS, SKILL_ALIAS_GROUPS, affirmative_skill_in_text,
                    canonical_skill, skill_mentions)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|\s*[\u2022•|]\s*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*\s*%?|\$\s*\d+")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_INFLATION_WORDS = {"led", "lead", "managed", "architected", "owned", "spearheaded",
                    "mentored", "directed", "supervised", "headed", "founded", "launched"}
# 'go', 'node', 'rest' are ordinary words: "go to market" is not a technology claim.
_ALL_TECH_TERMS = sorted({v for g in SKILL_ALIAS_GROUPS for v in g
                          if v not in AMBIGUOUS_VARIANTS and len(v) > 2 or v in ("ai", "ml", "c#")},
                         key=len, reverse=True)


def _norm_ws(s):
    return " ".join(str(s or "").split())


def resume_passages(resume_text):
    text = str(resume_text or "")
    out, pos = [], 0
    for part in _SENT_SPLIT_RE.split(text):
        p = part.strip()
        if len(p) < 3:
            continue
        off = text.find(p, pos)
        if off < 0:
            off = text.find(p)
        out.append((off, p))
        if off >= 0:
            pos = off + len(p)
    return out


def _passages_for_skill(skill, passages, surface_forms=None, limit=2):
    hits = []
    for off, p in passages:
        aff, _neg = skill_mentions(skill, p, surface_forms)
        if aff:
            hits.append({"source": "resume", "offset": off, "text": p[:300]})
            if len(hits) >= limit:
                break
    return hits


def rule_suggestions(parsed_resume, resume_text, job, requirements, score_result):
    passages = resume_passages(resume_text)
    surface = (requirements or {}).get("_surface_forms") or {}
    out = []

    matched = [s for s in (score_result.get("matched_skills") or []) if " / " not in s]
    for group in [s for s in (score_result.get("matched_skills") or []) if " / " in s]:
        for member in group.split(" / "):
            if affirmative_skill_in_text(member, resume_text, surface.get(member)):
                matched.append(member)
                break
    for skill in matched[:6]:
        ev = _passages_for_skill(skill, passages, surface.get(canonical_skill(skill)))
        if not ev:
            continue
        out.append({"kind": "highlight_skill", "original_text": ev[0]["text"],
                    "suggested_text": f"Make '{skill}' visible in your summary or skills section "
                                      f"and keep the supporting line shown here.",
                    "reason": f"The job requires {skill}, and your resume already shows it.",
                    "evidence": ev, "method": "rules", "status": "validated"})

    wanted = {canonical_skill(s) for s in (requirements or {}).get("required_skills", [])}
    wanted |= {canonical_skill(s) for s in (requirements or {}).get("preferred_skills", [])}
    for g in (requirements or {}).get("required_any_of", []) or []:
        wanted |= {canonical_skill(s) for s in g}
    for proj in (parsed_resume or {}).get("projects") or []:
        if not isinstance(proj, dict) or not proj.get("name"):
            continue
        overlap = [t for t in (proj.get("tech") or []) if canonical_skill(t) in wanted]
        overlap = [t for t in overlap if affirmative_skill_in_text(t, resume_text)]   # R10
        if not overlap:
            continue
        name = str(proj["name"])
        name_ev = [{"source": "resume", "offset": off, "text": p[:300]}
                   for off, p in passages if name.lower() in p.lower()][:1]
        out.append({"kind": "reorder_project",
                    "original_text": name_ev[0]["text"] if name_ev else name,
                    "suggested_text": f"Move the project '{name}' higher and name "
                                      f"{', '.join(overlap)} in its first line.",
                    "reason": f"It uses {', '.join(overlap)}, which this job asks for.",
                    "evidence": name_ev, "method": "rules",
                    "status": "validated" if name_ev else "needs_confirmation",
                    "validation_notes": None if name_ev else "project name not found verbatim in the resume text"})

    negated = set(score_result.get("negated_skills") or [])
    for gap in (score_result.get("missing_skills") or [])[:6]:
        is_neg = any(canonical_skill(n) == canonical_skill(gap) for n in negated)
        out.append({"kind": "gap", "original_text": None,
                    "suggested_text": (f"Your resume states you do not have {gap} experience; "
                                       f"do not add it unless that has changed."
                                       if is_neg else
                                       f"{gap} is required but not evidenced in your resume. Add it "
                                       f"only if you have real experience with it — confirm first."),
                    "reason": "Required by the job; no supporting passage was found.",
                    "evidence": [], "method": "rules", "status": "needs_confirmation"})
    return out


class _Rewrite(BaseModel):
    original_text: str = Field(..., min_length=3, max_length=400)
    suggested_text: str = Field(..., min_length=3, max_length=400)
    reason: str = Field(..., min_length=3, max_length=300)


class _RewriteList(BaseModel):
    rewrites: List[_Rewrite] = Field(default_factory=list, max_length=5)


def _tech_terms_in(text):
    low = f" {_norm_ws(text).lower()} "
    found = set()
    for t in _ALL_TECH_TERMS:
        if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9+#])", low):
            found.add(canonical_skill(t))
    return found


def validate_rewrite(original, suggested, resume_text, job_terms=()):
    """(status, notes): rejected / needs_confirmation / validated."""
    notes = []
    norm_resume = _norm_ws(resume_text)
    if _norm_ws(original) not in norm_resume:
        return "rejected", "original_text is not a verbatim passage of the resume"

    new_numbers = {n.strip() for n in set(_NUMBER_RE.findall(suggested))
                   - set(_NUMBER_RE.findall(original)) if n.strip()}
    if new_numbers:
        return "rejected", f"adds numbers not in the original: {sorted(new_numbers)[:5]}"
    new_years = set(_YEAR_RE.findall(suggested)) - set(_YEAR_RE.findall(original))
    if new_years:
        return "rejected", f"adds dates not in the original: {sorted(new_years)}"

    candidate_terms = _tech_terms_in(suggested) | {
        canonical_skill(t) for t in job_terms
        if re.search(r"(?<![a-z0-9])" + re.escape(str(t).lower()) + r"(?![a-z0-9+#])",
                     suggested.lower())}
    orig_terms = _tech_terms_in(original) | {
        canonical_skill(t) for t in job_terms if str(t).lower() in original.lower()}
    for term in sorted(candidate_terms - orig_terms):
        if not affirmative_skill_in_text(term, resume_text):
            return "rejected", f"introduces technology not evidenced in the resume: {term}"
        notes.append(f"moves '{term}' into this line (evidenced elsewhere in the resume)")

    orig_words = set(re.findall(r"[a-z]+", original.lower()))
    inflated = [w for w in re.findall(r"[a-z]+", suggested.lower())
                if w in _INFLATION_WORDS and w not in orig_words]
    if inflated:
        notes.append(f"adds responsibility wording not in the original: {sorted(set(inflated))}")
        return "needs_confirmation", "; ".join(notes)

    resume_low = norm_resume.lower()
    proper = []
    for m in re.finditer(r"[A-Z][a-zA-Z0-9&]{2,}", suggested):
        before = suggested[:m.start()].rstrip()
        if not before or before[-1] in ".!?:;-(\u2022":
            continue                     # sentence-initial capital, not a name
        if m.group(0).lower() not in resume_low:
            proper.append(m.group(0))
    if proper:
        return "rejected", f"introduces names not in the resume: {sorted(set(proper))[:5]}"
    return ("needs_confirmation" if notes else "validated"), ("; ".join(notes) or None)


def gemini_rewrites(verified_passages, job, requirements, run_id, step_id, budget):
    """Only verified passages are sent — never the full resume."""
    from llm import logged_llm_call
    from prompt_safety import HARDENING_PREAMBLE, wrap_untrusted
    facts = [p["text"] for p in verified_passages][:8]
    prompt = f"""{HARDENING_PREAMBLE}

You improve resume WORDING only. Rewrite up to 3 of the candidate's existing lines so
they are clearer for the job below. Rules:
- original_text MUST be copied exactly from CANDIDATE LINES.
- Do not add technologies, employers, dates, numbers, metrics, titles or
  responsibilities that the original line does not already state.
- Return ONLY JSON: {{"rewrites": [{{"original_text": "...", "suggested_text": "...", "reason": "..."}}]}}

CANDIDATE LINES: {wrap_untrusted(json.dumps(facts), "RESUME_LINES")}
JOB TITLE: {wrap_untrusted(job.get("title", ""), "JOB_TITLE")}
REQUIRED SKILLS: {wrap_untrusted(json.dumps((requirements or {}).get("required_skills", [])), "REQUIRED")}
"""
    raw = logged_llm_call(prompt, run_id, step_id, operation="resume_rewrite", budget=budget)
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    return _RewriteList(**json.loads(cleaned)).rewrites


def build_suggestions(parsed_resume, resume_text, job, requirements, score_result,
                      use_llm=False, run_id=None, step_id=None, budget=None):
    """Returns (suggestions, generation_note). Gemini failure never loses rules output."""
    suggestions = rule_suggestions(parsed_resume, resume_text, job, requirements, score_result)
    note = "rules"
    if not use_llm:
        return suggestions, note
    verified = [e for s in suggestions for e in (s.get("evidence") or [])]
    if not verified:
        return suggestions, "rules (no verified passages to reword)"
    try:
        rewrites = gemini_rewrites(verified, job, requirements, run_id, step_id, budget)
        job_terms = list((requirements or {}).get("required_skills", [])) + \
            list((requirements or {}).get("preferred_skills", []))
        for rw in rewrites:
            status, vnotes = validate_rewrite(rw.original_text, rw.suggested_text,
                                              resume_text, job_terms)
            off = _norm_ws(resume_text).find(_norm_ws(rw.original_text))
            suggestions.append({"kind": "rewrite", "original_text": rw.original_text,
                                "suggested_text": rw.suggested_text, "reason": rw.reason,
                                "evidence": [{"source": "resume", "offset": off,
                                              "text": rw.original_text}],
                                "method": "gemini", "status": status,
                                "validation_notes": vnotes})
        note = "rules + gemini wording"
    except (ValidationError, ValueError) as e:
        note = f"rules (gemini output invalid: {type(e).__name__})"
    except Exception as e:
        note = f"rules (gemini unavailable: {type(e).__name__})"
    return suggestions, note


def advice_summary(job, suggestions, note):
    ok = [s for s in suggestions if s["status"] == "validated"]
    confirm = [s for s in suggestions if s["status"] == "needs_confirmation"]
    lines = [f"RESUME SUGGESTIONS for {job.get('title', '')} ({note})"]
    for i, s in enumerate(ok, 1):
        lines.append(f"{i}. {s['suggested_text']}  — {s['reason']}")
    if confirm:
        lines.append("NEEDS YOUR CONFIRMATION:")
        for s in confirm:
            lines.append(f"- {s['suggested_text']}")
    return "\n".join(lines)