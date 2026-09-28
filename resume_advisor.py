"""
Resume edit suggestions that do NOT require a paid model.

Pipeline:
  1. verified_facts()      — resume passages (with offsets) that support each claim.
                             Unknown stays unknown: nothing is inferred.
  2. rule_suggestions()    — highlight supported required skills, move relevant
                             existing projects up, flag requirements with no evidence.
  3. gemini_rewrites()     — OPTIONAL wording improvements of EXISTING sentences,
                             sent only the relevant verified passages.
  4. validate_rewrite()    — rejects invented technologies, numbers, dates, employers
                             and inflated responsibilities; uncertain changes are
                             marked needs_confirmation. Valid JSON is not proof of
                             truthful advice, so every model suggestion goes through it.

Every suggestion carries: kind, original_text, suggested_text, reason, evidence
(source passages), method ('rules' | 'gemini') and status ('validated' |
'needs_confirmation' | 'rejected').
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
_FUNCTION_WORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
                   "by", "at", "as", "using", "via", "from", "into", "that", "which",
                   "its", "their", "this", "these", "my", "our"}
_POLARITY_RE = re.compile(r"(?<![a-z])(no|not|never|without|none|n't)(?![a-z])")


def _has_negation(text):
    return bool(_POLARITY_RE.search(str(text or "").lower()))


_INFLATION_WORDS = {"led", "lead", "managed", "architected", "owned", "spearheaded",
                    "mentored", "directed", "supervised", "headed", "founded", "launched"}
# Short aliases that are ordinary English words ('go', 'node', 'rest') are excluded:
# "go to market" must not be read as a new technology claim.
_ALL_TECH_TERMS = sorted({v for g in SKILL_ALIAS_GROUPS for v in g
                          if v not in AMBIGUOUS_VARIANTS and len(v) > 2 or v in ("ai", "ml", "c#")},
                         key=len, reverse=True)


def _norm_ws(s):
    return " ".join(str(s or "").split())


def resume_passages(resume_text):
    """[(offset, sentence)] — the resume split into citable passages."""
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
    """Passages that AFFIRMATIVELY mention the skill (negated mentions excluded)."""
    hits = []
    for off, p in passages:
        aff, _neg = skill_mentions(skill, p, surface_forms)
        if aff:
            hits.append({"source": "resume", "offset": off, "text": p[:300]})
            if len(hits) >= limit:
                break
    return hits


_HEADING_WORDS = {"projects", "project", "experience", "work experience", "education",
                  "skills", "technical skills", "certifications", "summary", "profile"}


def _project_window(name, all_names, passages, max_extra=3):
    """[(offset, passage)] belonging to project `name`: the passage that names it
    plus up to `max_extra` following passages, stopping at another project's name
    or a section heading. Empty if the name does not occur in the resume."""
    low = name.lower()
    others = [n.lower() for n in all_names if n.lower() != low]
    start = next((i for i, (_, t) in enumerate(passages) if low in t.lower()), None)
    if start is None:
        return []
    window = [passages[start]]
    for off, t in passages[start + 1:start + 1 + max_extra]:
        tl = t.lower().strip(" :")
        if tl in _HEADING_WORDS or any(o in tl for o in others):
            break
        window.append((off, t))
    return window


def rule_suggestions(parsed_resume, resume_text, job, requirements, score_result):
    """Deterministic suggestions; each one cites resume evidence or is explicitly a
    gap the user must confirm. Never adds a claim."""
    passages = resume_passages(resume_text)
    surface = (requirements or {}).get("_surface_forms") or {}
    out = []

    matched = [s for s in (score_result.get("matched_skills") or []) if " / " not in s]
    for group in [s for s in (score_result.get("matched_skills") or []) if " / " in s]:
        # any-of group: highlight the member the resume actually evidences
        for member in group.split(" / "):
            if affirmative_skill_in_text(member, resume_text, surface.get(member)):
                matched.append(member)
                break
    for skill in matched[:6]:
        ev = _passages_for_skill(skill, passages, surface.get(canonical_skill(skill)))
        if not ev:
            continue
        out.append({
            "kind": "highlight_skill",
            "original_text": ev[0]["text"],
            "suggested_text": f"Make '{skill}' visible in your summary or skills section "
                              f"and keep the supporting line shown here.",
            "reason": f"The job requires {skill}, and your resume already shows it.",
            "evidence": ev, "method": "rules", "status": "validated",
        })

    wanted = {canonical_skill(s) for s in (requirements or {}).get("required_skills", [])}
    wanted |= {canonical_skill(s) for s in (requirements or {}).get("preferred_skills", [])}
    for g in (requirements or {}).get("required_any_of", []) or []:
        wanted |= {canonical_skill(s) for s in g}
    projects = [p for p in ((parsed_resume or {}).get("projects") or [])
                if isinstance(p, dict) and p.get("name")]
    all_names = [str(p["name"]) for p in projects]
    for proj in projects:
        name = str(proj["name"])
        window = _project_window(name, all_names, passages)
        if not window:
            continue                    # project not locatable in the text: no claim
        window_text = "\n".join(t for _, t in window)
        # The technology must be evidenced IN THIS PROJECT'S OWN TEXT (N02/R10). A
        # model-supplied tech label, or the same word in another project, is not
        # evidence for this one.
        overlap = [t for t in (proj.get("tech") or [])
                   if canonical_skill(t) in wanted
                   and affirmative_skill_in_text(t, window_text)]
        if not overlap:
            continue
        ev = [{"source": "resume", "offset": off, "text": t[:300]} for off, t in window]
        out.append({
            "kind": "reorder_project",
            "original_text": window[0][1][:300],
            "suggested_text": f"Move the project '{name}' higher and name "
                              f"{', '.join(overlap)} in its first line.",
            "reason": f"Its own description mentions {', '.join(overlap)}, which this job asks for.",
            "evidence": ev, "method": "rules", "status": "validated",
        })

    negated = set(score_result.get("negated_skills") or [])
    for gap in (score_result.get("missing_skills") or [])[:6]:
        is_neg = any(canonical_skill(n) == canonical_skill(gap) for n in negated)
        out.append({
            "kind": "gap",
            "original_text": None,
            "suggested_text": (f"Your resume states you do not have {gap} experience; "
                               f"do not add it unless that has changed."
                               if is_neg else
                               f"{gap} is required but not evidenced in your resume. Add it "
                               f"only if you have real experience with it — confirm first."),
            "reason": "Required by the job; no supporting passage was found.",
            "evidence": [], "method": "rules", "status": "needs_confirmation",
        })
    return out


# ------------------------------------------------------------ Gemini (optional) --

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
    """(status, notes). 'rejected' for invented facts; 'needs_confirmation' for
    changes a human must check; 'validated' otherwise."""
    notes = []
    norm_resume = _norm_ws(resume_text)
    if _norm_ws(original) not in norm_resume:
        return "rejected", "original_text is not a verbatim passage of the resume"

    new_numbers = set(_NUMBER_RE.findall(suggested)) - set(_NUMBER_RE.findall(original))
    new_numbers = {n.strip() for n in new_numbers if n.strip()}
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

    # Polarity must not change: "Never built X" -> "Built X" inverts the claim (N02).
    if _has_negation(original) != _has_negation(suggested):
        return "rejected", "changes a negative statement into a positive one (or vice versa)"

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
    # Keyword checks cannot prove a new sentence keeps the original facts. Only a
    # rewrite that uses NO new content words (reordering / trimming the original's
    # own words) is auto-validated; everything else is a draft the user confirms.
    orig_content = set(re.findall(r"[a-z][a-z0-9+#]*", original.lower()))
    new_words = sorted({w for w in re.findall(r"[a-z][a-z0-9+#]*", suggested.lower())
                        if w not in orig_content and w not in _FUNCTION_WORDS})
    if new_words:
        notes.append(f"new wording to confirm: {new_words[:8]}")
        return "needs_confirmation", "; ".join(notes)
    return ("needs_confirmation" if notes else "validated"), ("; ".join(notes) or None)


def gemini_rewrites(verified_passages, job, requirements, run_id, step_id, budget):
    """Ask the model to reword up to 3 EXISTING passages. Returns a list of raw dicts
    (unvalidated) or raises. Only verified passages are sent — never the full resume."""
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
    """Rules always; Gemini wording optionally. Returns (suggestions, generation_note).
    A Gemini failure never loses the rule-based suggestions."""
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
        sent = {_norm_ws(p["text"]).lower() for p in verified}
        for rw in rewrites:
            if _norm_ws(rw.original_text).lower() not in sent:
                status, vnotes = "rejected", "original_text was not one of the verified passages"
            else:
                status, vnotes = validate_rewrite(rw.original_text, rw.suggested_text,
                                                  resume_text, job_terms)
            off = _norm_ws(resume_text).find(_norm_ws(rw.original_text))
            suggestions.append({
                "kind": "rewrite", "original_text": rw.original_text,
                "suggested_text": rw.suggested_text, "reason": rw.reason,
                "evidence": [{"source": "resume", "offset": off, "text": rw.original_text}],
                "method": "gemini", "status": status, "validation_notes": vnotes,
            })
        note = "rules + gemini wording"
    except (ValidationError, ValueError) as e:
        note = f"rules (gemini output invalid: {type(e).__name__})"
    except Exception as e:     # budget, quota, outage — explicit, not silent
        note = f"rules (gemini unavailable: {type(e).__name__})"
    return suggestions, note


def advice_summary(job, suggestions, note):
    """Human-readable summary stored in run_advice (kept for the existing UI)."""
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