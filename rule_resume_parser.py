"""
Deterministic resume extraction — ZERO model calls (N01).

Used when the run's model policy is "rules_only", when no GEMINI_API_KEY is
configured, when the quota breaker is open, when the run's LLM budget is spent,
or when the model parse fails. It extracts only what the text states:

  skills            vocabulary terms that the resume mentions AFFIRMATIVELY
                    (negated / aspirational mentions are excluded), each with
                    the verbatim passage that supports it
  years_experience  ONLY from an explicit statement such as "5 years of
                    experience"; otherwise None + experience_unknown=True.
                    Unknown is never turned into 0 or into "sufficient".
  projects          lines under a "Projects" heading (or "Project: X ..."),
                    with technologies mentioned IN THAT PROJECT'S OWN TEXT

Education and itemized experience are not guessed. The result carries
extraction_method="rule_based" so every consumer can see it is degraded.
"""
import re

from rule_requirements import SKILL_VOCAB
from skills import (SKILL_ALIAS_GROUPS, canonical_skill, skill_mention_states,
                    AMBIGUOUS_VARIANTS)

# Vocabulary: rule_requirements terms + canonical alias heads. Ambiguous short
# words ("go", "node", "rest", "r", "ai", "ml", "api") are excluded from free-text
# resume scanning because they collide with ordinary words.
_EXCLUDE = AMBIGUOUS_VARIANTS | {"r", "ai", "ml", "api", "plus"}
_VOCAB = sorted({canonical_skill(v) for v in SKILL_VOCAB if v not in _EXCLUDE}
                | {g[0] for g in SKILL_ALIAS_GROUPS if g[0] not in _EXCLUDE},
                key=len, reverse=True)

_YEARS_RE = re.compile(
    r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?"
    r"(?:professional\s+|industry\s+|work\s+|hands-on\s+|relevant\s+)?experience", re.I)
_HEADING_RE = re.compile(
    r"^\s*(projects?|personal projects|academic projects|experience|work experience|"
    r"professional experience|employment|education|skills|technical skills|"
    r"certifications?|summary|profile|awards|publications|languages|interests)\s*:?\s*$",
    re.I)
_PROJECT_LINE_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?"
    r"([A-Za-z0-9][\w.&+#' \-]{1,40}?)\s*(?::|\s[-\u2013\u2014|]\s|\s+(?:using|with|built with)\b)")
_NEG_YEARS_CUE = re.compile(r"\b(no|not|never|without|lack)\b[^.\n]{0,20}$", re.I)


def _lines(text):
    return [ln.rstrip() for ln in str(text or "").splitlines()]


def _passages(text):
    out = []
    for ln in _lines(text):
        for part in re.split(r"(?<=[.!?])\s+", ln):
            p = part.strip()
            if len(p) >= 3:
                out.append(p)
    return out


def extract_skills(text):
    """[(canonical_skill, evidence_passage)] for affirmatively mentioned skills."""
    found = {}
    passages = _passages(text)
    for skill in _VOCAB:
        for p in passages:
            st = skill_mention_states(skill, p)
            if st["affirmative"]:
                found.setdefault(skill, p[:300])
                break
    return sorted(found.items())


def extract_years(text):
    """Explicitly stated total years, or None. The largest explicit, non-negated
    statement is used; any value outside 0-50 is ignored."""
    best = None
    for m in _YEARS_RE.finditer(str(text or "")):
        prefix = str(text)[max(0, m.start() - 30):m.start()]
        if _NEG_YEARS_CUE.search(prefix):
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 0 <= v <= 50:
            best = v if best is None else max(best, v)
    return best


def extract_projects(text):
    """[{'name', 'tech', 'evidence'}] — tech only from the project's own lines."""
    lines = _lines(text)
    projects = []
    in_projects = False
    current = None
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        h = _HEADING_RE.match(stripped)
        if h:
            in_projects = h.group(1).lower().startswith(("project", "personal project",
                                                         "academic project"))
            current = None
            continue
        explicit = re.match(r"^\s*project\s*[:\-]", stripped, re.I)
        if in_projects or explicit:
            candidate = re.sub(r"^\s*project\s*[:\-]\s*", "", stripped, flags=re.I) \
                if explicit else stripped
            m = _PROJECT_LINE_RE.match(candidate)
            is_bullet = stripped[:1] in "-*\u2022" and current is not None and not explicit
            if m and not is_bullet:
                current = {"name": m.group(1).strip(), "lines": [stripped]}
                projects.append(current)
                continue
            if current is not None:
                current["lines"].append(stripped)
    out = []
    for p in projects:
        body = "\n".join(p["lines"])
        tech = [s for s, _ in extract_skills(body)]
        out.append({"name": p["name"], "tech": tech, "evidence": p["lines"][0][:300]})
    return out


def parse_resume_rules(resume_text):
    skills = extract_skills(resume_text)
    years = extract_years(resume_text)
    return {
        "skills": [s for s, _ in skills],
        "years_experience": years,
        "experience_unknown": years is None,
        "education": [],
        "projects": [{"name": p["name"], "tech": p["tech"]} for p in extract_projects(resume_text)],
        "experience": [],
        "skill_evidence": [{"skill": s, "evidence": ev} for s, ev in skills],
        "grounded_skills": [s for s, _ in skills],
        "ungrounded_skills": [],
        "extraction_method": "rule_based",
    }