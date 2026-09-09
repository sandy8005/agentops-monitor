"""
Rule-based requirement extraction — a NO-LLM fallback for extract_requirements.

Used when the LLM is unavailable (quota exhausted / budget spent). It pulls a
best-effort requirements dict from the job description using a skills vocabulary
and simple required-vs-preferred signal phrases. Cruder than the LLM, but it
means a job always gets SOME requirements and never drops out of the pipeline
just because Gemini quota ran out. Zero LLM calls.
"""
import re

# A practical skills vocabulary. Multi-word entries are matched as phrases;
# single tokens are matched whole-word. Extend as needed.
SKILL_VOCAB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "scala", "kotlin", "swift", "r", "sql", "nosql",
    "react", "angular", "vue", "node.js", "node", "django", "flask", "fastapi",
    "spring", "express", ".net", "rails",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "git", "ci/cd", "linux",
    "machine learning", "ml", "deep learning", "ai", "nlp", "computer vision",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "spark", "hadoop", "kafka", "airflow", "etl", "data engineering",
    "rest", "graphql", "microservices", "api", "agile", "scrum",
    "devops", "mlops", "security", "cybersecurity",
]

# Phrases that signal a REQUIRED skill vs a PREFERRED/nice-to-have one.
REQUIRED_SIGNALS = ["required", "must have", "must-have", "requirement",
                    "you have", "you must", "we require", "essential",
                    "minimum qualifications", "you'll need"]
PREFERRED_SIGNALS = ["preferred", "nice to have", "nice-to-have", "bonus",
                     "plus", "a plus", "desirable", "advantageous",
                     "good to have", "ideally"]


def _tokens(text):
    return set(re.findall(r"[a-z0-9\+\#\.]+", text.lower()))


def _skill_in(skill, text_lower, tokens):
    s = skill.lower()
    if " " in s or "." in s or "/" in s:
        return s in text_lower
    return s in tokens


def _years_from_text(text_lower):
    """Find a 'N years' minimum-experience mention, if any."""
    m = re.search(r"(\d+)\+?\s*years?", text_lower)
    return float(m.group(1)) if m else 0.0


def extract_requirements_rule_based(job):
    """
    Best-effort, LLM-free requirements from a job description.
    Returns the same shape as the LLM extractor:
      {required_skills, required_any_of, preferred_skills, min_years_experience, responsibilities}
    Heuristic: a matched skill mentioned near a PREFERRED signal goes to preferred;
    otherwise it's treated as required. Falls back to 'required' when ambiguous.
    """
    desc = (job.get("description") or "") + " " + (job.get("title") or "")
    text_lower = desc.lower()
    tokens = _tokens(text_lower)

    # Which skills appear at all
    present = [s for s in SKILL_VOCAB if _skill_in(s, text_lower, tokens)]

    required, preferred = [], []
    for skill in present:
        # Look at a window around the skill's first mention for a preferred-signal.
        idx = text_lower.find(skill.lower())
        window = text_lower[max(0, idx - 60): idx + 60]
        if any(sig in window for sig in PREFERRED_SIGNALS):
            preferred.append(skill)
        else:
            required.append(skill)

    # De-dupe, keep order
    def _uniq(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return {
        "required_skills": _uniq(required),
        "required_any_of": [],
        "preferred_skills": _uniq(preferred),
        "min_years_experience": _years_from_text(text_lower),
        "responsibilities": [],
    }