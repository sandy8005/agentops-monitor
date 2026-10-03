"""
The goal of an autonomous run, the user's FIXED constraints, and the hard limits.

The controller may choose which permitted tool to call next and may adapt the
SEARCH TITLE (e.g. "AI Engineer" -> "Machine Learning Engineer"). It may never
change anything in FixedConstraints: location, work mode, employment type and
seniority are applied by the backend to every search, and a query that tries to
change seniority is rejected. Work authorization is out of scope — nothing here
claims to have checked it.
"""
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Seniority = Literal["", "intern", "entry", "junior", "mid", "senior"]
# Live providers only. The "pool" practice source was retired (migration 0013
# retires every stored goal that still names it).
Provider = Literal["adzuna", "remotive"]
LIVE_PROVIDERS = ("adzuna", "remotive")

# Seniority words that may appear in a job TITLE, used to (a) reject a search
# query that silently changes seniority and (b) mark discovered jobs whose title
# contradicts the requested level as ineligible.
SENIORITY_WORDS = {
    "intern": {"intern", "internship"},
    "entry": {"entry", "entry-level", "graduate", "new grad", "junior", "jr", "associate", "i"},
    "junior": {"junior", "jr", "entry", "entry-level", "graduate", "associate"},
    "mid": {"mid", "mid-level", "ii", "intermediate"},
    "senior": {"senior", "sr", "lead", "principal", "staff", "iii", "head", "director"},
}
_ABOVE = {  # title words that are ABOVE the requested level (-> ineligible)
    "intern": SENIORITY_WORDS["senior"] | SENIORITY_WORDS["mid"] | {"manager"},
    "entry": SENIORITY_WORDS["senior"] | SENIORITY_WORDS["mid"] | {"manager"},
    "junior": SENIORITY_WORDS["senior"] | SENIORITY_WORDS["mid"] | {"manager"},
    "mid": SENIORITY_WORDS["senior"] | {"manager"},
    "senior": set(),
}
_BELOW = {  # title words BELOW the requested level (-> ineligible)
    "senior": SENIORITY_WORDS["intern"] | {"junior", "jr", "entry", "entry-level", "graduate"},
    "mid": SENIORITY_WORDS["intern"],
    "entry": SENIORITY_WORDS["intern"],
    "junior": SENIORITY_WORDS["intern"],
    "intern": set(),
}

_QUERY_ALLOWED_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 +#./&\-]{1,79}$")


def normalize_query(q):
    """Canonical form for duplicate detection: lower-case, single spaces."""
    return " ".join(str(q or "").lower().split())


def title_words(title):
    return set(re.findall(r"[a-z]+(?:-[a-z]+)?", str(title or "").lower()))


def seniority_conflict(text, seniority):
    """Return the conflicting word if `text` (a title or query) names a seniority
    level incompatible with the requested one, else None."""
    if not seniority:
        return None
    words = title_words(text)
    joined = " " + " ".join(str(text or "").lower().split()) + " "
    for w in sorted(_ABOVE.get(seniority, set()) | _BELOW.get(seniority, set())):
        if (" " in w and f" {w} " in joined) or w in words:
            # Roman-numeral "i" is too ambiguous in free text; only match as a suffix.
            if w == "i" and not re.search(r"\bi\s*$", str(text or "").lower()):
                continue
            return w
    return None


class FixedConstraints(BaseModel):
    """Set by the user. The controller can read these but never change them."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    location: str = Field("", max_length=200)
    work_mode: Literal["", "remote", "hybrid", "onsite"] = ""
    employment_type: Literal["", "full-time", "part-time", "contract", "internship"] = ""
    seniority: Seniority = ""


class Limits(BaseModel):
    """Hard stop conditions, all enforced by the backend before every tool call."""
    model_config = ConfigDict(extra="forbid")
    max_iterations: int = Field(20, ge=1, le=60)
    max_searches: int = Field(6, ge=1, le=20)
    max_llm_calls: int = Field(40, ge=0, le=200)
    max_runtime_seconds: int = Field(1800, ge=60, le=4 * 3600)
    max_cost_usd: float = Field(0.50, ge=0.0, le=20.0)
    no_progress_limit: int = Field(3, ge=1, le=10)
    max_jobs_per_search: int = Field(25, ge=1, le=50)
    max_evaluate_batch: int = Field(5, ge=1, le=10)


class AgentGoal(BaseModel):
    """e.g. 'Find 10 suitable entry-level AI roles'."""
    model_config = ConfigDict(extra="forbid")
    description: str = Field("", max_length=300)
    target_role: str = Field(..., min_length=2, max_length=80)
    # Titles the user explicitly allows the controller to try. The controller may
    # also propose its own title variants; those are validated the same way.
    alternative_titles: List[str] = Field(default_factory=list, max_length=8)
    target_count: int = Field(10, ge=1, le=50)
    # A job counts toward the goal when its AUTHORITATIVE final decision (human >
    # judge > score) is in this set and it is not awaiting review.
    qualifying_decisions: List[Literal["Apply", "Maybe"]] = Field(default_factory=lambda: ["Apply"])
    providers: List[Provider] = Field(default_factory=lambda: ["adzuna", "remotive"])
    constraints: FixedConstraints = Field(default_factory=FixedConstraints)
    limits: Limits = Field(default_factory=Limits)
    evaluate_quality: bool = False
    # What to do when the controller model is unavailable (quota, outage):
    #   "rules" -> continue with the deterministic policy, recorded as such
    #   "pause" -> stop and ask the user (never silently claims autonomy)
    on_model_unavailable: Literal["rules", "pause"] = "rules"
    use_llm_controller: bool = True
    use_llm_advice: bool = False
    # "rules_only": no model call anywhere in the run (parse, requirements, judge,
    # evaluation, controller, advice). Forces the two flags above off.
    model_policy: Literal["auto", "rules_only"] = "auto"
    # When True, a match counts toward the goal only if its requirements came from
    # the model, the judge ran, or a human reviewed it. Rule-based-only matches are
    # always REPORTED separately as unverified either way.
    require_verified_matches: bool = False

    @field_validator("target_role", "alternative_titles", mode="after")
    @classmethod
    def _titles_are_plain(cls, v):
        items = v if isinstance(v, list) else [v]
        for t in items:
            if not _QUERY_ALLOWED_RE.match(t.strip()):
                raise ValueError(f"title {t!r} must be 2-80 plain characters")
        return v

    @model_validator(mode="after")
    def _titles_respect_seniority(self):
        s = self.constraints.seniority
        for t in [self.target_role] + list(self.alternative_titles):
            w = seniority_conflict(t, s)
            if w:
                raise ValueError(f"title {t!r} contradicts the requested seniority ({w!r} vs {s!r})")
        if not self.providers:
            raise ValueError("at least one provider is required")
        if self.model_policy == "rules_only":
            object.__setattr__(self, "use_llm_controller", False)
            object.__setattr__(self, "use_llm_advice", False)
            object.__setattr__(self, "evaluate_quality", False)
        return self


def validate_search_query(query, goal: AgentGoal):
    """Return (ok, normalized_query, reason). Queries are plain job titles only:
    they cannot carry location, seniority changes, or instruction-like text."""
    q = " ".join(str(query or "").split())
    if not _QUERY_ALLOWED_RE.match(q):
        return False, None, "query must be 2-80 characters of letters, digits, spaces and + # . / & -"
    if len(q.split()) > 8:
        return False, None, "query must be a job title (at most 8 words)"
    w = seniority_conflict(q, goal.constraints.seniority)
    if w:
        return False, None, f"query changes the fixed seniority constraint ({w!r})"
    loc = normalize_query(goal.constraints.location)
    if loc and loc in normalize_query(q):
        return False, None, "location is a fixed constraint applied by the backend; do not put it in the query"
    for bad in ("remote", "onsite", "on-site", "hybrid", "contract", "part-time", "full-time",
                "internship", "anywhere", "worldwide"):
        if bad in title_words(q) or bad in normalize_query(q).split():
            return False, None, f"'{bad}' is a fixed constraint applied by the backend, not a title word"
    return True, normalize_query(q), None

# ------------------------------------------------------- title variants -----
# Only the TITLE changes; seniority words from the user's own title are carried
# over, never added or removed.
TITLE_VARIANTS = [
    (r"\bai engineer\b", ["machine learning engineer", "ml engineer", "applied ai engineer"]),
    (r"\bmachine learning engineer\b", ["ai engineer", "ml engineer", "applied scientist"]),
    (r"\bml engineer\b", ["machine learning engineer", "ai engineer"]),
    (r"\bdata scientist\b", ["machine learning scientist", "applied scientist"]),
    (r"\bdata engineer\b", ["analytics engineer", "etl developer"]),
    (r"\bsoftware engineer\b", ["software developer", "backend engineer"]),
    (r"\bbackend engineer\b", ["backend developer", "software engineer"]),
    (r"\bfrontend engineer\b", ["frontend developer", "ui engineer"]),
    (r"\bdevops engineer\b", ["site reliability engineer", "platform engineer"]),
    (r"\bdata analyst\b", ["business intelligence analyst", "analytics analyst"]),
]
_ALL_SENIORITY = frozenset(w for ws in SENIORITY_WORDS.values() for w in ws if " " not in w)


def title_variants(target_role):
    """Known alternative titles for `target_role`, keeping its leading seniority
    words (e.g. "Junior AI Engineer" -> "junior machine learning engineer")."""
    words = normalize_query(target_role).split()
    prefix_words = []
    for w in words:
        if w not in _ALL_SENIORITY:
            break
        prefix_words.append(w)
    core = " ".join(words[len(prefix_words):])
    prefix = (" ".join(prefix_words) + " ") if prefix_words else ""
    out = []
    for pattern, variants in TITLE_VARIANTS:
        if re.search(pattern, core):
            out.extend(prefix + v for v in variants)
    return out


def candidate_queries(goal: AgentGoal):
    """Every valid, distinct title this run may search: the target role, the
    user's alternative titles and known title variants, in that order. Together
    with the goal's providers this DEFINES the run's search space (and therefore
    when `finish` is permitted)."""
    seen, out = set(), []
    for q in [goal.target_role] + list(goal.alternative_titles) + title_variants(goal.target_role):
        ok, qn, _ = validate_search_query(q, goal)
        if ok and qn not in seen:
            seen.add(qn)
            out.append(qn)
    return out