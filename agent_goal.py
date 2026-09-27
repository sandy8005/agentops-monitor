"""
Goal, FIXED user constraints, and hard limits. The controller may adapt the search
TITLE; it can never change location, work mode, employment type or seniority.
Work authorization is out of scope and never claimed as checked.
"""
import re
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Seniority = Literal["", "intern", "entry", "junior", "mid", "senior"]
Provider = Literal["adzuna", "remotive", "pool"]

SENIORITY_WORDS = {
    "intern": {"intern", "internship"},
    "entry": {"entry", "entry-level", "graduate", "new grad", "junior", "jr", "associate", "i"},
    "junior": {"junior", "jr", "entry", "entry-level", "graduate", "associate"},
    "mid": {"mid", "mid-level", "ii", "intermediate"},
    "senior": {"senior", "sr", "lead", "principal", "staff", "iii", "head", "director"},
}
_ABOVE = {
    "intern": SENIORITY_WORDS["senior"] | SENIORITY_WORDS["mid"] | {"manager"},
    "entry": SENIORITY_WORDS["senior"] | SENIORITY_WORDS["mid"] | {"manager"},
    "junior": SENIORITY_WORDS["senior"] | SENIORITY_WORDS["mid"] | {"manager"},
    "mid": SENIORITY_WORDS["senior"] | {"manager"},
    "senior": set(),
}
_BELOW = {
    "senior": SENIORITY_WORDS["intern"] | {"junior", "jr", "entry", "entry-level", "graduate"},
    "mid": SENIORITY_WORDS["intern"],
    "entry": SENIORITY_WORDS["intern"],
    "junior": SENIORITY_WORDS["intern"],
    "intern": set(),
}
_QUERY_ALLOWED_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 +#./&\-]{1,79}$")


def normalize_query(q):
    return " ".join(str(q or "").lower().split())


def title_words(title):
    return set(re.findall(r"[a-z]+(?:-[a-z]+)?", str(title or "").lower()))


def seniority_conflict(text, seniority):
    if not seniority:
        return None
    words = title_words(text)
    joined = " " + " ".join(str(text or "").lower().split()) + " "
    for w in sorted(_ABOVE.get(seniority, set()) | _BELOW.get(seniority, set())):
        if (" " in w and f" {w} " in joined) or w in words:
            if w == "i" and not re.search(r"\bi\s*$", str(text or "").lower()):
                continue
            return w
    return None


class FixedConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    location: str = Field("", max_length=200)
    work_mode: Literal["", "remote", "hybrid", "onsite"] = ""
    employment_type: Literal["", "full-time", "part-time", "contract", "internship"] = ""
    seniority: Seniority = ""


class Limits(BaseModel):
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
    model_config = ConfigDict(extra="forbid")
    description: str = Field("", max_length=300)
    target_role: str = Field(..., min_length=2, max_length=80)
    alternative_titles: List[str] = Field(default_factory=list, max_length=8)
    target_count: int = Field(10, ge=1, le=50)
    qualifying_decisions: List[Literal["Apply", "Maybe"]] = Field(default_factory=lambda: ["Apply"])
    providers: List[Provider] = Field(default_factory=lambda: ["adzuna", "remotive"])
    constraints: FixedConstraints = Field(default_factory=FixedConstraints)
    limits: Limits = Field(default_factory=Limits)
    evaluate_quality: bool = False
    on_model_unavailable: Literal["rules", "pause"] = "rules"
    use_llm_controller: bool = True
    use_llm_advice: bool = False

    @field_validator("target_role", "alternative_titles", mode="after")
    @classmethod
    def _titles_are_plain(cls, v):
        for t in (v if isinstance(v, list) else [v]):
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
        return self


def validate_search_query(query, goal: AgentGoal):
    """(ok, normalized_query, reason). Queries are plain job titles only."""
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