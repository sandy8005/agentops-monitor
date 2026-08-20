import re
from pydantic import BaseModel, field_validator, Field
from typing import List
from enum import Enum


def _strict_float(v, field_name="value"):
    """
    Coerce recoverable numeric values to float, but FAIL LOUDLY on garbage.
    - 2, 2.0, "2", "2.5"          -> float (clean number)
    - "2 years", "24 months"      -> extracted number (months converted to years)
    - None                        -> raises (caller decides if None is acceptable)
    - "about two", "N/A", "lots"  -> raises ValueError (garbage must NOT become 0.0)

    The point: recoverable variation is coerced, but genuinely unparseable values
    raise instead of silently becoming 0.0 — hiding bad LLM output as a false fact
    is worse than surfacing it. A raised error appears in the run trace, where it
    belongs.
    """
    if v is None:
        raise ValueError(f"{field_name}: value is None")
    if isinstance(v, bool):
        raise ValueError(f"{field_name}: bool is not a valid number")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        m_months = re.fullmatch(r"(\d+(?:\.\d+)?)\s*months?", s)
        if m_months:
            return round(float(m_months.group(1)) / 12.0, 2)
        m_num = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?", s)
        if m_num:
            return float(m_num.group(1))
        raise ValueError(f"{field_name}: cannot parse '{v}' as a number")
    raise ValueError(f"{field_name}: unexpected type {type(v).__name__}")


class Education(BaseModel):
    degree: str
    institution: str
    year: str

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        return str(v) if v is not None else ""


class Project(BaseModel):
    name: str
    tech: List[str]


class Experience(BaseModel):
    title: str
    company: str
    years: float = Field(ge=0)   # negative years are impossible

    @field_validator("years", mode="before")
    @classmethod
    def coerce_years(cls, v):
        # None here means the LLM omitted it for this entry; treat as 0 (unknown).
        if v is None:
            return 0.0
        return _strict_float(v, field_name="experience.years")


class ParsedResume(BaseModel):
    skills: List[str]
    years_experience: float = Field(ge=0)   # negative total experience is impossible
    education: List[Education]
    projects: List[Project]
    experience: List[Experience]

    @field_validator("years_experience", mode="before")
    @classmethod
    def coerce_total_years(cls, v):
        if v is None:
            return 0.0
        return _strict_float(v, field_name="years_experience")


class JobRequirements(BaseModel):
    required_skills: List[str]
    required_any_of: List[List[str]] = []
    preferred_skills: List[str]
    min_years_experience: float = Field(ge=0)
    responsibilities: List[str]

    @field_validator("min_years_experience", mode="before")
    @classmethod
    def coerce_min_years(cls, v):
        if v is None:
            return 0.0   # "no minimum stated" is a legitimate default
        return _strict_float(v, field_name="min_years_experience")


class DecisionEnum(str, Enum):
    apply = "Apply"
    maybe = "Maybe"
    skip = "Skip"


class JobDecision(BaseModel):
    decision: DecisionEnum
    reason: str


class Evaluation(BaseModel):
    relevance_score: int = Field(ge=0, le=10)
    faithfulness_score: int = Field(ge=0, le=10)
    completeness_score: int = Field(ge=0, le=10)
    hallucination_detected: bool
    hallucinated_claims: List[str]
    notes: str