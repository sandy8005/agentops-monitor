from pydantic import BaseModel, field_validator
from typing import List
from enum import Enum


class Education(BaseModel):
    degree: str
    institution: str
    year: str

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        # LLM sometimes returns 2025 (int) instead of "2025" (str)
        return str(v) if v is not None else ""


class Project(BaseModel):
    name: str
    tech: List[str]


class Experience(BaseModel):
    title: str
    company: str
    years: float

    @field_validator("title", mode="before")
    @classmethod
    def coerce_title(cls, v):
        # accept 'role' as an alias if the LLM used it (handled in parser too)
        return v if v is not None else ""

    @field_validator("years", mode="before")
    @classmethod
    def coerce_years(cls, v):
        # LLM sometimes gives months (int) or a string; coerce to float years
        if v is None:
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0


class ParsedResume(BaseModel):
    skills: List[str]
    years_experience: float
    education: List[Education]
    projects: List[Project]
    experience: List[Experience]


class JobRequirements(BaseModel):
    required_skills: List[str]
    required_any_of: List[List[str]] = []
    preferred_skills: List[str]
    min_years_experience: float
    responsibilities: List[str]


class DecisionEnum(str, Enum):
    apply = "Apply"
    maybe = "Maybe"
    skip = "Skip"


class JobDecision(BaseModel):
    decision: DecisionEnum
    reason: str


class Evaluation(BaseModel):
    relevance_score: int
    faithfulness_score: int
    completeness_score: int
    hallucination_detected: bool
    hallucinated_claims: List[str]
    notes: str