from pydantic import BaseModel
from typing import List


class Education(BaseModel):
    degree: str
    institution: str
    year: str


class Project(BaseModel):
    name: str
    tech: List[str]


class Experience(BaseModel):
    title: str
    company: str
    years: float


class ParsedResume(BaseModel):
    skills: List[str]
    years_experience: float
    education: List[Education]
    projects: List[Project]
    experience: List[Experience]


class JobRequirements(BaseModel):
    required_skills: List[str]
    required_any_of: List[List[str]] = []   # groups; candidate needs ANY one per group
    preferred_skills: List[str]
    min_years_experience: float
    responsibilities: List[str]


class Evaluation(BaseModel):
    relevance_score: int
    faithfulness_score: int
    completeness_score: int
    hallucination_detected: bool
    hallucinated_claims: List[str]
    notes: str