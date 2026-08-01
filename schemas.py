from pydantic import BaseModel
from typing import List


class Education(BaseModel):
    degree: str
    institution: str
    year: int


class Project(BaseModel):
    name: str
    tech: List[str]
    description: str


class Experience(BaseModel):
    role: str
    company: str
    months: int


class ParsedResume(BaseModel):
    skills: List[str]
    years_experience: float
    education: List[Education]
    projects: List[Project]
    experience: List[Experience]


class JobRequirements(BaseModel):
    required_skills: List[str]
    preferred_skills: List[str]
    min_years_experience: float
    responsibilities: List[str]

class Evaluation(BaseModel):
    relevance_score: int        # 0-10
    faithfulness_score: int     # 0-10
    hallucination_detected: bool
    hallucinated_claims: List[str]
    notes: str