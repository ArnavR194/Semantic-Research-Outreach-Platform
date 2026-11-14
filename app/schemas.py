"""Pydantic models: the API contract the Lovable frontend codes against (see /openapi.json)."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    name: str = "Unknown"
    email: str = ""
    university: str | None = None
    major: str | None = None
    year: str | None = None  # e.g. "junior", "senior", "PhD student"
    skills: list[str] = Field(default_factory=list)
    coursework: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)


class StudentProfileUpdate(BaseModel):
    """Partial update; only provided fields change. Lets users fix parser mistakes."""
    name: str | None = None
    email: str | None = None
    university: str | None = None
    major: str | None = None
    year: str | None = None
    skills: list[str] | None = None
    coursework: list[str] | None = None
    projects: list[str] | None = None
    interests: list[str] | None = None


class Student(BaseModel):
    id: UUID
    profile: StudentProfile
    summary: str
    parser_confidence: float | None = None
    parser_used_llm: bool = False
    created_at: datetime
    updated_at: datetime


class ScoreBreakdown(BaseModel):
    final: float
    embedding: float
    skills: float
    interests: float
    coursework: float


class MatchEvidence(BaseModel):
    skills: list[str]
    interests: list[str]
    coursework: list[str]


class Match(BaseModel):
    rank: int
    professor_id: int
    name: str
    email: str | None
    title: str | None
    url: str | None
    research_summary: str | None
    scores: ScoreBreakdown
    matches: MatchEvidence


class MatchList(BaseModel):
    student_id: UUID
    matches: list[Match]


class Paper(BaseModel):
    id: int
    title: str | None
    abstract: str | None
    year: int | None
    url: str | None


class ProfessorSummary(BaseModel):
    id: int
    name: str
    title: str | None
    email: str | None
    url: str | None
    department: str | None
    university: str | None


class Professor(ProfessorSummary):
    bio: str | None
    research_interests: list[str]
    research_areas: list[str]
    research_topics: list[str]
    research_summary: str | None
    papers: list[Paper] = Field(default_factory=list)


Generator = Literal["template", "llm"]


class EmailRequest(BaseModel):
    student_id: UUID
    professor_id: int
    generator: Generator = "template"


class EmailContext(BaseModel):
    """The retrieved facts the email is grounded in. Show these in the UI as 'why this email'."""
    best_interest: str | None
    relevant_interests: list[str]
    skill_overlap: list[str]
    aligned_projects: list[str]
    best_project: str | None
    paper_title: str | None
    paper_year: int | None
    paper_similarity: float | None


class EmailDraft(BaseModel):
    student_id: UUID
    professor_id: int
    generator: Generator
    to: str | None
    subject: str
    body: str
    context: EmailContext
