"""Builds the grounding context for an outreach email. Shared by the template and LLM generators,
so both are grounded in exactly the same retrieved facts."""
import re
from dataclasses import dataclass

from app.schemas import EmailContext, StudentProfile

PROJECT_TOPIC_KEYWORDS = {
    "FPGA": ["fpga", "reconfigurable", "hardware acceleration", "hls"],
    "computer vision": ["vision", "image", "detection", "recognition", "visual"],
    "machine learning": ["machine learning", "deep learning", "neural", "ml", "ai"],
    "embedded systems": ["embedded", "firmware", "microcontroller", "hardware"],
    "signal processing": ["signal", "processing", "filtering", "acoustic"],
    "ASL/gestures": ["gesture", "hand", "interface", "human computer"],
    "robotics": ["robot", "autonomous", "control", "navigation"],
    "circuits": ["circuit", "analog", "amplifier", "vlsi"],
    "power": ["power", "converter", "inverter", "grid", "battery"],
}


@dataclass
class ProfessorInfo:
    name: str
    email: str | None
    summary: str
    interests: list[str]  # raw research_interests + areas + topics


def oxford_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def clean_interests(professor_interests: list[str]) -> list[str]:
    cleaned = []
    for interest in professor_interests:
        cleaned.extend(p.strip() for p in interest.split(","))
    return [i for i in cleaned if i]


def get_salutation(full_name: str) -> str:
    return full_name.strip().split()[-1]


def _contains_word(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def get_skill_overlap(student_skills: list[str], professor_summary: str) -> list[str]:
    summary_lower = professor_summary.lower()
    return [s for s in student_skills if len(s) > 1 and _contains_word(summary_lower, s.lower())]


def get_project_alignment(student_projects: list[str], professor_summary: str) -> list[tuple[str, str]]:
    summary_lower = professor_summary.lower()
    aligned = []
    for project in student_projects:
        project_lower = project.lower()
        for topic, terms in PROJECT_TOPIC_KEYWORDS.items():
            # word-boundary match so "ml" doesn't match "html", "ai" doesn't match "maintain"
            if any(_contains_word(project_lower, t) for t in terms) and any(
                _contains_word(summary_lower, t) for t in terms
            ):
                aligned.append((project, topic))
                break
    return aligned


def get_best_project(aligned: list[tuple[str, str]], student_projects: list[str], professor_summary: str) -> str | None:
    summary_lower = professor_summary.lower()
    if aligned:
        return max(aligned, key=lambda pt: sum(w in summary_lower for w in pt[1].lower().split()))[0]
    # Fallback: any project sharing a meaningful word with the professor summary
    for proj in student_projects:
        if any(_contains_word(summary_lower, w) for w in proj.lower().split() if len(w) > 4):
            return proj
    return None


def get_relevant_interests(student_interests: list[str], professor_summary: str) -> list[str]:
    summary_lower = professor_summary.lower()
    return [i for i in student_interests
            if any(_contains_word(summary_lower, w) for w in i.lower().split() if w not in _STOPWORDS)]


# Stopwords plus words so common in research summaries that they carry no signal on their own.
_STOPWORDS = {"and", "or", "of", "the", "for", "in", "to", "a", "an", "&",
              "systems", "system", "engineering", "science", "data", "computing", "design"}


def get_best_interest(cleaned_prof_interests: list[str], student_interests: list[str]) -> str | None:
    """Pick the single professor interest that best overlaps the student's background."""
    if not cleaned_prof_interests:
        return None
    student_text = " ".join(student_interests).lower()
    for interest in cleaned_prof_interests:
        words = [w for w in interest.lower().split() if len(w) > 3 and w not in _STOPWORDS]
        if any(_contains_word(student_text, w) for w in words):
            return interest
    return cleaned_prof_interests[0]


def build_context(prof: ProfessorInfo, student: StudentProfile, paper: dict | None) -> EmailContext:
    aligned = get_project_alignment(student.projects, prof.summary)
    return EmailContext(
        best_interest=get_best_interest(clean_interests(prof.interests), student.interests),
        relevant_interests=get_relevant_interests(student.interests, prof.summary),
        skill_overlap=get_skill_overlap(student.skills, prof.summary),
        aligned_projects=[p for p, _ in aligned],
        best_project=get_best_project(aligned, student.projects, prof.summary),
        paper_title=paper["title"] if paper else None,
        paper_year=paper["year"] if paper else None,
        paper_similarity=round(paper["similarity"], 4) if paper else None,
    )
