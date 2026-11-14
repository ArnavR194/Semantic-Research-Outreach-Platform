"""Unit tests for the pure service layer (no DB, no model download)."""
from pathlib import Path

import pytest

from app.config import RerankWeights
from app.schemas import StudentProfile
from app.services.email import template
from app.services.email.context import ProfessorInfo, build_context
from app.services.reranker import Candidate, coursework_relevance_score, rerank

ROOT = Path(__file__).resolve().parent.parent

STUDENT = StudentProfile(
    name="Jane Doe", email="jane@example.edu", university="Example University",
    major="Electrical Engineering", year="senior",
    skills=["MATLAB", "Cadence", "Python"],
    coursework=["Power Electronics", "Signals and Systems"],
    projects=["Buck Converter Design for Solar Charging"],
    interests=["power electronics", "energy systems"],
)


def _cand(pid, sim, terms, summary=""):
    return Candidate(pid, f"Prof {pid}", None, None, None, summary or " ".join(terms), terms, sim)


def test_coursework_unknown_course_does_not_crash():
    # Previously: course_keywords.get(course, [course.lower().split()]) -> TypeError
    score, matches = coursework_relevance_score(["Power Electronics"], "research in power electronics")
    assert matches == ["Power Electronics"] and score > 0


def test_rerank_promotes_explicit_domain_match():
    ranked = rerank(
        [_cand(1, 0.50, ["quantum optics"]), _cand(2, 0.45, ["power electronics", "smart grid"])],
        STUDENT, RerankWeights(),
    )
    assert ranked[0].candidate.professor_id == 2
    assert "power electronics" in ranked[0].matches["interests"]


def test_template_uses_student_details_not_hardcoded_ones():
    prof = ProfessorInfo("Ada Lovelace", None, "power electronics converter design for renewable energy",
                         ["Power electronics, renewable energy"])
    ctx = build_context(prof, STUDENT, paper=None)
    subject, body = template.generate(prof, STUDENT, ctx)
    assert "Dear Professor Lovelace" in body
    assert "a senior Electrical Engineering student at Example University" in body
    assert "SystemVerilog" not in body and "junior" not in body
    assert "Buck Converter" in body
    assert subject.startswith("Prospective Research Student")


def test_project_alignment_uses_word_boundaries():
    # "ml" must not match "html"; "ai" must not match "maintain"
    prof = ProfessorInfo("X Y", None, "we maintain html tooling", [])
    ctx = build_context(prof, StudentProfile(projects=["ML image classifier with AI"]), None)
    assert ctx.aligned_projects == []


@pytest.mark.skipif(not (ROOT / "ML_Resume.pdf").exists(), reason="sample resume not present")
def test_parse_sample_resume():
    from app.services.resume_parser import parse_resume
    r = parse_resume(ROOT / "ML_Resume.pdf", confidence_threshold=0)
    assert r.profile.skills and r.profile.projects and not r.used_llm
