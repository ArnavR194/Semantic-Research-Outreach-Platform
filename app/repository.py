"""All SQL for students and professors lives here."""
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from app.db import to_vector
from app.schemas import Paper, Professor, ProfessorSummary, Student, StudentProfile

_STUDENT_COLS = "id, profile, summary, parser_confidence, parser_used_llm, created_at, updated_at"


def _student(row: dict) -> Student:
    return Student(**{**row, "profile": StudentProfile(**row["profile"])})


def insert_student(conn: psycopg.Connection, profile: StudentProfile, summary: str,
                   embedding: list[float], embed_model: str,
                   confidence: float | None, used_llm: bool) -> Student:
    row = conn.execute(
        f"""
        INSERT INTO students (profile, summary, embedding, embed_model, parser_confidence, parser_used_llm)
        VALUES (%s, %s, %s::vector, %s, %s, %s)
        RETURNING {_STUDENT_COLS}
        """,
        (Jsonb(profile.model_dump()), summary, to_vector(embedding), embed_model, confidence, used_llm),
    ).fetchone()
    return _student(row)


def get_student(conn: psycopg.Connection, student_id: UUID) -> Student | None:
    row = conn.execute(f"SELECT {_STUDENT_COLS} FROM students WHERE id = %s", (student_id,)).fetchone()
    return _student(row) if row else None


def get_student_embedding(conn: psycopg.Connection, student_id: UUID) -> list[float] | None:
    row = conn.execute("SELECT embedding::text AS e FROM students WHERE id = %s", (student_id,)).fetchone()
    if not row or row["e"] is None:
        return None
    return [float(x) for x in row["e"].strip("[]").split(",")]


def update_student(conn: psycopg.Connection, student_id: UUID, profile: StudentProfile,
                   summary: str, embedding: list[float], embed_model: str) -> Student | None:
    row = conn.execute(
        f"""
        UPDATE students SET profile = %s, summary = %s, embedding = %s::vector,
               embed_model = %s, updated_at = now()
        WHERE id = %s
        RETURNING {_STUDENT_COLS}
        """,
        (Jsonb(profile.model_dump()), summary, to_vector(embedding), embed_model, student_id),
    ).fetchone()
    return _student(row) if row else None


def delete_student(conn: psycopg.Connection, student_id: UUID) -> bool:
    return conn.execute("DELETE FROM students WHERE id = %s", (student_id,)).rowcount > 0


def list_professors(conn: psycopg.Connection, q: str | None, limit: int, offset: int) -> list[ProfessorSummary]:
    rows = conn.execute(
        """
        SELECT id, name, title, email, url, department, university FROM professors
        WHERE %(q)s::text IS NULL OR name ILIKE '%%' || %(q)s || '%%' OR research_summary ILIKE '%%' || %(q)s || '%%'
        ORDER BY name LIMIT %(limit)s OFFSET %(offset)s
        """,
        {"q": q, "limit": limit, "offset": offset},
    ).fetchall()
    return [ProfessorSummary(**r) for r in rows]


def get_professor(conn: psycopg.Connection, professor_id: int, with_papers: bool = True) -> Professor | None:
    row = conn.execute(
        """
        SELECT id, name, title, email, url, department, university, bio, research_summary,
               COALESCE(research_interests, '{}') AS research_interests,
               COALESCE(research_areas, '{}') AS research_areas,
               COALESCE(research_topics, '{}') AS research_topics
        FROM professors WHERE id = %s
        """,
        (professor_id,),
    ).fetchone()
    if not row:
        return None
    papers = []
    if with_papers and row["email"]:
        papers = [Paper(**p) for p in conn.execute(
            "SELECT id, title, abstract, year, url FROM papers WHERE professor_email = %s "
            "ORDER BY year DESC NULLS LAST",
            (row["email"],),
        ).fetchall()]
    return Professor(**row, papers=papers)
