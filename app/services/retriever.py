"""Vector search over professors and papers (pgvector cosine distance, `<=>`)."""
import psycopg

from app.db import to_vector
from app.services.reranker import Candidate


def search_professors(conn: psycopg.Connection, embedding: list[float], k: int) -> list[Candidate]:
    """One query returns everything the reranker needs (no per-professor follow-up queries)."""
    vec = to_vector(embedding)
    rows = conn.execute(
        """
        SELECT id, name, email, title, url, research_summary,
               COALESCE(research_interests, '{}') || COALESCE(research_areas, '{}')
                 || COALESCE(research_topics, '{}') AS research_terms,
               1 - (embedding <=> %(v)s::vector) AS similarity
        FROM professors
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %(v)s::vector
        LIMIT %(k)s
        """,
        {"v": vec, "k": k},
    ).fetchall()
    return [
        Candidate(
            professor_id=r["id"], name=r["name"], email=r["email"], title=r["title"],
            url=r["url"], research_summary=r["research_summary"] or "",
            research_terms=list(r["research_terms"] or []), embedding_score=float(r["similarity"]),
        )
        for r in rows
    ]


def most_similar_paper(conn: psycopg.Connection, professor_email: str | None,
                       embedding: list[float], min_similarity: float = 0.15) -> dict | None:
    if not professor_email:
        return None
    vec = to_vector(embedding)
    row = conn.execute(
        """
        SELECT title, abstract, year, url, 1 - (embedding <=> %(v)s::vector) AS similarity
        FROM papers
        WHERE professor_email = %(e)s AND embedding IS NOT NULL
        ORDER BY embedding <=> %(v)s::vector
        LIMIT 1
        """,
        {"v": vec, "e": professor_email},
    ).fetchone()
    if not row or row["similarity"] < min_similarity:
        return None
    return {**row, "similarity": float(row["similarity"])}
