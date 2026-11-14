"""End-to-end orchestration: retrieve → rerank → ground → generate. Used by the API and CLI demo."""
from typing import Literal

import psycopg

from app.config import RerankWeights, get_settings
from app.schemas import EmailContext, Professor, StudentProfile
from app.services import retriever
from app.services.email import llm, template
from app.services.email.context import ProfessorInfo, build_context
from app.services.reranker import Ranked, rerank


def match(conn: psycopg.Connection, profile: StudentProfile, embedding: list[float],
          top_k: int = 10, weights: RerankWeights | None = None) -> list[Ranked]:
    s = get_settings()
    candidates = retriever.search_professors(conn, embedding, k=max(s.retrieval_k, top_k))
    return rerank(candidates, profile, weights or s.rerank_weights)[:top_k]


def draft_email(conn: psycopg.Connection, prof: Professor, profile: StudentProfile,
                embedding: list[float], generator: Literal["template", "llm"] = "template",
                ) -> tuple[str, str, EmailContext]:
    info = ProfessorInfo(
        name=prof.name,
        email=prof.email,
        summary=prof.research_summary or "",
        interests=prof.research_interests + prof.research_areas + prof.research_topics,
    )
    paper = retriever.most_similar_paper(conn, prof.email, embedding)
    ctx = build_context(info, profile, paper)

    if generator == "llm":
        snippet = (paper.get("abstract") or "")[:600] if paper else None
        subject, body = llm.generate(info, profile, ctx, paper_snippet=snippet)
    else:
        subject, body = template.generate(info, profile, ctx)
    return subject, body, ctx
