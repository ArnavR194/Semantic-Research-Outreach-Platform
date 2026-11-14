from fastapi import APIRouter, HTTPException

from app import repository as repo
from app.api.deps import Conn, require_student
from app.schemas import EmailDraft, EmailRequest
from app.services import pipeline
from app.services.email.llm import LLMUnavailable

router = APIRouter(tags=["emails"])


@router.post("/emails", response_model=EmailDraft)
def create_email(req: EmailRequest, conn: Conn):
    """Draft an outreach email from a student to a professor. generator: "template" | "llm"."""
    student = require_student(conn, req.student_id)
    prof = repo.get_professor(conn, req.professor_id, with_papers=False)
    if prof is None:
        raise HTTPException(404, "Professor not found")
    embedding = repo.get_student_embedding(conn, req.student_id)

    try:
        subject, body, ctx = pipeline.draft_email(conn, prof, student.profile, embedding, req.generator)
    except LLMUnavailable as e:
        raise HTTPException(503, str(e)) from e

    return EmailDraft(student_id=req.student_id, professor_id=req.professor_id,
                      generator=req.generator, to=prof.email, subject=subject, body=body, context=ctx)
