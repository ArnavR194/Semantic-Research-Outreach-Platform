from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app import repository as repo
from app.api.deps import Conn, EmbedderDep, require_student
from app.config import get_settings
from app.schemas import (Match, MatchEvidence, MatchList, ScoreBreakdown, Student,
                         StudentProfileUpdate)
from app.services import pipeline
from app.services.resume_parser import build_student_summary, parse_resume

router = APIRouter(tags=["students"])


@router.post("/resumes", response_model=Student, status_code=status.HTTP_201_CREATED)
async def upload_resume(conn: Conn, embedder: EmbedderDep, file: UploadFile = File(...)):
    """Upload a resume PDF. Parses it, embeds the profile, and creates a student."""
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(415, "Upload a PDF")
    data = await file.read()
    if len(data) > get_settings().max_upload_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "File is not a valid PDF")

    try:
        result = await run_in_threadpool(parse_resume, data)
    except Exception as e:
        raise HTTPException(422, f"Could not read resume: {e}") from e

    summary = build_student_summary(result.profile)
    embedding = await run_in_threadpool(embedder.encode, summary)
    return repo.insert_student(conn, result.profile, summary, embedding, embedder.name,
                               result.confidence, result.used_llm)


@router.get("/students/{student_id}", response_model=Student)
def get_student(student_id: UUID, conn: Conn):
    return require_student(conn, student_id)


@router.patch("/students/{student_id}", response_model=Student)
def update_student(student_id: UUID, update: StudentProfileUpdate, conn: Conn, embedder: EmbedderDep):
    """Correct parsed fields (skills, interests, ...). The profile is re-embedded."""
    student = require_student(conn, student_id)
    profile = student.profile.model_copy(update=update.model_dump(exclude_unset=True))
    summary = build_student_summary(profile)
    return repo.update_student(conn, student_id, profile, summary, embedder.encode(summary), embedder.name)


@router.delete("/students/{student_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_student(student_id: UUID, conn: Conn):
    if not repo.delete_student(conn, student_id):
        raise HTTPException(404, "Student not found")


@router.get("/students/{student_id}/matches", response_model=MatchList)
def get_matches(student_id: UUID, conn: Conn, k: int = Query(10, ge=1, le=50)):
    """Top-k professors after vector search + multi-signal reranking, with score breakdowns."""
    student = require_student(conn, student_id)
    embedding = repo.get_student_embedding(conn, student_id)
    ranked = pipeline.match(conn, student.profile, embedding, top_k=k)
    return MatchList(
        student_id=student_id,
        matches=[
            Match(
                rank=i,
                professor_id=r.candidate.professor_id,
                name=r.candidate.name,
                email=r.candidate.email,
                title=r.candidate.title,
                url=r.candidate.url,
                research_summary=r.candidate.research_summary,
                scores=ScoreBreakdown(**r.scores),
                matches=MatchEvidence(**r.matches),
            )
            for i, r in enumerate(ranked, 1)
        ],
    )
