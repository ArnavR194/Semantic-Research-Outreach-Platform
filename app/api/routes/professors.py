from fastapi import APIRouter, HTTPException, Query

from app import repository as repo
from app.api.deps import Conn
from app.schemas import Professor, ProfessorSummary

router = APIRouter(prefix="/professors", tags=["professors"])


@router.get("", response_model=list[ProfessorSummary])
def list_professors(conn: Conn, q: str | None = None,
                    limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    return repo.list_professors(conn, q, limit, offset)


@router.get("/{professor_id}", response_model=Professor)
def get_professor(professor_id: int, conn: Conn):
    prof = repo.get_professor(conn, professor_id)
    if prof is None:
        raise HTTPException(404, "Professor not found")
    return prof
