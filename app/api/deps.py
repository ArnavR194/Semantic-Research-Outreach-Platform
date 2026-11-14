from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import Depends, HTTPException

from app import repository as repo
from app.db import get_conn
from app.schemas import Student
from app.services.embedder import Embedder, get_embedder

Conn = Annotated[psycopg.Connection, Depends(get_conn)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def require_student(conn: psycopg.Connection, student_id: UUID) -> Student:
    student = repo.get_student(conn, student_id)
    if student is None:
        raise HTTPException(404, "Student not found")
    return student
