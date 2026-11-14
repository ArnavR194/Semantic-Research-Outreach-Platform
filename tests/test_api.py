"""API wiring tests with the DB and embedder stubbed out."""
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import repository as repo
from app.db import get_conn
from app.main import create_app
from app.schemas import Professor, Student, StudentProfile
from app.services import retriever
from app.services.embedder import get_embedder
from app.services.reranker import Candidate

SID = uuid4()
NOW = datetime.now(timezone.utc)
PROFILE = StudentProfile(name="Jane Doe", email="jane@x.edu", skills=["PyTorch"],
                         interests=["machine learning"], projects=["Neural network for image detection"])


class FakeEmbedder:
    name = "fake"
    def encode(self, text): return [0.0] * 384
    def encode_many(self, texts): return [[0.0] * 384 for _ in texts]


@pytest.fixture
def client(monkeypatch):
    student = Student(id=SID, profile=PROFILE, summary="s", created_at=NOW, updated_at=NOW)
    prof = Professor(id=7, name="Alan Turing", title="Professor", email="at@x.edu", url=None,
                     department="ECE", university="UIUC", bio=None,
                     research_interests=["machine learning, deep learning"], research_areas=[],
                     research_topics=[], research_summary="Research interests: machine learning, deep learning")

    monkeypatch.setattr(repo, "get_student", lambda c, i: student if i == SID else None)
    monkeypatch.setattr(repo, "get_student_embedding", lambda c, i: [0.0] * 384)
    monkeypatch.setattr(repo, "get_professor", lambda c, i, with_papers=True: prof if i == 7 else None)
    monkeypatch.setattr(retriever, "search_professors", lambda c, e, k: [
        Candidate(7, prof.name, prof.email, None, None, prof.research_summary,
                  prof.research_interests, 0.6),
        Candidate(8, "Grace Hopper", None, None, None, "compilers", ["compilers"], 0.62),
    ])
    monkeypatch.setattr(retriever, "most_similar_paper", lambda c, e, v: {
        "title": "Deep Nets", "year": 2024, "abstract": "...", "similarity": 0.4})

    app = create_app()
    app.router.lifespan_context = _no_lifespan
    app.dependency_overrides[get_conn] = lambda: None
    app.dependency_overrides[get_embedder] = FakeEmbedder
    return TestClient(app)


from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _no_lifespan(app):
    yield


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_matches_rerank_and_shape(client):
    r = client.get(f"/students/{SID}/matches?k=2")
    assert r.status_code == 200
    m = r.json()["matches"]
    assert [x["professor_id"] for x in m] == [7, 8]  # rerank beats raw similarity
    assert m[0]["matches"]["skills"] == ["PyTorch"]


def test_unknown_student_404(client):
    assert client.get(f"/students/{uuid4()}/matches").status_code == 404


def test_email_template(client):
    r = client.post("/emails", json={"student_id": str(SID), "professor_id": 7})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["to"] == "at@x.edu" and "Dear Professor Turing" in d["body"]
    assert d["context"]["paper_title"] == "Deep Nets" and "Deep Nets" in d["body"]


def test_email_llm_without_key_is_503(client, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "anthropic_api_key", None)
    r = client.post("/emails", json={"student_id": str(SID), "professor_id": 7, "generator": "llm"})
    assert r.status_code == 503


def test_upload_rejects_non_pdf(client):
    r = client.post("/resumes", files={"file": ("a.pdf", b"hello", "application/pdf")})
    assert r.status_code == 415
