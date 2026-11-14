"""FastAPI entry point.  Run:  uvicorn app.main:app --reload   Docs: http://localhost:8000/docs"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import emails, professors, students
from app.config import get_settings
from app.db import close_pool, init_pool
from app.services.embedder import get_embedder

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_pool()
    get_embedder()  # load the model once at startup, not on the first request
    yield
    close_pool()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="Research Outreach API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_origin_regex=s.cors_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(students.router)
    app.include_router(emails.router)
    app.include_router(professors.router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "embed_model": s.embed_model, "llm_enabled": s.llm_enabled}

    return app


app = create_app()
