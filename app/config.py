from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class RerankWeights(BaseModel):
    embedding: float = 0.45
    skills: float = 0.25
    interests: float = 0.20
    coursework: float = 0.10


class Settings(BaseSettings):
    """All runtime configuration. Values come from env vars or a .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_nested_delimiter="__", extra="ignore"
    )

    database_url: str = "postgresql:///research_outreach"
    db_pool_max: int = 10

    # Comma-separated in the env: CORS_ORIGINS=http://localhost:5173,https://myapp.com
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_origin_regex: str | None = r"https://.*\.lovable\.app|https://.*\.lovableproject\.com"

    embed_model: str = "all-MiniLM-L6-v2"
    retrieval_k: int = 116  # candidates pulled from pgvector before reranking
    rerank_weights: RerankWeights = Field(default_factory=RerankWeights)  # RERANK_WEIGHTS__SKILLS=0.3

    # LLM used for the resume-parse fallback and the "llm" email generator
    llm_provider: Literal["anthropic", "groq"] = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = Field("claude-opus-5-5", validation_alias=AliasChoices("anthropic_model", "llm_model"))
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    parser_confidence_threshold: float = 0.6
    max_upload_mb: int = 10

    semantic_scholar_api_key: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def llm_enabled(self) -> bool:
        key = self.groq_api_key if self.llm_provider == "groq" else self.anthropic_api_key
        return bool(key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
