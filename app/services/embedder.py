"""Single place that owns the sentence-transformer model. Load once, share everywhere."""
from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    name: str

    def encode(self, text: str) -> list[float]: ...
    def encode_many(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=32).tolist()


@lru_cache
def get_embedder() -> Embedder:
    return SentenceTransformerEmbedder(get_settings().embed_model)
