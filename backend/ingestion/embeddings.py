"""Embedding helper.

Uses OpenAI text-embedding-3-small (1536 dims) via the OpenAI SDK.
Batches requests to cut API calls roughly in half versus one call per
chunk. The client is created lazily so tests/CI run without a key.
"""
from __future__ import annotations

from typing import Iterable

from openai import OpenAI

from backend.config import OPENAI_API_KEY, mock_llm_enabled
from backend.mock_llm import MockEmbeddings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of texts, batching requests to reduce API overhead.

    Order of results matches order of inputs. In MOCK_LLM mode, routes to
    the deterministic bag-of-words mock (no key, no network).
    """
    if mock_llm_enabled():
        return MockEmbeddings().embed_texts(texts)
    vectors: list[list[float]] = []
    client = _get_client()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small", input=batch
        )
        vectors.extend(d.embedding for d in response.data)
    return vectors


def embed_one(text: str) -> list[float]:
    """Embed a single text (e.g. a user query)."""
    return embed_texts([text])[0]


__all__ = ["embed_texts", "embed_one"]
