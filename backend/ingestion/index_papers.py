"""Chunking + Qdrant indexing (Phase 2/3 ingestion).

Turns fetched papers into ~200-word chunks, embeds them, and upserts
into Qdrant with deterministic UUID5 IDs so re-running ingestion
updates points in place instead of duplicating them.

Supports three Qdrant backends via backend.config.get_qdrant_client():
QDRANT_URL (server/cloud), QDRANT_PATH (embedded on-disk, no Docker),
or in-memory (tests).
"""
from __future__ import annotations

import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import (
    CHUNK_SIZE_WORDS,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    get_qdrant_client,
)
from backend.ingestion.embeddings import embed_texts

EMBEDDING_DIM = 1536  # text-embedding-3-small


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS) -> list[str]:
    """Split text into word windows.

    Guide default was 500 words, but abstracts are short; 200-word
    chunks keep each chunk topical (better retrieval precision).
    """
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def point_id_for(url: str, chunk_index: int) -> str:
    """Deterministic UUID5 from (paper URL, chunk index).

    Re-running ingestion upserts the same IDs instead of duplicating
    chunks — the idempotency the guide's auto-increment IDs lack.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#chunk{chunk_index}"))


def index_papers(papers: list[dict], collection: str = QDRANT_COLLECTION) -> int:
    """Chunk, embed, and upsert papers into Qdrant. Returns chunk count."""
    if not papers:
        return 0

    client = get_qdrant_client()
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Flatten (chunk, metadata) pairs first so embeddings are one batched call.
    chunks: list[str] = []
    metas: list[dict] = []
    for paper in papers:
        for i, chunk in enumerate(chunk_text(paper["abstract"])):
            chunks.append(chunk)
            metas.append(
                {
                    "text": chunk,
                    "title": paper["title"],
                    "url": paper["url"],
                    "authors": paper.get("authors", []),
                    "published": paper.get("published", ""),
                    "chunk_index": i,
                }
            )
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    points = [
        PointStruct(
            id=point_id_for(meta["url"], meta["chunk_index"]),
            vector=vector,
            payload=meta,
        )
        for vector, meta in zip(vectors, metas)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


if __name__ == "__main__":
    from backend.ingestion.fetch_papers import fetch_arxiv_papers

    topic = input("Topic to index from arXiv: ") or "retrieval augmented generation"
    papers = fetch_arxiv_papers(topic, max_results=25)
    print(f"Fetched {len(papers)} papers on '{topic}'.")
    n = index_papers(papers)
    print(f"Indexed {n} chunks into collection '{QDRANT_COLLECTION}'.")
