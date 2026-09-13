"""Retriever Agent (Phase 3).

Embeds a query with the same model used at indexing time and returns
the top-k most similar chunks from Qdrant with their source metadata
and similarity scores. This is the R in RAG.
"""
from __future__ import annotations

from backend.config import QDRANT_COLLECTION, get_qdrant_client
from backend.ingestion.embeddings import embed_one


def retrieve(query: str, top_k: int = 5, min_score: float = 0.0) -> list[dict]:
    """Semantic search over the indexed paper collection.

    Returns a list of dicts with keys: text, title, url, authors,
    published, score — sorted by descending similarity.
    """
    client = get_qdrant_client()
    query_vector = embed_one(query)
    try:
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
    except ValueError:
        # Collection doesn't exist yet (ingestion not run) — fail soft so
        # the pipeline degrades to "insufficient evidence" downstream.
        return []
    return [
        {
            "text": h.payload["text"],
            "title": h.payload["title"],
            "url": h.payload["url"],
            "authors": h.payload.get("authors", []),
            "published": h.payload.get("published", ""),
            "score": round(h.score, 4),
        }
        for h in hits
        if h.score >= min_score
    ]


if __name__ == "__main__":
    q = input("Search query: ") or "what is retrieval augmented generation"
    for i, r in enumerate(retrieve(q, top_k=5), 1):
        print(f"{i}. [{r['score']:.3f}] {r['title']}")
        print(f"   {r['text'][:150]}...")
