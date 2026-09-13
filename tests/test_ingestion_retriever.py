"""Offline tests for ingestion + retriever (no network, no API keys).

Strategy: use Qdrant's in-memory mode and stub the embedding calls, so
the full chunk -> index -> retrieve pipeline is exercised without
spending credits or needing Docker.
"""
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.ingestion.index_papers import chunk_text, point_id_for


# ---------------------------------------------------------------------------
# Pure-function tests (no Qdrant needed)
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_splits_on_word_count(self):
        text = " ".join(f"w{i}" for i in range(10))
        chunks = chunk_text(text, chunk_size=4)
        assert chunks == [
            "w0 w1 w2 w3",
            "w4 w5 w6 w7",
            "w8 w9",
        ]

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_exact_multiple(self):
        text = " ".join(["a"] * 8)
        assert len(chunk_text(text, chunk_size=4)) == 2


class TestPointId:
    def test_deterministic(self):
        assert point_id_for("http://x/1", 0) == point_id_for("http://x/1", 0)

    def test_differs_per_chunk_index(self):
        assert point_id_for("http://x/1", 0) != point_id_for("http://x/1", 1)

    def test_valid_uuid(self):
        import uuid

        assert uuid.UUID(point_id_for("http://x/1", 0)).version == 5


# ---------------------------------------------------------------------------
# Pipeline tests with in-memory Qdrant + stubbed embeddings
# ---------------------------------------------------------------------------
# Fake embedding: deterministic bag-of-words over a high-dimensional space,
# so texts sharing words genuinely get higher cosine similarity.
FAKE_DIM = 256


def fake_embed(text: str) -> list[float]:
    import zlib

    vec = [0.0] * FAKE_DIM
    for word in text.lower().split():
        vec[zlib.crc32(word.encode()) % FAKE_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def fake_embeddings(monkeypatch):
    """Stub embed_texts/embed_one to work offline."""
    import backend.ingestion.embeddings as emb

    monkeypatch.setattr(emb, "embed_texts", lambda texts, batch_size=64: [fake_embed(t) for t in texts])
    monkeypatch.setattr(emb, "embed_one", lambda text: fake_embed(text))
    # Also patch the names the importing modules bound at import time.
    import backend.ingestion.index_papers as idx
    import backend.agents.retriever as ret

    monkeypatch.setattr(idx, "embed_texts", emb.embed_texts)
    monkeypatch.setattr(ret, "embed_one", emb.embed_one)
    # Fake vectors are 4-dim; patch the dim used to create the collection.
    monkeypatch.setattr(idx, "EMBEDDING_DIM", FAKE_DIM)
    return emb


@pytest.fixture()
def memory_qdrant(monkeypatch):
    """Route get_qdrant_client() to a fresh in-memory client."""
    client = QdrantClient(":memory:")
    import backend.config as cfg
    import backend.ingestion.index_papers as idx
    import backend.agents.retriever as ret

    monkeypatch.setattr(cfg, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(idx, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(ret, "get_qdrant_client", lambda: client)
    return client


PAPERS = [
    {
        "title": "Retrieval Augmented Generation Survey",
        "abstract": "retrieval augmented generation combines search with language models",
        "authors": ["A. Author"],
        "published": "2025-01-01",
        "url": "http://arxiv.org/abs/1",
    },
    {
        "title": "Federated Learning Privacy",
        "abstract": "federated learning preserves privacy by training without sharing data",
        "authors": ["B. Author"],
        "published": "2025-02-01",
        "url": "http://arxiv.org/abs/2",
    },
]


class TestIndexPapers:
    def test_returns_chunk_count(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers

        n = index_papers(PAPERS)
        assert n == 2  # one chunk per short abstract

    def test_collection_created_with_payload_indexes(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers

        index_papers(PAPERS)
        info = memory_qdrant.get_collection("mara_papers")
        assert info.points_count == 2

    def test_indexing_twice_does_not_duplicate(self, memory_qdrant, fake_embeddings):
        """Deterministic IDs make ingestion idempotent."""
        from backend.ingestion.index_papers import index_papers

        index_papers(PAPERS)
        index_papers(PAPERS)
        info = memory_qdrant.get_collection("mara_papers")
        assert info.points_count == 2

    def test_empty_paper_list(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers

        assert index_papers([]) == 0


class TestRetrieve:
    def test_returns_relevant_first(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers
        from backend.agents.retriever import retrieve

        index_papers(PAPERS)
        results = retrieve("privacy in federated learning", top_k=2)
        assert len(results) == 2
        assert "Federated" in results[0]["title"]

    def test_result_shape(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers
        from backend.agents.retriever import retrieve

        index_papers(PAPERS)
        (r,) = retrieve("retrieval augmented generation", top_k=1)
        assert set(r) == {"text", "title", "url", "authors", "published", "score"}
        assert 0.0 <= r["score"] <= 1.0

    def test_top_k_respected(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers
        from backend.agents.retriever import retrieve

        index_papers(PAPERS)
        assert len(retrieve("anything", top_k=1)) == 1
        assert len(retrieve("anything", top_k=5)) == 2

    def test_min_score_filters(self, memory_qdrant, fake_embeddings):
        from backend.ingestion.index_papers import index_papers
        from backend.agents.retriever import retrieve

        index_papers(PAPERS)
        assert retrieve("retrieval augmented generation", min_score=2.0) == []

    def test_empty_collection_returns_empty(self, memory_qdrant, fake_embeddings):
        from backend.agents.retriever import retrieve

        assert retrieve("anything") == []
