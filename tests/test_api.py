"""Offline tests for the FastAPI backend (Phase 8).

Two layers, both with zero API keys and zero network:
- Unit: /health, /trace and /trace/latest against a temp TRACE_DIR,
  plus source-deduplication logic.
- Integration: /query with backend.graph.run_query monkeypatched to a
  canned result (the real pipeline is covered by test_writer_graph.py
  and test_mock_tracing.py); verifies payload shape, validation and
  error mapping.
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.config import mock_llm_enabled
from backend.main import _sources_from, app


client = TestClient(app)


CANNED_RESULT = {
    "question": "How does federated learning preserve data privacy?",
    "final_answer": "Federated learning keeps raw data on-device [Source: FedAvg Paper].",
    "confidence": 0.75,
    "supported_claims": 3,
    "total_claims": 4,
    "verified_findings": [
        {"claim": "Raw data never leaves the device.",
         "sources": [{"title": "FedAvg Paper", "url": "https://arxiv.org/abs/1602.05629"}]},
        {"claim": "Updates are averaged on the server.",
         "sources": [{"title": "FedAvg Paper", "url": "https://arxiv.org/abs/1602.05629"}]},
        {"claim": "Secure aggregation hides individual updates.",
         "sources": [{"title": "SecAgg Paper", "url": "https://arxiv.org/abs/1610.04436"}]},
    ],
    "sub_questions": ["What is federated learning?", "How is privacy preserved?"],
    "rejected_claims": [{"claim": "FL is unbreakable.", "verdict": "NOT_SUPPORTED"}],
    "timing": {"planner_s": 0.4, "retriever_s": 0.1, "writer_s": 0.9},
    "trace_path": "traces/20260917-000000_test.json",
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mock_llm"] is mock_llm_enabled()


# ---------------------------------------------------------------------------
# Source deduplication
# ---------------------------------------------------------------------------

def test_sources_deduplicated_preserving_order():
    sources = _sources_from(CANNED_RESULT["verified_findings"])
    assert sources == [
        {"title": "FedAvg Paper", "url": "https://arxiv.org/abs/1602.05629"},
        {"title": "SecAgg Paper", "url": "https://arxiv.org/abs/1610.04436"},
    ]


def test_sources_empty_for_no_findings():
    assert _sources_from([]) == []


def test_sources_accept_flat_legacy_shape():
    findings = [{"claim": "c", "title": "T", "url": "https://x/1"}]
    assert _sources_from(findings) == [{"title": "T", "url": "https://x/1"}]


# ---------------------------------------------------------------------------
# /query
# ---------------------------------------------------------------------------

def test_query_returns_full_payload(monkeypatch):
    import backend.graph as graph

    calls = []
    monkeypatch.setattr(graph, "run_query", lambda q: (calls.append(q), dict(CANNED_RESULT))[1])

    response = client.post("/query", json={"question": CANNED_RESULT["question"]})
    assert response.status_code == 200
    body = response.json()

    # The question flows through to the pipeline unchanged.
    assert calls == [CANNED_RESULT["question"]]

    assert body["answer"] == CANNED_RESULT["final_answer"]
    assert body["confidence"] == 0.75
    assert body["supported_claims"] == 3
    assert body["total_claims"] == 4
    assert len(body["sources"]) == 2                       # deduplicated
    assert body["sources"][0]["url"].startswith("https://arxiv.org")
    assert body["sub_questions"] == CANNED_RESULT["sub_questions"]
    assert body["rejected_claims"] == CANNED_RESULT["rejected_claims"]
    assert body["timing_s"]["writer_s"] == 0.9
    assert body["trace_path"].endswith(".json")


def test_query_maps_pipeline_errors_to_500(monkeypatch):
    import backend.graph as graph

    def boom(_q):
        raise RuntimeError("Qdrant unreachable")

    monkeypatch.setattr(graph, "run_query", boom)
    response = client.post("/query", json={"question": "anything at all"})
    assert response.status_code == 500
    assert "Qdrant unreachable" in response.json()["detail"]


def test_query_rejects_too_short_question():
    response = client.post("/query", json={"question": "hi"})
    assert response.status_code == 422  # pydantic min_length=3


# ---------------------------------------------------------------------------
# /trace and /trace/latest
# ---------------------------------------------------------------------------

@pytest.fixture()
def trace_dir(tmp_path, monkeypatch):
    """Point TRACE_DIR at a temp dir; write one canned trace into it."""
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    record = {
        "timestamp_utc": "2026-09-17T10:00:00+00:00",
        "question": "What is federated learning?",
        "sub_questions": ["define FL"],
        "retrieved_count": 6,
        "sources": ["https://arxiv.org/abs/1602.05629"],
        "candidate_claims": ["FL trains on-device."],
        "verdicts": [{"claim": "FL trains on-device.", "verdict": "SUPPORTED"}],
        "verified_findings": [{"claim": "FL trains on-device.",
                               "title": "FedAvg", "url": "https://arxiv.org/abs/1602.05629"}],
        "rejected_claims": [],
        "final_answer": "FL trains models on-device.",
        "confidence": 1.0,
        "total_claims": 1,
        "supported_claims": 1,
        "timing_s": {"planner_s": 0.3},
        "latency_s": 1.8,
    }
    path = tmp_path / "20260917-100000_what-is-federated-learning.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def test_trace_aggregates_metrics(trace_dir):
    response = client.get("/trace")
    assert response.status_code == 200
    body = response.json()
    assert body["runs"] == 1
    assert body["avg_latency_s"] == 1.8
    assert body["factual_consistency_rate"] == 1.0
    assert body["avg_confidence"] == 1.0
    assert body["avg_sources_per_run"] == 6.0
    assert body["trace_dir"] == str(trace_dir)


def test_trace_latest_returns_most_recent_record(trace_dir):
    response = client.get("/trace/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is federated learning?"
    assert body["trace_file"] == "20260917-100000_what-is-federated-learning.json"


def test_trace_latest_404_when_no_traces(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path))
    response = client.get("/trace/latest")
    assert response.status_code == 404


def test_trace_empty_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "does-not-exist"))
    body = client.get("/trace").json()
    assert body["runs"] == 0
