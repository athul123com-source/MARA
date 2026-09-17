"""Offline tests for MOCK_LLM mode and per-run trace logging.

Verifies that the entire pipeline runs with zero API keys in mock mode,
that mock outputs are deterministic, and that traces are written and
summarized correctly (the raw material for Phase 11 metrics).
"""
import json

import pytest

from backend.config import mock_llm_enabled
from backend.mock_llm import MockAnthropic, MockEmbeddings, _respond
from backend.tracing import summarize_traces, write_trace


# ---------------------------------------------------------------------------
# Mock mode: config flag
# ---------------------------------------------------------------------------
class TestMockFlag:
    def test_flag_off_by_default(self, monkeypatch):
        monkeypatch.delenv("MOCK_LLM", raising=False)
        assert mock_llm_enabled() is False

    def test_flag_on_variants(self, monkeypatch):
        for v in ("1", "true", "YES", "yes "):
            monkeypatch.setenv("MOCK_LLM", v)
            assert mock_llm_enabled() is True


# ---------------------------------------------------------------------------
# Mock mode: deterministic outputs per agent
# ---------------------------------------------------------------------------
class TestMockAnthropic:
    def test_planner_prompt_gets_json_list(self):
        out = MockAnthropic().messages.create(
            messages=[{"role": "user", "content": "break it into sub-questions\nResearch question: How does FL preserve privacy?"}]
        )
        parsed = json.loads(out.content[0].text)
        assert isinstance(parsed, list) and len(parsed) == 4

    def test_claims_prompt_gets_json_list(self):
        out = MockAnthropic().messages.create(
            messages=[{"role": "user", "content": "Extract factual claims\nEXCERPTS:\n[1] Federated learning keeps data on device.\n---\n[2] RAG reduces hallucinations."}]
        )
        parsed = json.loads(out.content[0].text)
        assert len(parsed) == 2 and "[1]" in parsed[0]

    def test_verdict_supported_when_claim_words_in_excerpts(self):
        prompt = (
            "VERDICT: SUPPORTED | NOT_SUPPORTED\n\n"
            "CLAIM: Federated learning keeps data on device.\n\n"
            "SOURCE EXCERPTS:\nEXCERPTS:\n[1] Federated learning keeps data on device."
        )
        out = MockAnthropic().messages.create(messages=[{"role": "user", "content": prompt}])
        assert "VERDICT: SUPPORTED" in out.content[0].text

    def test_verdict_not_supported_for_unrelated_claim(self):
        prompt = (
            "VERDICT: SUPPORTED | NOT_SUPPORTED\n\n"
            "CLAIM: Quantum computers use entangled qubits.\n\n"
            "SOURCE EXCERPTS:\nEXCERPTS:\n[1] Federated learning keeps data on device."
        )
        out = MockAnthropic().messages.create(messages=[{"role": "user", "content": prompt}])
        assert "VERDICT: NOT_SUPPORTED" in out.content[0].text

    def test_writer_prompt_gets_summary(self):
        out = MockAnthropic().messages.create(
            messages=[{"role": "user", "content": "write a well-organized summary\nVERIFIED FINDINGS:\n- Claim one"}]
        )
        assert "Claim one" in out.content[0].text

    def test_deterministic_across_calls(self):
        p = "break it into sub-questions\nResearch question: Test?"
        a = MockAnthropic().messages.create(messages=[{"role": "user", "content": p}]).content[0].text
        b = MockAnthropic().messages.create(messages=[{"role": "user", "content": p}]).content[0].text
        assert a == b


# ---------------------------------------------------------------------------
# Mock mode: embeddings + end-to-end pipeline without keys
# ---------------------------------------------------------------------------
class TestMockEmbeddings:
    def test_same_text_same_vector(self):
        e = MockEmbeddings(dim=64)
        assert e.embed_texts(["hello world"]) == e.embed_texts(["hello world"])

    def test_related_texts_more_similar_than_unrelated(self):
        e = MockEmbeddings(dim=256)
        a = e.embed_texts(["federated learning privacy"])[0]
        b = e.embed_texts(["federated learning security"])[0]
        c = e.embed_texts(["chocolate cake recipe"])[0]
        cos = lambda x, y: sum(i * j for i, j in zip(x, y))
        assert cos(a, b) > cos(a, c)


class TestMockPipelineEndToEnd:
    def test_run_query_with_no_keys(self, monkeypatch, tmp_path):
        """Full pipeline with MOCK_LLM=1: no Anthropic/OpenAI keys needed."""
        monkeypatch.setenv("MOCK_LLM", "1")
        monkeypatch.setenv("TRACE_DIR", str(tmp_path))  # keep traces out of repo
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("backend.graph.retrieve", lambda q, top_k=3: [
            {"text": "Federated learning trains models without sharing raw data.",
             "title": "FL Paper", "url": "http://x/1", "score": 0.9},
        ])
        # Import inside so patched retriever is used; reimport config fresh.
        import importlib
        import backend.graph as g
        importlib.reload(g)
        try:
            result = g.run_query("How does federated learning preserve privacy?")
        finally:
            importlib.reload(g)  # restore original graph for other tests
        assert result["sub_questions"]
        assert result["final_answer"]
        assert result["trace_path"]  # tracing wrote a file


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
def _sample_result(question: str = "What is RAG?") -> dict:
    return {
        "question": question,
        "sub_questions": ["q1", "q2"],
        "retrieved": [
            {"text": "t", "title": "A", "url": "http://a", "score": 0.9},
            {"text": "u", "title": "B", "url": "http://b", "score": 0.8},
        ],
        "candidate_claims": ["claim ok", "claim bad"],
        "verified_findings": [{"claim": "claim ok", "verdict": "SUPPORTED", "sources": []}],
        "rejected_claims": [{"claim": "claim bad", "verdict": "NOT_SUPPORTED"}],
        "final_answer": "answer text",
        "confidence": 0.5,
        "total_claims": 2,
        "supported_claims": 1,
        "timing": {"planner_s": 0.1, "retriever_s": 0.2, "writer_s": 0.3},
    }


class TestWriteTrace:
    def test_writes_json_trace(self, tmp_path):
        path = write_trace(_sample_result(), traces_dir=str(tmp_path))
        assert path and path.endswith(".json")
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["question"] == "What is RAG?"
        assert data["confidence"] == 0.5
        assert data["latency_s"] == 0.6
        assert len(data["verdicts"]) == 2

    def test_disabled_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRACE", "0")
        assert write_trace(_sample_result(), traces_dir=str(tmp_path)) is None

    def test_empty_question_returns_none(self, tmp_path):
        assert write_trace({"question": ""}, traces_dir=str(tmp_path)) is None


class TestSummarizeTraces:
    def test_aggregates_metrics(self, tmp_path):
        write_trace(_sample_result("q one"), traces_dir=str(tmp_path))
        write_trace(_sample_result("q two"), traces_dir=str(tmp_path))
        s = summarize_traces(str(tmp_path))
        assert s["runs"] == 2
        assert s["factual_consistency_rate"] == 0.5
        assert s["avg_confidence"] == 0.5
        assert s["avg_latency_s"] == 0.6

    def test_empty_dir(self, tmp_path):
        assert summarize_traces(str(tmp_path))["runs"] == 0
