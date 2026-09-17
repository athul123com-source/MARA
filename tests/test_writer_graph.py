"""Offline tests for the Writer agent and the LangGraph orchestrator.

No network, no API keys: Anthropic clients are stubbed at _get_client,
and the retriever is stubbed so graph tests exercise real orchestration
logic (state flow, cross-verification, confidence math) without Qdrant.
"""
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.writer import FALLBACK_ANSWER, format_findings, write_summary
from backend.graph import (
    MARAState,
    _merged_evidence,
    build_graph,
    drafter_node,
    verifier_node,
    writer_node,
)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
def _mock_writer_client(response_text: str):
    fake = MagicMock()
    fake.content = [MagicMock(text=response_text)]
    client = MagicMock()
    client.messages.create.return_value = fake
    return patch("backend.agents.writer._get_client", return_value=client)


class TestWriter:
    def test_writes_summary_from_findings(self):
        findings = [{
            "claim": "FL keeps raw data on-device.",
            "sources": [{"title": "Paper A", "url": "u1"}],
        }]
        with _mock_writer_client("FL preserves privacy [Source: Paper A]."):
            out = write_summary("Why FL?", findings)
        assert "[Source: Paper A]" in out

    def test_empty_findings_returns_fallback(self):
        assert write_summary("Why FL?", []) == FALLBACK_ANSWER

    def test_empty_model_output_falls_back(self):
        with _mock_writer_client(""):
            assert write_summary("Why FL?", [{"claim": "c", "sources": []}]) == FALLBACK_ANSWER

    def test_prompt_contains_question_and_findings(self):
        with _mock_writer_client("answer") as ctx:
            client = ctx.return_value
            write_summary("THE QUESTION", [{"claim": "claim text", "sources": []}])
        prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "THE QUESTION" in prompt and "claim text" in prompt

    def test_format_findings_dedupes_titles(self):
        findings = [{
            "claim": "c",
            "sources": [{"title": "B", "url": "2"}, {"title": "A", "url": "1"}, {"title": "B", "url": "3"}],
        }]
        text = format_findings(findings)
        assert "A, B" in text and "B, A" not in text


# ---------------------------------------------------------------------------
# Orchestrator nodes (unit-level, stubbed agents)
# ---------------------------------------------------------------------------
class TestMergedEvidence:
    def test_dedupes_by_url_and_keeps_best_score(self):
        state = {"retrieved": [
            {"text": "t1", "title": "A", "url": "u", "score": 0.7},
            {"text": "t1", "title": "A", "url": "u", "score": 0.9},
            {"text": "t2", "title": "B", "url": "v", "score": 0.5},
        ]}
        merged = _merged_evidence(state)
        assert len(merged) == 2
        assert merged[0]["score"] == 0.9


class TestVerifierNode:
    def _evidence(self):
        return [
            {"text": "Dense evidence about X.", "title": "Paper X", "url": "x", "score": 0.9},
            {"text": "Sparse evidence about Y.", "title": "Paper Y", "url": "y", "score": 0.8},
        ]

    def test_cross_verification_filters_unsupported(self):
        state = {
            "retrieved": self._evidence(),
            "candidate_claims": ["Claim A.", "Claim B."],
        }

        def fake_check(claim, sources, **kw):
            verdict = "SUPPORTED" if claim == "Claim A." else "NOT_SUPPORTED"
            return {"verdict": verdict, "justification": "j", "verdict_raw": ""}

        with patch("backend.graph.check_claim", side_effect=fake_check):
            out = verifier_node(state)
        assert len(out["verified_findings"]) == 1
        assert out["verified_findings"][0]["claim"] == "Claim A."
        assert out["rejected_claims"][0]["claim"] == "Claim B."
        assert out["confidence"] == 0.5
        assert out["total_claims"] == 2

    def test_each_claim_checked_against_all_evidence(self):
        state = {"retrieved": self._evidence(), "candidate_claims": ["A."]}
        seen_sources = []

        def fake_check(claim, sources, **kw):
            seen_sources.append(sources)
            return {"verdict": "SUPPORTED", "justification": "", "verdict_raw": ""}

        with patch("backend.graph.check_claim", side_effect=fake_check):
            verifier_node(state)
        # The claim was checked against BOTH chunks, not just its own.
        assert len(seen_sources[0]) == 2

    def test_empty_candidates_zero_confidence(self):
        out = verifier_node({"retrieved": self._evidence(), "candidate_claims": []})
        assert out["confidence"] == 0.0 and out["verified_findings"] == []


class TestDrafterNode:
    def test_no_evidence_short_circuits(self):
        out = drafter_node({"retrieved": [], "question": "q"})
        assert out["candidate_claims"] == []
        assert out["verified_findings"] == [] and out["confidence"] == 0.0


class TestWriterNode:
    def test_writes_from_verified_findings(self):
        state = {"question": "q", "verified_findings": [{"claim": "c", "sources": []}]}
        with patch("backend.graph.write_summary", return_value="final answer"):
            out = writer_node(state)
        assert out["final_answer"] == "final answer"


# ---------------------------------------------------------------------------
# Full graph (integration over stubbed agents)
# ---------------------------------------------------------------------------
def _stub_all_agents(evidence, verdicts):
    """Patch planner/retriever/critic/writer inside backend.graph."""
    p_planner = patch("backend.graph.plan", return_value=["sub q1", "sub q2"])
    p_retriever = patch("backend.graph.retrieve", side_effect=lambda q, top_k=3: evidence)
    p_extract = patch(
        "backend.graph.extract_claims",
        return_value=[{"claim": c, "sources": []} for c in verdicts],
    )
    p_check = patch(
        "backend.graph.check_claim",
        side_effect=lambda claim, sources, **kw: {
            "verdict": verdicts[claim],
            "justification": "j",
            "verdict_raw": "",
        },
    )
    p_writer = patch("backend.graph.write_summary", return_value="FINAL ANSWER")
    return p_planner, p_retriever, p_extract, p_check, p_writer


class TestFullGraph:
    def test_end_to_end_happy_path(self):
        evidence = [
            {"text": "chunk about A", "title": "T1", "url": "u1", "score": 0.9},
            {"text": "chunk about B", "title": "T2", "url": "u2", "score": 0.8},
        ]
        patches = _stub_all_agents(evidence, {"A claim.": "SUPPORTED", "B claim.": "NOT_SUPPORTED"})
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = build_graph().invoke({"question": "test question?"})
        assert result["final_answer"] == "FINAL ANSWER"
        assert result["supported_claims"] == 1
        assert result["total_claims"] == 2
        assert result["confidence"] == 0.5
        assert "planner_s" in result["timing"] and "writer_s" in result["timing"]

    def test_no_evidence_degrades_to_fallback(self):
        patches = _stub_all_agents([], {})
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = build_graph().invoke({"question": "obscure question?"})
        # Writer stub returns FINAL ANSWER even with no findings, but the
        # pipeline must reach the writer with empty findings and zero conf.
        assert result["verified_findings"] == []
        assert result["confidence"] == 0.0

    def test_all_claims_rejected_still_answers_via_fallback(self):
        evidence = [{"text": "e", "title": "T", "url": "u", "score": 0.9}]
        patches = _stub_all_agents(evidence, {"Only claim.": "NOT_SUPPORTED"})
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with patch("backend.graph.write_summary", return_value=FALLBACK_ANSWER):
                result = build_graph().invoke({"question": "q?"})
        assert result["final_answer"] == FALLBACK_ANSWER
        assert result["supported_claims"] == 0

    def test_state_type_covers_telemetry(self):
        # MARAState must declare the keys nodes write (LangGraph TypedDict).
        assert "confidence" in MARAState.__annotations__
        assert "timing" in MARAState.__annotations__
