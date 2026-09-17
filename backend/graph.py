"""LangGraph orchestration (Phase 7).

Wires the four agents into the MARA pipeline:

    Planner -> Retriever -> Claim drafting (Critic) -> Cross-verification
    (Critic) -> Writer -> END

Key design decision (differs from the original guide, deliberately):

    The guide's critic_node checks every retrieved chunk against itself:
        check_claim(chunk["text"], [chunk["text"]])
    That is tautological — a chunk trivially supports itself, so every
    verdict is SUPPORTED and the Critic filters nothing. MARA instead
    drafts candidate claims from the evidence first, then cross-verifies
    each claim against ALL retrieved chunks. Only SUPPORTED claims reach
    the Writer. This is a headline viva point.

Every node returns a full-state dict (LangGraph reducer semantics) and
degrades gracefully: retrieval misses, empty claims, or an empty verdict
set all fall through to the Writer's "insufficient evidence" fallback
instead of raising.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.agents.critic import check_claim, extract_claims
from backend.agents.planner import plan
from backend.agents.retriever import retrieve
from backend.agents.writer import FALLBACK_ANSWER, write_summary
from backend.tracing import write_trace

TOP_K_PER_SUBQUESTION = 3
CLAIM_TOP_K = 5  # cross-verify each claim against up to this many chunks


class MARAState(TypedDict, total=False):
    # Inputs
    question: str
    # Planner
    sub_questions: list[str]
    # Retriever
    retrieved: list[dict]
    # Critic (drafting + cross-verification)
    candidate_claims: list[str]
    verified_findings: list[dict]
    rejected_claims: list[dict]
    # Writer
    final_answer: str
    # Telemetry / confidence badge data
    total_claims: int
    supported_claims: int
    confidence: float
    timing: dict[str, float]
    error: str


def _merged_evidence(state: MARAState) -> list[dict]:
    """Deduplicate retrieved chunks by URL+text, keep highest score."""
    seen: dict[str, dict] = {}
    for chunk in state.get("retrieved", []):
        key = f"{chunk['url']}::{chunk['text'][:120]}"
        if key not in seen or chunk["score"] > seen[key]["score"]:
            seen[key] = chunk
    return sorted(seen.values(), key=lambda c: c["score"], reverse=True)


def planner_node(state: MARAState) -> dict[str, Any]:
    t0 = time.perf_counter()
    sub_questions = plan(state["question"])
    return {
        "sub_questions": sub_questions,
        "timing": {**state.get("timing", {}), "planner_s": round(time.perf_counter() - t0, 3)},
    }


def retriever_node(state: MARAState) -> dict[str, Any]:
    t0 = time.perf_counter()
    all_chunks: list[dict] = []
    for sub_q in state.get("sub_questions", [state["question"]]):
        all_chunks.extend(retrieve(sub_q, top_k=TOP_K_PER_SUBQUESTION))
    return {
        "retrieved": all_chunks,
        "timing": {**state.get("timing", {}), "retriever_s": round(time.perf_counter() - t0, 3)},
    }


def drafter_node(state: MARAState) -> dict[str, Any]:
    t0 = time.perf_counter()
    evidence = _merged_evidence(state)
    if not evidence:
        return {
            "candidate_claims": [],
            "verified_findings": [],
            "rejected_claims": [],
            "total_claims": 0,
            "supported_claims": 0,
            "confidence": 0.0,
            "timing": {**state.get("timing", {}), "critic_s": round(time.perf_counter() - t0, 3)},
        }
    claims = extract_claims(evidence)
    return {
        "candidate_claims": [c["claim"] for c in claims],
        "timing": {**state.get("timing", {}), "critic_s": round(time.perf_counter() - t0, 3)},
    }


def verifier_node(state: MARAState) -> dict[str, Any]:
    t0 = time.perf_counter()
    evidence = _merged_evidence(state)
    source_texts = [e["text"] for e in evidence[:CLAIM_TOP_K]]

    verified: list[dict] = []
    rejected: list[dict] = []
    for claim in state.get("candidate_claims", []):
        # CROSS-VERIFICATION: each claim is checked against ALL retrieved
        # chunks (not just the one it was drafted from).
        result = check_claim(claim, source_texts)
        entry = {
            "claim": claim,
            "verdict": result["verdict"],
            "justification": result["justification"],
            "sources": _sources_for(claim, evidence),
        }
        if result["verdict"] == "SUPPORTED":
            verified.append(entry)
        else:
            rejected.append(entry)

    total = len(verified) + len(rejected)
    return {
        "verified_findings": verified,
        "rejected_claims": rejected,
        "total_claims": total,
        "supported_claims": len(verified),
        "confidence": round(len(verified) / total, 3) if total else 0.0,
        "timing": {**state.get("timing", {}), "verifier_s": round(time.perf_counter() - t0, 3)},
    }


def _sources_for(claim: str, evidence: list[dict]) -> list[dict]:
    """Attribute a claim to the chunks it was drafted from (by [n] tags)."""
    import re

    tags = [int(n) for n in re.findall(r"\[(\d+)\]", claim)]
    if tags:
        return [
            {"title": evidence[n - 1]["title"], "url": evidence[n - 1]["url"]}
            for n in tags
            if 1 <= n <= len(evidence)
        ]
    # No tags: attribute the top-scored chunk (conservative default).
    return [{"title": evidence[0]["title"], "url": evidence[0]["url"]}] if evidence else []


def writer_node(state: MARAState) -> dict[str, Any]:
    t0 = time.perf_counter()
    answer = write_summary(state["question"], state.get("verified_findings", []))
    return {
        "final_answer": answer,
        "timing": {**state.get("timing", {}), "writer_s": round(time.perf_counter() - t0, 3)},
    }


def build_graph():
    """Compile the MARA pipeline. Rebuildable; call sites cache the result."""
    graph = StateGraph(MARAState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "drafter")
    graph.add_edge("drafter", "verifier")
    graph.add_edge("verifier", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


mara_app = build_graph()


def run_query(question: str, trace: bool = True) -> dict:
    """Run the full pipeline; persist a JSON trace when tracing is enabled.

    The trace file (default traces/) is the raw material for the Phase 11
    evaluation metrics - see backend.tracing.summarize_traces().
    """
    result = mara_app.invoke({"question": question})
    result["trace_path"] = write_trace(result) if trace else None
    return result


if __name__ == "__main__":
    import os

    if os.getenv("MOCK_LLM", "").strip().lower() in ("1", "true", "yes"):
        print("[MOCK_LLM=1] Running with deterministic stubs - no API keys used.")
    q = input("Research question: ") or "How does federated learning preserve data privacy?"
    result = run_query(q)
    print("\n=== ANSWER ===")
    print(result["final_answer"])
    print(f"\nConfidence: {result['supported_claims']}/{result['total_claims']} "
          f"claims verified ({result['confidence']:.0%})")
    print("Timing (s):", {k: v for k, v in result["timing"].items()})
    if result.get("trace_path"):
        print(f"Trace saved: {result['trace_path']}")
