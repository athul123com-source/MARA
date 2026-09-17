"""MARA backend API (Phase 8).

Serves the LangGraph pipeline over HTTP:

    POST /query          -> run the full pipeline, return answer + confidence
                            + sources + per-agent timing
    GET  /trace          -> aggregated Phase 11 metrics over all trace files
    GET  /trace/latest   -> the full trace of the most recent run
    GET  /health         -> liveness probe (used by Streamlit and Render)

The graph import is deferred to request time so the module imports cleanly
without API keys (and so tests can monkeypatch backend.graph.run_query).
Run:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import mock_llm_enabled, trace_dir

app = FastAPI(
    title="MARA API",
    description="Multi-Agent Research Assistant: Planner -> Retriever -> "
    "Drafter -> Verifier -> Writer, with cross-verification.",
    version="0.1.0",
)

# Wide-open CORS for local development (Streamlit on another port).
# Tighten origins before any public deployment (Phase 10).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=[
        "How does federated learning preserve data privacy?"])


def _sources_from(verified_findings: list[dict]) -> list[dict]:
    """Deduplicate sources from verified findings, preserving first-seen order.

    The verifier stores each claim's provenance in finding["sources"] (a list
    of {title, url}); a flat finding["title"]/["url"] shape is also accepted
    for robustness against older traces.
    """
    seen: dict[str, dict] = {}
    for finding in verified_findings:
        candidates = list(finding.get("sources") or [])
        if finding.get("url"):
            candidates.append({"title": finding.get("title", ""), "url": finding["url"]})
        for src in candidates:
            url = src.get("url", "")
            if url and url not in seen:
                seen[url] = {"title": src.get("title", ""), "url": url}
    return list(seen.values())


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """Run the full MARA pipeline for one research question."""
    from backend.graph import run_query  # deferred: no keys needed to import app

    try:
        result = run_query(request.question)
    except Exception as exc:  # noqa: BLE001 - surface pipeline errors to the caller
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return {
        "question": result["question"],
        "answer": result["final_answer"],
        "confidence": result["confidence"],
        "supported_claims": result["supported_claims"],
        "total_claims": result["total_claims"],
        "sources": _sources_from(result.get("verified_findings", [])),
        "sub_questions": result.get("sub_questions", []),
        "rejected_claims": result.get("rejected_claims", []),
        "timing_s": result.get("timing", {}),
        "trace_path": result.get("trace_path"),
    }


@app.get("/trace")
def trace() -> dict:
    """Aggregated Phase 11 metrics across every recorded run."""
    from backend.tracing import summarize_traces

    summary = summarize_traces()
    summary["trace_dir"] = trace_dir()
    return summary


@app.get("/trace/latest")
def trace_latest() -> dict:
    """The full JSON trace of the most recent pipeline run."""
    directory = Path(trace_dir())
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise HTTPException(status_code=404, detail="No traces recorded yet.")
    record = json.loads(files[-1].read_text(encoding="utf-8"))
    record["trace_file"] = files[-1].name
    return record


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_llm": mock_llm_enabled()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
