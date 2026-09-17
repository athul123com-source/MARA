"""Per-run trace logging (master guide P1 item 6).

Every pipeline run writes traces/<slug>.json containing the question,
sub-questions, retrieved-chunk count, candidate claims, verdicts with
justifications, the final answer, confidence and per-agent timing.

This is deliberate instrumentation, not debugging sugar: it AUTOMATICALLY
generates the Phase 11 evaluation data (factual-consistency rate, latency
per agent, source counts) instead of having to reconstruct it before the
report deadline. Disable with TRACE=0; relocate with TRACE_DIR=<path>.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.config import trace_dir, trace_enabled


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:max_len].strip("-") or "query"


def write_trace(result: dict, traces_dir: str | None = None) -> str | None:
    """Persist one pipeline run's full state as a JSON trace.

    Returns the written path, or None when tracing is disabled or the
    result lacks the fields a trace needs (e.g. error states).
    """
    if not trace_enabled():
        return None
    question = result.get("question", "")
    if not question:
        return None

    directory = Path(traces_dir or trace_dir())
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{timestamp}_{_slugify(question)}.json"

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "sub_questions": result.get("sub_questions", []),
        "retrieved_count": len(result.get("retrieved", [])),
        "sources": sorted(
            {c["url"] for c in result.get("retrieved", []) if c.get("url")}
        ),
        "candidate_claims": result.get("candidate_claims", []),
        "verified_findings": result.get("verified_findings", []),
        "rejected_claims": result.get("rejected_claims", []),
        "verdicts": [
            {"claim": f["claim"], "verdict": "SUPPORTED"}
            for f in result.get("verified_findings", [])
        ]
        + [
            {"claim": r["claim"], "verdict": r.get("verdict", "NOT_SUPPORTED")}
            for r in result.get("rejected_claims", [])
        ],
        "final_answer": result.get("final_answer", ""),
        "confidence": result.get("confidence", 0.0),
        "total_claims": result.get("total_claims", 0),
        "supported_claims": result.get("supported_claims", 0),
        "timing_s": result.get("timing", {}),
        "latency_s": round(sum(result.get("timing", {}).values()), 3),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return str(path)


def summarize_traces(traces_dir: str | None = None) -> dict:
    """Aggregate all traces into the headline Phase 11 metrics.

    Returns: runs, avg_latency_s, factual_consistency_rate (supported /
    total claims across all runs), avg_confidence, avg_sources_per_run.
    """
    directory = Path(traces_dir or trace_dir())
    if not directory.exists():
        return {"runs": 0}

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]
    if not records:
        return {"runs": 0}

    total_claims = sum(r.get("total_claims", 0) for r in records)
    supported = sum(r.get("supported_claims", 0) for r in records)
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    return {
        "runs": len(records),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "factual_consistency_rate": round(supported / total_claims, 3) if total_claims else None,
        "avg_confidence": round(
            sum(r.get("confidence", 0.0) for r in records) / len(records), 3
        ),
        "avg_sources_per_run": round(
            sum(r.get("retrieved_count", 0) for r in records) / len(records), 1
        ),
    }


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps(summarize_traces(d), indent=2))
