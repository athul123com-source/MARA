"""MARA frontend (Phase 9).

A Streamlit chat interface over the FastAPI backend:

    streamlit run ui/app.py            # expects the API on http://localhost:8000
    MOCK_LLM=1 uvicorn backend.main:app --port 8000   # keyless demo backend

Design notes:
- Pure helpers (validation, confidence labels, timing formatting) live at the
  top and are unit-tested offline in tests/test_ui.py - no Streamlit runtime
  or network needed.
- The UI is a thin, honest client: one POST /query per question, no fake
  streaming. Per-agent progress is shown from the real timing_s data the
  pipeline returns.
"""
from __future__ import annotations

import httpx
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
REQUEST_TIMEOUT_S = 180
MAX_QUESTION_LEN = 2000  # mirrors QueryRequest.max_length on the API

DEMO_QUESTIONS = [
    "How does federated learning preserve data privacy?",
    "What are the main approaches to detecting hallucinations in LLMs?",
    "How do transformer attention mechanisms work?",
]

AGENT_ORDER = ["planner_s", "retriever_s", "critic_s", "verifier_s", "writer_s"]
AGENT_LABELS = {
    "planner_s": "Planner",
    "retriever_s": "Retriever",
    "critic_s": "Critic (draft)",
    "verifier_s": "Verifier",
    "writer_s": "Writer",
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# ---------------------------------------------------------------------------
def validate_question(question: str) -> str | None:
    """Return an error message, or None if the question is API-valid."""
    q = (question or "").strip()
    if len(q) < 3:
        return "Please enter a research question (at least 3 characters)."
    if len(q) > MAX_QUESTION_LEN:
        return f"Question too long ({len(q)} chars) - the API accepts up to {MAX_QUESTION_LEN}."
    return None


def confidence_label(confidence: float) -> str:
    """Map claim-verification confidence to a demo-friendly label."""
    if confidence >= 0.75:
        return "High"
    if confidence >= 0.4:
        return "Medium"
    return "Low"


def format_timing(timing: dict[str, float]) -> list[tuple[str, float]]:
    """Return [(label, seconds)] in pipeline order, dropping missing agents."""
    return [
        (AGENT_LABELS[key], float(timing[key]))
        for key in AGENT_ORDER
        if key in timing
    ]


def total_latency(timing: dict[str, float]) -> float:
    """Sum of all per-agent timings (the pipeline is sequential)."""
    return round(sum(float(v) for v in timing.values()), 3)


# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------
def check_health(base_url: str) -> dict | None:
    """GET /health; None when the API is unreachable."""
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def run_pipeline(base_url: str, question: str) -> tuple[dict | None, str | None]:
    """POST /query. Returns (payload, error_message)."""
    url = f"{base_url.rstrip('/')}/query"
    try:
        response = httpx.post(url, json={"question": question}, timeout=REQUEST_TIMEOUT_S)
    except httpx.ConnectError:
        return None, (
            "Cannot reach the MARA API. Start it first:\n"
            "```\nMOCK_LLM=1 uvicorn backend.main:app --port 8000\n```"
        )
    except httpx.HTTPError as exc:
        return None, f"Request failed: {exc}"

    if response.status_code == 500:
        detail = response.json().get("detail", "pipeline error")
        return None, f"The pipeline failed: {detail}"
    if response.status_code == 422:
        return None, "The API rejected that question (length limits?)."
    response.raise_for_status()
    return response.json(), None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_answer(payload: dict) -> None:
    """Render one pipeline result inside an assistant chat bubble."""
    timing = payload.get("timing_s", {})
    supported = payload.get("supported_claims", 0)
    total = payload.get("total_claims", 0)
    confidence = float(payload.get("confidence", 0.0))

    st.markdown(payload.get("answer", "_No answer returned._"))

    # Confidence badge + bar (headline metric of the Critic stage)
    label = confidence_label(confidence)
    st.progress(min(max(confidence, 0.0), 1.0),
                text=f"Confidence: {supported}/{total} claims verified "
                     f"({confidence:.0%}) — {label}")

    # Per-agent timing from the real pipeline telemetry
    stages = format_timing(timing)
    if stages:
        cols = st.columns(len(stages))
        for col, (name, seconds) in zip(cols, stages):
            col.metric(name, f"{seconds:.2f}s")
        st.caption(f"Total pipeline latency: {total_latency(timing):.2f}s")

    # Evidence: sources, planner plan, claim verdicts
    sources = payload.get("sources", [])
    with st.expander(f"📚 Sources ({len(sources)})", expanded=bool(sources)):
        for i, src in enumerate(sources, 1):
            title = src.get("title") or src.get("url", "source")
            st.markdown(f"{i}. [{title}]({src.get('url', '')})")

    sub_questions = payload.get("sub_questions", [])
    if sub_questions:
        with st.expander(f"🧭 Planner sub-questions ({len(sub_questions)})"):
            for q in sub_questions:
                st.markdown(f"- {q}")

    with st.expander("✅ Claim verification detail"):
        for finding in payload.get("verified_findings", []):
            n_src = len(finding.get("sources", []))
            st.markdown(f"- ✔️ {finding.get('claim', '')}  \n\n"
                        f"  {finding.get('justification', '')} _(sources: {n_src})_")
    rejected = payload.get("rejected_claims", [])
    if rejected:
        with st.expander(f"🚫 Filtered by the Critic ({len(rejected)})"):
            for finding in rejected:
                st.markdown(f"- ❌ {finding.get('claim', '')}")

    if payload.get("trace_path"):
        st.caption(f"Trace saved: {payload['trace_path']}")


def main() -> None:
    st.set_page_config(page_title="MARA", page_icon="🔬",
                       layout="centered", initial_sidebar_state="expanded")

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.header("🔬 MARA")
        st.caption("Multi-Agent Research Assistant\n\n"
                   "Planner → Retriever → Critic → Writer")

        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        health = check_health(api_url)
        if health:
            if health.get("mock_llm"):
                st.info("🟢 API online — **mock mode** (no API keys used)")
            else:
                st.success("🟢 API online — live LLM mode")
        else:
            st.error("🔴 API offline — start it with:\n"
                     "`MOCK_LLM=1 uvicorn backend.main:app --port 8000`")

        st.divider()
        st.subheader("Try a demo question")
        for q in DEMO_QUESTIONS:
            if st.button(q, key=f"demo:{q}"):
                st.session_state.pending_question = q
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    # ---------------- chat ----------------
    st.title("Ask MARA anything 🔬")
    st.caption("Research questions in, verified summaries with citations out — "
               "every claim is cross-checked against the retrieved evidence.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        with st.chat_message(role):
            if role == "assistant":
                render_answer(content)
            else:
                st.markdown(content)

    prompt = st.chat_input("e.g. How does federated learning preserve data privacy?")
    pending = st.session_state.pop("pending_question", None) or prompt
    if not pending:
        return

    error = validate_question(pending)
    if error:
        st.error(error)
        return

    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user"):
        st.markdown(pending)

    with st.chat_message("assistant"):
        with st.spinner("Running the agent pipeline: planning → retrieving → "
                        "verifying → writing..."):
            payload, api_error = run_pipeline(api_url, pending)
        if api_error:
            st.error(api_error)
            return
        render_answer(payload)
        st.session_state.messages.append({"role": "assistant", "content": payload})


if __name__ == "__main__":
    main()
