"""Writer Agent (Phase 6).

Composes the final, citation-backed summary from the *verified* findings
that survived the Critic. Design rules:

- Uses ONLY verified findings — never adds information on its own.
- Cites every claim inline as [Source: <title>].
- Graceful degradation: with no verified findings it returns the
  "insufficient evidence" fallback (orchestrator also handles this, but
  the agent stays safe when called directly).
- Same conventions as the other agents: lazy client, retry, offline
  testability, standalone demo.
"""
from __future__ import annotations

import anthropic

from backend.config import ANTHROPIC_API_KEY, WRITER_MODEL, mock_llm_enabled
from backend.mock_llm import MockAnthropic

WRITER_PROMPT = """You are a research writing assistant. Using ONLY the verified
findings below, write a clear, well-organized summary answering the original
question.

Rules:
- Cite each claim inline using [Source: <title>] format.
- Do not add any information not present in the verified findings.
- If findings conflict, present both positions and attribute each.
- 3-5 short paragraphs maximum.

ORIGINAL QUESTION: {question}

VERIFIED FINDINGS:
{findings}
"""

FALLBACK_ANSWER = (
    "Insufficient verified evidence was found to answer this question reliably. "
    "Try rephrasing the question or indexing more papers on this topic."
)


def _get_client() -> anthropic.Anthropic:
    """Return the mock client in MOCK_LLM mode, else the real Anthropic client."""
    if mock_llm_enabled():
        return MockAnthropic()  # type: ignore[return-value]
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def format_findings(findings: list[dict]) -> str:
    """Render verified findings (claim + sources) for the prompt."""
    blocks = []
    for f in findings:
        titles = ", ".join(sorted({s["title"] for s in f.get("sources", [])})) or "Unknown"
        blocks.append(f"- {f['claim']} [Source: {titles}]")
    return "\n\n".join(blocks)


def write_summary(question: str, findings: list[dict], max_retries: int = 1) -> str:
    """Write the final cited summary from verified findings.

    Falls back to FALLBACK_ANSWER when there is nothing verified to write
    about, so the pipeline never ends empty or crashed.
    """
    if not findings:
        return FALLBACK_ANSWER

    client = _get_client()
    prompt = WRITER_PROMPT.format(question=question, findings=format_findings(findings))

    last_text = ""
    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=WRITER_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        last_text = response.content[0].text.strip()
        if last_text:
            return last_text

    # Model returned empty content repeatedly — degrade gracefully.
    return FALLBACK_ANSWER


__all__ = ["write_summary", "format_findings", "FALLBACK_ANSWER", "WRITER_PROMPT"]


if __name__ == "__main__":
    q = input("Research question: ") or "How does federated learning preserve privacy?"
    demo_findings = [
        {
            "claim": (
                "Federated learning lets clients train a shared model while raw "
                "data stays on-device."
            ),
            "sources": [{"title": "Communication-Efficient Learning of Deep Networks", "url": ""}],
        },
    ]
    print("\nWriting summary...\n")
    print(write_summary(q, demo_findings))
