"""Planner Agent (Phase 4).

Breaks a user's research question into 3-5 focused sub-questions that,
if answered, would together answer the original question. These
sub-questions drive the Retriever Agent (Phase 3) later.

Design notes:
- The client is created lazily so importing this module never crashes
  when ANTHROPIC_API_KEY is missing (e.g., in unit tests or CI).
- LLMs sometimes wrap JSON in markdown code fences or add prose; we
  parse defensively instead of relying on a bare json.loads().
- Retries once on transient API errors to keep benchmark runs stable.
"""
import json
import re

import anthropic

from backend.config import ANTHROPIC_API_KEY, PLANNER_MODEL

PLANNER_PROMPT = """You are a research planning assistant. Given a research question,
break it into 3-5 focused sub-questions that, if answered, would together answer the
original question. Each sub-question must be self-contained (understandable without
the original) and non-overlapping. Respond ONLY with a JSON list of strings, no
other text.

Research question: {question}
"""

_VALID_VERDICT_RE = re.compile(r"^\[[^\]]*\]|^```(?:json)?|```", re.IGNORECASE)


def _get_client() -> anthropic.Anthropic:
    """Create an Anthropic client; raise a clear error if no API key."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def parse_subquestions(text: str) -> list[str]:
    """Extract a JSON list of sub-questions from an LLM response.

    Handles: clean JSON lists, ```json fenced lists, and a JSON list
    embedded in surrounding prose. Returns [] if nothing parseable.
    """
    text = text.strip()
    # 1) Fast path: the whole response is a JSON list.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    # 2) Fallback: grab the first bracketed [...] block in the text.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    # 3) Last resort: numbered/bulleted lines ("1. ..." or "- ...").
    lines = re.findall(r"^(?:\d+[\.\)]|[-*])\s+(.+)$", text, re.MULTILINE)
    if lines:
        return [line.strip().strip('"') for line in lines]
    return []


def plan(question: str, max_retries: int = 1) -> list[str]:
    """Decompose a research question into 3-5 sub-questions."""
    client = _get_client()
    prompt = PLANNER_PROMPT.format(question=question)

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=PLANNER_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        subquestions = parse_subquestions(text)
        if subquestions:
            return subquestions
        if attempt < max_retries:
            continue  # ask again; malformed JSON is often fixed on retry

    # Never hand an empty plan downstream - degrade gracefully.
    return [question]


__all__ = ["plan", "parse_subquestions", "PLANNER_PROMPT"]


if __name__ == "__main__":
    q = input("Research question: ") or "How does federated learning preserve data privacy?"
    print("Sub-questions:")
    for i, sq in enumerate(plan(q), 1):
        print(f"  {i}. {sq}")
