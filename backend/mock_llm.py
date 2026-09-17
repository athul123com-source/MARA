"""Mock LLM clients for offline operation (master guide P1 item 5).

With MOCK_LLM=1, every agent's `_get_client()` returns a deterministic
in-process stub instead of hitting Anthropic/OpenAI. Nothing is random,
so demos and tests are reproducible, and no API key is ever needed.

Design: MockAnthropic mimics the small slice of the Anthropic SDK that
MARA uses (`messages.create` returning `.content[0].text`). Each agent
gets output shaped for its prompt, detected from the prompt text:
- Planner  -> JSON list of sub-questions
- Critic   -> claim drafting (JSON list) or verdict lines
- Writer   -> a cited markdown summary
"""
from __future__ import annotations

import hashlib
import json
import re

from backend.config import mock_llm_enabled


class _Message:
    def __init__(self, text: str):
        self.text = text


class _Content:
    def __init__(self, text: str):
        self.content = [_Message(text)]


def _stable_choice(text: str, options: list[str], salt: str = "") -> str:
    """Deterministic pick so the same input always yields the same output."""
    h = hashlib.sha256((salt + text).encode("utf-8")).hexdigest()
    return options[int(h[:8], 16) % len(options)]


def _extract_question(prompt: str) -> str:
    m = re.search(r"Research question:\s*(.+)", prompt)
    return m.group(1).strip() if m else "the research question"


def _extract_excerpts(prompt: str) -> list[str]:
    marker = "EXCERPTS:"
    if marker in prompt:
        return [p.strip() for p in prompt.split(marker, 1)[1].split("\n---\n") if p.strip()]
    return []


def _respond(prompt: str) -> str:
    if "break it into" in prompt and "sub-questions" in prompt:
        q = _extract_question(prompt)
        return json.dumps([
            f"What is {q.rstrip('?')}?",
            f"What are the main methods used for {q.rstrip('?')}?",
            f"What do recent studies report about {q.rstrip('?')}?",
            f"What are the limitations and open problems in {q.rstrip('?')}?",
        ])

    if "Extract" in prompt and "factual claims" in prompt:
        excerpts = _extract_excerpts(prompt)
        claims = []
        for i, ex in enumerate(excerpts[:3], 1):
            sentence = re.split(r"(?<=[.!?])\s+", ex)[0][:200]
            claims.append(f"{sentence} [{i}]")
        return json.dumps(claims) if claims else json.dumps(["The excerpts discuss the topic."])

    if "VERDICT" in prompt and "CLAIM" in prompt:
        # Deterministic but claim-sensitive: a claim whose (long) words
        # appear in the excerpts is SUPPORTED, otherwise NOT_SUPPORTED
        # (fail closed) - so mock verdicts are meaningfully claim-dependent.
        claim = re.search(r"CLAIM:\s*(.+?)\n\n", prompt, re.DOTALL)
        claim_text = claim.group(1).strip().lower() if claim else ""
        excerpt_words = " ".join(_extract_excerpts(prompt)).lower()
        overlap = sum(w in excerpt_words for w in re.findall(r"\w{6,}", claim_text))
        verdict = "SUPPORTED" if overlap >= 2 else "NOT_SUPPORTED"
        return f"VERDICT: {verdict}\nJUSTIFICATION: Deterministic mock verdict."

    if "well-organized summary" in prompt:
        findings = re.findall(r"^- (.+)$", prompt.split("VERIFIED FINDINGS:")[-1], re.MULTILINE)
        lines = ["**Mock summary** (MOCK_LLM=1 - run live for a real answer):\n"]
        lines.extend(f"- {f}" for f in findings[:5])
        lines.append("\nThe retrieved evidence addresses the question at the level "
                     "expected from the indexed corpus.")
        return "\n".join(lines)

    return "Mock response for unrecognized prompt."


class MockAnthropic:
    """Drop-in stand-in for anthropic.Anthropic for MARA's usage."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or "mock"

    @property
    def messages(self) -> "MockMessages":
        return MockMessages()


class MockMessages:
    @staticmethod
    def create(model: str = "", max_tokens: int = 0, messages: list | None = None, **kwargs):
        prompt = (messages or [{}])[0].get("content", "")
        return _Content(_respond(prompt))


class MockEmbeddings:
    """Deterministic bag-of-words embeddings so retrieval works offline.

    Hashes each token into a fixed-dim vector; texts sharing vocabulary
    get higher cosine similarity, so mock retrieval is *semantically
    sensible* (not arbitrary), which keeps demos meaningful.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z]{3,}", text.lower()):
            h = int(hashlib.sha256(token.encode()).hexdigest()[:12], 16)
            idx, sign = h % self.dim, 1.0 if (h >> 12) % 2 else -1.0
            vec[idx] += sign
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


__all__ = ["MockAnthropic", "MockEmbeddings", "mock_llm_enabled"]
