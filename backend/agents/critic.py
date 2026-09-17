"""Critic Agent (Phase 5) — the fact-checking gate.

Checks each candidate claim against the retrieved source excerpts and
labels it SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED. Only claims
marked SUPPORTED may reach the Writer (design rule from the MARA guide:
this is what makes the system traceable instead of just another chatbot).

Design notes (mirroring planner.py conventions):
- Lazy client creation so the module imports cleanly without an API key.
- The guide's `verdict = text.split()[0]` is fragile: the model may
  prepend Markdown (e.g. "**SUPPORTED** — the excerpts state ...") or a
  JSON object. We scan tokens for a known verdict word instead.
- Any unrecognized/missing verdict is treated as NOT_SUPPORTED (fail
  closed) — an unverifiable claim must never reach the Writer.
- Retries once on transient API errors to keep benchmark runs stable.
"""
from __future__ import annotations

import re

import anthropic

from backend.agents.llm_parsing import parse_json_list
from backend.config import ANTHROPIC_API_KEY, CRITIC_MODEL, mock_llm_enabled
from backend.mock_llm import MockAnthropic

CRITIC_PROMPT = """You are a fact-checking assistant. Given a CLAIM and a set of SOURCE
EXCERPTS, determine whether the claim is directly supported by the excerpts.

Respond in exactly this format (one line):
VERDICT: SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED
JUSTIFICATION: <one sentence>

CLAIM: {claim}

SOURCE EXCERPTS:
{sources}
"""

# Canonical verdicts, most lenient first. Used for token-scan parsing.
VERDICTS = ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")

# Matches a verdict with optional Markdown decoration: **SUPPORTED**,
# `NOT_SUPPORTED`, "PARTIALLY SUPPORTED", or a bare word. Underscore and
# space forms both accepted. Alternation order matters: longer verdicts
# first, so "NOT SUPPORTED" is never read as just "SUPPORTED".
_VERDICT_TOKEN_RE = re.compile(
    r"[`*_\"']*\b(NOT[ _]SUPPORTED|PARTIALLY[ _]SUPPORTED|SUPPORTED)\b[`*_\"']*",
    re.IGNORECASE,
)

_MAX_SOURCES = 8          # keep prompts bounded; critic checks per-chunk claims
_MAX_SOURCE_CHARS = 2000  # truncate pathological chunks


def _get_client() -> anthropic.Anthropic:
    """Return the mock client in MOCK_LLM mode, else the real Anthropic client."""
    if mock_llm_enabled():
        return MockAnthropic()  # type: ignore[return-value]
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example with your key."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def parse_verdict(text: str) -> str:
    """Extract a canonical verdict from an LLM response.

    Scans tokens for SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED,
    accepting underscore ("NOT_SUPPORTED") and space ("NOT SUPPORTED")
    forms plus Markdown decoration. Returns "NOT_SUPPORTED" when nothing
    recognizable is found — the critic must fail closed.
    """
    for match in _VERDICT_TOKEN_RE.finditer(text):
        return match.group(1).upper().replace(" ", "_")
    return "NOT_SUPPORTED"


def check_claim(claim: str, sources: list[str], max_retries: int = 1) -> dict:
    """Fact-check one claim against source excerpts.

    Returns {"claim", "verdict", "justification", "verdict_raw"}.
    verdict is always one of VERDICTS; justification is one sentence
    (the raw response if the model ignored the format).
    """
    client = _get_client()
    joined = "\n---\n".join(
        s[:_MAX_SOURCE_CHARS] for s in sources[:_MAX_SOURCES]
    )
    prompt = CRITIC_PROMPT.format(claim=claim, sources=joined)
    last_text = ""

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=CRITIC_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        last_text = text
        verdict = parse_verdict(text)
        if verdict != "NOT_SUPPORTED" or text:
            break  # parse only "fails" on empty/malformed output; retry then

    justification = _extract_justification(last_text)
    return {
        "claim": claim,
        "verdict": verdict,
        "justification": justification or last_text,
        "verdict_raw": last_text,
    }


def _extract_justification(text: str) -> str:
    """Pull the one-sentence justification out of a formatted response.

    Prefers the text after "JUSTIFICATION:"; falls back to the first
    non-verdict line, then to the whole text.
    """
    match = re.search(r"JUSTIFICATION:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        if not _VERDICT_TOKEN_RE.search(line):
            return line
    return text


# ---------------------------------------------------------------------------
# Claim drafting: candidate claims are drafted FROM the evidence first, then
# cross-verified against ALL retrieved chunks. The original guide's design
# (critic_node checks chunk['text'] against [chunk['text']]) is tautological
# - a chunk trivially supports itself, so every verdict would be SUPPORTED
# and the Critic would filter nothing. Cross-verification is the fix.
# ---------------------------------------------------------------------------

CLAIMS_PROMPT = """You are a research analysis assistant. Below are excerpts from
academic papers. Extract 3-6 short, self-contained factual claims that the
excerpts support (each claim must state a specific finding or technique, be
understandable without the excerpts, and be attributable to at least one of
them). Respond ONLY with a JSON list of strings.

EXCERPTS:
{excerpts}
"""


def extract_claims(evidence: list[dict], max_retries: int = 1) -> list[dict]:
    """Draft candidate claims from retrieved evidence chunks.

    Args:
        evidence: retrieved chunks (dicts with at least "text"; "title"/"url"
            used to track provenance).
    Returns:
        List of {"claim", "sources"} where sources is a list of {"title",
        "url"} dicts naming the chunk(s) each claim came from.
    """
    client = _get_client()
    excerpts = "\n---\n".join(
        f"[{i + 1}] {e['text'][:_MAX_SOURCE_CHARS]}" for i, e in enumerate(evidence[:_MAX_SOURCES])
    )
    prompt = CLAIMS_PROMPT.format(excerpts=excerpts)

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=CRITIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        claims = parse_json_list(text)
        if claims:
            break

    # Track which chunk(s) each claim was drafted from (by number tag).
    # If the model doesn't tag sources, attribute all chunks conservatively.
    out: list[dict] = []
    for claim in claims:
        tag_refs = [int(n) for n in re.findall(r"\[(\d+)\]", claim)]
        if tag_refs:
            sources = [
                {"title": evidence[n - 1]["title"], "url": evidence[n - 1]["url"]}
                for n in tag_refs
                if 1 <= n <= len(evidence)
            ]
        else:
            sources = [
                {"title": e["title"], "url": e["url"]} for e in evidence[:_MAX_SOURCES]
            ]
        out.append({"claim": claim, "sources": sources})
    return out


__all__ = ["check_claim", "parse_verdict", "extract_claims", "VERDICTS", "CRITIC_PROMPT"]


if __name__ == "__main__":
    claim = input("Claim to check: ") or (
        "Federated learning allows model training without centralizing raw user data."
    )
    source = input(
        "Source excerpt (press Enter for a demo excerpt): "
    ) or (
        "Federated learning is a machine learning setting where many clients "
        "(e.g. mobile devices) collaboratively train a shared model under the "
        "coordination of a central server, while the training data remains "
        "decentralized on the clients."
    )
    result = check_claim(claim, [source])
    print(f"\nVerdict: {result['verdict']}")
    print(f"Justification: {result['justification']}")
