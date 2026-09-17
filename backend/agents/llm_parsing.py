"""Shared defensive LLM-output parsing.

LLMs rarely return *exactly* what a prompt asks for: JSON gets wrapped in
markdown code fences, embedded in prose, or replaced by numbered lists.
Both the Planner (sub-questions) and the Critic (candidate claims) need
the same tolerant extraction, so it lives here.
"""
from __future__ import annotations

import json
import re


def parse_json_list(text: str) -> list[str]:
    """Extract a JSON list of strings from an LLM response.

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


__all__ = ["parse_json_list"]
