"""Offline unit tests for the Critic agent (no network, no API key).

Strategy mirrors test_planner.py: exercise the parsing/normalization
logic directly and stub the Anthropic client for check_claim, so the
fact-checking behavior is fully verified without spending credits.
"""
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.critic import (
    VERDICTS,
    _extract_justification,
    check_claim,
    parse_verdict,
)


class TestParseVerdict:
    def test_plain_supported(self):
        assert parse_verdict("SUPPORTED: the excerpt states exactly this") == "SUPPORTED"

    def test_plain_not_supported(self):
        assert parse_verdict("NOT_SUPPORTED: the excerpt does not mention it") == "NOT_SUPPORTED"

    def test_plain_partially_supported(self):
        assert parse_verdict("PARTIALLY_SUPPORTED: only half the claim is covered") == "PARTIALLY_SUPPORTED"

    def test_markdown_bold(self):
        assert parse_verdict("**SUPPORTED** — the excerpts state this directly") == "SUPPORTED"

    def test_markdown_bold_not_supported(self):
        assert parse_verdict("**NOT_SUPPORTED** — unrelated evidence") == "NOT_SUPPORTED"

    def test_space_form(self):
        assert parse_verdict("NOT SUPPORTED: no evidence found") == "NOT_SUPPORTED"

    def test_partially_space_form(self):
        assert parse_verdict("Partially Supported: the excerpt only covers X") == "PARTIALLY_SUPPORTED"

    def test_code_fence_form(self):
        assert parse_verdict("`NOT_SUPPORTED` because sources contradict it") == "NOT_SUPPORTED"

    def test_case_insensitive(self):
        assert parse_verdict("supported") == "SUPPORTED"

    def test_garbage_fails_closed(self):
        assert parse_verdict("I cannot help with that.") == "NOT_SUPPORTED"

    def test_empty_fails_closed(self):
        assert parse_verdict("") == "NOT_SUPPORTED"

    def test_word_inside_sentence(self):
        text = "After careful review, the claim is NOT_SUPPORTED by the sources."
        assert parse_verdict(text) == "NOT_SUPPORTED"

    def test_supported_is_not_misread_from_not_supported(self):
        # "NOT_SUPPORTED" must never be parsed as plain "SUPPORTED".
        assert parse_verdict("NOT_SUPPORTED") == "NOT_SUPPORTED"
        assert parse_verdict("PARTIALLY_SUPPORTED") == "PARTIALLY_SUPPORTED"


class TestExtractJustification:
    def test_labeled_justification(self):
        text = "VERDICT: SUPPORTED\nJUSTIFICATION: The excerpt explicitly states this."
        assert _extract_justification(text) == "The excerpt explicitly states this."

    def test_falls_back_to_first_non_verdict_line(self):
        text = "SUPPORTED\nThe excerpt directly states the claim."
        assert _extract_justification(text) == "The excerpt directly states the claim."

    def test_falls_back_to_whole_text(self):
        text = "VERDICT: SUPPORTED"
        assert _extract_justification(text) == "VERDICT: SUPPORTED"


def _mock_critic_client(response_text: str):
    """Patch _get_client to return a client whose API yields response_text."""
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response
    patcher = patch("backend.agents.critic._get_client", return_value=mock_client)
    return patcher, mock_client


class TestCheckClaim:
    def test_supported_claim_passes_through(self):
        patcher, mock_client = _mock_critic_client(
            "VERDICT: SUPPORTED\nJUSTIFICATION: Directly stated."
        )
        with patcher:
            result = check_claim("The sky is blue.", ["Excerpt about blue skies."])
        assert result["verdict"] == "SUPPORTED"
        assert result["justification"] == "Directly stated."
        assert result["claim"] == "The sky is blue."

    def test_verdict_raw_preserved(self):
        raw = "**NOT_SUPPORTED** — sources discuss something else."
        patcher, _ = _mock_critic_client(raw)
        with patcher:
            result = check_claim("Claim.", ["Source."])
        assert result["verdict_raw"] == raw
        assert result["verdict"] == "NOT_SUPPORTED"

    def test_malformed_response_fails_closed(self):
        patcher, _ = _mock_critic_client("Sorry, I don't understand.")
        with patcher:
            result = check_claim("Claim.", ["Source."])
        assert result["verdict"] == "NOT_SUPPORTED"
        assert result["justification"]  # raw text kept as justification

    def test_prompt_contains_claim_and_sources(self):
        patcher, mock_client = _mock_critic_client("SUPPORTED")
        with patcher:
            check_claim("My claim here.", ["Source A.", "Source B."])
        sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "My claim here." in sent_prompt
        assert "Source A." in sent_prompt
        assert "Source B." in sent_prompt

    def test_sources_truncated_to_bound_prompt(self):
        patcher, mock_client = _mock_critic_client("SUPPORTED")
        long_source = "x" * 5000
        with patcher:
            check_claim("Claim.", [long_source] * 12)
        sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "x" * 2001 not in sent_prompt  # each source truncated to 2000 chars
        # 12 sources offered -> at most 8 included (7 separators between them).
        assert sent_prompt.count("\n---\n") == 7

    def test_missing_api_key_raises_clear_error(self):
        with patch("backend.agents.critic.ANTHROPIC_API_KEY", ""):
            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                check_claim("Claim.", ["Source."])


class TestConstants:
    def test_verdicts_canonical(self):
        assert VERDICTS == ("SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED")

    def test_prompt_formats_without_key_error(self):
        formatted = __import__(
            "backend.agents.critic", fromlist=["CRITIC_PROMPT"]
        ).CRITIC_PROMPT.format(claim="C", sources="S")
        assert "C" in formatted and "S" in formatted
