"""Offline unit tests for the Planner agent's parsing logic.

These run without an API key or network access, so you can verify the
pipeline works before spending any API credits.
"""
import json

import pytest

from backend.agents.planner import parse_subquestions


class TestCleanJson:
    def test_clean_list(self):
        text = '["What is X?", "How does Y work?", "Why Z?"]'
        assert parse_subquestions(text) == ["What is X?", "How does Y work?", "Why Z?"]

    def test_single_item(self):
        assert parse_subquestions('["One?"]') == ["One?"]

    def test_drops_empty_strings(self):
        assert parse_subquestions('["A?", "", "B?"]') == ["A?", "B?"]


class TestFencedJson:
    def test_json_code_fence(self):
        text = '```json\n["A?", "B?"]\n```'
        assert parse_subquestions(text) == ["A?", "B?"]

    def test_plain_code_fence(self):
        text = '```\n["A?", "B?"]\n```'
        assert parse_subquestions(text) == ["A?", "B?"]


class TestEmbeddedJson:
    def test_prose_around_json(self):
        text = 'Here are the sub-questions:\n["A?", "B?"]\nDone.'
        assert parse_subquestions(text) == ["A?", "B?"]

    def test_preamble_only(self):
        text = 'Sure! Here they are:\n["A?", "B?"]'
        assert parse_subquestions(text) == ["A?", "B?"]


class TestNumberedListFallback:
    def test_numbered_lines(self):
        text = "1. What is X?\n2. How does Y work?"
        assert parse_subquestions(text) == ["What is X?", "How does Y work?"]

    def test_bulleted_lines(self):
        text = "- What is X?\n- How does Y work?"
        assert parse_subquestions(text) == ["What is X?", "How does Y work?"]

    def test_unparseable_returns_empty(self):
        assert parse_subquestions("I cannot help with that.") == []


class TestPlanDegradesGracefully:
    def test_plan_never_returns_empty(self):
        """If parsing fails entirely, plan() falls back to the raw question."""
        from unittest.mock import MagicMock, patch

        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="I cannot help with that.")]
        with patch("backend.agents.planner._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = fake_response
            mock_get_client.return_value = mock_client
            from backend.agents.planner import plan

            assert plan("some question") == ["some question"]


class TestPromptFormat:
    def test_prompt_formats_without_key_error(self):
        from backend.agents.planner import PLANNER_PROMPT

        formatted = PLANNER_PROMPT.format(question="Test?")
        assert "Test?" in formatted
