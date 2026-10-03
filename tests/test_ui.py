"""Offline tests for the Streamlit UI layer (Phase 9).

The pure helpers in ui/app.py are tested directly - importing the module
does NOT require a Streamlit runtime or a network connection. The Streamlit
rendering itself is verified live (streamlit run ui/app.py + browser check).
"""
import httpx
import pytest

try:
    from ui import app as ui
except RuntimeError as exc:  # bare-mode Streamlit guard (no script runtime)
    pytest.skip(f"ui.app import unavailable outside Streamlit runtime: {exc}",
                allow_module_level=True)


# ---------------------------------------------------------------------------
# validate_question
# ---------------------------------------------------------------------------

def test_validate_question_accepts_normal_question():
    assert ui.validate_question("How does federated learning work?") is None


def test_validate_question_strips_whitespace():
    assert ui.validate_question("   ok question   ") is None


def test_validate_question_rejects_too_short():
    assert ui.validate_question("hi") is not None
    assert ui.validate_question("   ") is not None
    assert ui.validate_question("") is not None


def test_validate_question_rejects_too_long():
    long_q = "x" * (ui.MAX_QUESTION_LEN + 1)
    assert ui.validate_question(long_q) is not None


# ---------------------------------------------------------------------------
# confidence_label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence,expected", [
    (1.0, "High"), (0.75, "High"),
    (0.74, "Medium"), (0.4, "Medium"), (0.5, "Medium"),
    (0.39, "Low"), (0.0, "Low"),
])
def test_confidence_labels(confidence, expected):
    assert ui.confidence_label(confidence) == expected


# ---------------------------------------------------------------------------
# timing formatting
# ---------------------------------------------------------------------------

def test_format_timing_orders_agents_and_drops_missing():
    timing = {"writer_s": 0.5, "planner_s": 1.2, "critic_s": 0.3}
    stages = ui.format_timing(timing)
    assert stages == [("Planner", 1.2), ("Critic (draft)", 0.3), ("Writer", 0.5)]


def test_format_timing_empty_when_no_timing():
    assert ui.format_timing({}) == []


def test_total_latency_sums_all_agents():
    assert ui.total_latency({"planner_s": 0.5, "writer_s": 1.0}) == 1.5


# ---------------------------------------------------------------------------
# API client helpers (httpx MockTransport - no real network)
# ---------------------------------------------------------------------------

def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def _response(status_code, json_body, method="POST", url="http://test/query"):
    """Build an httpx.Response with its request attached (raise_for_status needs it)."""
    return httpx.Response(status_code, json=json_body, request=httpx.Request(method, url))


def test_check_health_online(monkeypatch):
    def fake_get(url, timeout):
        assert url.endswith("/health")
        return _response(200, {"status": "ok", "mock_llm": True}, method="GET",
                         url=url)

    monkeypatch.setattr(ui.httpx, "get", fake_get)
    result = ui.check_health("http://test")
    assert result == {"status": "ok", "mock_llm": True}


def test_check_health_offline_returns_none(monkeypatch):
    def boom(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ui.httpx, "get", boom)
    assert ui.check_health("http://test") is None


def test_run_pipeline_success(monkeypatch):
    payload = {"answer": "ok", "confidence": 1.0}

    def fake_post(url, json, timeout):
        assert url.endswith("/query")
        assert json["question"] == "What is RAG?"
        return _response(200, payload)

    monkeypatch.setattr(ui.httpx, "post", fake_post)
    result, error = ui.run_pipeline("http://test", "What is RAG?")
    assert error is None
    assert result == payload


def test_run_pipeline_connect_error_has_helpful_message(monkeypatch):
    def boom(url, json, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(ui.httpx, "post", boom)
    result, error = ui.run_pipeline("http://test", "What is RAG?")
    assert result is None
    assert "uvicorn backend.main:app" in error


def test_run_pipeline_maps_500(monkeypatch):
    def fail(url, json, timeout):
        return _response(500, {"detail": "Pipeline error: Qdrant down"})

    monkeypatch.setattr(ui.httpx, "post", fail)
    result, error = ui.run_pipeline("http://test", "What is RAG?")
    assert result is None
    assert "Qdrant down" in error


def test_run_pipeline_maps_422(monkeypatch):
    def reject(url, json, timeout):
        return _response(422, {"detail": "bad question"})

    monkeypatch.setattr(ui.httpx, "post", reject)
    result, error = ui.run_pipeline("http://test", "What is RAG?")
    assert result is None
    assert "rejected" in error.lower()
