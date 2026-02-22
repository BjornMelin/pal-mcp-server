"""Unit tests for provider discovery normalizers."""

from __future__ import annotations

from utils.model_catalog import discovery_client


def test_discover_openai_normalizes_metadata(monkeypatch) -> None:
    def _mock_fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None):  # noqa: ARG001
        return {"data": [{"id": "gpt-5"}, {"id": "text-embedding-3-large"}]}

    monkeypatch.setattr(discovery_client, "_fetch_json", _mock_fetch_json)

    entries = discovery_client.discover_openai(api_key="sk-openai", timeout_ms=1000)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "openai"
    assert entry["model_name"] == "gpt-5"
    assert entry["model_id"] == "gpt-5"
    assert entry["provider_type"] == "openai"
    assert entry["display_name"] == "gpt-5"
    assert entry["openrouter_counterpart"] == "openai/gpt-5"
    assert entry["release_metadata"]["source_url"] == discovery_client.OPENAI_RELEASE_NOTES_URL
    assert entry["release_metadata"]["last_verified_at"]


def test_discover_gemini_normalizes_token_limits(monkeypatch) -> None:
    def _mock_fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None):  # noqa: ARG001
        return {
            "models": [
                {
                    "name": "models/gemini-2.5-pro",
                    "supportedGenerationMethods": ["generateContent"],
                    "inputTokenLimit": 1048576,
                    "outputTokenLimit": 65536,
                },
                {
                    "name": "models/text-embedding-004",
                    "supportedGenerationMethods": ["embedContent"],
                },
            ]
        }

    monkeypatch.setattr(discovery_client, "_fetch_json", _mock_fetch_json)

    entries = discovery_client.discover_gemini(api_key="gemini-key", timeout_ms=1000)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "google"
    assert entry["model_name"] == "gemini-2.5-pro"
    assert entry["context_window"] == 1048576
    assert entry["max_output_tokens"] == 65536
    assert entry["openrouter_counterpart"] == "google/gemini-2.5-pro"


def test_discover_xai_normalizes_entries(monkeypatch) -> None:
    def _mock_fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None):  # noqa: ARG001
        return {"data": [{"id": "grok-4"}, {"id": "xai-unknown"}]}

    monkeypatch.setattr(discovery_client, "_fetch_json", _mock_fetch_json)

    entries = discovery_client.discover_xai(api_key="xai-key", timeout_ms=1000)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "xai"
    assert entry["model_name"] == "grok-4"
    assert entry["openrouter_counterpart"] == "x-ai/grok-4"


def test_discover_openrouter_parses_modality_and_limits(monkeypatch) -> None:
    def _mock_fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None):  # noqa: ARG001
        return {
            "data": [
                {
                    "id": "openai/gpt-5",
                    "context_length": 400000,
                    "architecture": {"modality": ["text", "image"]},
                    "top_provider": {"max_completion_tokens": 128000},
                }
            ]
        }

    monkeypatch.setattr(discovery_client, "_fetch_json", _mock_fetch_json)

    entries = discovery_client.discover_openrouter(api_key="or-key", timeout_ms=1000)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "openrouter"
    assert entry["model_name"] == "openai/gpt-5"
    assert entry["context_window"] == 400000
    assert entry["max_output_tokens"] == 128000
    assert entry["supports_images"] is True
    assert entry["openrouter_counterpart"] == "openai/gpt-5"


def test_discover_vercel_gateway_normalizes_entries(monkeypatch) -> None:
    def _mock_fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None):  # noqa: ARG001
        return {"data": [{"id": "openai/gpt-5", "context_window": 400000, "max_output_tokens": 128000}]}

    monkeypatch.setattr(discovery_client, "_fetch_json", _mock_fetch_json)
    monkeypatch.setattr(discovery_client, "get_env", lambda *_args, **_kwargs: "https://ai-gateway.vercel.sh/v1")

    entries = discovery_client.discover_vercel_gateway(api_key="vercel-key", timeout_ms=1000)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["provider"] == "vercel_gateway"
    assert entry["model_name"] == "openai/gpt-5"
    assert entry["context_window"] == 400000
    assert entry["max_output_tokens"] == 128000


def test_discover_all_collects_errors(monkeypatch) -> None:
    values = {
        "OPENAI_API_KEY": "sk-live-openai",
        "GEMINI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "XAI_API_KEY": "xai-live",
        "OPENROUTER_API_KEY": "",
        "VERCEL_AI_GATEWAY_API_KEY": "",
    }

    monkeypatch.setattr(discovery_client, "get_env", lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(
        discovery_client,
        "discover_openai",
        lambda _api_key, _timeout_ms: [{"provider": "openai", "model_name": "gpt-5"}],
    )

    def _raise_xai(*_args, **_kwargs):
        raise RuntimeError("xai timeout")

    monkeypatch.setattr(discovery_client, "discover_xai", _raise_xai)

    entries, errors = discovery_client.discover_all(timeout_ms=1000)

    assert entries == [{"provider": "openai", "model_name": "gpt-5"}]
    assert len(errors) == 1
    assert "OPENAI_API_KEY" not in errors[0]
    assert errors[0].startswith("XAI_API_KEY:")
