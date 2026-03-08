"""Tests for model catalog merge and generated manifest materialization."""

from __future__ import annotations

import json

from providers.registries.azure import AzureModelRegistry
from utils.model_catalog.merge_engine import (
    QUARANTINE_PREFIX,
    is_quarantine_description,
    materialize_provider_manifests,
    merge_catalogs,
)


def _find_model(entries: list[dict], model_name: str) -> dict:
    for entry in entries:
        if entry.get("model_name") == model_name:
            return entry
    raise AssertionError(f"Model not found: {model_name}")


def test_merge_catalogs_static_precedence_and_discovery_enrichment() -> None:
    merged = merge_catalogs(
        static_by_provider={
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI GPT-5",
                    "intelligence_score": 10,
                    "description": "static",
                }
            ]
        },
        cache_by_provider={
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI GPT-5",
                    "intelligence_score": 11,
                    "description": "cache",
                }
            ]
        },
        discovered_entries=[
            {
                "provider": "openai",
                "model_name": "gpt-5",
                "context_window": 400000,
                "max_output_tokens": 128000,
            },
            {
                "provider": "openai",
                "model_name": "gpt-5-very-new",
                "context_window": 262144,
            },
        ],
        quarantine_enabled=True,
    )

    openai_entries = merged["openai"]
    gpt5 = _find_model(openai_entries, "gpt-5")

    assert gpt5["intelligence_score"] == 10
    assert gpt5["description"] == "static"
    assert gpt5["context_window"] == 400000
    assert gpt5["max_output_tokens"] == 128000

    new_model = _find_model(openai_entries, "gpt-5-very-new")
    assert new_model["description"].startswith(QUARANTINE_PREFIX)
    assert is_quarantine_description(new_model["description"])


def test_merge_catalogs_cache_enriches_existing_static_model_without_overrides() -> None:
    merged = merge_catalogs(
        static_by_provider={
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI GPT-5",
                    "intelligence_score": 10,
                    "description": "static",
                }
            ]
        },
        cache_by_provider={
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI GPT-5",
                    "intelligence_score": 19,
                    "description": "cache",
                    "context_window": 400000,
                    "max_output_tokens": 128000,
                    "supports_function_calling": True,
                }
            ]
        },
        discovered_entries=[],
        quarantine_enabled=True,
    )

    gpt5 = _find_model(merged["openai"], "gpt-5")
    assert gpt5["intelligence_score"] == 10
    assert gpt5["description"] == "static"
    assert gpt5["context_window"] == 400000
    assert gpt5["max_output_tokens"] == 128000
    assert gpt5["supports_function_calling"] is True


def test_merge_catalogs_without_quarantine_keeps_new_models_active() -> None:
    merged = merge_catalogs(
        static_by_provider={},
        cache_by_provider={},
        discovered_entries=[
            {
                "provider": "vercel_gateway",
                "model_name": "openai/gpt-future",
                "context_window": 999999,
                "max_output_tokens": 12345,
            }
        ],
        quarantine_enabled=False,
    )

    vercel_entries = merged["vercel_gateway"]
    model = _find_model(vercel_entries, "openai/gpt-future")

    assert model["context_window"] == 999999
    assert model["max_output_tokens"] == 12345
    assert not is_quarantine_description(model.get("description"))


def test_merge_catalogs_quarantine_preserves_discovery_release_metadata() -> None:
    merged = merge_catalogs(
        static_by_provider={},
        cache_by_provider={},
        discovered_entries=[
            {
                "provider": "openrouter",
                "model_name": "anthropic/claude-new",
                "release_metadata": {
                    "source_url": "https://openrouter.ai/docs/overview/models",
                    "release_date": "2026-01-31",
                },
            }
        ],
        quarantine_enabled=True,
    )

    model = _find_model(merged["openrouter"], "anthropic/claude-new")
    metadata = model.get("release_metadata")

    assert isinstance(metadata, dict)
    assert metadata.get("source_url") == "https://openrouter.ai/docs/overview/models"
    assert metadata.get("release_date") == "2026-01-31"
    assert metadata.get("last_verified_at")


def test_materialize_provider_manifests_contains_generation_metadata() -> None:
    payloads = materialize_provider_manifests(
        {
            "openrouter": [
                {
                    "model_name": "anthropic/claude-sonnet-4.5",
                    "friendly_name": "OpenRouter Sonnet",
                    "intelligence_score": 13,
                    "description": "test",
                }
            ]
        },
        source_mode="discovery+static",
    )

    openrouter_payload = payloads["openrouter"]
    generated = openrouter_payload["_GENERATED"]

    assert generated["source_mode"] == "discovery+static"
    assert generated["entry_count"] == 1
    assert "generated_at" not in generated
    assert openrouter_payload["models"][0]["model_name"] == "anthropic/claude-sonnet-4.5"


def test_merge_catalogs_keeps_static_metadata_when_cache_differs() -> None:
    warnings: list[str] = []
    merged = merge_catalogs(
        static_by_provider={
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI GPT-5",
                    "intelligence_score": 10,
                    "description": "static",
                }
            ]
        },
        cache_by_provider={
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI GPT-5",
                    "intelligence_score": 19,
                    "description": "cache",
                }
            ]
        },
        discovered_entries=[],
        quarantine_enabled=True,
        warnings_out=warnings,
    )

    assert _find_model(merged["openai"], "gpt-5")["intelligence_score"] == 10
    assert warnings == []


def test_merge_catalogs_preserves_azure_deployment_fields() -> None:
    merged = merge_catalogs(
        static_by_provider={
            "azure": [
                {
                    "model_name": "gpt-4o",
                    "friendly_name": "Azure OpenAI (gpt-4o)",
                    "intelligence_score": 10,
                    "description": "static",
                    "deployment": "azure-gpt4o-static",
                }
            ]
        },
        cache_by_provider={
            "azure": [
                {
                    "model_name": "gpt-4o",
                    "friendly_name": "Azure OpenAI (gpt-4o)",
                    "intelligence_score": 11,
                    "description": "cache",
                    "deployment": "azure-gpt4o-cache",
                }
            ]
        },
        discovered_entries=[],
        quarantine_enabled=True,
    )

    model = _find_model(merged["azure"], "gpt-4o")
    assert model["deployment"] == "azure-gpt4o-static"


def test_materialize_provider_manifests_keeps_azure_deployment_fields() -> None:
    payloads = materialize_provider_manifests(
        {
            "azure": [
                {
                    "model_name": "gpt-4o-mini",
                    "friendly_name": "Azure OpenAI (gpt-4o-mini)",
                    "intelligence_score": 9,
                    "description": "azure",
                    "deployment": "azure-gpt4o-mini-prod",
                }
            ]
        },
        source_mode="static",
    )

    model = _find_model(payloads["azure"]["models"], "gpt-4o-mini")
    assert model["deployment"] == "azure-gpt4o-mini-prod"


def test_generated_azure_manifest_remains_loadable_by_azure_registry(tmp_path) -> None:
    merged = merge_catalogs(
        static_by_provider={
            "azure": [
                {
                    "model_name": "gpt-4o",
                    "friendly_name": "Azure OpenAI (gpt-4o)",
                    "intelligence_score": 10,
                    "description": "azure",
                    "deployment": "azure-gpt4o-prod",
                }
            ]
        },
        cache_by_provider={},
        discovered_entries=[],
        quarantine_enabled=True,
    )
    payloads = materialize_provider_manifests(merged, source_mode="static")

    manifest_path = tmp_path / "azure_models.json"
    manifest_path.write_text(json.dumps(payloads["azure"]), encoding="utf-8")

    registry = AzureModelRegistry(config_path=str(manifest_path))
    assert "gpt-4o" in registry.list_models()
    assert registry.get_entry("gpt-4o") == {"deployment": "azure-gpt4o-prod"}


def test_merge_catalogs_preserves_azure_deployment_name_alias() -> None:
    merged = merge_catalogs(
        static_by_provider={
            "azure": [
                {
                    "model_name": "gpt-4.1-mini",
                    "friendly_name": "Azure OpenAI (gpt-4.1-mini)",
                    "intelligence_score": 8,
                    "description": "azure",
                    "deployment_name": "azure-gpt41-mini",
                }
            ]
        },
        cache_by_provider={},
        discovered_entries=[],
        quarantine_enabled=True,
    )

    model = _find_model(merged["azure"], "gpt-4.1-mini")
    assert model["deployment_name"] == "azure-gpt41-mini"
