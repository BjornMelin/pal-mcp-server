"""Tests for Vercel AI Gateway provider behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from providers.vercel_gateway import VercelGatewayProvider


def _write_vercel_registry(path: Path) -> None:
    payload = {
        "models": [
            {
                "model_name": "openai/gpt-5",
                "aliases": ["gpt5-vg"],
                "friendly_name": "Vercel GPT-5",
                "intelligence_score": 18,
                "context_window": 400000,
                "max_output_tokens": 128000,
                "supports_extended_thinking": True,
                "supports_function_calling": True,
                "supports_json_mode": True,
                "supports_images": True,
                "description": "GPT-5 via Vercel",
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_openrouter_registry(path: Path) -> None:
    payload = {
        "models": [
            {
                "model_name": "openai/gpt-5",
                "aliases": ["gpt5-or"],
                "friendly_name": "OpenRouter GPT-5",
                "intelligence_score": 18,
                "description": "GPT-5 via OpenRouter",
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_vercel_provider_registry_alias_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify registry alias resolution for the Vercel provider.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest environment patch fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "vercel_gateway_models.json"
    _write_vercel_registry(config_path)

    monkeypatch.setenv("VERCEL_GATEWAY_MODELS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", raising=False)

    VercelGatewayProvider.reload_registry()
    provider = VercelGatewayProvider(api_key="test-key")

    assert provider.get_provider_type() == ProviderType.VERCEL_GATEWAY
    assert provider.validate_model_name("openai/gpt-5")
    assert provider.validate_model_name("gpt5-vg")

    caps = provider.get_capabilities("gpt5-vg")
    assert caps.model_name == "openai/gpt-5"
    assert caps.provider == ProviderType.VERCEL_GATEWAY

    listed = provider.list_models(respect_restrictions=False)
    assert "openai/gpt-5" in listed
    assert "gpt5-vg" in listed


def test_vercel_provider_provider_model_requires_opt_in_for_generic_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require opt-in for provider/model generic fallback.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest environment patch fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "vercel_gateway_models.json"
    _write_vercel_registry(config_path)

    monkeypatch.setenv("VERCEL_GATEWAY_MODELS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", raising=False)

    VercelGatewayProvider.reload_registry()
    strict_provider = VercelGatewayProvider(api_key="test-key")
    assert not strict_provider.validate_model_name("openai/gpt-future")

    monkeypatch.setenv("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", "true")
    VercelGatewayProvider.reload_registry()
    permissive_provider = VercelGatewayProvider(api_key="test-key")

    assert permissive_provider.validate_model_name("openai/gpt-future")
    caps = permissive_provider.get_capabilities("openai/gpt-future")

    assert caps.model_name == "openai/gpt-future"
    assert caps.provider == ProviderType.VERCEL_GATEWAY
    assert getattr(caps, "_is_generic", False) is True


def test_vercel_provider_plain_model_name_requires_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require opt-in for plain model-name generic fallback.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest environment patch fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "vercel_gateway_models.json"
    _write_vercel_registry(config_path)

    monkeypatch.setenv("VERCEL_GATEWAY_MODELS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", raising=False)

    VercelGatewayProvider.reload_registry()
    strict_provider = VercelGatewayProvider(api_key="test-key")
    assert not strict_provider.validate_model_name("gpt-future")

    monkeypatch.setenv("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", "true")
    VercelGatewayProvider.reload_registry()
    permissive_provider = VercelGatewayProvider(api_key="test-key")

    assert permissive_provider.validate_model_name("gpt-future")
    caps = permissive_provider.get_capabilities("gpt-future")
    assert caps.model_name == "gpt-future"
    assert getattr(caps, "_is_generic", False) is True


def test_registry_routes_unknown_provider_model_to_openrouter_when_vercel_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route unknown provider/model IDs to OpenRouter when Vercel is strict.

    Args:
        tmp_path: Pytest temporary directory fixture.
        monkeypatch: Pytest environment patch fixture.

    Returns:
        None.
    """
    vercel_config_path = tmp_path / "vercel_gateway_models.json"
    openrouter_config_path = tmp_path / "openrouter_models.json"
    _write_vercel_registry(vercel_config_path)
    _write_openrouter_registry(openrouter_config_path)

    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-vercel-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("VERCEL_GATEWAY_MODELS_CONFIG_PATH", str(vercel_config_path))
    monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(openrouter_config_path))
    monkeypatch.delenv("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", raising=False)

    VercelGatewayProvider.reload_registry()
    OpenRouterProvider.reload_registry()

    try:
        ModelProviderRegistry.reset_for_testing()
        ModelProviderRegistry.register_provider(ProviderType.VERCEL_GATEWAY, VercelGatewayProvider)
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

        configured = ModelProviderRegistry.get_provider_for_model("openai/gpt-5")
        unknown_prefixed = ModelProviderRegistry.get_provider_for_model("openai/gpt-future")

        assert configured is not None
        assert configured.get_provider_type() == ProviderType.VERCEL_GATEWAY
        assert unknown_prefixed is not None
        assert unknown_prefixed.get_provider_type() == ProviderType.OPENROUTER
    finally:
        OpenRouterProvider.reload_registry()
        VercelGatewayProvider.reload_registry()
        ModelProviderRegistry.reset_for_testing()
