"""Routing tests for quarantine-aware fallback behavior."""

from __future__ import annotations

import importlib
import json

import pytest

import utils.env as env_config
import utils.model_restrictions as model_restrictions
from providers.openrouter import OpenRouterProvider
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType


def test_auto_fallback_skips_quarantined_models(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "openrouter_models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_name": "openai/gpt-quarantined",
                        "friendly_name": "OpenRouter Quarantined",
                        "intelligence_score": 20,
                        "description": "[QUARANTINE] pending ranking",
                        "quarantine": True,
                        "lifecycle": "quarantined",
                        "aliases": ["gptq"],
                    },
                    {
                        "model_name": "openai/gpt-safe",
                        "friendly_name": "OpenRouter Safe",
                        "intelligence_score": 12,
                        "description": "safe model",
                        "aliases": ["gptsafe"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    env_config.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    try:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
        monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(config_path))
        monkeypatch.setenv("MODEL_CATALOG_QUARANTINE_ENABLED", "true")
        for key in [
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "DIAL_API_KEY",
            "CUSTOM_API_URL",
            "VERCEL_AI_GATEWAY_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        OpenRouterProvider.reload_registry()

        ModelProviderRegistry.reset_for_testing()
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

        selected = ModelProviderRegistry.get_preferred_fallback_model()

        assert selected in {"openai/gpt-safe", "gptsafe"}
        assert selected not in {"openai/gpt-quarantined", "gptq"}
    finally:
        OpenRouterProvider.reload_registry()
        ModelProviderRegistry.reset_for_testing()
        env_config.reload_env()


def test_available_models_exclude_quarantined_entries(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "openrouter_models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_name": "openai/gpt-quarantined",
                        "friendly_name": "OpenRouter Quarantined",
                        "intelligence_score": 20,
                        "description": "[QUARANTINE] pending ranking",
                        "quarantine": True,
                        "lifecycle": "quarantined",
                        "aliases": ["gptq"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_config.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    try:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
        monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(config_path))
        monkeypatch.setenv("MODEL_CATALOG_QUARANTINE_ENABLED", "true")
        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", "gptq")
        for key in [
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "DIAL_API_KEY",
            "CUSTOM_API_URL",
            "VERCEL_AI_GATEWAY_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        OpenRouterProvider.reload_registry()

        ModelProviderRegistry.reset_for_testing()
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)

        available_models = ModelProviderRegistry.get_available_models(respect_restrictions=True)

        assert "openai/gpt-quarantined" not in available_models
        assert "gptq" not in available_models
        assert available_models == {}
    finally:
        OpenRouterProvider.reload_registry()
        ModelProviderRegistry.reset_for_testing()
        env_config.reload_env()


def test_auto_mode_startup_fails_when_only_quarantined_models_remain(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "openrouter_models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_name": "openai/gpt-quarantined",
                        "friendly_name": "OpenRouter Quarantined",
                        "intelligence_score": 20,
                        "description": "[QUARANTINE] pending ranking",
                        "quarantine": True,
                        "lifecycle": "quarantined",
                        "aliases": ["gptq"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    env_config.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    try:
        monkeypatch.setenv("PAL_MCP_FORCE_ENV_OVERRIDE", "false")
        monkeypatch.setenv("DEFAULT_MODEL", "auto")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter")
        monkeypatch.setenv("OPENROUTER_MODELS_CONFIG_PATH", str(config_path))
        monkeypatch.setenv("OPENROUTER_ALLOWED_MODELS", "gptq")
        monkeypatch.setenv("MODEL_CATALOG_QUARANTINE_ENABLED", "true")

        for key in [
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "DIAL_API_KEY",
            "CUSTOM_API_URL",
            "CUSTOM_API_KEY",
            "VERCEL_AI_GATEWAY_API_KEY",
        ]:
            monkeypatch.delenv(key, raising=False)

        OpenRouterProvider.reload_registry()

        import config
        import server

        model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()
        importlib.reload(config)
        importlib.reload(server)

        model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()

        try:
            server.configure_providers()
            pytest.fail("Expected auto-mode startup validation to fail when only quarantined models are available")
        except ValueError as exc:
            message = str(exc)
            assert "No models available for auto mode due to restrictions" in message
    finally:
        OpenRouterProvider.reload_registry()
        model_restrictions._restriction_service = None
        ModelProviderRegistry.reset_for_testing()
        env_config.reload_env()
