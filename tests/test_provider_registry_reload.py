"""Tests for provider registry reload side effects."""

from __future__ import annotations

import utils.model_restrictions as model_restrictions
from providers import registry as provider_registry_module
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType


def test_reload_provider_catalogs_invalidates_restriction_alias_cache() -> None:
    service = model_restrictions.ModelRestrictionService()
    service.restrictions[ProviderType.OPENAI] = {"pro"}
    service._alias_resolution_cache[ProviderType.OPENAI]["pro"] = "gpt-old"
    service._alias_resolution_provider_cache[ProviderType.OPENAI] = object()
    model_restrictions._restriction_service = service

    try:
        ModelProviderRegistry.reload_provider_catalogs(["openai"])

        assert service._alias_resolution_cache.get(ProviderType.OPENAI) in (None, {})
        assert ProviderType.OPENAI not in service._alias_resolution_provider_cache
    finally:
        model_restrictions._restriction_service = None


def test_provider_api_key_env_mapping_covers_priority_providers() -> None:
    for provider_type in ModelProviderRegistry.PROVIDER_PRIORITY_ORDER:
        assert provider_type in provider_registry_module._PROVIDER_API_KEY_ENV


def test_reload_provider_name_map_covers_provider_mappings() -> None:
    reload_provider_types = set(provider_registry_module._RELOAD_PROVIDER_NAME_MAP.values())
    mapped_provider_types = set(provider_registry_module._PROVIDER_API_KEY_ENV.keys())
    assert reload_provider_types == mapped_provider_types
