"""
Model Restriction Service

This module provides centralized management of model usage restrictions
based on environment variables. It allows organizations to limit which
models can be used from each provider for cost control, compliance, or
standardization purposes.

Environment Variables:
- OPENAI_ALLOWED_MODELS: Comma-separated list of allowed OpenAI models
- GOOGLE_ALLOWED_MODELS: Comma-separated list of allowed Gemini models
- XAI_ALLOWED_MODELS: Comma-separated list of allowed X.AI GROK models
- OPENROUTER_ALLOWED_MODELS: Comma-separated list of allowed OpenRouter models
- VERCEL_GATEWAY_ALLOWED_MODELS: Comma-separated list of allowed Vercel AI Gateway models
- DIAL_ALLOWED_MODELS: Comma-separated list of allowed DIAL models

Example:
    OPENAI_ALLOWED_MODELS=o3-mini,o4-mini
    GOOGLE_ALLOWED_MODELS=flash
    XAI_ALLOWED_MODELS=grok-4,grok-4.1-fast-reasoning
    OPENROUTER_ALLOWED_MODELS=opus,sonnet,mistral
"""

import logging
from collections import defaultdict
from typing import Any, Optional

from providers.shared import ProviderType
from utils.env import get_env

logger = logging.getLogger(__name__)


class ModelRestrictionService:
    """Central authority for environment-driven model allowlists.

    Role
        Interpret ``*_ALLOWED_MODELS`` environment variables, keep their
        entries normalised (lowercase), and answer whether a provider/model
        pairing is permitted.

    Responsibilities
        * Parse, cache, and expose per-provider restriction sets
        * Validate configuration by cross-checking each entry against the
          provider’s alias-aware model list
        * Offer helper methods such as ``is_allowed`` and ``filter_models`` to
          enforce policy everywhere model names appear (tool selection, CLI
          commands, etc.).
    """

    # Environment variable names
    ENV_VARS = {
        ProviderType.OPENAI: "OPENAI_ALLOWED_MODELS",
        ProviderType.GOOGLE: "GOOGLE_ALLOWED_MODELS",
        ProviderType.XAI: "XAI_ALLOWED_MODELS",
        ProviderType.OPENROUTER: "OPENROUTER_ALLOWED_MODELS",
        ProviderType.VERCEL_GATEWAY: "VERCEL_GATEWAY_ALLOWED_MODELS",
        ProviderType.DIAL: "DIAL_ALLOWED_MODELS",
    }

    def __init__(self):
        """Initialize the restriction service by loading from environment."""
        self.restrictions: dict[ProviderType, set[str]] = {}
        self._alias_resolution_cache: dict[ProviderType, dict[str, str]] = defaultdict(dict)
        self._alias_resolution_provider_cache: dict[ProviderType, Any] = {}
        self._load_from_env()

    def _get_alias_resolution_provider(self, provider_type: ProviderType) -> Optional[Any]:
        """Return a provider instance suitable for alias-to-canonical resolution."""

        try:
            from providers.registry import ModelProviderRegistry

            provider = ModelProviderRegistry.get_provider(provider_type)
            if provider:
                return provider
        except Exception:  # pragma: no cover - defensive fallback
            provider = None

        if provider_type in self._alias_resolution_provider_cache:
            return self._alias_resolution_provider_cache[provider_type]

        try:
            if provider_type == ProviderType.OPENAI:
                from providers.openai import OpenAIModelProvider

                provider = OpenAIModelProvider(api_key=(get_env("OPENAI_API_KEY") or "restriction-alias-resolution"))
            elif provider_type == ProviderType.GOOGLE:
                from providers.gemini import GeminiModelProvider

                provider_kwargs = {"api_key": (get_env("GEMINI_API_KEY") or "restriction-alias-resolution")}
                gemini_base_url = get_env("GEMINI_BASE_URL")
                if gemini_base_url:
                    provider_kwargs["base_url"] = gemini_base_url
                provider = GeminiModelProvider(**provider_kwargs)
            elif provider_type == ProviderType.XAI:
                from providers.xai import XAIModelProvider

                provider = XAIModelProvider(api_key=(get_env("XAI_API_KEY") or "restriction-alias-resolution"))
            elif provider_type == ProviderType.OPENROUTER:
                from providers.openrouter import OpenRouterProvider

                provider = OpenRouterProvider(api_key=(get_env("OPENROUTER_API_KEY") or "restriction-alias-resolution"))
            elif provider_type == ProviderType.VERCEL_GATEWAY:
                from providers.vercel_gateway import VercelGatewayProvider

                provider = VercelGatewayProvider(
                    api_key=(get_env("VERCEL_AI_GATEWAY_API_KEY") or "restriction-alias-resolution")
                )
            elif provider_type == ProviderType.DIAL:
                from providers.dial import DIALModelProvider

                provider = DIALModelProvider(api_key=(get_env("DIAL_API_KEY") or "restriction-alias-resolution"))
            else:
                provider = None
        except Exception as exc:  # pragma: no cover - resolution helpers must not break policy checks
            logger.debug(
                "Could not initialize %s provider for alias resolution: %s",
                provider_type.value,
                exc,
            )
            provider = None

        if provider:
            self._alias_resolution_provider_cache[provider_type] = provider
        return provider

    def _load_from_env(self) -> None:
        """Load restrictions from environment variables."""
        for provider_type, env_var in self.ENV_VARS.items():
            env_value = get_env(env_var)

            if env_value is None or env_value == "":
                # Not set or empty - no restrictions (allow all models)
                logger.debug(f"{env_var} not set or empty - all {provider_type.value} models allowed")
                continue

            # Parse comma-separated list
            models = set()
            for model in env_value.split(","):
                cleaned = model.strip().lower()
                if cleaned:
                    models.add(cleaned)

            if models:
                self.restrictions[provider_type] = models
                self._alias_resolution_cache[provider_type] = {}
                logger.info(f"{provider_type.value} allowed models: {sorted(models)}")
            else:
                # All entries were empty after cleaning - treat as no restrictions
                logger.debug(f"{env_var} contains only whitespace - all {provider_type.value} models allowed")

    def invalidate_alias_resolution_cache(self, provider_types: Optional[set[ProviderType]] = None) -> None:
        """Clear cached alias resolutions so future checks use fresh provider metadata.

        Args:
            provider_types: Optional set of provider types to clear. If omitted,
                clears all cached providers and aliases.

        Returns:
            None.
        """

        if provider_types is None:
            target_types = set(self._alias_resolution_cache.keys()) | set(self._alias_resolution_provider_cache.keys())
        else:
            target_types = {
                provider_type for provider_type in provider_types if isinstance(provider_type, ProviderType)
            }

        for provider_type in target_types:
            self._alias_resolution_cache.pop(provider_type, None)
            provider = self._alias_resolution_provider_cache.pop(provider_type, None)
            if provider:
                try:
                    provider.close()
                except Exception:  # pragma: no cover - defensive cleanup
                    logger.debug("Ignoring alias-resolution provider close failure for %s", provider_type.value)

    def validate_against_known_models(self, provider_instances: dict[ProviderType, any]) -> None:
        """
        Validate restrictions against known models from providers.

        This should be called after providers are initialized to warn about
        typos or invalid model names in the restriction lists.

        Args:
            provider_instances: Dictionary of provider type to provider instance
        """
        for provider_type, allowed_models in self.restrictions.items():
            provider = provider_instances.get(provider_type)
            if not provider:
                continue

            # Get all supported models using the clean polymorphic interface
            try:
                # Gather canonical models and aliases with consistent formatting
                all_models = provider.list_models(
                    respect_restrictions=False,
                    include_aliases=True,
                    lowercase=True,
                    unique=True,
                )
                supported_models = set(all_models)
            except Exception as e:
                logger.debug(f"Could not get model list from {provider_type.value} provider: {e}")
                supported_models = set()

            # Check each allowed model
            for allowed_model in allowed_models:
                if allowed_model not in supported_models:
                    logger.warning(
                        f"Model '{allowed_model}' in {self.ENV_VARS[provider_type]} "
                        f"is not a recognized {provider_type.value} model. "
                        f"Please check for typos. Known models: {sorted(supported_models)}"
                    )

    def is_allowed(self, provider_type: ProviderType, model_name: str, original_name: Optional[str] = None) -> bool:
        """
        Check if a model is allowed for a specific provider.

        Args:
            provider_type: The provider type (OPENAI, GOOGLE, etc.)
            model_name: The canonical model name (after alias resolution)
            original_name: The original model name before alias resolution (optional)

        Returns:
            True if allowed (or no restrictions), False if restricted
        """
        if provider_type not in self.restrictions:
            # No restrictions for this provider
            return True

        allowed_set = self.restrictions[provider_type]

        if len(allowed_set) == 0:
            # Empty set - allowed
            return True

        # Check both the resolved name and original name (if different)
        names_to_check = {model_name.lower()}
        if original_name and original_name.lower() != model_name.lower():
            names_to_check.add(original_name.lower())

        # If any of the names is in the allowed set, it's allowed
        if any(name in allowed_set for name in names_to_check):
            return True

        # Attempt to resolve canonical names for allowed aliases using provider metadata.
        provider = self._get_alias_resolution_provider(provider_type)

        if provider:
            cache = self._alias_resolution_cache.setdefault(provider_type, {})

            for allowed_entry in list(allowed_set):
                normalized_resolved = cache.get(allowed_entry)

                if not normalized_resolved:
                    try:
                        resolved = provider._resolve_model_name(allowed_entry)
                    except Exception:  # pragma: no cover - resolution failures are treated as non-matches
                        continue

                    if not resolved:
                        continue

                    normalized_resolved = resolved.lower()
                    cache[allowed_entry] = normalized_resolved

                if normalized_resolved in names_to_check:
                    allowed_set.add(normalized_resolved)
                    cache[normalized_resolved] = normalized_resolved
                    return True

        return False

    def get_allowed_models(self, provider_type: ProviderType) -> Optional[set[str]]:
        """
        Get the set of allowed models for a provider.

        Args:
            provider_type: The provider type

        Returns:
            Set of allowed model names, or None if no restrictions
        """
        return self.restrictions.get(provider_type)

    def has_restrictions(self, provider_type: ProviderType) -> bool:
        """
        Check if a provider has any restrictions.

        Args:
            provider_type: The provider type

        Returns:
            True if restrictions exist, False otherwise
        """
        return provider_type in self.restrictions

    def filter_models(self, provider_type: ProviderType, models: list[str]) -> list[str]:
        """
        Filter a list of models based on restrictions.

        Args:
            provider_type: The provider type
            models: List of model names to filter

        Returns:
            Filtered list containing only allowed models
        """
        if not self.has_restrictions(provider_type):
            return models

        return [m for m in models if self.is_allowed(provider_type, m)]

    def get_restriction_summary(self) -> dict[str, any]:
        """
        Get a summary of all restrictions for logging/debugging.

        Returns:
            Dictionary with provider names and their restrictions
        """
        summary = {}
        for provider_type, allowed_set in self.restrictions.items():
            if allowed_set:
                summary[provider_type.value] = sorted(allowed_set)
            else:
                summary[provider_type.value] = "none (provider disabled)"

        return summary


# Global instance (singleton pattern)
_restriction_service: Optional[ModelRestrictionService] = None


def invalidate_restriction_alias_caches(provider_types: Optional[set[ProviderType]] = None) -> None:
    """Clear cached alias resolutions on the global restriction service instance.

    Args:
        provider_types: Optional set of providers to clear. If ``None``, clears
            all cached alias data.

    Returns:
        None.
    """

    if _restriction_service is None:
        return

    _restriction_service.invalidate_alias_resolution_cache(provider_types)


def get_restriction_service() -> ModelRestrictionService:
    """
    Get the global restriction service instance.

    Returns:
        The singleton ModelRestrictionService instance
    """
    global _restriction_service
    if _restriction_service is None:
        _restriction_service = ModelRestrictionService()
    return _restriction_service
