"""Vercel AI Gateway provider implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from utils.env import get_env, get_env_bool

from .openai_compatible import OpenAICompatibleProvider
from .registries.vercel_gateway import VercelGatewayModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class VercelGatewayProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """OpenAI-compatible provider backed by the Vercel AI Gateway."""

    FRIENDLY_NAME = "Vercel AI Gateway"
    REGISTRY_CLASS = VercelGatewayModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    def __init__(self, api_key: str, **kwargs):
        self._ensure_registry()

        base_url = get_env("VERCEL_AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
        kwargs.setdefault("base_url", (base_url or "https://ai-gateway.vercel.sh/v1").rstrip("/"))

        self._alias_cache: dict[str, str] = {}
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: str | None = None,
    ) -> ModelCapabilities | None:
        """Resolve registry metadata, falling back to generic provider/model entries."""

        capabilities = super()._lookup_capabilities(canonical_name, requested_name)
        if capabilities is not None:
            return capabilities

        allow_generic_models = get_env_bool("VERCEL_GATEWAY_ALLOW_GENERIC_MODELS", False)

        if not allow_generic_models:
            logger.debug(
                "Rejecting unknown Vercel Gateway model '%s' (not in registry; set "
                "VERCEL_GATEWAY_ALLOW_GENERIC_MODELS=true to opt in to generic fallback)",
                canonical_name,
            )
            return None

        logger.debug("Using generic Vercel Gateway capabilities for %s", canonical_name)
        generic = ModelCapabilities(
            provider=ProviderType.VERCEL_GATEWAY,
            model_name=canonical_name,
            friendly_name=f"Vercel Gateway ({canonical_name})",
            display_name=f"Vercel Gateway ({canonical_name})",
            intelligence_score=9,
            context_window=32_768,
            max_output_tokens=32_768,
            supports_extended_thinking=False,
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_images=False,
            supports_json_mode=False,
            supports_temperature=True,
            lifecycle="quarantined",
            quarantine=True,
            catalog_source="generic_fallback",
            description=(
                "Generic Vercel AI Gateway model entry inferred at runtime. "
                "Allowed for explicit user requests; excluded from auto-mode selection."
            ),
        )
        generic._is_generic = True
        return generic

    def _raise_unsupported_model(self, model_name: str) -> None:
        raise ValueError(
            "Unsupported Vercel Gateway model: "
            f"{model_name}. Use a configured Vercel registry model/alias or set "
            "VERCEL_GATEWAY_ALLOW_GENERIC_MODELS=true to allow unknown generic model IDs."
        )

    def get_provider_type(self) -> ProviderType:
        return ProviderType.VERCEL_GATEWAY

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve aliases from the static registry before using model names directly."""

        cache_key = model_name.lower()
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]

        if self._registry is not None:
            config = self._registry.resolve(model_name)
            if config:
                resolved = config.model_name
                self._alias_cache[cache_key] = resolved
                self._alias_cache.setdefault(resolved.lower(), resolved)
                return resolved

        self._alias_cache[cache_key] = model_name
        return model_name

    def get_all_model_capabilities(self) -> dict[str, ModelCapabilities]:
        self._ensure_registry()
        if self._registry is None:
            return {}
        return dict(self._registry.model_map)

    def get_preferred_model(self, category: ToolModelCategory, allowed_models: list[str]) -> str | None:
        """Return the top-ranked allowed model from this provider."""

        _ = category  # Category-aware routing can be added later if needed.
        if not allowed_models:
            return None

        capabilities = self.get_all_model_capabilities()

        def _score(model_name: str) -> int:
            caps = capabilities.get(model_name)
            if not caps:
                return 0
            return caps.get_effective_capability_rank()

        ranked = sorted(
            allowed_models,
            key=lambda name: (-_score(name), name),
        )
        return ranked[0]
