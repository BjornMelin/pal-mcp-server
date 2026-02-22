"""Model provider abstractions for supporting multiple AI providers.

This package intentionally keeps heavyweight provider imports lazy so startup
code can prepare runtime configuration (for example generated model manifests)
before provider modules construct registries.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import ModelProvider
from .registry import ModelProviderRegistry
from .shared import ModelCapabilities, ModelResponse

_LAZY_EXPORTS = {
    "AzureOpenAIProvider": "providers.azure_openai",
    "GeminiModelProvider": "providers.gemini",
    "OpenAIModelProvider": "providers.openai",
    "OpenAICompatibleProvider": "providers.openai_compatible",
    "OpenRouterProvider": "providers.openrouter",
    "VercelGatewayProvider": "providers.vercel_gateway",
}

__all__ = [
    "ModelProvider",
    "ModelResponse",
    "ModelCapabilities",
    "ModelProviderRegistry",
    *sorted(_LAZY_EXPORTS.keys()),
]


def __getattr__(name: str) -> Any:
    """Lazily import concrete provider classes on demand."""

    module_name = _LAZY_EXPORTS.get(name)
    if not module_name:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
