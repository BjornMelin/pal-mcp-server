"""Registry implementations for provider capability manifests."""

from .azure import AzureModelRegistry
from .custom import CustomEndpointModelRegistry
from .dial import DialModelRegistry
from .gemini import GeminiModelRegistry
from .openai import OpenAIModelRegistry
from .openrouter import OpenRouterModelRegistry
from .vercel_gateway import VercelGatewayModelRegistry
from .xai import XAIModelRegistry

__all__ = [
    "AzureModelRegistry",
    "CustomEndpointModelRegistry",
    "DialModelRegistry",
    "GeminiModelRegistry",
    "OpenAIModelRegistry",
    "OpenRouterModelRegistry",
    "VercelGatewayModelRegistry",
    "XAIModelRegistry",
]
