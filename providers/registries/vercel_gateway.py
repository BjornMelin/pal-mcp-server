"""Registry loader for Vercel AI Gateway model capabilities."""

from __future__ import annotations

from ..shared import ProviderType
from .base import CapabilityModelRegistry


class VercelGatewayModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/vercel_gateway_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="VERCEL_GATEWAY_MODELS_CONFIG_PATH",
            default_filename="vercel_gateway_models.json",
            provider=ProviderType.VERCEL_GATEWAY,
            friendly_prefix="Vercel Gateway ({model})",
            config_path=config_path,
        )
