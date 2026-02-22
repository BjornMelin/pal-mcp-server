"""Shared types for the model catalog control-plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CatalogStatus:
    """Runtime status for the model catalog service."""

    initialized: bool = False
    enabled: bool = False
    periodic_refresh_enabled: bool = False
    refresh_task_running: bool = False
    refresh_in_progress: bool = False
    refresh_interval_seconds: int = 21600
    refresh_attempt_count: int = 0
    refresh_success_count: int = 0
    last_refresh_reason: str | None = None
    last_refresh_started_utc: str | None = None
    last_refresh_completed_utc: str | None = None
    last_refresh_duration_ms: int = 0
    last_refresh_success: bool | None = None
    last_refresh_error: str | None = None
    source_mode: str = "static"
    fallback_mode: str = "none"
    cache_path: str | None = None
    cache_enabled: bool = True
    cache_loaded: bool = False
    cache_saved: bool = False
    discovery_enabled: bool = False
    discovery_errors: list[str] = field(default_factory=list)
    discovered_models: int = 0
    merged_models: int = 0
    merge_warnings: list[str] = field(default_factory=list)
    changed_providers: list[str] = field(default_factory=list)
    generated_paths: dict[str, str] = field(default_factory=dict)
    skipped_overrides: list[str] = field(default_factory=list)
    last_updated_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the status."""

        return asdict(self)
