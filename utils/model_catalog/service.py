"""Model catalog control-plane service.

This module materializes merged provider manifests and injects them through the
existing `*_MODELS_CONFIG_PATH` registry override mechanism.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.env import get_env, get_env_bool, has_runtime_override, set_runtime_envs
from utils.model_catalog.cache_store import load_cache, save_cache
from utils.model_catalog.discovery_client import discover_all
from utils.model_catalog.merge_engine import materialize_provider_manifests, merge_catalogs
from utils.model_catalog.types import CatalogStatus

logger = logging.getLogger(__name__)

_CATALOG_STATUS = CatalogStatus()
_STATUS_LOCK = threading.Lock()
_REFRESH_LOCK = threading.Lock()
_REFRESH_TASK: asyncio.Task | None = None

_DEFAULT_REFRESH_INTERVAL_SECONDS = 21600
_DEFAULT_DISCOVERY_TIMEOUT_MS = 3000

_PROVIDER_CONFIG = {
    "openai": ("OPENAI_MODELS_CONFIG_PATH", "openai_models.json"),
    "google": ("GEMINI_MODELS_CONFIG_PATH", "gemini_models.json"),
    "xai": ("XAI_MODELS_CONFIG_PATH", "xai_models.json"),
    "openrouter": ("OPENROUTER_MODELS_CONFIG_PATH", "openrouter_models.json"),
    "dial": ("DIAL_MODELS_CONFIG_PATH", "dial_models.json"),
    "custom": ("CUSTOM_MODELS_CONFIG_PATH", "custom_models.json"),
    "azure": ("AZURE_MODELS_CONFIG_PATH", "azure_models.json"),
    "vercel_gateway": ("VERCEL_GATEWAY_MODELS_CONFIG_PATH", "vercel_gateway_models.json"),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _conf_dir() -> Path:
    return _project_root() / "conf"


def _is_writable_directory(path: Path) -> bool:
    probe_path = path / ".pal_catalog_write_probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            probe_path.unlink(missing_ok=True)
        return False


def _default_catalog_runtime_root() -> Path:
    candidates: list[Path] = []
    xdg_cache_home = get_env("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home).expanduser() / "pal-mcp-server" / "model_catalog")
    candidates.append(Path.home() / ".cache" / "pal-mcp-server" / "model_catalog")
    candidates.append(Path(tempfile.gettempdir()) / "pal-mcp-server" / "model_catalog")

    for candidate in candidates:
        if _is_writable_directory(candidate):
            return candidate
    return candidates[-1]


def _resolve_catalog_storage_paths() -> tuple[Path, str]:
    generated_raw = get_env("MODEL_CATALOG_GENERATED_DIR")
    cache_raw = get_env("MODEL_CATALOG_CACHE_PATH")
    if generated_raw and cache_raw:
        return Path(generated_raw).expanduser(), str(Path(cache_raw).expanduser())

    runtime_root = _default_catalog_runtime_root()
    generated_dir = Path(generated_raw).expanduser() if generated_raw else runtime_root / "generated"
    cache_path = Path(cache_raw).expanduser() if cache_raw else runtime_root / "model_catalog_cache.json"
    return generated_dir, str(cache_path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_positive_int(raw_value: str | None, default_value: int, *, min_value: int = 1) -> int:
    try:
        parsed = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return default_value
    return parsed if parsed >= min_value else default_value


def _refresh_interval_seconds() -> int:
    raw = get_env("MODEL_CATALOG_REFRESH_INTERVAL_SECONDS", str(_DEFAULT_REFRESH_INTERVAL_SECONDS))
    return _safe_positive_int(raw, _DEFAULT_REFRESH_INTERVAL_SECONDS, min_value=0)


def _read_manifest_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []
    return [item for item in models if isinstance(item, dict)]


def _normalize_models_for_diff(payload: dict[str, Any]) -> list[dict[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, list):
        return []

    normalized = [item for item in models if isinstance(item, dict)]
    return sorted(normalized, key=lambda item: str(item.get("model_name", "")).lower())


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> tuple[bool, bool]:
    try:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return False, False

    next_models = _normalize_models_for_diff(payload)

    try:
        if path.exists() and path.is_file():
            existing = path.read_text(encoding="utf-8")
            try:
                existing_payload = json.loads(existing)
            except json.JSONDecodeError:
                existing_payload = None

            if isinstance(existing_payload, dict):
                if _normalize_models_for_diff(existing_payload) == next_models:
                    if existing_payload.get("_GENERATED") == payload.get("_GENERATED"):
                        return True, False
            elif existing == serialized:
                return True, False

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
        return True, True
    except OSError:
        return False, False


def _load_static_catalog() -> dict[str, list[dict[str, Any]]]:
    manifests: dict[str, list[dict[str, Any]]] = {}
    conf_dir = _conf_dir()

    for provider_name, (_, filename) in _PROVIDER_CONFIG.items():
        entries = _read_manifest_file(conf_dir / filename)
        if entries:
            manifests[provider_name] = entries

    return manifests


def _compute_static_manifest_fingerprint(static_catalog: dict[str, list[dict[str, Any]]]) -> str:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for provider_name, entries in static_catalog.items():
        normalized[str(provider_name)] = sorted(
            [item for item in entries if isinstance(item, dict)],
            key=lambda item: str(item.get("model_name", "")).lower(),
        )

    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_catalog(
    cache_path: str,
    *,
    expected_static_fingerprint: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    payload = load_cache(cache_path)
    if not payload:
        return {}

    if expected_static_fingerprint:
        cached_fingerprint = str(payload.get("static_manifest_fingerprint") or "").strip()
        # Backwards compatibility: caches written before fingerprint support remain valid.
        # Only enforce strict invalidation when a fingerprint exists and mismatches.
        if cached_fingerprint and cached_fingerprint != expected_static_fingerprint:
            logger.warning(
                "Invalidating model catalog cache at %s due to static manifest fingerprint mismatch "
                "(expected=%s, cached=%s).",
                cache_path,
                expected_static_fingerprint,
                cached_fingerprint,
            )
            return {}

    providers = payload.get("providers")
    if not isinstance(providers, dict):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for provider_name, entries in providers.items():
        if isinstance(entries, list):
            normalized[str(provider_name)] = [item for item in entries if isinstance(item, dict)]
    return normalized


def _snapshot_status() -> CatalogStatus:
    with _STATUS_LOCK:
        return CatalogStatus(**_CATALOG_STATUS.to_dict())


def _store_status(status: CatalogStatus) -> None:
    global _CATALOG_STATUS
    with _STATUS_LOCK:
        _CATALOG_STATUS = status


def _compute_fallback_mode(
    *,
    enabled: bool,
    discovered_models: int,
    cache_catalog: dict[str, list[dict[str, Any]]],
) -> str:
    if not enabled:
        return "disabled"
    if discovered_models > 0:
        return "none"
    if cache_catalog:
        return "cache+static"
    return "static"


def _notify_provider_registry_reload(changed_providers: list[str]) -> None:
    if not changed_providers:
        return

    try:
        from providers.registry import ModelProviderRegistry

        ModelProviderRegistry.reload_provider_catalogs(changed_providers)
    except Exception as exc:  # pragma: no cover - import/initialization varies in tests
        logger.warning("Unable to hot-reload provider registries: %s", exc)


def get_model_catalog_status() -> dict[str, Any]:
    """Return runtime status of the model catalog control-plane.

    Returns:
        Runtime status snapshot for catalog refresh and storage behavior.
    """

    status = _snapshot_status()
    status.refresh_task_running = bool(_REFRESH_TASK and not _REFRESH_TASK.done())
    return status.to_dict()


def refresh_model_catalog_once(
    reason: str = "manual",
    *,
    enable_discovery: bool | None = None,
) -> dict[str, Any]:
    """Build merged catalog manifests and inject runtime config path overrides.

    Args:
        reason: Label describing why the refresh was triggered.
        enable_discovery: Optional runtime override for discovery behavior.

    Returns:
        Runtime status snapshot after refresh attempt.
    """

    started_at = datetime.now(timezone.utc)

    with _REFRESH_LOCK:
        status = _snapshot_status()
        status.refresh_in_progress = True
        status.last_refresh_reason = reason
        status.last_refresh_started_utc = started_at.isoformat()
        status.last_refresh_error = None
        status.last_refresh_success = None
        status.last_refresh_duration_ms = 0
        status.merge_warnings = []
        status.changed_providers = []
        status.cache_loaded = False
        status.cache_saved = False
        status.refresh_attempt_count += 1
        status.refresh_interval_seconds = _refresh_interval_seconds()
        status.periodic_refresh_enabled = status.refresh_interval_seconds > 0
        _store_status(status)

        try:
            status.enabled = get_env_bool("MODEL_CATALOG_ENABLED", False)

            running_under_pytest = bool(get_env("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules
            discovery_default = False
            cache_enabled_default = False if running_under_pytest else True
            cache_enabled = get_env_bool("MODEL_CATALOG_ENABLE_CACHE", cache_enabled_default)
            configured_discovery = get_env_bool("MODEL_CATALOG_ENABLE_DISCOVERY", discovery_default)
            status.discovery_enabled = configured_discovery if enable_discovery is None else bool(enable_discovery)
            status.cache_enabled = cache_enabled

            if not status.enabled:
                status.initialized = True
                status.source_mode = "disabled"
                status.fallback_mode = "disabled"
                status.merged_models = 0
                status.discovered_models = 0
                status.discovery_errors = []
                status.last_updated_utc = _now_utc()
                status.refresh_success_count += 1
                status.last_refresh_success = True
                return status.to_dict()

            generated_dir, cache_path = _resolve_catalog_storage_paths()
            status.cache_path = cache_path
            writes_enabled = _is_writable_directory(generated_dir) and _is_writable_directory(Path(cache_path).parent)
            if not writes_enabled:
                warning = (
                    "Model catalog writes disabled for this run: "
                    f"generated_dir='{generated_dir}', cache_path='{cache_path}'. "
                    "Set MODEL_CATALOG_GENERATED_DIR and MODEL_CATALOG_CACHE_PATH to writable locations."
                )
                status.merge_warnings.append(warning)
                logger.warning(warning)

            static_catalog = _load_static_catalog()
            static_manifest_fingerprint = _compute_static_manifest_fingerprint(static_catalog)
            cache_catalog = (
                _load_cached_catalog(
                    cache_path,
                    expected_static_fingerprint=static_manifest_fingerprint,
                )
                if cache_enabled
                else {}
            )
            status.cache_loaded = bool(cache_catalog)
            source_mode = "cache+static" if cache_catalog else "static"

            discovered_entries: list[dict[str, Any]] = []
            if status.discovery_enabled:
                discovery_timeout_ms = _safe_positive_int(
                    get_env("MODEL_CATALOG_DISCOVERY_TIMEOUT_MS", str(_DEFAULT_DISCOVERY_TIMEOUT_MS)),
                    _DEFAULT_DISCOVERY_TIMEOUT_MS,
                )
                discovered_entries, discovery_errors = discover_all(discovery_timeout_ms)
                status.discovery_errors = discovery_errors
                status.discovered_models = len(discovered_entries)
                if discovered_entries:
                    source_mode = "discovery+cache+static" if cache_catalog else "discovery+static"
            else:
                status.discovery_errors = []
                status.discovered_models = 0

            quarantine_enabled = get_env_bool("MODEL_CATALOG_QUARANTINE_ENABLED", True)
            merge_warnings: list[str] = []

            merged = merge_catalogs(
                static_by_provider=static_catalog,
                cache_by_provider=cache_catalog,
                discovered_entries=discovered_entries,
                quarantine_enabled=quarantine_enabled,
                warnings_out=merge_warnings,
            )
            status.merge_warnings.extend(merge_warnings)

            generated_payloads = materialize_provider_manifests(
                merged,
                source_mode=source_mode,
            )

            runtime_overrides: dict[str, str] = {}
            changed_providers: list[str] = []
            status.generated_paths = {}
            status.skipped_overrides = []

            if writes_enabled:
                for provider_name, payload in generated_payloads.items():
                    env_var_name, default_filename = _PROVIDER_CONFIG.get(provider_name, (None, None))
                    if not env_var_name or not default_filename:
                        continue

                    explicit_value = None if has_runtime_override(env_var_name) else get_env(env_var_name)
                    if explicit_value:
                        status.skipped_overrides.append(env_var_name)
                        continue

                    generated_path = generated_dir / default_filename
                    write_ok, changed = _write_json_if_changed(generated_path, payload)
                    if write_ok:
                        runtime_overrides[env_var_name] = str(generated_path)
                        status.generated_paths[provider_name] = str(generated_path)
                        if changed:
                            changed_providers.append(provider_name)
                    else:
                        warning = f"failed to write generated catalog for provider '{provider_name}'"
                        status.merge_warnings.append(warning)
                        logger.warning(warning)

            if runtime_overrides:
                set_runtime_envs(runtime_overrides)

            cache_saved = False
            if cache_enabled and writes_enabled:
                cache_saved = save_cache(
                    cache_path,
                    {
                        "saved_at": _now_utc(),
                        "source_mode": source_mode,
                        "static_manifest_fingerprint": static_manifest_fingerprint,
                        "providers": merged,
                    },
                )

            status.cache_saved = cache_saved
            status.source_mode = source_mode
            status.fallback_mode = _compute_fallback_mode(
                enabled=status.enabled,
                discovered_models=status.discovered_models,
                cache_catalog=cache_catalog,
            )
            status.merged_models = sum(len(entries) for entries in merged.values())
            status.initialized = True
            status.last_updated_utc = _now_utc()
            status.changed_providers = sorted(set(changed_providers))

            _notify_provider_registry_reload(status.changed_providers)

            status.refresh_success_count += 1
            status.last_refresh_success = True

            logger.info(
                "Model catalog refresh complete: reason=%s mode=%s merged=%s discovered=%s changed=%s",
                reason,
                status.source_mode,
                status.merged_models,
                status.discovered_models,
                ",".join(status.changed_providers) if status.changed_providers else "none",
            )
            return status.to_dict()
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            status.last_refresh_error = str(exc)
            status.last_refresh_success = False
            status.fallback_mode = status.fallback_mode or "static"
            logger.exception("Model catalog refresh failed: %s", exc)
            return status.to_dict()
        finally:
            completed_at = datetime.now(timezone.utc)
            status.refresh_in_progress = False
            status.last_refresh_completed_utc = completed_at.isoformat()
            status.last_refresh_duration_ms = int((completed_at - started_at).total_seconds() * 1000)
            if status.last_refresh_success:
                status.last_updated_utc = completed_at.isoformat()
            _store_status(status)


def initialize_model_catalog() -> dict[str, Any]:
    """Initialize the model catalog once at startup.

    Returns:
        Runtime status snapshot after startup initialization.
    """

    return refresh_model_catalog_once(reason="startup", enable_discovery=False)


async def _periodic_refresh_loop(interval_seconds: int) -> None:
    try:
        await asyncio.to_thread(refresh_model_catalog_once, "startup-background")
    except asyncio.CancelledError:  # pragma: no cover - cancellation timing is nondeterministic
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Background model catalog refresh failed: %s", exc)

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await asyncio.to_thread(refresh_model_catalog_once, "periodic")
        except asyncio.CancelledError:  # pragma: no cover - cancellation timing is nondeterministic
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Periodic model catalog refresh failed: %s", exc)


async def start_model_catalog_refresh_task() -> None:
    """Start periodic model-catalog refresh task if enabled.

    Returns:
        None.
    """

    global _REFRESH_TASK

    status = _snapshot_status()
    status.refresh_interval_seconds = _refresh_interval_seconds()
    status.periodic_refresh_enabled = status.refresh_interval_seconds > 0

    if not get_env_bool("MODEL_CATALOG_ENABLED", False) or not status.periodic_refresh_enabled:
        status.refresh_task_running = False
        _store_status(status)
        logger.info("Model catalog periodic refresh disabled")
        return

    if _REFRESH_TASK and not _REFRESH_TASK.done():
        status.refresh_task_running = True
        _store_status(status)
        return

    loop = asyncio.get_running_loop()
    _REFRESH_TASK = loop.create_task(
        _periodic_refresh_loop(status.refresh_interval_seconds),
        name="pal-model-catalog-refresh",
    )
    status.refresh_task_running = True
    _store_status(status)
    logger.info(
        "Started model catalog periodic refresh task (interval=%ss, discovery=%s)",
        status.refresh_interval_seconds,
        str(get_env_bool("MODEL_CATALOG_ENABLE_DISCOVERY", False)).lower(),
    )


async def stop_model_catalog_refresh_task() -> None:
    """Stop periodic model-catalog refresh task.

    Returns:
        None.
    """

    global _REFRESH_TASK

    task = _REFRESH_TASK
    _REFRESH_TASK = None

    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    status = _snapshot_status()
    status.refresh_task_running = False
    _store_status(status)
