"""Merge logic for static, cache, and discovered model catalogs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from providers.registries.base import CAPABILITY_FIELD_NAMES

QUARANTINE_PREFIX = "[QUARANTINE] "
_PROVIDER_EXTRA_FIELDS: dict[str, set[str]] = {
    # Azure registries require deployment metadata outside ModelCapabilities.
    "azure": {"deployment", "deployment_name"},
}


def is_quarantine_description(description: str | None) -> bool:
    """Return whether a model description marks the model as quarantined.

    Args:
        description: Candidate model description.

    Returns:
        True when description starts with the quarantine prefix.
    """

    return bool(description and description.startswith(QUARANTINE_PREFIX))


def _friendly_prefix(provider_name: str) -> str:
    mapping = {
        "openai": "OpenAI",
        "google": "Gemini",
        "xai": "X.AI",
        "openrouter": "OpenRouter",
        "dial": "DIAL",
        "custom": "Custom",
        "azure": "Azure",
        "vercel_gateway": "Vercel Gateway",
    }
    return mapping.get(provider_name, provider_name.replace("_", " ").title())


def _provider_extra_fields(provider_name: str) -> set[str]:
    return _PROVIDER_EXTRA_FIELDS.get(provider_name.strip().lower(), set())


def sanitize_capability_entry(
    entry: dict[str, Any],
    provider_name: str,
    *,
    source: str | None = None,
) -> dict[str, Any] | None:
    """Normalize a capability entry to fields accepted by ``ModelCapabilities``.

    Args:
        entry: Raw capability entry payload.
        provider_name: Provider identifier for the entry.
        source: Optional catalog source tag.

    Returns:
        Normalized capability entry, or ``None`` when the input is invalid.
    """

    model_name = str(entry.get("model_name", "")).strip()
    if not model_name:
        return None

    friendly_name = str(
        entry.get("friendly_name") or entry.get("display_name") or f"{_friendly_prefix(provider_name)} ({model_name})"
    )
    display_name = str(entry.get("display_name") or friendly_name)
    model_id = str(entry.get("model_id") or model_name)
    provider_type = str(entry.get("provider_type") or provider_name)
    aliases = entry.get("aliases")
    if isinstance(aliases, str):
        aliases = [alias.strip() for alias in aliases.split(",") if alias.strip()]
    if not isinstance(aliases, list):
        aliases = []

    intelligence_score = entry.get("intelligence_score", 10)
    try:
        intelligence_score = int(intelligence_score)
    except (TypeError, ValueError):
        intelligence_score = 10
    intelligence_score = max(1, min(20, intelligence_score))

    description = str(entry.get("description") or "")
    release_metadata = entry.get("release_metadata")
    if not isinstance(release_metadata, dict):
        release_metadata = {}
    else:
        release_metadata = {
            str(k): str(v)
            for k, v in release_metadata.items()
            if k in {"release_date", "source_url", "last_verified_at"} and isinstance(v, str) and v.strip()
        }
    openrouter_counterpart = entry.get("openrouter_counterpart")
    if openrouter_counterpart is not None:
        openrouter_counterpart = str(openrouter_counterpart).strip() or None

    normalized: dict[str, Any] = {
        "model_name": model_name,
        "model_id": model_id,
        "provider_type": provider_type,
        "friendly_name": friendly_name,
        "display_name": display_name,
        "aliases": aliases,
        "intelligence_score": intelligence_score,
        "description": description,
        "release_metadata": release_metadata,
        "openrouter_counterpart": openrouter_counterpart,
    }

    if source and "catalog_source" in CAPABILITY_FIELD_NAMES:
        normalized["catalog_source"] = source

    for key in CAPABILITY_FIELD_NAMES:
        if key in {"provider", "model_name", "friendly_name", "aliases", "intelligence_score", "description"}:
            continue
        if key in entry and entry[key] is not None:
            normalized[key] = entry[key]

    # Preserve provider-scoped extension fields required by specific registries.
    for key in _provider_extra_fields(provider_name):
        if key in entry and entry[key] is not None:
            normalized[key] = entry[key]

    # Cast known numeric fields defensively.
    for numeric_key in ["context_window", "max_output_tokens", "max_thinking_tokens"]:
        if numeric_key in normalized:
            try:
                normalized[numeric_key] = int(normalized[numeric_key])
            except (TypeError, ValueError):
                normalized[numeric_key] = 0

    return normalized


def build_quarantine_entry(provider_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Create a conservative quarantine entry for newly discovered models.

    Args:
        provider_name: Provider identifier for the discovered entry.
        entry: Discovered capability metadata.

    Returns:
        Quarantined capability entry with safe defaults.
    """

    model_name = str(entry.get("model_name", "")).strip()
    context_window = int(entry.get("context_window", 0) or 0)
    max_output_tokens = int(entry.get("max_output_tokens", 0) or 0)
    supports_images = bool(entry.get("supports_images", False))

    now = datetime.now(timezone.utc).isoformat()
    source_metadata = entry.get("release_metadata")
    if not isinstance(source_metadata, dict):
        source_metadata = {}

    source_url = str(source_metadata.get("source_url") or entry.get("source_url") or "").strip()
    release_date = str(source_metadata.get("release_date") or "").strip()

    release_metadata = {"last_verified_at": now}
    if source_url:
        release_metadata["source_url"] = source_url
    if release_date:
        release_metadata["release_date"] = release_date
    counterpart = entry.get("openrouter_counterpart")
    if counterpart is not None:
        counterpart = str(counterpart).strip() or None

    return {
        "model_name": model_name,
        "model_id": model_name,
        "provider_type": provider_name,
        "friendly_name": f"{_friendly_prefix(provider_name)} ({model_name})",
        "display_name": f"{_friendly_prefix(provider_name)} ({model_name})",
        "aliases": [],
        "intelligence_score": 1,
        "context_window": context_window if context_window > 0 else 32768,
        "max_output_tokens": max_output_tokens if max_output_tokens > 0 else 8192,
        "supports_extended_thinking": False,
        "supports_system_prompts": True,
        "supports_streaming": True,
        "supports_function_calling": False,
        "supports_images": supports_images,
        "supports_json_mode": False,
        "supports_temperature": True,
        "quarantine": True,
        "lifecycle": "quarantined",
        "catalog_source": "discovery",
        "release_metadata": release_metadata,
        "openrouter_counterpart": counterpart,
        "description": (
            f"{QUARANTINE_PREFIX}Discovered automatically at {now}. This model is visible but excluded from auto "
            "selection until curated ranking metadata is published."
        ),
    }


def _append_warning(
    warnings_out: list[str] | None,
    warning_seen: set[str],
    message: str,
) -> None:
    if warnings_out is None:
        return

    if message in warning_seen:
        return

    warning_seen.add(message)
    if len(warnings_out) >= 100:
        return

    warnings_out.append(message)


def _merge_source(existing_source: str | None, new_source: str) -> str:
    if not existing_source:
        return new_source

    parts = {part.strip() for part in existing_source.split("+") if part.strip()}
    parts.add(new_source)
    return "+".join(sorted(parts))


def _find_alias_owner(bucket: dict[str, dict[str, Any]], candidate_model_key: str) -> tuple[str, str] | None:
    for existing_key, existing_entry in bucket.items():
        aliases = existing_entry.get("aliases")
        if not isinstance(aliases, list):
            continue
        for alias in aliases:
            if str(alias).strip().lower() == candidate_model_key:
                model_name = str(existing_entry.get("model_name", "")).strip() or existing_key
                return existing_key, model_name
    return None


def merge_catalogs(
    *,
    static_by_provider: dict[str, list[dict[str, Any]]],
    cache_by_provider: dict[str, list[dict[str, Any]]],
    discovered_entries: list[dict[str, Any]],
    quarantine_enabled: bool,
    warnings_out: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Merge static, cache, and discovery sources into provider manifests.

    Args:
        static_by_provider: Provider-scoped static manifest entries.
        cache_by_provider: Provider-scoped cached manifest entries.
        discovered_entries: Discovery entries from provider APIs.
        quarantine_enabled: Whether unknown discovered entries are quarantined.
        warnings_out: Optional warning collector for merge side effects.

    Returns:
        Provider-scoped merged capability entries.
    """

    merged: dict[str, dict[str, dict[str, Any]]] = {}
    warning_seen: set[str] = set()

    def coerce_optional_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return None

    def upsert(provider_name: str, raw_entry: dict[str, Any], *, replace: bool, source: str) -> None:
        normalized = sanitize_capability_entry(raw_entry, provider_name, source=source)
        if not normalized:
            return
        model_key = normalized["model_name"].lower()
        bucket = merged.setdefault(provider_name, {})

        alias_owner = _find_alias_owner(bucket, model_key)
        if alias_owner and model_key not in bucket:
            _, owner_name = alias_owner
            _append_warning(
                warnings_out,
                warning_seen,
                (
                    f"{provider_name}:{normalized['model_name']} skipped insert from {source} because it collides "
                    f"with existing alias owned by {owner_name}"
                ),
            )
            return

        if replace and model_key in bucket:
            existing = bucket[model_key]
            for field_name in ["aliases", "intelligence_score", "description", "lifecycle", "quarantine"]:
                if field_name not in normalized or field_name not in existing:
                    continue
                if normalized[field_name] == existing[field_name]:
                    continue
                _append_warning(
                    warnings_out,
                    warning_seen,
                    (
                        f"{provider_name}:{normalized['model_name']} field '{field_name}' replaced by {source} "
                        f"(old={existing[field_name]!r}, new={normalized[field_name]!r})"
                    ),
                )
        if replace or model_key not in bucket:
            bucket[model_key] = normalized

    # 1) static base
    for provider_name, entries in static_by_provider.items():
        for entry in entries:
            upsert(provider_name, entry, replace=True, source="static")

    def enrich_from_cache(existing: dict[str, Any], cached: dict[str, Any]) -> None:
        # Fill-only enrichment fields; never override curated/static values when already present.
        numeric_fields = ["context_window", "max_output_tokens", "max_thinking_tokens"]
        bool_fields = [
            "supports_images",
            "supports_function_calling",
            "supports_json_mode",
            "supports_temperature",
        ]

        for key in numeric_fields:
            if key not in cached:
                continue
            try:
                cached_value = int(cached[key])
            except (TypeError, ValueError):
                continue
            if cached_value <= 0:
                continue
            existing_value = existing.get(key)
            if not isinstance(existing_value, int) or existing_value <= 0:
                existing[key] = cached_value

        if "max_image_size_mb" in cached:
            try:
                cached_max_image = float(cached["max_image_size_mb"])
            except (TypeError, ValueError):
                cached_max_image = 0.0
            existing_max_image = existing.get("max_image_size_mb")
            if cached_max_image > 0 and (not isinstance(existing_max_image, (int, float)) or existing_max_image <= 0):
                existing["max_image_size_mb"] = cached_max_image

        for key in bool_fields:
            if key not in cached:
                continue
            cached_value = coerce_optional_bool(cached[key])
            if cached_value is None:
                continue
            existing_value = existing.get(key)
            if existing_value is None:
                existing[key] = cached_value

        cached_release = cached.get("release_metadata")
        if isinstance(cached_release, dict) and cached_release:
            existing_release = existing.get("release_metadata")
            if not isinstance(existing_release, dict):
                existing_release = {}
                existing["release_metadata"] = existing_release
            for meta_key, meta_value in cached_release.items():
                if meta_key not in existing_release and isinstance(meta_value, str) and meta_value.strip():
                    existing_release[meta_key] = meta_value

        existing["catalog_source"] = _merge_source(str(existing.get("catalog_source") or ""), "cache")

    # 2) cache overlay (authoritative for cache-only models, enrichment-only for static models)
    for provider_name, entries in cache_by_provider.items():
        for entry in entries:
            normalized = sanitize_capability_entry(entry, provider_name, source="cache")
            if not normalized:
                continue

            bucket = merged.setdefault(provider_name, {})
            model_key = normalized["model_name"].lower()
            existing = bucket.get(model_key)

            if existing is None:
                upsert(provider_name, entry, replace=False, source="cache")
                continue

            enrich_from_cache(existing, normalized)

    # 3) provider discovery (enrich existing or quarantine unknown)
    for entry in discovered_entries:
        provider_name = str(entry.get("provider", "")).strip().lower()
        model_name = str(entry.get("model_name", "")).strip()
        if not provider_name or not model_name:
            continue

        bucket = merged.setdefault(provider_name, {})
        model_key = model_name.lower()

        existing = bucket.get(model_key)
        if existing:
            # enrich only missing fields from discovery to preserve curated metadata
            for key in ["context_window", "max_output_tokens", "supports_images", "supports_function_calling"]:
                existing_val = existing.get(key)
                discovered_val = entry.get(key)
                if discovered_val in (None, ""):
                    continue
                if key in ["context_window", "max_output_tokens"]:
                    try:
                        discovered_int = int(discovered_val)
                    except (TypeError, ValueError):
                        continue
                    if (not isinstance(existing_val, int) or existing_val <= 0) and discovered_int > 0:
                        existing[key] = discovered_int
                    elif (
                        isinstance(existing_val, int)
                        and existing_val > 0
                        and discovered_int > 0
                        and existing_val != discovered_int
                    ):
                        _append_warning(
                            warnings_out,
                            warning_seen,
                            (
                                f"{provider_name}:{model_name} retained authoritative "
                                f"{key}={existing_val}; "
                                f"discovery reported {discovered_int}"
                            ),
                        )
                elif existing_val in (None, False, 0):
                    existing[key] = discovered_val
                elif existing_val != discovered_val:
                    _append_warning(
                        warnings_out,
                        warning_seen,
                        (
                            f"{provider_name}:{model_name} retained authoritative "
                            f"{key}={existing_val!r}; "
                            f"discovery reported {discovered_val!r}"
                        ),
                    )
            existing["catalog_source"] = _merge_source(str(existing.get("catalog_source") or ""), "discovery")
            continue

        alias_owner = _find_alias_owner(bucket, model_key)
        if alias_owner:
            _, owner_name = alias_owner
            _append_warning(
                warnings_out,
                warning_seen,
                (
                    f"{provider_name}:{model_name} skipped discovery insert because "
                    "it collides with existing alias "
                    f"owned by {owner_name}"
                ),
            )
            continue

        if quarantine_enabled:
            bucket[model_key] = build_quarantine_entry(provider_name, entry)
        else:
            upsert(provider_name, entry, replace=False, source="discovery")

    return {provider_name: list(entries.values()) for provider_name, entries in merged.items()}


def materialize_provider_manifests(
    merged_by_provider: dict[str, list[dict[str, Any]]],
    *,
    source_mode: str,
) -> dict[str, dict[str, Any]]:
    """Build provider manifest payloads ready for JSON serialization.

    Args:
        merged_by_provider: Provider-scoped merged entries.
        source_mode: Effective source mode used by the merge.

    Returns:
        Provider-scoped JSON payloads for registry manifests.
    """

    manifests: dict[str, dict[str, Any]] = {}

    for provider_name, entries in merged_by_provider.items():
        sorted_entries = sorted(entries, key=lambda item: item.get("model_name", ""))
        manifests[provider_name] = {
            "_GENERATED": {
                "source_mode": source_mode,
                "entry_count": len(sorted_entries),
            },
            "models": sorted_entries,
        }

    return manifests
