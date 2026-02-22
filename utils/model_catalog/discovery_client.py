"""Provider API discovery clients for model catalog enrichment."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from utils.env import get_env

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
OPENAI_RELEASE_NOTES_URL = "https://platform.openai.com/docs/changelog"
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_RELEASE_NOTES_URL = "https://ai.google.dev/gemini-api/docs/changelog"
XAI_MODELS_URL = "https://api.x.ai/v1/models"
XAI_RELEASE_NOTES_URL = "https://docs.x.ai/docs/api-reference"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_RELEASE_NOTES_URL = "https://docs.anthropic.com/en/release-notes/api"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_DOCS_URL = "https://openrouter.ai/docs/overview/models"
VERCEL_GATEWAY_DOCS_URL = "https://vercel.com/docs/ai-gateway"


def _looks_like_placeholder_api_key(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True

    placeholder_prefixes = (
        "your_",
        "dummy",
        "test-",
        "sk-test",
        "example",
    )
    return normalized.startswith(placeholder_prefixes)


def _fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None) -> dict[str, Any] | None:
    request = urllib.request.Request(url=url, headers=headers or {"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=max(timeout_ms / 1000.0, 0.25)) as response:
        raw = response.read().decode("utf-8")
        payload = json.loads(raw)
    return payload if isinstance(payload, dict) else None


def _entry(provider: str, model_name: str, **kwargs) -> dict[str, Any]:
    source_url = kwargs.pop("source_url", None)
    last_verified_at = kwargs.pop("last_verified_at", None)
    display_name = kwargs.pop("display_name", None) or model_name
    model_id = kwargs.pop("model_id", None) or model_name

    result = {
        "provider": provider,
        "model_name": model_name,
        "model_id": model_id,
        "provider_type": provider,
        "display_name": str(display_name),
    }
    release_metadata = kwargs.pop("release_metadata", None)
    if not isinstance(release_metadata, dict):
        release_metadata = {}
    if source_url:
        release_metadata["source_url"] = str(source_url)
    if last_verified_at:
        release_metadata["last_verified_at"] = str(last_verified_at)
    if release_metadata:
        result["release_metadata"] = release_metadata

    result.update({k: v for k, v in kwargs.items() if v is not None})
    return result


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _openrouter_counterpart(provider: str, model_name: str) -> str | None:
    prefixes = {
        "openai": "openai",
        "google": "google",
        "xai": "x-ai",
        "anthropic": "anthropic",
    }
    prefix = prefixes.get(provider)
    if not prefix or not model_name:
        return None
    return f"{prefix}/{model_name}"


def discover_openai(api_key: str, timeout_ms: int) -> list[dict[str, Any]]:
    now = _now_utc()
    payload = _fetch_json(
        OPENAI_MODELS_URL,
        timeout_ms,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for model in payload.get("data", []):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id", "")).strip()
        if not model_id:
            continue
        if not model_id.startswith(("gpt", "o1", "o3", "o4", "o5", "chatgpt", "codex")):
            continue
        entries.append(
            _entry(
                "openai",
                model_id,
                source_url=OPENAI_RELEASE_NOTES_URL,
                last_verified_at=now,
                openrouter_counterpart=_openrouter_counterpart("openai", model_id),
            )
        )
    return entries


def discover_gemini(api_key: str, timeout_ms: int) -> list[dict[str, Any]]:
    now = _now_utc()
    query = urllib.parse.urlencode({"key": api_key})
    payload = _fetch_json(f"{GEMINI_MODELS_URL}?{query}", timeout_ms)
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for model in payload.get("models", []):
        if not isinstance(model, dict):
            continue

        full_name = str(model.get("name", "")).strip()
        if not full_name.startswith("models/"):
            continue

        model_id = full_name.split("/", 1)[1]
        if not model_id.startswith("gemini"):
            continue

        generation_methods = model.get("supportedGenerationMethods") or []
        if generation_methods and "generateContent" not in generation_methods:
            continue

        entries.append(
            _entry(
                "google",
                model_id,
                context_window=int(model.get("inputTokenLimit", 0) or 0),
                max_output_tokens=int(model.get("outputTokenLimit", 0) or 0),
                source_url=GEMINI_RELEASE_NOTES_URL,
                last_verified_at=now,
                openrouter_counterpart=_openrouter_counterpart("google", model_id),
            )
        )
    return entries


def discover_xai(api_key: str, timeout_ms: int) -> list[dict[str, Any]]:
    now = _now_utc()
    payload = _fetch_json(
        XAI_MODELS_URL,
        timeout_ms,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for model in payload.get("data", []):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id", "")).strip()
        if not model_id or not model_id.startswith("grok"):
            continue
        entries.append(
            _entry(
                "xai",
                model_id,
                source_url=XAI_RELEASE_NOTES_URL,
                last_verified_at=now,
                openrouter_counterpart=_openrouter_counterpart("xai", model_id),
            )
        )
    return entries


def discover_anthropic(api_key: str, timeout_ms: int) -> list[dict[str, Any]]:
    now = _now_utc()
    payload = _fetch_json(
        ANTHROPIC_MODELS_URL,
        timeout_ms,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Accept": "application/json",
        },
    )
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for model in payload.get("data", []):
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id", "")).strip()
        if not model_id or not model_id.startswith("claude"):
            continue

        # PAL does not ship a direct Anthropic provider yet. We emit the
        # OpenRouter-style counterpart so users with aggregator providers can
        # immediately route to newly released Anthropic models.
        counterpart = _openrouter_counterpart("anthropic", model_id)
        entries.append(
            _entry(
                "openrouter",
                counterpart or f"anthropic/{model_id}",
                source_url=ANTHROPIC_RELEASE_NOTES_URL,
                last_verified_at=now,
                openrouter_counterpart=counterpart or f"anthropic/{model_id}",
            )
        )

    return entries


def discover_openrouter(api_key: str, timeout_ms: int) -> list[dict[str, Any]]:
    now = _now_utc()
    payload = _fetch_json(
        OPENROUTER_MODELS_URL,
        timeout_ms,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for model in payload.get("data", []):
        if not isinstance(model, dict):
            continue

        model_id = str(model.get("id", "")).strip()
        if not model_id or "/" not in model_id:
            continue

        architecture = model.get("architecture") if isinstance(model.get("architecture"), dict) else {}
        modalities = architecture.get("modality") if isinstance(architecture.get("modality"), list) else []
        supports_images = "image" in [str(mod).lower() for mod in modalities]

        context_window = int(model.get("context_length", 0) or 0)
        top_provider = model.get("top_provider") if isinstance(model.get("top_provider"), dict) else {}
        max_output_tokens = int(top_provider.get("max_completion_tokens", 0) or 0)

        entries.append(
            _entry(
                "openrouter",
                model_id,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                supports_images=supports_images,
                source_url=OPENROUTER_DOCS_URL,
                last_verified_at=now,
                openrouter_counterpart=model_id,
            )
        )

    return entries


def discover_vercel_gateway(api_key: str, timeout_ms: int) -> list[dict[str, Any]]:
    now = _now_utc()
    base_url = (
        get_env("VERCEL_AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1") or "https://ai-gateway.vercel.sh/v1"
    )
    endpoint = f"{base_url.rstrip('/')}/models"

    payload = _fetch_json(
        endpoint,
        timeout_ms,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for model in payload.get("data", []):
        if not isinstance(model, dict):
            continue

        model_id = str(model.get("id", "")).strip()
        if not model_id:
            continue

        context_window = int(model.get("context_window", 0) or 0)
        max_output_tokens = int(model.get("max_output_tokens", 0) or 0)

        entries.append(
            _entry(
                "vercel_gateway",
                model_id,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                source_url=VERCEL_GATEWAY_DOCS_URL,
                last_verified_at=now,
            )
        )

    return entries


def discover_all(timeout_ms: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Discover models from configured providers.

    Returns:
        (entries, errors)
    """

    discovered: list[dict[str, Any]] = []
    errors: list[str] = []

    providers = [
        ("OPENAI_API_KEY", discover_openai),
        ("GEMINI_API_KEY", discover_gemini),
        ("ANTHROPIC_API_KEY", discover_anthropic),
        ("XAI_API_KEY", discover_xai),
        ("OPENROUTER_API_KEY", discover_openrouter),
        ("VERCEL_AI_GATEWAY_API_KEY", discover_vercel_gateway),
    ]

    for env_key, discovery_fn in providers:
        api_key = get_env(env_key)
        if not api_key or _looks_like_placeholder_api_key(api_key):
            continue
        try:
            discovered.extend(discovery_fn(api_key, timeout_ms))
        except Exception as exc:  # pragma: no cover - network behaviour varies by environment
            errors.append(f"{env_key}: {exc}")

    return discovered, errors
