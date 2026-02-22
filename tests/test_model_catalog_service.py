"""Tests for model catalog service runtime behavior."""

from __future__ import annotations

import json

import pytest

import utils.env as env_utils
import utils.model_catalog.service as catalog_service
from utils.model_catalog import (
    discovery_client,
    get_model_catalog_status,
    initialize_model_catalog,
    refresh_model_catalog_once,
    start_model_catalog_refresh_task,
    stop_model_catalog_refresh_task,
)


def test_discover_anthropic_maps_to_openrouter_counterparts(monkeypatch) -> None:
    def _mock_fetch_json(url: str, timeout_ms: int, headers: dict[str, str] | None = None):  # noqa: ARG001
        return {
            "data": [
                {"id": "claude-sonnet-4-5"},
                {"id": "claude-opus-4-1"},
                {"id": "not-claude"},
            ]
        }

    monkeypatch.setattr(discovery_client, "_fetch_json", _mock_fetch_json)

    entries = discovery_client.discover_anthropic(api_key="sk-ant-test", timeout_ms=1000)

    sonnet = next((item for item in entries if item.get("model_name") == "anthropic/claude-sonnet-4-5"), None)
    opus = next((item for item in entries if item.get("model_name") == "anthropic/claude-opus-4-1"), None)

    assert sonnet is not None and sonnet.get("provider") == "openrouter"
    assert sonnet.get("openrouter_counterpart") == "anthropic/claude-sonnet-4-5"
    assert isinstance(sonnet.get("release_metadata"), dict)

    assert opus is not None and opus.get("provider") == "openrouter"
    assert opus.get("openrouter_counterpart") == "anthropic/claude-opus-4-1"
    assert all(item["model_name"].startswith("anthropic/claude") for item in entries)


def test_refresh_model_catalog_disabled_mode(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "false")

    status = refresh_model_catalog_once("test-disabled")

    assert status["enabled"] is False
    assert status["source_mode"] == "disabled"
    assert status["fallback_mode"] == "disabled"
    assert status["last_refresh_success"] is True


def test_initialize_model_catalog_skips_discovery_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_CACHE", "false")
    monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))

    called = {"count": 0}

    def _discover_all(_timeout: int):
        called["count"] += 1
        return ([], [])

    monkeypatch.setattr(catalog_service, "discover_all", _discover_all)

    status = initialize_model_catalog()

    assert called["count"] == 0
    assert status["discovery_enabled"] is False
    assert status["source_mode"] == "static"
    assert status["last_refresh_success"] is True


@pytest.mark.asyncio
async def test_periodic_refresh_task_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "false")
    monkeypatch.setenv("MODEL_CATALOG_REFRESH_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setenv("MODEL_CATALOG_CACHE_PATH", str(tmp_path / "cache.json"))

    await stop_model_catalog_refresh_task()
    initialize_model_catalog()
    await start_model_catalog_refresh_task()

    running_status = get_model_catalog_status()
    assert running_status["refresh_task_running"] is True
    assert running_status["periodic_refresh_enabled"] is True
    assert running_status["refresh_interval_seconds"] == 3600

    await stop_model_catalog_refresh_task()
    stopped_status = get_model_catalog_status()
    assert stopped_status["refresh_task_running"] is False


def test_startup_without_network_uses_cache_and_static(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.json"
    cache_payload = {
        "saved_at": "2026-02-21T00:00:00+00:00",
        "providers": {
            "openai": [
                {
                    "model_name": "gpt-cache-only",
                    "friendly_name": "OpenAI (gpt-cache-only)",
                    "intelligence_score": 7,
                    "description": "cached",
                }
            ]
        },
    }
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")

    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_CACHE", "true")
    monkeypatch.setenv("MODEL_CATALOG_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))

    monkeypatch.setattr(catalog_service, "discover_all", lambda _timeout: ([], ["OPENAI_API_KEY: timeout"]))

    status = refresh_model_catalog_once("test-no-network")

    assert status["last_refresh_success"] is True
    assert status["cache_loaded"] is True
    assert status["cache_enabled"] is True
    assert status["fallback_mode"] == "cache+static"
    assert status["discovery_errors"] == ["OPENAI_API_KEY: timeout"]
    assert status["merged_models"] > 0


def test_discovery_api_outage_does_not_break_refresh(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_CACHE", "false")
    monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))

    monkeypatch.setattr(catalog_service, "discover_all", lambda _timeout: ([], ["OPENAI_API_KEY: timeout"]))

    status = refresh_model_catalog_once("test-discovery-outage")

    assert status["last_refresh_success"] is True
    assert status["discovery_errors"] == ["OPENAI_API_KEY: timeout"]
    assert status["merged_models"] > 0
    assert status["generated_paths"]


def test_refresh_adds_openrouter_counterpart_for_new_discovered_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_CACHE", "false")
    monkeypatch.setenv("MODEL_CATALOG_QUARANTINE_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))

    monkeypatch.setattr(
        catalog_service,
        "discover_all",
        lambda _timeout: (
            [
                {
                    "provider": "openrouter",
                    "model_name": "anthropic/claude-new",
                    "openrouter_counterpart": "anthropic/claude-new",
                    "context_window": 200000,
                    "release_metadata": {"source_url": "https://docs.anthropic.com"},
                }
            ],
            [],
        ),
    )

    status = refresh_model_catalog_once("test-openrouter-counterpart")
    assert status["last_refresh_success"] is True

    openrouter_payload = json.loads((tmp_path / "generated" / "openrouter_models.json").read_text(encoding="utf-8"))
    discovered = next(model for model in openrouter_payload["models"] if model["model_name"] == "anthropic/claude-new")
    assert discovered["quarantine"] is True
    assert discovered["openrouter_counterpart"] == "anthropic/claude-new"


def test_refresh_model_catalog_does_not_persist_cache_when_disabled(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.json"
    sentinel_payload = {"sentinel": "unchanged"}
    cache_path.write_text(json.dumps(sentinel_payload), encoding="utf-8")

    monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "false")
    monkeypatch.setenv("MODEL_CATALOG_ENABLE_CACHE", "false")
    monkeypatch.setenv("MODEL_CATALOG_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))

    status = refresh_model_catalog_once("test-cache-disabled-write")

    assert status["last_refresh_success"] is True
    assert status["cache_enabled"] is False
    assert status["cache_saved"] is False
    assert json.loads(cache_path.read_text(encoding="utf-8")) == sentinel_payload


def test_write_json_if_changed_ignores_generated_metadata(tmp_path) -> None:
    path = tmp_path / "openai_models.json"
    base_payload = {
        "_GENERATED": {"source_mode": "static", "entry_count": 1, "generated_at": "2026-02-21T00:00:00+00:00"},
        "models": [{"model_name": "gpt-5", "friendly_name": "OpenAI GPT-5"}],
    }
    metadata_only_change = {
        "_GENERATED": {
            "source_mode": "discovery+static",
            "entry_count": 1,
            "generated_at": "2026-02-21T01:00:00+00:00",
        },
        "models": [{"model_name": "gpt-5", "friendly_name": "OpenAI GPT-5"}],
    }

    first_ok, first_changed = catalog_service._write_json_if_changed(path, base_payload)
    second_ok, second_changed = catalog_service._write_json_if_changed(path, metadata_only_change)

    assert first_ok is True and first_changed is True
    assert second_ok is True and second_changed is False


def test_refresh_model_catalog_second_run_without_model_changes_has_no_changed_providers(tmp_path, monkeypatch) -> None:
    env_utils.clear_all_runtime_env()
    try:
        monkeypatch.setenv("MODEL_CATALOG_ENABLED", "true")
        monkeypatch.setenv("MODEL_CATALOG_ENABLE_DISCOVERY", "false")
        monkeypatch.setenv("MODEL_CATALOG_ENABLE_CACHE", "false")
        monkeypatch.setenv("MODEL_CATALOG_GENERATED_DIR", str(tmp_path / "generated"))

        for env_var_name, _ in catalog_service._PROVIDER_CONFIG.values():
            monkeypatch.delenv(env_var_name, raising=False)

        first_status = refresh_model_catalog_once("test-first-refresh")
        second_status = refresh_model_catalog_once("test-second-refresh")

        assert first_status["last_refresh_success"] is True
        assert first_status["changed_providers"]
        assert second_status["last_refresh_success"] is True
        assert second_status["changed_providers"] == []
    finally:
        env_utils.clear_all_runtime_env()
