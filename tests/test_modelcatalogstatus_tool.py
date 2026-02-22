"""Tests for the modelcatalogstatus tool."""

from __future__ import annotations

import json

import pytest

from tools.modelcatalogstatus import ModelCatalogStatusTool


@pytest.mark.asyncio
async def test_modelcatalogstatus_tool_renders_refresh_section(monkeypatch) -> None:
    sample_status = {
        "enabled": True,
        "initialized": True,
        "source_mode": "discovery+cache+static",
        "fallback_mode": "none",
        "merged_models": 123,
        "changed_providers": ["openai", "openrouter"],
        "periodic_refresh_enabled": True,
        "refresh_task_running": True,
        "refresh_in_progress": False,
        "refresh_interval_seconds": 21600,
        "refresh_attempt_count": 4,
        "refresh_success_count": 4,
        "last_refresh_reason": "periodic",
        "last_refresh_started_utc": "2026-02-21T01:00:00+00:00",
        "last_refresh_completed_utc": "2026-02-21T01:00:02+00:00",
        "last_refresh_duration_ms": 2100,
        "last_refresh_success": True,
        "last_refresh_error": None,
        "discovery_enabled": True,
        "discovered_models": 12,
        "discovery_errors": [],
        "cache_path": "/tmp/cache.json",
        "cache_enabled": True,
        "cache_loaded": True,
        "cache_saved": True,
        "merge_warnings": ["example warning"],
        "generated_paths": {"openai": "/tmp/openai.json"},
    }

    monkeypatch.setattr("tools.modelcatalogstatus.get_model_catalog_status", lambda: sample_status)

    tool = ModelCatalogStatusTool()
    result = await tool.execute({})

    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["status"] == "success"
    content = payload["content"]
    assert "# Model Catalog Status" in content
    assert "## Refresh" in content
    assert "## Discovery and Cache" in content
    assert "Cache enabled: ✅" in content
    assert "## Merge Warnings (1)" in content
