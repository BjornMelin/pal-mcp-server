"""Unit tests for model-catalog shadow cache helpers."""

from __future__ import annotations

import json

from utils.model_catalog import cache_store


def test_cache_round_trip(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    payload = {
        "saved_at": "2026-02-21T00:00:00+00:00",
        "providers": {
            "openai": [
                {
                    "model_name": "gpt-5",
                    "friendly_name": "OpenAI (gpt-5)",
                    "intelligence_score": 19,
                    "description": "test",
                }
            ]
        },
    }

    assert cache_store.save_cache(str(cache_path), payload) is True

    loaded = cache_store.load_cache(str(cache_path))
    assert loaded == payload


def test_load_cache_handles_corruption(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{invalid", encoding="utf-8")

    assert cache_store.load_cache(str(cache_path)) is None


def test_save_cache_failure_keeps_existing_file(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    original_payload = {"providers": {"openai": [{"model_name": "gpt-existing"}]}}
    cache_path.write_text(json.dumps(original_payload), encoding="utf-8")
    original_text = cache_path.read_text(encoding="utf-8")

    def _raise_replace(self, target):  # noqa: ARG001
        raise OSError("disk full")

    monkeypatch.setattr(cache_store.Path, "replace", _raise_replace)

    updated_payload = {"providers": {"openai": [{"model_name": "gpt-new"}]}}
    assert cache_store.save_cache(str(cache_path), updated_payload) is False

    assert cache_path.read_text(encoding="utf-8") == original_text
