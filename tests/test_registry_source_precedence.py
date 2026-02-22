"""Tests for model registry source precedence and fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from providers.registries.base import CapabilityModelRegistry
from providers.shared import ProviderType


class _FakeResourceFile:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read_text(self, encoding: str = "utf-8") -> str:  # noqa: ARG002
        return json.dumps(self._payload)


class _FakeResourceRoot:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._file = _FakeResourceFile(payload)

    def joinpath(self, _filename: str) -> _FakeResourceFile:
        return self._file


class _TestRegistry(CapabilityModelRegistry):
    def __init__(self) -> None:
        super().__init__(
            env_var_name="PAL_TEST_MODELS_CONFIG_PATH",
            default_filename="pal_test_models.json",
            provider=ProviderType.OPENROUTER,
            friendly_prefix="Test ({model})",
        )


def test_packaged_registry_precedes_workspace_fallback(tmp_path, monkeypatch) -> None:
    packaged_payload = {"models": [{"model_name": "packaged/model", "aliases": ["packaged"]}]}
    workspace_payload = {"models": [{"model_name": "workspace/model", "aliases": ["workspace"]}]}

    workspace_path = tmp_path / "conf" / "pal_test_models.json"
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_text(json.dumps(workspace_payload), encoding="utf-8")

    monkeypatch.delenv("PAL_TEST_MODELS_CONFIG_PATH", raising=False)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        "providers.registries.base.importlib.resources.files",
        lambda _package: _FakeResourceRoot(packaged_payload),
    )

    registry = _TestRegistry()
    metadata = registry.get_source_metadata()

    assert "packaged/model" in registry.list_models()
    assert "workspace/model" not in registry.list_models()
    assert metadata["source_label"] == "packaged_static"
    assert "path" not in metadata


def test_workspace_fallback_still_used_when_packaged_resource_unavailable(tmp_path, monkeypatch) -> None:
    workspace_payload = {"models": [{"model_name": "workspace/model", "aliases": ["workspace"]}]}

    workspace_path = tmp_path / "conf" / "pal_test_models.json"
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_text(json.dumps(workspace_payload), encoding="utf-8")

    def _raise_missing_resource(_package: str):  # noqa: ARG001
        raise RuntimeError("resource not available")

    monkeypatch.delenv("PAL_TEST_MODELS_CONFIG_PATH", raising=False)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr("providers.registries.base.importlib.resources.files", _raise_missing_resource)

    registry = _TestRegistry()
    metadata = registry.get_source_metadata()

    assert "workspace/model" in registry.list_models()
    assert metadata["source_label"] == "workspace_static"
    assert metadata.get("path") == str(workspace_path)
