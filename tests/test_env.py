"""Tests for centralized environment value precedence."""

from __future__ import annotations

import utils.env as env_utils


def test_get_env_prefers_runtime_override_over_process_env(monkeypatch) -> None:
    key = "PAL_TEST_RUNTIME_OVERRIDE_KEY"
    env_utils.clear_all_runtime_env()
    env_utils.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    monkeypatch.setenv(key, "process-value")
    env_utils.set_runtime_env(key, "runtime-value")

    try:
        assert env_utils.get_env(key) == "runtime-value"
    finally:
        env_utils.clear_all_runtime_env()
        env_utils.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})


def test_get_env_prefers_runtime_override_over_empty_process_env(monkeypatch) -> None:
    key = "PAL_TEST_RUNTIME_OVERRIDE_EMPTY_KEY"
    env_utils.clear_all_runtime_env()
    env_utils.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
    monkeypatch.setenv(key, "")
    env_utils.set_runtime_env(key, "runtime-value")

    try:
        assert env_utils.get_env(key) == "runtime-value"
    finally:
        env_utils.clear_all_runtime_env()
        env_utils.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})


def test_get_env_prefers_runtime_override_when_force_override_enabled(monkeypatch) -> None:
    key = "PAL_TEST_RUNTIME_OVERRIDE_FORCE_KEY"
    env_utils.clear_all_runtime_env()
    env_utils.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "true", key: "dotenv-value"})
    monkeypatch.setenv(key, "process-value")
    env_utils.set_runtime_env(key, "runtime-value")

    try:
        assert env_utils.get_env(key) == "runtime-value"
    finally:
        env_utils.clear_all_runtime_env()
        env_utils.reload_env({"PAL_MCP_FORCE_ENV_OVERRIDE": "false"})
