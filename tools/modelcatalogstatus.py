"""Model catalog diagnostics tool."""

from __future__ import annotations

import os
from typing import Any

from mcp.types import TextContent

from tools.models import ToolModelCategory, ToolOutput
from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool
from utils.model_catalog import get_model_catalog_status


def _bool_icon(value: bool) -> str:
    return "✅" if value else "❌"


def _format_path(path_value: str | None, *, show_paths: bool) -> str:
    if not path_value:
        return "none"
    if show_paths:
        return path_value
    basename = os.path.basename(path_value)
    return basename or "<redacted>"


class ModelCatalogStatusTool(BaseTool):
    """Read-only diagnostics for the model catalog control-plane."""

    def get_name(self) -> str:
        return "modelcatalogstatus"

    def get_description(self) -> str:
        return "Show model-catalog status including refresh history, fallback mode, and merge warnings."

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "show_paths": {
                    "type": "boolean",
                    "description": "When true, include full local paths in output.",
                    "default": False,
                }
            },
            "required": [],
            "additionalProperties": False,
        }

    def get_annotations(self) -> dict[str, Any] | None:
        return {"readOnlyHint": True}

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return ToolRequest

    def requires_model(self) -> bool:
        return False

    async def prepare_prompt(self, request: ToolRequest) -> str:
        return ""

    def format_response(self, response: str, request: ToolRequest, model_info: dict | None = None) -> str:
        return response

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        status = get_model_catalog_status()
        show_paths = bool(arguments.get("show_paths", False))

        output_lines = ["# Model Catalog Status", ""]

        output_lines.append("## Runtime")
        output_lines.append(f"- Enabled: {_bool_icon(bool(status.get('enabled', False)))}")
        output_lines.append(f"- Initialized: {_bool_icon(bool(status.get('initialized', False)))}")
        output_lines.append(f"- Source mode: `{status.get('source_mode', 'unknown')}`")
        output_lines.append(f"- Fallback mode: `{status.get('fallback_mode', 'none')}`")
        output_lines.append(f"- Merged models: {int(status.get('merged_models', 0) or 0)}")
        changed_providers = [str(item) for item in (status.get("changed_providers") or [])]
        output_lines.append(f"- Changed providers (last run): {', '.join(changed_providers) or 'none'}")
        output_lines.append("")

        output_lines.append("## Refresh")
        output_lines.append(f"- Periodic enabled: {_bool_icon(bool(status.get('periodic_refresh_enabled', False)))}")
        output_lines.append(f"- Task running: {_bool_icon(bool(status.get('refresh_task_running', False)))}")
        output_lines.append(f"- In progress: {_bool_icon(bool(status.get('refresh_in_progress', False)))}")
        output_lines.append(f"- Interval (seconds): {int(status.get('refresh_interval_seconds', 0) or 0)}")
        output_lines.append(f"- Attempts: {int(status.get('refresh_attempt_count', 0) or 0)}")
        output_lines.append(f"- Successes: {int(status.get('refresh_success_count', 0) or 0)}")
        output_lines.append(f"- Last reason: `{status.get('last_refresh_reason') or 'unknown'}`")
        output_lines.append(f"- Last started: `{status.get('last_refresh_started_utc') or 'never'}`")
        output_lines.append(f"- Last completed: `{status.get('last_refresh_completed_utc') or 'never'}`")
        output_lines.append(f"- Last duration (ms): {int(status.get('last_refresh_duration_ms', 0) or 0)}")
        output_lines.append(f"- Last refresh success: `{status.get('last_refresh_success')}`")
        output_lines.append(f"- Last refresh error: `{status.get('last_refresh_error') or 'none'}`")
        output_lines.append("")

        output_lines.append("## Discovery and Cache")
        output_lines.append(f"- Discovery enabled: {_bool_icon(bool(status.get('discovery_enabled', False)))}")
        output_lines.append(f"- Discovered models (last run): {int(status.get('discovered_models', 0) or 0)}")
        discovery_errors = [str(item) for item in (status.get("discovery_errors") or [])]
        output_lines.append(f"- Discovery errors: {', '.join(discovery_errors) or 'none'}")
        cache_path = _format_path(status.get("cache_path"), show_paths=show_paths)
        output_lines.append(f"- Cache path: `{cache_path}`")
        output_lines.append(f"- Cache enabled: {_bool_icon(bool(status.get('cache_enabled', False)))}")
        output_lines.append(f"- Cache loaded: {_bool_icon(bool(status.get('cache_loaded', False)))}")
        output_lines.append(f"- Cache saved: {_bool_icon(bool(status.get('cache_saved', False)))}")
        output_lines.append("")

        merge_warnings = status.get("merge_warnings") or []
        output_lines.append(f"## Merge Warnings ({len(merge_warnings)})")
        if merge_warnings:
            for warning in merge_warnings[:20]:
                output_lines.append(f"- {warning}")
            if len(merge_warnings) > 20:
                output_lines.append(f"- ... and {len(merge_warnings) - 20} more")
        else:
            output_lines.append("- none")
        output_lines.append("")

        generated_paths = status.get("generated_paths") or {}
        output_lines.append("## Generated Manifests")
        if generated_paths:
            for provider_name in sorted(generated_paths.keys()):
                path_value = _format_path(str(generated_paths[provider_name]), show_paths=show_paths)
                output_lines.append(f"- `{provider_name}`: `{path_value}`")
        else:
            output_lines.append("- none")

        content = "\n".join(output_lines)
        tool_output = ToolOutput(
            status="success",
            content=content,
            content_type="text",
            metadata={
                "tool_name": self.name,
                "catalog_enabled": bool(status.get("enabled", False)),
                "source_mode": status.get("source_mode", "unknown"),
                "fallback_mode": status.get("fallback_mode", "none"),
                "discovery_enabled": bool(status.get("discovery_enabled", False)),
                "cache_enabled": bool(status.get("cache_enabled", False)),
                "last_refresh_success": status.get("last_refresh_success"),
            },
        )
        return [TextContent(type="text", text=tool_output.model_dump_json())]

    def get_model_category(self) -> ToolModelCategory:
        return ToolModelCategory.FAST_RESPONSE
