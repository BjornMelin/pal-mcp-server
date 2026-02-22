"""Disk cache helpers for merged model catalog snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def load_cache(cache_path: str) -> dict[str, Any] | None:
    """Load a previously merged catalog snapshot from disk."""

    path = Path(cache_path)
    if not path.exists() or not path.is_file():
        return None

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    return payload if isinstance(payload, dict) else None


def save_cache(cache_path: str, payload: dict[str, Any]) -> bool:
    """Persist a merged catalog snapshot atomically."""

    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)
        temp_path.replace(path)
        return True
    except OSError:
        return False
