"""Small JSON helpers for run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: str | Path, payload: dict[str, Any], *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=indent, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data
