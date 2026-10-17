"""Load project YAML config files."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: Path | str) -> dict:
    """Read a YAML file into a dict; missing/empty files yield {}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}
