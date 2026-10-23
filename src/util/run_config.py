"""Snapshot of settings used by ``main.py`` for reproducible ``eval.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .json_io import load_json, save_json

SCHEMA_VERSION = 1


def build_main_run_config(
    *,
    tickers: list[str],
    start: str,
    end: str | None,
    price_col: str,
    weights: dict[str, float] | None,
    cache_format: str,
    walkforward: dict[str, str],
    strategy_run: str | list[str],
    position_run: str | list[str],
    position_overlay_scale: float,
    candidate: dict[str, dict[str, list]],
    model_run: str | list[str],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Serialize the knobs that define a main → eval run (JSON-friendly)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tickers": list(tickers),
        "start": start,
        "end": end,
        "price_col": price_col,
        "weights": weights,
        "cache_format": cache_format,
        "walkforward": dict(walkforward),
        "strategy_run": strategy_run,
        "position_run": position_run,
        "position_overlay_scale": float(position_overlay_scale),
        "candidate": candidate,
        "model_run": model_run,
        "dataset": dict(dataset_cfg),
        "feature": dict(feature_cfg),
        "model": dict(model_cfg),
    }


def save_main_run_config(path: str | Path, payload: dict[str, Any]) -> Path:
    return save_json(path, payload)


def load_main_run_config(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported main_run_config schema_version={version!r}; "
            f"expected {SCHEMA_VERSION}. Re-run main.py."
        )
    required = (
        "tickers",
        "start",
        "price_col",
        "walkforward",
        "candidate",
        "position_overlay_scale",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"main_run_config missing keys: {missing}")
    return data
