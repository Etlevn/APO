"""Load SPO strategy recipes from config/strategy.yaml."""

from __future__ import annotations

from pathlib import Path

from util.yaml_io import load_yaml


def load_strategy_recipes(path: Path | str) -> dict[str, dict]:
    """Return ``{strategy_id: {"params": grid}}``."""
    raw = load_yaml(path)
    if not raw:
        raise ValueError(f"strategy file is empty: {path}")

    recipes: dict[str, dict] = {}
    for strategy_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise TypeError(f"strategy {strategy_id!r} must be a mapping")
        if "features" in spec:
            raise ValueError(
                f"strategy {strategy_id!r} no longer lists features. "
                "Put market features in config/feature.yaml."
            )
        params = spec.get("params")
        if not isinstance(params, dict) or not params:
            raise ValueError(f"strategy {strategy_id!r} params must be a non-empty mapping")
        grids: dict[str, list] = {}
        for key, values in params.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"strategy {strategy_id!r} param {key!r} must be a non-empty list")
            grids[str(key)] = list(values)
        recipes[str(strategy_id)] = {"params": grids}
    return recipes


def param_grids(recipes: dict[str, dict]) -> dict[str, dict[str, list]]:
    """Parameter grids only, for candidate expansion and the fixed baseline."""
    return {strategy_id: dict(spec["params"]) for strategy_id, spec in recipes.items()}
