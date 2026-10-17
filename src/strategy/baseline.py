"""Derive fixed baseline strategy params from SPO candidate grids."""

from __future__ import annotations


def baseline_strategy_cfg(
    candidate_cfg: dict,
    *,
    price_col: str = "adj_close",
) -> dict:
    """
    Build a strategy_cfg for run_strategies from candidate grids.

    Baseline uses the first value of each parameter list.
    """
    if not candidate_cfg:
        raise ValueError("candidate_cfg must be a non-empty mapping of strategy grids")

    cfg: dict[str, object] = {"price_col": price_col}
    for strategy_id, param_grid in candidate_cfg.items():
        if not isinstance(param_grid, dict) or not param_grid:
            raise ValueError(f"candidate grid for {strategy_id!r} must be a non-empty mapping")
        baseline_params: dict[str, object] = {}
        for key, values in param_grid.items():
            if isinstance(values, list):
                if not values:
                    raise ValueError(f"candidate grid {strategy_id}.{key} is empty")
                baseline_params[key] = values[0]
            else:
                baseline_params[key] = values
        cfg[strategy_id] = baseline_params
    return cfg
