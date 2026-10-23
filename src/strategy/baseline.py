"""Derive fixed baseline strategy params from SPO candidate grids."""

from __future__ import annotations


def _first_grid_value(strategy_id: str, key: str, values: object) -> object:
    if isinstance(values, list):
        if not values:
            raise ValueError(f"candidate grid {strategy_id}.{key} is empty")
        return values[0]
    return values


def baseline_candidate_cfg(candidate_cfg: dict) -> dict[str, dict[str, list]]:
    """
    Shrink SPO grids to a single point per strategy (first value of each list).

    Output keeps the candidate-grid shape expected by ``build_candidates`` /
    ``build_training_table``. Full grids remain for ``main.py``.
    """
    if not candidate_cfg:
        raise ValueError("candidate_cfg must be a non-empty mapping of strategy grids")

    out: dict[str, dict[str, list]] = {}
    for strategy_id, param_grid in candidate_cfg.items():
        if not isinstance(param_grid, dict) or not param_grid:
            raise ValueError(f"candidate grid for {strategy_id!r} must be a non-empty mapping")
        out[strategy_id] = {
            key: [_first_grid_value(strategy_id, key, values)]
            for key, values in param_grid.items()
        }
    return out


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
            baseline_params[key] = _first_grid_value(strategy_id, key, values)
        cfg[strategy_id] = baseline_params
    return cfg
