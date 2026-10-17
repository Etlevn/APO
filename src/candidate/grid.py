"""Build finite, auditable SPO parameter candidate tables."""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import pandas as pd

PARAM_COLUMNS = ("lookback", "ma_window", "entry_z")


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return [value]


def _product_rows(strategy_id: str, param_grid: dict) -> list[dict[str, object]]:
    if not param_grid:
        raise ValueError(f"candidate grid for {strategy_id!r} is empty")

    keys = list(param_grid.keys())
    value_lists = [_as_list(param_grid[k]) for k in keys]
    rows: list[dict[str, object]] = []
    for combo in itertools.product(*value_lists):
        row: dict[str, object] = {"strategy_id": strategy_id}
        for key in PARAM_COLUMNS:
            row[key] = None
        for key, value in zip(keys, combo, strict=True):
            row[key] = value
        rows.append(row)
    return rows


def build_candidates(candidate_cfg: dict) -> pd.DataFrame:
    """
    Expand configured grids into a candidate table.

    Columns:
    strategy_id, candidate_id, lookback, ma_window, entry_z
    Unused parameters for a strategy are left as missing values.
    """
    if not candidate_cfg:
        raise ValueError("candidate_cfg must be a non-empty mapping of strategy grids")

    rows: list[dict[str, object]] = []
    for strategy_id, param_grid in candidate_cfg.items():
        if not isinstance(param_grid, dict):
            raise TypeError(
                f"candidate grid for {strategy_id!r} must be a mapping, got {type(param_grid)}"
            )
        rows.extend(_product_rows(str(strategy_id), param_grid))

    frame = pd.DataFrame(rows)
    frame["candidate_id"] = [
        f"{strategy}_{idx:03d}"
        for strategy, idx in zip(
            frame["strategy_id"],
            frame.groupby("strategy_id").cumcount(),
            strict=True,
        )
    ]
    ordered = ["strategy_id", "candidate_id", *PARAM_COLUMNS]
    frame = frame.loc[:, ordered]
    for col in ("lookback", "ma_window"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    frame["entry_z"] = pd.to_numeric(frame["entry_z"], errors="coerce")
    return frame.reset_index(drop=True)


def iter_candidate_params(candidates: pd.DataFrame) -> Iterator[tuple[str, str, dict]]:
    """Yield (strategy_id, candidate_id, params_dict) with null parameters dropped."""
    for row in candidates.to_dict(orient="records"):
        strategy_id = str(row["strategy_id"])
        candidate_id = str(row["candidate_id"])
        params: dict[str, object] = {}
        for key in PARAM_COLUMNS:
            value = row[key]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            if key in {"lookback", "ma_window"}:
                params[key] = int(value)
            else:
                params[key] = float(value)
        yield strategy_id, candidate_id, params
