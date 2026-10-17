"""Slice walk-forward selection rows for evaluate-only views."""

from __future__ import annotations

import pandas as pd


def _as_id_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def filter_selections(
    selections: pd.DataFrame,
    *,
    model_ids: str | list[str] | None = None,
    strategy_ids: str | list[str] | None = None,
    position_modes: str | list[str] | None = None,
    date_start: str | pd.Timestamp | None = None,
    date_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Filter selection rows. ``None`` means no constraint on that axis.

    Raises if a requested model/strategy/position id is absent from the file.
    """
    if selections.empty:
        return selections.copy()

    frame = selections.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    if "position_mode" not in frame.columns:
        frame["position_mode"] = "long_short"

    models = _as_id_list(model_ids)
    strategies = _as_id_list(strategy_ids)
    modes = _as_id_list(position_modes)

    if models is not None:
        available = set(frame["model_id"].astype(str).unique())
        missing = [m for m in models if m not in available]
        if missing:
            raise ValueError(
                f"MODEL_IDS not found in selections: {missing}; available={sorted(available)}"
            )
        frame = frame.loc[frame["model_id"].astype(str).isin(models)]

    if strategies is not None:
        # "blend" is derived in evaluate, not stored in selections.csv.
        raw_strategies = [s for s in strategies if s != "blend"]
        if raw_strategies:
            available = set(frame["strategy_id"].astype(str).unique())
            missing = [s for s in raw_strategies if s not in available]
            if missing:
                raise ValueError(
                    f"STRATEGY_IDS not found in selections: {missing}; "
                    f"available={sorted(available)}"
                )
            frame = frame.loc[frame["strategy_id"].astype(str).isin(raw_strategies)]

    if modes is not None:
        available = set(frame["position_mode"].astype(str).unique())
        missing = [m for m in modes if m not in available]
        if missing:
            raise ValueError(
                f"POSITION_IDS not found in selections: {missing}; available={sorted(available)}"
            )
        frame = frame.loc[frame["position_mode"].astype(str).isin(modes)]

    if date_start is not None:
        start = pd.Timestamp(date_start).normalize()
        frame = frame.loc[frame["decision_date"] >= start]
    if date_end is not None:
        end = pd.Timestamp(date_end).normalize()
        frame = frame.loc[frame["decision_date"] <= end]

    return frame.sort_values(
        ["model_id", "position_mode", "decision_date", "strategy_id"]
    ).reset_index(drop=True)
