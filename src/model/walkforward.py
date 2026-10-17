"""Walk-forward fit of F(E, x) and candidate selection at each decision date."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from candidate import PARAM_COLUMNS
from dataset.calendar import parse_span
from feature import FEATURE_COLUMNS
from util import ProgressBar

from .registry import create_model, resolve_model_ids

INPUT_COLUMNS = (*FEATURE_COLUMNS, *PARAM_COLUMNS)


def _design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.loc[:, list(INPUT_COLUMNS)].copy()
    for col in PARAM_COLUMNS:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    return matrix.fillna(0.0)


def _train_mask(
    table: pd.DataFrame,
    *,
    strategy_id: str,
    position_mode: str,
    decision: pd.Timestamp,
    walkforward_cfg: dict,
) -> pd.Series:
    decision = pd.Timestamp(decision).normalize()
    realized = (
        (table["strategy_id"] == strategy_id)
        & (table["position_mode"].astype(str) == str(position_mode))
        & (pd.to_datetime(table["decision_date"]) < decision)
        & (pd.to_datetime(table["label_end"]) <= decision)
    )
    mode = walkforward_cfg.get("mode", "rolling")
    if mode == "expanding":
        return realized
    if mode != "rolling":
        raise ValueError(f"Unsupported walkforward mode: {mode!r}")
    train_start = decision - parse_span(walkforward_cfg["train_window"])
    return realized & (pd.to_datetime(table["decision_date"]) >= train_start)


def _select_for_decision(
    table: pd.DataFrame,
    *,
    model_id: str,
    model_cfg: dict,
    strategy_id: str,
    position_mode: str,
    decision: pd.Timestamp,
    walkforward_cfg: dict,
) -> dict[str, object] | None:
    train = table.loc[
        _train_mask(
            table,
            strategy_id=strategy_id,
            position_mode=position_mode,
            decision=decision,
            walkforward_cfg=walkforward_cfg,
        )
    ]
    min_rows = int(model_cfg.get("min_train_rows", 3))
    if len(train) < min_rows:
        return None

    # Current-date scoring rows: reuse feature snapshot from any row at this decision.
    today = table.loc[
        (table["strategy_id"] == strategy_id)
        & (table["position_mode"].astype(str) == str(position_mode))
        & (pd.to_datetime(table["decision_date"]) == pd.Timestamp(decision).normalize())
    ]
    if today.empty:
        return None

    x_train = _design_matrix(train)
    y_train = train["label"].astype(float).to_numpy()
    x_today = _design_matrix(today)

    if model_cfg.get("standardize", True):
        scaler = StandardScaler()
        x_train_arr = scaler.fit_transform(x_train.to_numpy(dtype=float))
        x_today_arr = scaler.transform(x_today.to_numpy(dtype=float))
    else:
        x_train_arr = x_train.to_numpy(dtype=float)
        x_today_arr = x_today.to_numpy(dtype=float)

    model = create_model(model_id, model_cfg)
    model.fit(x_train_arr, y_train)
    preds = model.predict(x_today_arr)

    best_idx = int(np.argmax(preds))
    best = today.iloc[best_idx]
    return {
        "decision_date": pd.Timestamp(decision).normalize(),
        "model_id": model_id,
        "position_mode": str(position_mode),
        "strategy_id": strategy_id,
        "candidate_id": best["candidate_id"],
        "lookback": best["lookback"],
        "ma_window": best["ma_window"],
        "entry_z": best["entry_z"],
        "predicted_label": float(preds[best_idx]),
        "realized_label": float(best["label"]),
        "train_rows": int(len(train)),
    }


def run_walkforward_selection(
    table: pd.DataFrame,
    *,
    walkforward_cfg: dict,
    model_cfg: dict,
    model_ids: str | list[str] | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    For each model and model-decision date with enough history, select the best
    candidate per strategy by predicted forward label.

    ``model_ids`` may be one id (``\"hgb\"``) or several (``[\"hgb\", \"ridge\"]``).
    If omitted, uses ``model_cfg[\"run\"]`` (see ``resolve_model_ids``).

    Warmup sample dates (before first_decision_date) are used only for training.
    """
    if table.empty:
        return pd.DataFrame()

    if "position_mode" not in table.columns:
        table = table.copy()
        table["position_mode"] = "long_short"

    ids = resolve_model_ids(model_cfg, model_ids)
    first_decision = pd.Timestamp(walkforward_cfg["first_decision_date"]).normalize()
    if "is_model_decision" in table.columns:
        decision_frame = table.loc[table["is_model_decision"]]
    else:
        decision_frame = table.loc[pd.to_datetime(table["decision_date"]) >= first_decision]
    decisions = sorted(pd.to_datetime(decision_frame["decision_date"]).unique())
    strategies = sorted(table["strategy_id"].astype(str).unique())
    position_modes = sorted(table["position_mode"].astype(str).unique())

    jobs = [
        (model_id, position_mode, decision, strategy_id)
        for model_id in ids
        for position_mode in position_modes
        for decision in decisions
        for strategy_id in strategies
    ]
    rows: list[dict[str, object]] = []
    pbar = ProgressBar(len(jobs), desc="model") if show_progress else None
    try:
        for model_id, position_mode, decision, strategy_id in jobs:
            if pbar is not None:
                pbar.set_description(f"model/{model_id}/{position_mode}")
                pbar.set_postfix(
                    date=pd.Timestamp(decision).date().isoformat(),
                    strategy=str(strategy_id),
                )
            selected = _select_for_decision(
                table,
                model_id=model_id,
                model_cfg=model_cfg,
                strategy_id=strategy_id,
                position_mode=position_mode,
                decision=decision,
                walkforward_cfg=walkforward_cfg,
            )
            if selected is not None:
                rows.append(selected)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["model_id", "position_mode", "decision_date", "strategy_id"]
    ).reset_index(drop=True)
