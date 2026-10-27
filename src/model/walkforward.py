"""Walk-forward fit of conditional models and candidate selection.

SPO: F(E, strategy params) → pick parameters per strategy × position_mode.
PWO: F(E, weights) → pick long-only ETF weight vectors per decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from candidate import PARAM_COLUMNS
from dataset.calendar import parse_span
from feature import FEATURE_COLUMNS
from util import ProgressBar

from .registry import create_model, resolve_model_ids

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fit_predict_best(
    train: pd.DataFrame,
    today: pd.DataFrame,
    *,
    model_id: str,
    model_cfg: dict,
    design_fn,
) -> tuple[pd.Series, float, np.ndarray] | None:
    """Fit on ``train``, score ``today``; return (best_row, pred, all_preds)."""
    x_train = design_fn(train)
    y_train = train["label"].astype(float).to_numpy()
    x_today = design_fn(today)

    if model_cfg.get("standardize", True):
        scaler = StandardScaler()
        x_train_arr = scaler.fit_transform(x_train.to_numpy(dtype=float))
        x_today_arr = scaler.transform(x_today.to_numpy(dtype=float))
    else:
        x_train_arr = x_train.to_numpy(dtype=float)
        x_today_arr = x_today.to_numpy(dtype=float)

    model = create_model(model_id, model_cfg)
    model.fit(x_train_arr, y_train)
    preds = np.asarray(model.predict(x_today_arr), dtype=float)
    best_idx = int(np.argmax(preds))
    return today.iloc[best_idx], float(preds[best_idx]), preds


def _weight_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if str(c).startswith("w_")]


# ---------------------------------------------------------------------------
# SPO
# ---------------------------------------------------------------------------

SPO_INPUT_COLUMNS = (*FEATURE_COLUMNS, *PARAM_COLUMNS)
# Backward-compatible alias
INPUT_COLUMNS = SPO_INPUT_COLUMNS


def _spo_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame.loc[:, list(SPO_INPUT_COLUMNS)].copy()
    for col in PARAM_COLUMNS:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    return matrix.fillna(0.0)


def _spo_train_mask(
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


# Backward-compatible aliases used by hist_mean SPO path
_train_mask = _spo_train_mask


def _select_spo_for_decision(
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
        _spo_train_mask(
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

    today = table.loc[
        (table["strategy_id"] == strategy_id)
        & (table["position_mode"].astype(str) == str(position_mode))
        & (pd.to_datetime(table["decision_date"]) == pd.Timestamp(decision).normalize())
    ]
    if today.empty:
        return None

    fitted = _fit_predict_best(
        train,
        today,
        model_id=model_id,
        model_cfg=model_cfg,
        design_fn=_spo_design_matrix,
    )
    if fitted is None:
        return None
    best, predicted, _ = fitted
    return {
        "decision_date": pd.Timestamp(decision).normalize(),
        "model_id": model_id,
        "position_mode": str(position_mode),
        "strategy_id": strategy_id,
        "candidate_id": best["candidate_id"],
        "lookback": best["lookback"],
        "ma_window": best["ma_window"],
        "entry_z": best["entry_z"],
        "predicted_label": predicted,
        "realized_label": float(best["label"]),
        "train_rows": int(len(train)),
    }


def _spo_selection_calendar(
    table: pd.DataFrame,
    walkforward_cfg: dict,
) -> tuple[list[pd.Timestamp], list[str], list[str]]:
    """Live decision dates × strategies × position modes present in ``table``."""
    first_decision = pd.Timestamp(walkforward_cfg["first_decision_date"]).normalize()
    if "is_model_decision" in table.columns:
        decision_frame = table.loc[table["is_model_decision"]]
    else:
        decision_frame = table.loc[pd.to_datetime(table["decision_date"]) >= first_decision]
    decisions = sorted(pd.to_datetime(decision_frame["decision_date"]).unique())
    strategies = sorted(table["strategy_id"].astype(str).unique())
    position_modes = sorted(table["position_mode"].astype(str).unique())
    return decisions, strategies, position_modes


# Backward-compatible alias
_selection_calendar = _spo_selection_calendar


def run_spo_walkforward_selection(
    table: pd.DataFrame,
    *,
    walkforward_cfg: dict,
    model_cfg: dict,
    model_ids: str | list[str] | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    SPO walk-forward: for each model and live decision, select the best parameter
    candidate per strategy × position_mode by predicted forward label.

    ``model_ids`` may be one id (``\"hgb\"``) or several (``[\"hgb\", \"ridge\"]``).
    If omitted, uses ``model_cfg[\"run\"]`` (see ``resolve_model_ids``).
    """
    if table.empty:
        return pd.DataFrame()

    if "position_mode" not in table.columns:
        table = table.copy()
        table["position_mode"] = "long_short"

    ids = resolve_model_ids(model_cfg, model_ids)
    decisions, strategies, position_modes = _spo_selection_calendar(table, walkforward_cfg)

    jobs = [
        (model_id, position_mode, decision, strategy_id)
        for model_id in ids
        for position_mode in position_modes
        for decision in decisions
        for strategy_id in strategies
    ]
    rows: list[dict[str, object]] = []
    pbar = ProgressBar(len(jobs), desc="spo_model") if show_progress else None
    try:
        for model_id, position_mode, decision, strategy_id in jobs:
            if pbar is not None:
                pbar.set_description(f"spo/{model_id}/{position_mode}")
                pbar.set_postfix(
                    date=pd.Timestamp(decision).date().isoformat(),
                    strategy=str(strategy_id),
                )
            selected = _select_spo_for_decision(
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


# Backward-compatible alias
run_walkforward_selection = run_spo_walkforward_selection


# ---------------------------------------------------------------------------
# PWO
# ---------------------------------------------------------------------------


def _pwo_train_mask(
    table: pd.DataFrame,
    *,
    decision: pd.Timestamp,
    walkforward_cfg: dict,
) -> pd.Series:
    decision = pd.Timestamp(decision).normalize()
    realized = (pd.to_datetime(table["decision_date"]) < decision) & (
        pd.to_datetime(table["label_end"]) <= decision
    )
    mode = walkforward_cfg.get("mode", "rolling")
    if mode == "expanding":
        return realized
    if mode != "rolling":
        raise ValueError(f"Unsupported walkforward mode: {mode!r}")
    train_start = decision - parse_span(walkforward_cfg["train_window"])
    return realized & (pd.to_datetime(table["decision_date"]) >= train_start)


def _pwo_decision_dates(table: pd.DataFrame, walkforward_cfg: dict) -> list[pd.Timestamp]:
    first_decision = pd.Timestamp(walkforward_cfg["first_decision_date"]).normalize()
    if "is_model_decision" in table.columns:
        decision_frame = table.loc[table["is_model_decision"]]
    else:
        decision_frame = table.loc[pd.to_datetime(table["decision_date"]) >= first_decision]
    return sorted(pd.to_datetime(decision_frame["decision_date"]).unique())


def _pwo_design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    w_cols = _weight_columns(frame)
    cols = [*FEATURE_COLUMNS, *w_cols]
    matrix = frame.loc[:, cols].copy()
    for col in w_cols:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce")
    return matrix.fillna(0.0)


def _select_pwo_for_decision(
    table: pd.DataFrame,
    *,
    model_id: str,
    model_cfg: dict,
    decision: pd.Timestamp,
    walkforward_cfg: dict,
) -> dict[str, object] | None:
    train = table.loc[
        _pwo_train_mask(table, decision=decision, walkforward_cfg=walkforward_cfg)
    ]
    min_rows = int(model_cfg.get("min_train_rows", 3))
    if len(train) < min_rows:
        return None

    today = table.loc[
        pd.to_datetime(table["decision_date"]) == pd.Timestamp(decision).normalize()
    ]
    if today.empty:
        return None

    fitted = _fit_predict_best(
        train,
        today,
        model_id=model_id,
        model_cfg=model_cfg,
        design_fn=_pwo_design_matrix,
    )
    if fitted is None:
        return None
    best, predicted, _ = fitted
    row: dict[str, object] = {
        "decision_date": pd.Timestamp(decision).normalize(),
        "model_id": model_id,
        "candidate_id": best["candidate_id"],
        "predicted_label": predicted,
        "realized_label": float(best["label"]),
        "train_rows": int(len(train)),
    }
    if "is_equal_weight" in best.index:
        row["is_equal_weight"] = bool(best["is_equal_weight"])
    for col in _weight_columns(table):
        row[col] = float(best[col])
    return row


def run_pwo_walkforward_selection(
    table: pd.DataFrame,
    *,
    walkforward_cfg: dict,
    model_cfg: dict,
    model_ids: str | list[str] | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    PWO walk-forward: for each model and live decision, select the best weight
    candidate by predicted forward portfolio label.

    Output schema matches ``run_pwo_hist_mean_selection`` (plus ML ``model_id``).
    """
    if table.empty:
        return pd.DataFrame()

    ids = resolve_model_ids(model_cfg, model_ids)
    decisions = _pwo_decision_dates(table, walkforward_cfg)
    jobs = [(model_id, decision) for model_id in ids for decision in decisions]
    rows: list[dict[str, object]] = []
    pbar = ProgressBar(len(jobs), desc="pwo_model") if show_progress else None
    try:
        for model_id, decision in jobs:
            if pbar is not None:
                pbar.set_description(f"pwo/{model_id}")
                pbar.set_postfix(date=pd.Timestamp(decision).date().isoformat())
            selected = _select_pwo_for_decision(
                table,
                model_id=model_id,
                model_cfg=model_cfg,
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
    w_cols = _weight_columns(frame)
    ordered = [
        "decision_date",
        "model_id",
        "candidate_id",
        *(["is_equal_weight"] if "is_equal_weight" in frame.columns else []),
        *w_cols,
        "predicted_label",
        "realized_label",
        "train_rows",
    ]
    frame = frame.loc[:, ordered]
    return frame.sort_values(["model_id", "decision_date"]).reset_index(drop=True)
