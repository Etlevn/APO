"""Walk-forward candidate selection by historical mean label (no ML).

SPO: pick parameter candidates (strategy × position_mode).
PWO: pick asset-weight candidates.
"""

from __future__ import annotations

import pandas as pd

from util import ProgressBar

from .walkforward import (
    _pwo_decision_dates,
    _pwo_train_mask,
    _spo_selection_calendar,
    _spo_train_mask,
    _weight_columns,
)


# ---------------------------------------------------------------------------
# SPO
# ---------------------------------------------------------------------------


def _select_spo_hist_mean_for_decision(
    table: pd.DataFrame,
    *,
    strategy_id: str,
    position_mode: str,
    decision: pd.Timestamp,
    walkforward_cfg: dict,
    min_train_rows: int = 3,
) -> dict[str, object] | None:
    """
    Non-ML baseline: pick the parameter candidate with the highest mean realized
    label over the same causal train window used by conditional models.
    """
    train = table.loc[
        _spo_train_mask(
            table,
            strategy_id=strategy_id,
            position_mode=position_mode,
            decision=decision,
            walkforward_cfg=walkforward_cfg,
        )
    ]
    if len(train) < min_train_rows:
        return None

    today = table.loc[
        (table["strategy_id"] == strategy_id)
        & (table["position_mode"].astype(str) == str(position_mode))
        & (pd.to_datetime(table["decision_date"]) == pd.Timestamp(decision).normalize())
    ]
    if today.empty:
        return None

    scores = train.groupby("candidate_id", sort=False)["label"].mean()
    scored = today.copy()
    scored["_hist_mean"] = scored["candidate_id"].map(scores)
    scored = scored.dropna(subset=["_hist_mean"])
    if scored.empty:
        return None

    scored = scored.sort_values(
        ["_hist_mean", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    best = scored.iloc[0]
    return {
        "decision_date": pd.Timestamp(decision).normalize(),
        "model_id": "hist_mean",
        "position_mode": str(position_mode),
        "strategy_id": strategy_id,
        "candidate_id": best["candidate_id"],
        "lookback": best["lookback"],
        "ma_window": best["ma_window"],
        "entry_z": best["entry_z"],
        "predicted_label": float(best["_hist_mean"]),
        "realized_label": float(best["label"]),
        "train_rows": int(len(train)),
    }


def run_spo_hist_mean_selection(
    table: pd.DataFrame,
    *,
    walkforward_cfg: dict,
    min_train_rows: int = 3,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Walk-forward SPO candidate selection using historical mean labels (no ML fit).

    Output schema matches ``run_spo_walkforward_selection`` / ``spo_sel.csv``:
    one row per (decision_date, position_mode, strategy_id) with ``model_id='hist_mean'``.
    ``predicted_label`` stores the selected candidate's historical mean label.
    """
    if table.empty:
        return pd.DataFrame()

    if "position_mode" not in table.columns:
        table = table.copy()
        table["position_mode"] = "long_short"

    decisions, strategies, position_modes = _spo_selection_calendar(table, walkforward_cfg)
    jobs = [
        (position_mode, decision, strategy_id)
        for position_mode in position_modes
        for decision in decisions
        for strategy_id in strategies
    ]
    rows: list[dict[str, object]] = []
    pbar = ProgressBar(len(jobs), desc="spo_hist_mean") if show_progress else None
    try:
        for position_mode, decision, strategy_id in jobs:
            if pbar is not None:
                pbar.set_description(f"spo_hist_mean/{position_mode}")
                pbar.set_postfix(
                    date=pd.Timestamp(decision).date().isoformat(),
                    strategy=str(strategy_id),
                )
            selected = _select_spo_hist_mean_for_decision(
                table,
                strategy_id=strategy_id,
                position_mode=position_mode,
                decision=decision,
                walkforward_cfg=walkforward_cfg,
                min_train_rows=min_train_rows,
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
run_hist_mean_selection = run_spo_hist_mean_selection


# ---------------------------------------------------------------------------
# PWO
# ---------------------------------------------------------------------------


def _select_pwo_hist_mean_for_decision(
    table: pd.DataFrame,
    *,
    decision: pd.Timestamp,
    walkforward_cfg: dict,
    min_train_rows: int = 3,
) -> dict[str, object] | None:
    train = table.loc[
        _pwo_train_mask(table, decision=decision, walkforward_cfg=walkforward_cfg)
    ]
    if len(train) < min_train_rows:
        return None

    today = table.loc[
        pd.to_datetime(table["decision_date"]) == pd.Timestamp(decision).normalize()
    ]
    if today.empty:
        return None

    scores = train.groupby("candidate_id", sort=False)["label"].mean()
    scored = today.copy()
    scored["_hist_mean"] = scored["candidate_id"].map(scores)
    scored = scored.dropna(subset=["_hist_mean"])
    if scored.empty:
        return None

    scored = scored.sort_values(
        ["_hist_mean", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    best = scored.iloc[0]
    row: dict[str, object] = {
        "decision_date": pd.Timestamp(decision).normalize(),
        "model_id": "hist_mean",
        "candidate_id": best["candidate_id"],
        "predicted_label": float(best["_hist_mean"]),
        "realized_label": float(best["label"]),
        "train_rows": int(len(train)),
    }
    if "is_equal_weight" in best.index:
        row["is_equal_weight"] = bool(best["is_equal_weight"])
    for col in _weight_columns(table):
        row[col] = float(best[col])
    return row


def run_pwo_hist_mean_selection(
    table: pd.DataFrame,
    *,
    walkforward_cfg: dict,
    min_train_rows: int = 3,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Walk-forward PWO weight selection using historical mean labels (no ML fit).

    One row per live decision_date with ``model_id='hist_mean'``.
    ``predicted_label`` is the selected candidate's mean historical label.
    """
    if table.empty:
        return pd.DataFrame()

    decisions = _pwo_decision_dates(table, walkforward_cfg)
    rows: list[dict[str, object]] = []
    pbar = ProgressBar(len(decisions), desc="pwo_hist_mean") if show_progress else None
    try:
        for decision in decisions:
            if pbar is not None:
                pbar.set_description("pwo_hist_mean")
                pbar.set_postfix(date=pd.Timestamp(decision).date().isoformat())
            selected = _select_pwo_hist_mean_for_decision(
                table,
                decision=decision,
                walkforward_cfg=walkforward_cfg,
                min_train_rows=min_train_rows,
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
