"""Diagnose conditional ML vs hist_mean selections (no refit)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pwo_report import load_pwo_selections
from .spo_report import load_selections

BASELINE_MODEL_ID = "hist_mean"
_KEY = ["decision_date", "position_mode", "strategy_id"]
_PWO_KEY = ["decision_date"]


def _as_id_list(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    ids = [str(item) for item in value]
    return ids or None


def _filter_frame(
    frame: pd.DataFrame,
    *,
    position_ids: list[str] | None,
    strategy_ids: list[str] | None,
) -> pd.DataFrame:
    out = frame
    if position_ids is not None:
        out = out.loc[out["position_mode"].astype(str).isin(position_ids)]
    if strategy_ids is not None:
        out = out.loc[out["strategy_id"].astype(str).isin(strategy_ids)]
    return out


def _compare_one_group(
    ml: pd.DataFrame,
    hm: pd.DataFrame,
    *,
    model_id: str,
    position_mode: str,
    strategy_id: str,
) -> dict[str, object]:
    """
    One summary row: agree_rate on all aligned dates; realized gap / win rate
    only on periods where ML and hist_mean picked different candidates.
    """
    left = ml.loc[:, _KEY + ["candidate_id", "realized_label"]].rename(
        columns={
            "candidate_id": "candidate_id_ml",
            "realized_label": "realized_ml",
        }
    )
    right = hm.loc[:, _KEY + ["candidate_id", "realized_label"]].rename(
        columns={
            "candidate_id": "candidate_id_hm",
            "realized_label": "realized_hm",
        }
    )
    joined = left.merge(right, on=_KEY, how="inner")
    if joined.empty:
        return {
            "model_id": model_id,
            "position_mode": position_mode,
            "strategy_id": strategy_id,
            "n_disagree": 0,
            "agree_rate": float("nan"),
            "mean_realized_diff": float("nan"),
            "ml_win_rate": float("nan"),
        }

    agree = joined["candidate_id_ml"].astype(str) == joined["candidate_id_hm"].astype(str)
    disagree = joined.loc[~agree]
    n_disagree = int(len(disagree))
    if n_disagree == 0:
        mean_diff = 0.0
        win_rate = float("nan")
    else:
        diff = disagree["realized_ml"].astype(float) - disagree["realized_hm"].astype(float)
        mean_diff = float(diff.mean())
        win_rate = float((diff > 0).mean())

    return {
        "model_id": model_id,
        "position_mode": position_mode,
        "strategy_id": strategy_id,
        "n_disagree": n_disagree,
        "agree_rate": float(agree.mean()),
        "mean_realized_diff": mean_diff,
        "ml_win_rate": win_rate,
    }


def run_ml_vs_hist_mean_diagnosis(
    selections: pd.DataFrame | str | Path,
    *,
    model_ids: list[str] | str | None = None,
    position_ids: list[str] | str | None = None,
    strategy_ids: list[str] | str | None = None,
    baseline_model_id: str = BASELINE_MODEL_ID,
) -> pd.DataFrame:
    """
    Compare each ML model_id against hist_mean on aligned decision keys.

    One row per (model_id, position_mode, strategy_id):
    - ``n_disagree`` / ``agree_rate``: how often picks differ vs match
    - ``mean_realized_diff``, ``ml_win_rate``: only on disagree dates
      (``mean_realized_diff`` = mean(ML realized − HM realized))
    """
    if isinstance(selections, (str, Path)):
        frame = load_selections(selections)
    else:
        frame = selections.copy()
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
        if "position_mode" not in frame.columns:
            frame["position_mode"] = "long_short"

    if baseline_model_id not in set(frame["model_id"].astype(str)):
        raise ValueError(
            f"baseline model {baseline_model_id!r} not found in selections; "
            f"available={sorted(frame['model_id'].astype(str).unique())}"
        )

    models = _as_id_list(model_ids)
    if models is None:
        models = sorted(
            m
            for m in frame["model_id"].astype(str).unique()
            if m != baseline_model_id
        )
    else:
        models = [m for m in models if m != baseline_model_id]

    if not models:
        raise ValueError("No ML model_ids left to compare against hist_mean.")

    missing = [m for m in models if m not in set(frame["model_id"].astype(str))]
    if missing:
        raise ValueError(
            f"MODEL_IDS not found in selections: {missing}; "
            f"available={sorted(frame['model_id'].astype(str).unique())}"
        )

    positions = _as_id_list(position_ids)
    strategies = _as_id_list(strategy_ids)
    frame = _filter_frame(frame, position_ids=positions, strategy_ids=strategies)
    if frame.empty:
        return pd.DataFrame()

    hm_all = frame.loc[frame["model_id"].astype(str) == baseline_model_id]
    rows: list[dict[str, object]] = []

    for model_id in models:
        ml_all = frame.loc[frame["model_id"].astype(str) == model_id]
        pos_values = sorted(ml_all["position_mode"].astype(str).unique())
        strat_values = sorted(ml_all["strategy_id"].astype(str).unique())
        for position_mode in pos_values:
            for strategy_id in strat_values:
                ml = ml_all.loc[
                    (ml_all["position_mode"].astype(str) == position_mode)
                    & (ml_all["strategy_id"].astype(str) == strategy_id)
                ]
                hm = hm_all.loc[
                    (hm_all["position_mode"].astype(str) == position_mode)
                    & (hm_all["strategy_id"].astype(str) == strategy_id)
                ]
                rows.append(
                    _compare_one_group(
                        ml,
                        hm,
                        model_id=model_id,
                        position_mode=position_mode,
                        strategy_id=strategy_id,
                    )
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["model_id", "position_mode", "strategy_id"]
    ).reset_index(drop=True)


def _compare_pwo_one(
    ml: pd.DataFrame,
    hm: pd.DataFrame,
    *,
    model_id: str,
) -> dict[str, object]:
    left = ml.loc[:, _PWO_KEY + ["candidate_id", "realized_label"]].rename(
        columns={
            "candidate_id": "candidate_id_ml",
            "realized_label": "realized_ml",
        }
    )
    right = hm.loc[:, _PWO_KEY + ["candidate_id", "realized_label"]].rename(
        columns={
            "candidate_id": "candidate_id_hm",
            "realized_label": "realized_hm",
        }
    )
    joined = left.merge(right, on=_PWO_KEY, how="inner")
    if joined.empty:
        return {
            "model_id": model_id,
            "n_disagree": 0,
            "agree_rate": float("nan"),
            "mean_realized_diff": float("nan"),
            "ml_win_rate": float("nan"),
        }

    agree = joined["candidate_id_ml"].astype(str) == joined["candidate_id_hm"].astype(str)
    disagree = joined.loc[~agree]
    n_disagree = int(len(disagree))
    if n_disagree == 0:
        mean_diff = 0.0
        win_rate = float("nan")
    else:
        diff = disagree["realized_ml"].astype(float) - disagree["realized_hm"].astype(float)
        mean_diff = float(diff.mean())
        win_rate = float((diff > 0).mean())

    return {
        "model_id": model_id,
        "n_disagree": n_disagree,
        "agree_rate": float(agree.mean()),
        "mean_realized_diff": mean_diff,
        "ml_win_rate": win_rate,
    }


def run_pwo_ml_vs_hist_mean_diagnosis(
    selections: pd.DataFrame | str | Path,
    *,
    model_ids: list[str] | str | None = None,
    baseline_model_id: str = BASELINE_MODEL_ID,
) -> pd.DataFrame:
    """
    PWO ML vs hist_mean on aligned decision_date keys (no position/strategy).

    One row per model_id with the same disagree metrics as the SPO diagnosis.
    """
    if isinstance(selections, (str, Path)):
        frame = load_pwo_selections(selections)
    else:
        frame = selections.copy()
        frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()

    if "candidate_id" not in frame.columns:
        raise ValueError("PWO selections must include candidate_id for diagnosis.")

    if baseline_model_id not in set(frame["model_id"].astype(str)):
        raise ValueError(
            f"baseline model {baseline_model_id!r} not found in selections; "
            f"available={sorted(frame['model_id'].astype(str).unique())}"
        )

    models = _as_id_list(model_ids)
    if models is None:
        models = sorted(
            m
            for m in frame["model_id"].astype(str).unique()
            if m != baseline_model_id
        )
    else:
        models = [m for m in models if m != baseline_model_id]

    if not models:
        raise ValueError("No ML model_ids left to compare against hist_mean.")

    missing = [m for m in models if m not in set(frame["model_id"].astype(str))]
    if missing:
        raise ValueError(
            f"MODEL_IDS not found in selections: {missing}; "
            f"available={sorted(frame['model_id'].astype(str).unique())}"
        )

    hm_all = frame.loc[frame["model_id"].astype(str) == baseline_model_id]
    rows: list[dict[str, object]] = []
    for model_id in models:
        ml = frame.loc[frame["model_id"].astype(str) == model_id]
        rows.append(_compare_pwo_one(ml, hm_all, model_id=model_id))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model_id"]).reset_index(drop=True)
