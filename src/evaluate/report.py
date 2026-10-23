"""Orchestrate evaluate: filter selections, build sleeves, summarize metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import summarize_period_returns
from .paths import (
    baseline_period_returns,
    blend_strategies,
    const_mix_period_returns,
    cpo_period_returns,
    decision_label_ends,
)


def load_selections(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"selections file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"selections file is empty: {path}")
    required = {
        "decision_date",
        "label_end",
        "model_id",
        "strategy_id",
        "realized_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"selections missing columns: {sorted(missing)}. "
            "Re-run main.py so main_selections.csv includes label_end from the training table."
        )
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["label_end"] = pd.to_datetime(frame["label_end"]).dt.normalize()
    if frame["label_end"].isna().any():
        raise ValueError("selections.label_end contains nulls; re-run main.py.")
    if "position_mode" not in frame.columns:
        frame["position_mode"] = "long_short"
    return frame


def _periods_per_year(label_horizon: str) -> float:
    """Approximate how many label horizons fit in one year (1y→1, 6m→2, 1m→12)."""
    text = str(label_horizon).strip().lower()
    if text.endswith("y"):
        years = float(text[:-1])
        return 1.0 / years if years > 0 else 1.0
    if text.endswith("m"):
        months = float(text[:-1])
        return 12.0 / months if months > 0 else 1.0
    if text.endswith("w"):
        weeks = float(text[:-1])
        return 52.0 / weeks if weeks > 0 else 1.0
    if text.endswith("d"):
        days = float(text[:-1])
        return 365.25 / days if days > 0 else 1.0
    return 1.0


def _summary_rows(
    periods: pd.DataFrame,
    *,
    periods_per_year: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if periods.empty:
        return rows
    group_cols = ["sleeve", "model_id", "position_mode", "strategy_id"]
    for keys, group in periods.groupby(group_cols, sort=True):
        sleeve, model_id, position_mode, strategy_id = keys
        metrics = summarize_period_returns(
            group.sort_values("decision_date")["period_return"],
            periods_per_year=periods_per_year,
        )
        rows.append(
            {
                "sleeve": sleeve,
                "model_id": model_id if model_id else "",
                "position_mode": position_mode if position_mode else "",
                "strategy_id": strategy_id if strategy_id else "",
                **metrics,
            }
        )
    return rows


def run_evaluation(
    panel: pd.DataFrame,
    *,
    selections_path: str | Path,
    walkforward_cfg: dict,
    candidate_cfg: dict,
    price_col: str,
    tickers: list[str],
    weights: dict[str, float] | None,
    overlay_scale: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return full ``(summary, periods)`` for everything in ``selections_path``.

    Hold windows use ``label_end`` stored in the selections CSV (from main);
    they are not recomputed from the current ``config.py``.

    Always includes cpo/baseline singles + blends, and const_mix.
    Display filters belong in the standalone ``plot.py`` entry, not here.
    """
    filtered = load_selections(selections_path)
    if filtered.empty:
        raise ValueError("selections file is empty.")

    label_ends = decision_label_ends(filtered)
    strategies = sorted(filtered["strategy_id"].astype(str).unique())
    modes = sorted(filtered["position_mode"].astype(str).unique())
    active_candidate_cfg = {
        sid: candidate_cfg[sid] for sid in strategies if sid in candidate_cfg
    }
    missing = [sid for sid in strategies if sid not in candidate_cfg]
    if missing:
        raise KeyError(f"strategies in selections missing from CANDIDATE: {missing}")
    if not active_candidate_cfg:
        raise ValueError("No overlapping strategies between selections and CANDIDATE grids.")

    ppy = _periods_per_year(walkforward_cfg["label_horizon"])

    cpo = cpo_period_returns(filtered)
    baseline = baseline_period_returns(
        panel,
        label_ends=label_ends,
        strategy_ids=strategies,
        position_modes=modes,
        candidate_cfg=active_candidate_cfg,
        price_col=price_col,
        tickers=tickers,
        weights=weights,
        overlay_scale=overlay_scale,
    )
    const_mix = const_mix_period_returns(
        panel,
        label_ends=label_ends,
        price_col=price_col,
        tickers=tickers,
        weights=weights,
    )

    periods = pd.concat(
        [cpo, blend_strategies(cpo), baseline, blend_strategies(baseline), const_mix],
        ignore_index=True,
    )
    periods = periods.sort_values(
        ["sleeve", "model_id", "position_mode", "strategy_id", "decision_date"]
    ).reset_index(drop=True)

    summary = pd.DataFrame(_summary_rows(periods, periods_per_year=ppy))
    summary = summary.sort_values(
        ["sleeve", "model_id", "position_mode", "strategy_id"]
    ).reset_index(drop=True)
    return summary, periods
