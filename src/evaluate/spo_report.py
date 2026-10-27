"""Orchestrate evaluate: filter selections, build sleeves, summarize metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.calendar import parse_span

from .metrics import summarize_period_returns
from .paths import (
    baseline_period_returns,
    blend_strategies,
    cpo_period_returns,
    market_period_returns,
)


def load_selections(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"selections file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"selections file is empty: {path}")
    required = {"decision_date", "model_id", "strategy_id", "realized_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"selections missing columns: {sorted(missing)}")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
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


def _attach_label_end(
    selections: pd.DataFrame,
    walkforward_cfg: dict,
) -> pd.DataFrame:
    frame = selections.copy()
    if "label_end" not in frame.columns or frame["label_end"].isna().all():
        offset = parse_span(walkforward_cfg["label_horizon"])
        frame["label_end"] = frame["decision_date"] + offset
    frame["label_end"] = pd.to_datetime(frame["label_end"]).dt.normalize()
    return frame


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

    Always includes cpo/baseline singles + blends, and market.
    Display filters belong in the standalone ``plot_spo.py`` entry, not here.
    """
    raw = load_selections(selections_path)
    if raw.empty:
        raise ValueError("selections file is empty.")

    filtered = _attach_label_end(raw, walkforward_cfg)
    decisions = sorted(filtered["decision_date"].unique())
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
        decision_dates=decisions,
        strategy_ids=strategies,
        position_modes=modes,
        candidate_cfg=active_candidate_cfg,
        walkforward_cfg=walkforward_cfg,
        price_col=price_col,
        tickers=tickers,
        weights=weights,
        overlay_scale=overlay_scale,
    )
    market = market_period_returns(
        panel,
        decision_dates=decisions,
        walkforward_cfg=walkforward_cfg,
        price_col=price_col,
        tickers=tickers,
        weights=weights,
    )

    periods = pd.concat(
        [cpo, blend_strategies(cpo), baseline, blend_strategies(baseline), market],
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
