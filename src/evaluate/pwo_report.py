"""PWO evaluate: selected weights vs equal-weight baseline vs market buy-and-hold."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.calendar import parse_span

from .metrics import summarize_period_returns
from .paths import market_period_returns
from .spo_report import _periods_per_year


_PWO_PERIOD_COLS = [
    "decision_date",
    "label_end",
    "model_id",
    "position_mode",
    "strategy_id",
    "sleeve",
    "period_return",
]


def load_pwo_selections(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"selections file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"selections file is empty: {path}")
    required = {"decision_date", "model_id", "realized_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"PWO selections missing columns: {sorted(missing)}")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    return frame


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


def _pwo_cpo_period_returns(selections: pd.DataFrame) -> pd.DataFrame:
    """Model-selected weight path: use stored realized_label as hold-period return."""
    if selections.empty:
        return pd.DataFrame(columns=_PWO_PERIOD_COLS)
    out = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(selections["decision_date"]).dt.normalize(),
            "label_end": pd.to_datetime(selections["label_end"]).dt.normalize(),
            "model_id": selections["model_id"].astype(str),
            "position_mode": "",
            "strategy_id": "",
            "sleeve": "cpo",
            "period_return": selections["realized_label"].astype(float),
        }
    )
    return out.reset_index(drop=True)


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


def run_pwo_evaluation(
    panel: pd.DataFrame,
    *,
    selections_path: str | Path,
    walkforward_cfg: dict,
    price_col: str,
    tickers: list[str],
    weights: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return ``(summary, periods)`` for PWO selections.

    Sleeves (same names as SPO for plot parity):
    - ``cpo``: model-selected weights (``realized_label``)
    - ``baseline``: equal-weight buy-and-hold over the same hold windows
    - ``market``: buy-and-hold under ``weights`` (``None`` → equal weight)
    """
    raw = load_pwo_selections(selections_path)
    filtered = _attach_label_end(raw, walkforward_cfg)
    decisions = sorted(filtered["decision_date"].unique())
    ppy = _periods_per_year(walkforward_cfg["label_horizon"])

    cpo = _pwo_cpo_period_returns(filtered)
    # Equal-weight baseline (PWO natural benchmark).
    baseline = market_period_returns(
        panel,
        decision_dates=decisions,
        walkforward_cfg=walkforward_cfg,
        price_col=price_col,
        tickers=tickers,
        weights=None,
    )
    if not baseline.empty:
        baseline = baseline.copy()
        baseline["sleeve"] = "baseline"

    market = market_period_returns(
        panel,
        decision_dates=decisions,
        walkforward_cfg=walkforward_cfg,
        price_col=price_col,
        tickers=tickers,
        weights=weights,
    )

    periods = pd.concat([cpo, baseline, market], ignore_index=True)
    periods = periods.sort_values(
        ["sleeve", "model_id", "position_mode", "strategy_id", "decision_date"]
    ).reset_index(drop=True)

    summary = pd.DataFrame(_summary_rows(periods, periods_per_year=ppy))
    summary = summary.sort_values(
        ["sleeve", "model_id", "position_mode", "strategy_id"]
    ).reset_index(drop=True)
    return summary, periods
