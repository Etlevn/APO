"""Single equity-curve figure for evaluate sleeves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import equity_curve

_MARKET_COLOR = "#888888"
_BASELINE_COLOR = "#000000"
_CPO_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
)


def _as_id_list(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    ids = [str(item) for item in value]
    return ids or None


def _series_label(
    sleeve: str,
    position_mode: str,
    strategy_id: str,
    model_id: str,
) -> str:
    """Legend: sleeve / position / strategy / model (model only for cpo)."""
    if sleeve == "market":
        return "market"
    parts = [sleeve]
    if position_mode:
        parts.append(position_mode)
    if strategy_id:
        parts.append(strategy_id)
    if sleeve == "cpo" and model_id:
        parts.append(model_id)
    return "/".join(parts)


def _filter_plot_rows(
    periods: pd.DataFrame,
    *,
    sleeve_ids: list[str] | None,
    position_ids: list[str] | None,
    strategy_ids: list[str] | None,
    model_ids: list[str] | None,
    date_start: str | None,
    date_end: str | None,
) -> pd.DataFrame:
    frame = periods.copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["strategy_id"] = frame["strategy_id"].fillna("").astype(str)
    frame["model_id"] = frame["model_id"].fillna("").astype(str)
    if "position_mode" not in frame.columns:
        frame["position_mode"] = ""
    else:
        frame["position_mode"] = frame["position_mode"].fillna("").astype(str)

    # Apply in conceptual order: sleeve → position → strategy → model → date.
    if sleeve_ids is not None:
        frame = frame.loc[frame["sleeve"].astype(str).isin(sleeve_ids)]

    if position_ids is not None:
        keep = (frame["sleeve"] == "market") | frame["position_mode"].isin(position_ids)
        frame = frame.loc[keep]

    if strategy_ids is not None:
        keep = (frame["sleeve"] == "market") | frame["strategy_id"].isin(strategy_ids)
        frame = frame.loc[keep]

    if model_ids is not None:
        keep = (frame["sleeve"] != "cpo") | frame["model_id"].isin(model_ids)
        frame = frame.loc[keep]

    if date_start is not None:
        frame = frame.loc[frame["decision_date"] >= pd.Timestamp(date_start).normalize()]
    if date_end is not None:
        frame = frame.loc[frame["decision_date"] <= pd.Timestamp(date_end).normalize()]

    return frame.reset_index(drop=True)


def save_equity_plot(
    periods: pd.DataFrame,
    path: str | Path,
    *,
    sleeve_ids: list[str] | str | None = None,
    position_ids: list[str] | str | None = None,
    strategy_ids: list[str] | str | None = None,
    model_ids: list[str] | str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> Path:
    """
    Write one PNG. Filters apply only to the figure (full ``periods`` unchanged).

    Colors: market = gray, baseline = black, cpo = colored cycle.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if periods.empty:
        raise ValueError("No period rows available to plot.")

    frame = _filter_plot_rows(
        periods,
        sleeve_ids=_as_id_list(sleeve_ids),
        position_ids=_as_id_list(position_ids),
        strategy_ids=_as_id_list(strategy_ids),
        model_ids=_as_id_list(model_ids),
        date_start=date_start,
        date_end=date_end,
    )
    if frame.empty:
        raise ValueError("No period rows left to plot after PLOT_* filters.")

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    # Legend / draw order: sleeve → position → strategy → model
    group_cols = ["sleeve", "position_mode", "strategy_id", "model_id"]
    cpo_color_i = 0
    for keys, group in frame.groupby(group_cols, sort=True):
        sleeve, position_mode, strategy_id, model_id = keys
        ordered = group.sort_values("decision_date")
        equity = equity_curve(ordered["period_return"])
        sleeve_s = str(sleeve)
        if sleeve_s == "market":
            color = _MARKET_COLOR
        elif sleeve_s == "baseline":
            color = _BASELINE_COLOR
        else:
            color = _CPO_COLORS[cpo_color_i % len(_CPO_COLORS)]
            cpo_color_i += 1
        ax.plot(
            pd.to_datetime(ordered["decision_date"]),
            equity.to_numpy(),
            label=_series_label(
                sleeve_s,
                str(position_mode),
                str(strategy_id),
                str(model_id),
            ),
            color=color,
            linewidth=1.8,
        )

    ax.axhline(1.0, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
