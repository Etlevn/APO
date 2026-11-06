"""PWO equity-curve figure (same visual style as SPO; no position/strategy dims)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .spo_plot import spo_equity_figure


def pwo_equity_figure(
    periods: pd.DataFrame,
    *,
    sleeve_ids: list[str] | str | None = None,
    model_ids: list[str] | str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> plt.Figure:
    """
    Build one PWO equity figure. Filters apply only to the figure.

    Colors match SPO: market = gray, baseline = black, hist_mean = red, other cpo = cycle.
    """
    return spo_equity_figure(
        periods,
        sleeve_ids=sleeve_ids,
        position_ids=None,
        strategy_ids=None,
        model_ids=model_ids,
        date_start=date_start,
        date_end=date_end,
    )


def save_pwo_equity_plot(
    periods: pd.DataFrame,
    path: str | Path,
    *,
    sleeve_ids: list[str] | str | None = None,
    model_ids: list[str] | str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> Path:
    """
    Write one PWO PNG.

    Reuses SPO styling (market gray / baseline black / hist_mean red / other cpo
    colored). Filters are sleeve × model × date only (no position/strategy).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = pwo_equity_figure(
        periods,
        sleeve_ids=sleeve_ids,
        model_ids=model_ids,
        date_start=date_start,
        date_end=date_end,
    )
    fig.savefig(path)
    plt.close(fig)
    return path
