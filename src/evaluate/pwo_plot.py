"""PWO equity-curve figure (same visual style as SPO; no position/strategy dims)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .spo_plot import save_spo_equity_plot


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
    return save_spo_equity_plot(
        periods,
        path,
        sleeve_ids=sleeve_ids,
        position_ids=None,
        strategy_ids=None,
        model_ids=model_ids,
        date_start=date_start,
        date_end=date_end,
    )
