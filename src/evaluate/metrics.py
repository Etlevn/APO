"""Summary metrics from discrete hold-period returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(period_returns: pd.Series) -> pd.Series:
    """Cumulative wealth starting at 1.0 from a series of period returns."""
    clean = period_returns.astype(float).fillna(0.0)
    return (1.0 + clean).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def summarize_period_returns(
    period_returns: pd.Series,
    *,
    periods_per_year: float = 1.0,
) -> dict[str, float]:
    """
    Aggregate metrics for non-overlapping hold periods.

    ``periods_per_year`` is 1.0 when each period is about one year.
    """
    r = period_returns.astype(float).dropna()
    n = int(len(r))
    if n == 0:
        return {
            "n_periods": 0.0,
            "cumulative_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
        }

    cum = float(np.prod(1.0 + r.to_numpy()) - 1.0)
    years = n / periods_per_year if periods_per_year > 0 else float("nan")
    if years and years > 0 and np.isfinite(years):
        ann_return = float((1.0 + cum) ** (1.0 / years) - 1.0)
    else:
        ann_return = float("nan")

    std = float(r.std(ddof=1)) if n > 1 else float("nan")
    mean = float(r.mean())
    ann_vol = float(std * np.sqrt(periods_per_year)) if np.isfinite(std) else float("nan")
    if np.isfinite(std) and std > 0:
        sharpe = float((mean * periods_per_year) / ann_vol) if ann_vol > 0 else float("nan")
    else:
        sharpe = float("nan")

    equity = equity_curve(r)
    return {
        "n_periods": float(n),
        "cumulative_return": cum,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((r > 0).mean()),
    }
