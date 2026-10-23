"""Build hold-period returns for CPO picks, fixed baseline, and constant-mix basket."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataset.weights import resolve_weights
from strategy import (
    apply_position_mode,
    baseline_strategy_cfg,
    run_mean_reversion,
    run_momentum,
    run_trend,
)

_PERIOD_COLS = [
    "decision_date",
    "label_end",
    "model_id",
    "position_mode",
    "strategy_id",
    "sleeve",
    "period_return",
]


def _run_one_candidate(
    panel: pd.DataFrame,
    strategy_id: str,
    params: dict,
    *,
    price_col: str,
) -> pd.DataFrame:
    if strategy_id == "momentum":
        return run_momentum(panel, lookback=int(params["lookback"]), price_col=price_col)
    if strategy_id == "trend":
        return run_trend(panel, ma_window=int(params["ma_window"]), price_col=price_col)
    if strategy_id == "mean_reversion":
        return run_mean_reversion(
            panel,
            lookback=int(params["lookback"]),
            entry_z=float(params["entry_z"]),
            price_col=price_col,
        )
    raise ValueError(f"Unsupported strategy_id: {strategy_id!r}")


def _portfolio_returns(
    signal_frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    frame = signal_frame.copy()
    frame["weight"] = frame["asset_id"].map(weights)
    if frame["weight"].isna().any():
        bad = sorted(frame.loc[frame["weight"].isna(), "asset_id"].astype(str).unique())
        raise KeyError(f"signal assets missing from WEIGHTS: {bad}")
    frame["weighted_return"] = frame["strategy_return"].astype(float) * frame["weight"]
    series = frame.groupby("timestamp", sort=True)["weighted_return"].sum().sort_index()
    series.index = pd.to_datetime(series.index)
    return series


def _forward_cumulative_return(
    returns: pd.Series,
    decision_date: pd.Timestamp,
    label_end: pd.Timestamp,
) -> float | None:
    window = returns.loc[(returns.index > decision_date) & (returns.index <= label_end)]
    window = window.dropna()
    if window.empty:
        return None
    return float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0)


def const_mix_returns(
    panel: pd.DataFrame,
    *,
    price_col: str,
    weights: dict[str, float],
) -> pd.Series:
    """Daily target-weight returns (constant mix; rebalanced each bar, not drift B&H)."""
    frame = panel.loc[:, ["timestamp", "asset_id", price_col]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    pieces: list[pd.DataFrame] = []
    for asset_id, group in frame.groupby("asset_id", sort=False):
        g = group.sort_values("timestamp").copy()
        w = weights.get(str(asset_id))
        if w is None:
            raise KeyError(f"asset {asset_id!r} missing from WEIGHTS")
        g["asset_return"] = g[price_col].astype(float).pct_change()
        g["weighted_return"] = g["asset_return"] * float(w)
        pieces.append(g.loc[:, ["timestamp", "weighted_return"]])
    stacked = pd.concat(pieces, ignore_index=True)
    series = stacked.groupby("timestamp", sort=True)["weighted_return"].sum().sort_index()
    series.index = pd.to_datetime(series.index)
    return series


def decision_label_ends(selections: pd.DataFrame) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each decision_date to its stored label_end from main selections."""
    if selections.empty:
        return {}
    frame = selections.loc[:, ["decision_date", "label_end"]].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"]).dt.normalize()
    frame["label_end"] = pd.to_datetime(frame["label_end"]).dt.normalize()
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision, group in frame.groupby("decision_date", sort=True):
        ends = group["label_end"].dropna().unique()
        if len(ends) != 1:
            raise ValueError(
                f"inconsistent label_end for decision_date={pd.Timestamp(decision).date()}: "
                f"{sorted(pd.Timestamp(x).date().isoformat() for x in ends)}"
            )
        mapping[pd.Timestamp(decision)] = pd.Timestamp(ends[0])
    return mapping


def cpo_period_returns(selections: pd.DataFrame) -> pd.DataFrame:
    """One row per selection using stored realized_label as the hold-period return."""
    if selections.empty:
        return pd.DataFrame(columns=_PERIOD_COLS)
    if "position_mode" not in selections.columns:
        selections = selections.copy()
        selections["position_mode"] = "long_short"
    out = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(selections["decision_date"]).dt.normalize(),
            "label_end": pd.to_datetime(selections["label_end"]).dt.normalize(),
            "model_id": selections["model_id"].astype(str),
            "position_mode": selections["position_mode"].astype(str),
            "strategy_id": selections["strategy_id"].astype(str),
            "sleeve": "cpo",
            "period_return": selections["realized_label"].astype(float),
        }
    )
    return out.reset_index(drop=True)


def baseline_period_returns(
    panel: pd.DataFrame,
    *,
    label_ends: dict[pd.Timestamp, pd.Timestamp],
    strategy_ids: list[str],
    position_modes: list[str],
    candidate_cfg: dict,
    price_col: str,
    tickers: list[str],
    weights: dict[str, float] | None,
    overlay_scale: float = 1.0,
) -> pd.DataFrame:
    """Fixed first-grid-point params under each position_mode, same hold windows."""
    if not label_ends or not strategy_ids or not position_modes:
        return pd.DataFrame(columns=_PERIOD_COLS)

    resolved = resolve_weights(tickers, weights)
    baseline_cfg = baseline_strategy_cfg(candidate_cfg, price_col=price_col)

    cache: dict[tuple[str, str], pd.Series] = {}
    for position_mode in position_modes:
        for strategy_id in strategy_ids:
            if strategy_id not in baseline_cfg:
                raise KeyError(f"baseline params missing for strategy {strategy_id!r}")
            params = dict(baseline_cfg[strategy_id])  # type: ignore[arg-type]
            raw = _run_one_candidate(panel, strategy_id, params, price_col=price_col)
            signal = apply_position_mode(raw, position_mode, overlay_scale=overlay_scale)
            cache[(position_mode, strategy_id)] = _portfolio_returns(signal, resolved)

    rows: list[dict[str, object]] = []
    for decision, label_end in sorted(label_ends.items()):
        decision = pd.Timestamp(decision).normalize()
        label_end = pd.Timestamp(label_end).normalize()
        for position_mode in position_modes:
            for strategy_id in strategy_ids:
                value = _forward_cumulative_return(
                    cache[(position_mode, strategy_id)],
                    decision,
                    label_end,
                )
                if value is None:
                    continue
                rows.append(
                    {
                        "decision_date": decision,
                        "label_end": label_end,
                        "model_id": "",
                        "position_mode": position_mode,
                        "strategy_id": strategy_id,
                        "sleeve": "baseline",
                        "period_return": value,
                    }
                )
    return pd.DataFrame(rows)


def const_mix_period_returns(
    panel: pd.DataFrame,
    *,
    label_ends: dict[pd.Timestamp, pd.Timestamp],
    price_col: str,
    tickers: list[str],
    weights: dict[str, float] | None,
) -> pd.DataFrame:
    """Constant-mix (target-weight) basket over the same hold windows as CPO/baseline."""
    if not label_ends:
        return pd.DataFrame(columns=_PERIOD_COLS)

    resolved = resolve_weights(tickers, weights)
    mix = const_mix_returns(panel, price_col=price_col, weights=resolved)

    rows: list[dict[str, object]] = []
    for decision, label_end in sorted(label_ends.items()):
        decision = pd.Timestamp(decision).normalize()
        label_end = pd.Timestamp(label_end).normalize()
        value = _forward_cumulative_return(mix, decision, label_end)
        if value is None:
            continue
        rows.append(
            {
                "decision_date": decision,
                "label_end": label_end,
                "model_id": "",
                "position_mode": "",
                "strategy_id": "",
                "sleeve": "const_mix",
                "period_return": value,
            }
        )
    return pd.DataFrame(rows)


def blend_strategies(period_frame: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average across strategies on each decision_date (within mode/model)."""
    if period_frame.empty:
        return pd.DataFrame(columns=_PERIOD_COLS)

    group_cols = ["decision_date", "label_end", "sleeve", "position_mode"]
    if period_frame["model_id"].astype(str).str.len().gt(0).any():
        group_cols = ["decision_date", "label_end", "model_id", "position_mode", "sleeve"]

    blended = (
        period_frame.groupby(group_cols, sort=False)["period_return"]
        .mean()
        .reset_index()
    )
    blended["strategy_id"] = "blend"
    if "model_id" not in blended.columns:
        blended["model_id"] = ""
    return blended.loc[:, _PERIOD_COLS]
