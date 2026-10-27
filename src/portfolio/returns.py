"""Asset returns and fixed-weight portfolio returns (no trading strategies)."""

from __future__ import annotations

import pandas as pd

from dataset.weights import resolve_weights


def build_asset_returns(
    panel: pd.DataFrame,
    *,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Per-asset simple returns from the price panel.

    Columns: timestamp, asset_id, asset_return
    ``asset_return[t]`` = price[t] / price[t-1] - 1 (first bar NaN).
    """
    required = {"timestamp", "asset_id", price_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for asset_id, group in panel.sort_values("timestamp").groupby("asset_id", sort=False):
        g = group.reset_index(drop=True)
        px = g[price_col].astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(g["timestamp"]),
                    "asset_id": str(asset_id),
                    "asset_return": px.pct_change(),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Daily portfolio return under fixed weights: sum_i w_i * R_{i,t}.

    Missing weight for an asset raises; missing return on a date contributes NaN
    for that asset (day is NaN only if no asset contributes).
    """
    frame = asset_returns.copy()
    frame["weight"] = frame["asset_id"].map(weights)
    if frame["weight"].isna().any():
        bad = sorted(frame.loc[frame["weight"].isna(), "asset_id"].astype(str).unique())
        raise KeyError(f"assets missing from weights: {bad}")
    frame["contrib"] = frame["asset_return"].astype(float) * frame["weight"]
    series = frame.groupby("timestamp", sort=True)["contrib"].sum(min_count=1)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def build_weighted_return_panel(
    panel: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    tickers: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Asset returns with baseline weights and contributions.

    Columns: timestamp, asset_id, asset_return, weight, contrib
    ``weights=None`` → equal weight over ``tickers`` (or assets present in panel).
    """
    asset_ret = build_asset_returns(panel, price_col=price_col)
    if tickers is None:
        tickers = sorted(asset_ret["asset_id"].astype(str).unique().tolist())
    resolved = resolve_weights(tickers, weights)
    frame = asset_ret.copy()
    frame["weight"] = frame["asset_id"].map(resolved)
    if frame["weight"].isna().any():
        bad = sorted(frame.loc[frame["weight"].isna(), "asset_id"].astype(str).unique())
        raise KeyError(f"assets missing from weights: {bad}")
    frame["contrib"] = frame["asset_return"].astype(float) * frame["weight"]
    return frame.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
