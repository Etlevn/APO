"""Shared helpers for strategy signal and return construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _asset_frame(panel: pd.DataFrame, price_col: str) -> list[tuple[str, pd.DataFrame]]:
    required = {"timestamp", "asset_id", price_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    frames: list[tuple[str, pd.DataFrame]] = []
    for asset_id, group in panel.sort_values("timestamp").groupby("asset_id", sort=False):
        g = group.reset_index(drop=True).copy()
        g["timestamp"] = pd.to_datetime(g["timestamp"])
        g["_price"] = g[price_col].astype(float)
        g["_asset_return"] = g["_price"].pct_change()
        frames.append((str(asset_id), g))
    return frames


def positions_to_strategy_frame(
    asset_id: str,
    strategy_id: str,
    timestamps: pd.Series,
    position: pd.Series,
    asset_return: pd.Series,
) -> pd.DataFrame:
    """
    Convert end-of-day positions into a strategy return panel.

    position[t] is decided with information through t and is held over the next bar.
    strategy_return[t] = position[t-1] * asset_return[t]
    """
    pos = pd.Series(position, dtype=float).fillna(0.0)
    ret = pd.Series(asset_return, dtype=float)
    strategy_return = pos.shift(1) * ret

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "asset_id": asset_id,
            "strategy_id": strategy_id,
            "position": pos.to_numpy(),
            "asset_return": ret.to_numpy(),
            "strategy_return": strategy_return.to_numpy(),
        }
    )


def sign_position(values: pd.Series) -> pd.Series:
    """Map positive/negative values to +1/-1; zeros and NaNs become 0."""
    arr = values.to_numpy(dtype=float)
    out = np.zeros(len(arr), dtype=float)
    out[arr > 0] = 1.0
    out[arr < 0] = -1.0
    return pd.Series(out, index=values.index)
