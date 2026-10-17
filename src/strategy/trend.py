"""Trend following: long above slow MA, short below."""

from __future__ import annotations

import pandas as pd

from .base import _asset_frame, positions_to_strategy_frame, sign_position

STRATEGY_ID = "trend"


def run_trend(
    panel: pd.DataFrame,
    *,
    ma_window: int = 50,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    if ma_window < 2:
        raise ValueError("ma_window must be >= 2")

    frames: list[pd.DataFrame] = []
    for asset_id, g in _asset_frame(panel, price_col):
        ma = g["_price"].rolling(ma_window, min_periods=ma_window).mean()
        position = sign_position(g["_price"] - ma)
        frames.append(
            positions_to_strategy_frame(
                asset_id=asset_id,
                strategy_id=STRATEGY_ID,
                timestamps=g["timestamp"],
                position=position,
                asset_return=g["_asset_return"],
            )
        )
    return pd.concat(frames, ignore_index=True)
