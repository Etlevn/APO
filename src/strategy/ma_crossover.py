"""Moving-average crossover: long when the fast average is above the slow one."""

from __future__ import annotations

import pandas as pd

from .base import _asset_frame, positions_to_strategy_frame, sign_position

STRATEGY_ID = "ma_crossover"


def run_ma_crossover(
    panel: pd.DataFrame,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    if fast_window < 2:
        raise ValueError("fast_window must be >= 2")
    if slow_window < 2:
        raise ValueError("slow_window must be >= 2")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")

    frames: list[pd.DataFrame] = []
    for asset_id, g in _asset_frame(panel, price_col):
        fast = g["_price"].rolling(fast_window, min_periods=fast_window).mean()
        slow = g["_price"].rolling(slow_window, min_periods=slow_window).mean()
        position = sign_position(fast - slow)
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
