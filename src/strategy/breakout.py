"""Channel breakout: enter on a new N-day extreme and hold until the opposite break."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import _asset_frame, positions_to_strategy_frame

STRATEGY_ID = "breakout"


def run_breakout(
    panel: pd.DataFrame,
    *,
    lookback: int = 20,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    frames: list[pd.DataFrame] = []
    for asset_id, g in _asset_frame(panel, price_col):
        prior_high = g["_price"].rolling(lookback, min_periods=lookback).max().shift(1)
        prior_low = g["_price"].rolling(lookback, min_periods=lookback).min().shift(1)
        signal = pd.Series(np.nan, index=g.index, dtype=float)
        signal = signal.mask(g["_price"] > prior_high, 1.0)
        signal = signal.mask(g["_price"] < prior_low, -1.0)
        position = signal.ffill().fillna(0.0)
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
