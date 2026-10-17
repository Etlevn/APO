"""Mean reversion: fade large z-score deviations from a moving average."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import _asset_frame, positions_to_strategy_frame

STRATEGY_ID = "mean_reversion"


def run_mean_reversion(
    panel: pd.DataFrame,
    *,
    lookback: int = 20,
    entry_z: float = 1.0,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if entry_z <= 0:
        raise ValueError("entry_z must be > 0")

    frames: list[pd.DataFrame] = []
    for asset_id, g in _asset_frame(panel, price_col):
        ma = g["_price"].rolling(lookback, min_periods=lookback).mean()
        sd = g["_price"].rolling(lookback, min_periods=lookback).std(ddof=0)
        z = (g["_price"] - ma) / sd.replace(0.0, np.nan)

        position = pd.Series(0.0, index=g.index)
        position = position.mask(z > entry_z, -1.0)
        position = position.mask(z < -entry_z, 1.0)
        position = position.fillna(0.0)

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
