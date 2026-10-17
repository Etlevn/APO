"""Time-series momentum: follow the sign of past N-day return."""

from __future__ import annotations

import pandas as pd

from .base import _asset_frame, positions_to_strategy_frame, sign_position

STRATEGY_ID = "momentum"


def run_momentum(
    panel: pd.DataFrame,
    *,
    lookback: int = 20,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    frames: list[pd.DataFrame] = []
    for asset_id, g in _asset_frame(panel, price_col):
        past_return = g["_price"].pct_change(lookback)
        position = sign_position(past_return)
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
