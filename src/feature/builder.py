"""Point-in-time market features from a price panel."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = (
    "ret_1",
    "ret_5",
    "ret_21",
    "vol_21",
    "ma_gap_21",
)


def _default_windows() -> dict[str, int]:
    return {
        "ret_1": 1,
        "ret_5": 5,
        "ret_21": 21,
        "vol": 21,
        "ma": 21,
    }


def build_features(
    panel: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Build per-asset features using only prices available through each timestamp.

    Output columns:
    timestamp, asset_id, availability_time, and FEATURE_COLUMNS.
    """
    required = {"timestamp", "asset_id", price_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    win = {**_default_windows(), **(windows or {})}
    frames: list[pd.DataFrame] = []

    for asset_id, group in panel.sort_values("timestamp").groupby("asset_id", sort=False):
        g = group.reset_index(drop=True)
        px = g[price_col].astype(float)
        daily_ret = px.pct_change()

        feat = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(g["timestamp"]),
                "asset_id": asset_id,
                "availability_time": pd.to_datetime(
                    g["availability_time"] if "availability_time" in g.columns else g["timestamp"]
                ),
                "ret_1": px.pct_change(win["ret_1"]),
                "ret_5": px.pct_change(win["ret_5"]),
                "ret_21": px.pct_change(win["ret_21"]),
                "vol_21": daily_ret.rolling(win["vol"], min_periods=win["vol"]).std(),
                "ma_gap_21": px / px.rolling(win["ma"], min_periods=win["ma"]).mean() - 1.0,
            }
        )
        frames.append(feat)

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
    return out
