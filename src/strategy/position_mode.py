"""Position / exposure modes applied on top of raw strategy signals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

SUPPORTED_POSITION_MODES = ("long_short", "long_only", "market_overlay")


def resolve_position_ids(position_ids: str | Sequence[str]) -> list[str]:
    """Accept one id (``\"long_only\"``) or several (``[\"long_short\", \"long_only\"]``)."""
    if isinstance(position_ids, str):
        ids = [position_ids]
    else:
        ids = [str(item) for item in position_ids]

    if not ids:
        raise ValueError("POSITION_RUN must contain at least one position_mode.")

    unknown = [mode for mode in ids if mode not in SUPPORTED_POSITION_MODES]
    if unknown:
        raise ValueError(
            f"Unsupported position_mode(s) {unknown}; choose from {list(SUPPORTED_POSITION_MODES)}"
        )
    return ids


def _recompute_strategy_return(frame: pd.DataFrame) -> pd.DataFrame:
    """strategy_return[t] = position[t-1] * asset_return[t] within each asset."""
    out = frame.copy()
    pieces: list[pd.DataFrame] = []
    for _, group in out.groupby("asset_id", sort=False):
        g = group.sort_values("timestamp").copy()
        pos = g["position"].astype(float).fillna(0.0)
        ret = g["asset_return"].astype(float)
        g["strategy_return"] = pos.shift(1) * ret
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def apply_position_mode(
    signal_frame: pd.DataFrame,
    position_mode: str,
    *,
    overlay_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Transform raw long/short signal positions into the requested exposure mode.

    - ``long_short``: keep +1 / 0 / -1
    - ``long_only``: clip at 0 → 0 / +1
    - ``market_overlay``: position = 1 + overlay_scale * raw_signal
      (buy-and-hold baseline plus signed overlay; scale=1 → 0 / 1 / 2)
    """
    mode = str(position_mode)
    if mode not in SUPPORTED_POSITION_MODES:
        raise ValueError(
            f"Unsupported position_mode {mode!r}; choose from {list(SUPPORTED_POSITION_MODES)}"
        )

    required = {"timestamp", "asset_id", "position", "asset_return"}
    missing = required - set(signal_frame.columns)
    if missing:
        raise ValueError(f"signal_frame missing columns: {sorted(missing)}")

    if mode == "long_short":
        out = signal_frame.copy()
        out["position_mode"] = mode
        return out

    out = signal_frame.copy()
    raw = out["position"].astype(float).fillna(0.0)

    if mode == "long_only":
        out["position"] = np.maximum(raw.to_numpy(dtype=float), 0.0)
    else:  # market_overlay
        if overlay_scale < 0:
            raise ValueError("overlay_scale must be >= 0")
        out["position"] = 1.0 + float(overlay_scale) * raw.to_numpy(dtype=float)

    out = _recompute_strategy_return(out)
    out["position_mode"] = mode
    return out
