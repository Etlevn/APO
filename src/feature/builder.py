"""Point-in-time market features from a price panel."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .recipe import FEATURE_FUNCTIONS

DEFAULT_FEATURE_SPECS: tuple[dict[str, str | int], ...] = (
    {"column": "ret_1", "fn": "pct_change", "window": 1},
    {"column": "ret_5", "fn": "pct_change", "window": 5},
    {"column": "ret_21", "fn": "pct_change", "window": 21},
    {"column": "vol_21", "fn": "rolling_std", "window": 21},
    {"column": "ma_gap_21", "fn": "ma_gap", "window": 21},
)

FEATURE_COLUMNS = tuple(str(spec["column"]) for spec in DEFAULT_FEATURE_SPECS)


def resolve_feature_columns(feature_columns: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Use ``feature_columns`` when provided; otherwise the default five columns."""
    if feature_columns is None:
        return FEATURE_COLUMNS
    columns = [str(col) for col in feature_columns]
    if not columns:
        raise ValueError("feature_columns must contain at least one column.")
    duplicates = sorted({col for col in columns if columns.count(col) > 1})
    if duplicates:
        raise ValueError(f"feature_columns has duplicates: {duplicates}")
    return tuple(columns)


def _pct_change(px: pd.Series, window: int) -> pd.Series:
    return px.pct_change(window)


def _rolling_std(px: pd.Series, window: int) -> pd.Series:
    daily_ret = px.pct_change()
    return daily_ret.rolling(window, min_periods=window).std()


def _ma_gap(px: pd.Series, window: int) -> pd.Series:
    return px / px.rolling(window, min_periods=window).mean() - 1.0


_CALCULATORS = {
    "pct_change": _pct_change,
    "rolling_std": _rolling_std,
    "ma_gap": _ma_gap,
}


def normalize_feature_specs(
    specs: Sequence[dict] | None,
) -> list[dict[str, str | int]]:
    """Validate specs. ``None`` uses the built-in five features."""
    source = DEFAULT_FEATURE_SPECS if specs is None else specs
    if not source:
        raise ValueError("feature specs must contain at least one feature.")
    out: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for spec in source:
        if not isinstance(spec, dict):
            raise TypeError("each feature spec must be a mapping with column, fn, and window")
        missing = {"column", "fn", "window"} - set(spec)
        if missing:
            raise ValueError(f"feature spec missing {sorted(missing)}")
        column = str(spec["column"])
        fn = str(spec["fn"])
        window = int(spec["window"])
        if column in seen:
            raise ValueError(f"feature column {column!r} is listed more than once")
        if fn not in FEATURE_FUNCTIONS:
            raise ValueError(
                f"feature {column!r} fn {fn!r} is unknown. Choices: {list(FEATURE_FUNCTIONS)}"
            )
        if window < 1:
            raise ValueError(f"feature {column!r} window must be >= 1")
        seen.add(column)
        out.append({"column": column, "fn": fn, "window": window})
    return out


def build_features(
    panel: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    specs: Sequence[dict] | None = None,
) -> pd.DataFrame:
    """
    Build per-asset features using only prices available through each timestamp.

    ``specs`` comes from ``config/feature.yaml``. Each entry names the output
    column, the calculation (``pct_change``, ``rolling_std``, or ``ma_gap``),
    and the window. When omitted, the built-in five features are used.

    Output columns: timestamp, asset_id, availability_time, then the spec columns.
    """
    required = {"timestamp", "asset_id", price_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    feature_specs = normalize_feature_specs(specs)
    frames: list[pd.DataFrame] = []

    for asset_id, group in panel.sort_values("timestamp").groupby("asset_id", sort=False):
        g = group.reset_index(drop=True)
        px = g[price_col].astype(float)
        values: dict[str, object] = {
            "timestamp": pd.to_datetime(g["timestamp"]),
            "asset_id": asset_id,
            "availability_time": pd.to_datetime(
                g["availability_time"] if "availability_time" in g.columns else g["timestamp"]
            ),
        }
        for spec in feature_specs:
            calculator = _CALCULATORS[str(spec["fn"])]
            values[str(spec["column"])] = calculator(px, int(spec["window"]))
        frames.append(pd.DataFrame(values))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
    return out
