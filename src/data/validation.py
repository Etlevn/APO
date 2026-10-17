"""Lightweight panel checks for the data module."""

from __future__ import annotations

import pandas as pd

from .loader import REQUIRED_COLUMNS


def validate_panel(panel: pd.DataFrame) -> list[str]:
    """Return a list of validation issues; empty means the panel looks usable."""
    issues: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        return issues

    if panel.empty:
        issues.append("panel is empty")
        return issues

    if panel.duplicated(subset=["timestamp", "asset_id"]).any():
        issues.append("duplicate (timestamp, asset_id) rows")

    if panel["timestamp"].isna().any() or panel["availability_time"].isna().any():
        issues.append("null timestamps present")

    price_cols = ["open", "high", "low", "close", "adj_close"]
    if (panel[price_cols] <= 0).any().any():
        issues.append("non-positive prices present")

    if panel["volume"].isna().any():
        issues.append("null volume present")

    unordered = (
        panel.sort_values(["asset_id", "timestamp"])
        .groupby("asset_id", sort=False)["timestamp"]
        .apply(lambda s: not s.is_monotonic_increasing)
    )
    if unordered.any():
        bad = unordered[unordered].index.tolist()
        issues.append(f"non-monotonic timestamps for assets: {bad}")

    return issues


def summarize_panel(panel: pd.DataFrame) -> dict[str, object]:
    """Compact summary for terminal previews."""
    return {
        "rows": int(len(panel)),
        "assets": sorted(panel["asset_id"].astype(str).unique().tolist()),
        "start": pd.Timestamp(panel["timestamp"].min()).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(panel["timestamp"].max()).strftime("%Y-%m-%d"),
        "columns": list(panel.columns),
        "null_counts": panel.isna().sum().to_dict(),
    }


def describe_panel(
    panel: pd.DataFrame,
    *,
    with_dtype: bool = True,
    **describe_kwargs: object,
) -> pd.DataFrame:
    """pandas describe with calendar-day formatting for datetime columns."""
    summary = panel.describe(**describe_kwargs).T
    if with_dtype:
        summary.insert(0, "dtype", panel.dtypes.reindex(summary.index).astype(str))
    day_stats = ("mean", "min", "25%", "50%", "75%", "max")
    for col in summary.index:
        if col not in panel.columns or not pd.api.types.is_datetime64_any_dtype(panel[col]):
            continue
        for stat in day_stats:
            if stat not in summary.columns:
                continue
            value = summary.at[col, stat]
            if pd.isna(value):
                continue
            summary.at[col, stat] = pd.Timestamp(value).strftime("%Y-%m-%d")
    return summary
