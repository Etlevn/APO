"""Resolve a named ETF universe from config/ticker.yaml."""

from __future__ import annotations

from pathlib import Path

from util.yaml_io import load_yaml


def resolve_ticker_set(set_id: str, path: Path | str) -> list[str]:
    """Return the ticker list for one preset in ``ticker.yaml``."""
    sets = load_yaml(path).get("sets", {})
    if set_id not in sets:
        known = ", ".join(sorted(str(key) for key in sets))
        raise KeyError(f"TICKER_SET {set_id!r} is not in {path}. Known: {known}")
    tickers = [str(item) for item in sets[set_id]]
    if not tickers:
        raise ValueError(f"TICKER_SET {set_id!r} is empty.")
    duplicates = sorted({item for item in tickers if tickers.count(item) > 1})
    if duplicates:
        raise ValueError(f"TICKER_SET {set_id!r} has duplicate tickers: {duplicates}")
    return tickers
