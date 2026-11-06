"""
SPO preview entry for the feature module.

Usage (from project root):
    python script/run_spo_2_feature.py

Fetches from yfinance and prints a terminal preview. Does not read or write cache.
Caching is handled only by main_spo.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from data import describe_panel, load_etf_panel
from feature import FeaturePipeline, build_features


def main() -> None:
    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    features = build_features(
        panel,
        price_col=config.PRICE_COL,
        specs=config.FEATURE_SPECS,
    )

    # Preview-only: fit on all non-null rows. Walk-forward must fit train-only.
    usable = features.dropna()
    features = FeaturePipeline(config.SPO_FEATURE_COLUMNS).fit_transform(usable)

    print(features.head(10).to_string(index=False))
    print("\n...\n")
    print(features.tail(10).to_string(index=False))
    print()
    print(describe_panel(features, with_dtype=False).to_string())


if __name__ == "__main__":
    main()
