"""
Manual preview entry for the feature module.

Usage (from project root):
    python script/run_2_feature.py

Fetches from yfinance and prints a terminal preview. Does not read or write cache.
Full production runs (complete grids / modes / models) are handled only by main.py.

Preview scope (no simplification vs main.py):
  - CANDIDATE:    (n/a — features only)
  - POSITION_RUN: (n/a)
  - MODEL_RUN:    (n/a)
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
from util import load_yaml


def main() -> None:
    feature_cfg = load_yaml(config.CONFIG_DIR / "default.yaml").get("feature", {})

    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    features = build_features(
        panel,
        price_col=config.PRICE_COL,
        windows=feature_cfg.get("windows"),
    )

    if feature_cfg.get("standardize", True):
        # Preview-only: fit on all non-null rows. Walk-forward must fit train-only.
        usable = features.dropna()
        features = FeaturePipeline().fit_transform(usable)

    print(features.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(features.tail(config.PREVIEW_ROWS).to_string(index=False))
    print()
    print(describe_panel(features, with_dtype=False).to_string())


if __name__ == "__main__":
    main()
