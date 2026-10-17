"""
Manual preview entry for the dataset module.

Usage (from project root):
    python script/run_5_dataset.py

Fetches from yfinance and prints a terminal preview of the CPO training table.
Does not read or write cache. Caching is handled only by main.py.
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
from dataset import build_training_table
from strategy import select_candidate_cfg
from util import load_yaml


def main() -> None:
    default_cfg = load_yaml(config.CONFIG_DIR / "default.yaml")
    candidate_cfg = select_candidate_cfg(config.CANDIDATE, config.STRATEGY_RUN)

    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    table = build_training_table(
        panel,
        candidate_cfg=candidate_cfg,
        feature_cfg=default_cfg.get("feature", {}),
        walkforward_cfg=config.WALKFORWARD,
        dataset_cfg=default_cfg.get("dataset", {}),
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.WEIGHTS,
        position_modes=config.POSITION_RUN,
        overlay_scale=config.POSITION_OVERLAY_SCALE,
    )

    print(table.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(table.tail(config.PREVIEW_ROWS).to_string(index=False))
    print()
    print(describe_panel(table, with_dtype=False).to_string())


if __name__ == "__main__":
    main()
