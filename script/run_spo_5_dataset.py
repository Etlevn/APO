"""
SPO preview entry for the dataset module.

Usage (from project root):
    python script/run_spo_5_dataset.py

Fetches from yfinance and prints a terminal preview of the CPO training table,
then a hist_mean (non-ML) selection table with the same schema as
``spo_sel.csv``. Does not read or write cache / output files.

Preview scope: only the first POSITION_RUN mode; STRATEGY_RUN and full
CANDIDATE grids unchanged. Full multi-position runs belong in main_spo.py.
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
from dataset import build_spo_training_table
from model import run_spo_hist_mean_selection
from strategy import resolve_position_ids, select_candidate_cfg
from util import load_yaml


def main() -> None:
    default_cfg = load_yaml(config.CONFIG_DIR / "default.yaml")
    model_cfg = load_yaml(config.CONFIG_DIR / "model.yaml")
    candidate_cfg = select_candidate_cfg(config.CANDIDATE, config.STRATEGY_RUN)
    preview_position = resolve_position_ids(config.POSITION_RUN)[0]

    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    table = build_spo_training_table(
        panel,
        candidate_cfg=candidate_cfg,
        feature_cfg=default_cfg.get("feature", {}),
        walkforward_cfg=config.WALKFORWARD,
        dataset_cfg=default_cfg.get("dataset", {}),
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.WEIGHTS,
        position_modes=preview_position,
        overlay_scale=config.POSITION_OVERLAY_SCALE,
        show_progress=True,
    )

    print(table.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(table.tail(config.PREVIEW_ROWS).to_string(index=False))
    print()
    print(describe_panel(table, with_dtype=False).to_string())

    selections = run_spo_hist_mean_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        min_train_rows=int(model_cfg.get("min_train_rows", 3)),
        show_progress=True,
    )
    print()
    if selections.empty:
        print("(empty — check WALKFORWARD / training-table coverage)")
        return
    print(selections.head(config.PREVIEW_ROWS).to_string(index=False))
    if len(selections) > config.PREVIEW_ROWS:
        print("\n...\n")
        print(selections.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
