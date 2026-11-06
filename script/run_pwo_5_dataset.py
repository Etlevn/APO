"""
PWO preview entry for the dataset module.

Usage (from project root):
    python script/run_pwo_5_dataset.py

Fetches from yfinance and prints a terminal preview of the PWO training table,
then a hist_mean weight-selection table. Does not read or write cache / output files.
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
from dataset import build_pwo_training_table
from model import run_pwo_hist_mean_selection
from util import load_yaml


def main() -> None:
    model_cfg = load_yaml(config.CONFIG_DIR / "model.yaml")

    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    table = build_pwo_training_table(
        panel,
        feature_cfg={"specs": config.FEATURE_SPECS},
        walkforward_cfg=config.WALKFORWARD,
        dataset_cfg={"label_metric": config.LABEL_METRIC},
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        n_weight_candidates=config.PWO_N_WEIGHT_CANDIDATES,
        weight_seed=config.PWO_WEIGHT_SEED,
        include_equal_weight=config.PWO_INCLUDE_EQUAL_WEIGHT,
        dirichlet_alpha=config.PWO_DIRICHLET_ALPHA,
        feature_columns=config.PWO_FEATURE_COLUMNS,
        show_progress=True,
    )

    print(table.head(10).to_string(index=False))
    print("\n...\n")
    print(table.tail(10).to_string(index=False))
    print()
    print(describe_panel(table, with_dtype=False).to_string())

    selections = run_pwo_hist_mean_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        min_train_rows=int(model_cfg.get("min_train_rows", 3)),
        show_progress=True,
    )
    print()
    if selections.empty:
        print("(empty — check WALKFORWARD / training-table coverage)")
        return
    print(selections.head(10).to_string(index=False))
    if len(selections) > 10:
        print("\n...\n")
        print(selections.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
