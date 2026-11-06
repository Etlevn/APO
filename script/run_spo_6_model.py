"""
SPO preview entry for the model module.

Usage (from project root):
    python script/run_spo_6_model.py

Fetches from yfinance, builds the CPO training table, runs walk-forward
selection, and prints a terminal preview. Does not read or write cache.
Caching is handled only by main_spo.py.

Preview scope: first SPO_POSITION_RUN mode and first SPO_MODEL_RUN model;
SPO_STRATEGY_RUN and full SPO_CANDIDATE grids unchanged. Full runs belong in main_spo.py.
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
from data import load_etf_panel
from dataset import build_spo_training_table
from model import resolve_model_ids, run_spo_walkforward_selection
from strategy import resolve_position_ids, select_candidate_cfg
from util import load_yaml


def main() -> None:
    model_yaml = load_yaml(config.CONFIG_DIR / "model.yaml")
    preview_model = resolve_model_ids({**model_yaml, "run": config.SPO_MODEL_RUN})[0]
    model_cfg = {**model_yaml, "run": preview_model}
    candidate_cfg = select_candidate_cfg(config.SPO_CANDIDATE, config.SPO_STRATEGY_RUN)
    preview_position = resolve_position_ids(config.SPO_POSITION_RUN)[0]

    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    table = build_spo_training_table(
        panel,
        candidate_cfg=candidate_cfg,
        feature_cfg={"specs": config.FEATURE_SPECS},
        walkforward_cfg=config.WALKFORWARD,
        dataset_cfg={"label_metric": config.LABEL_METRIC},
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.SPO_WEIGHTS,
        position_modes=preview_position,
        overlay_scale=config.SPO_POSITION_OVERLAY_SCALE,
        feature_columns=config.SPO_FEATURE_COLUMNS,
        show_progress=True,
    )
    selections = run_spo_walkforward_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        model_cfg=model_cfg,
        feature_columns=config.SPO_FEATURE_COLUMNS,
        show_progress=True,
    )

    print(selections.head(10).to_string(index=False))
    print("\n...\n")
    print(selections.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
