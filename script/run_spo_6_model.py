"""
SPO preview entry for the model module.

Usage (from project root):
    python script/run_spo_6_model.py

Fetches from yfinance, builds the CPO training table, runs walk-forward
selection, and prints a terminal preview. Does not read or write cache.
Caching is handled only by main_spo.py.

Preview scope: first POSITION_RUN mode and first MODEL_RUN model;
STRATEGY_RUN and full CANDIDATE grids unchanged. Full runs belong in main_spo.py.
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
    default_cfg = load_yaml(config.CONFIG_DIR / "default.yaml")
    model_yaml = load_yaml(config.CONFIG_DIR / "model.yaml")
    preview_model = resolve_model_ids({**model_yaml, "run": config.MODEL_RUN})[0]
    model_cfg = {**model_yaml, "run": preview_model}
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
    selections = run_spo_walkforward_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        model_cfg=model_cfg,
        show_progress=True,
    )

    print(selections.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(selections.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
