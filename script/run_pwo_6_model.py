"""
PWO preview entry for conditional ML weight selection.

Usage (from project root):
    python script/run_pwo_6_model.py

Fetches from yfinance, builds the PWO training table, runs walk-forward
selection, and prints a terminal preview. Does not read or write cache.
Caching is handled only by main_pwo.py.

Preview scope: only the first PWO_MODEL_RUN model; full weight-candidate set
unchanged (no POSITION_RUN in PWO). Full multi-model runs belong in main_pwo.py.
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
from dataset import build_pwo_training_table
from model import run_pwo_walkforward_selection
from util import load_yaml


def _first_model_id(run: str | list[str]) -> str:
    if isinstance(run, str):
        return run
    if not run:
        raise ValueError("config.PWO_MODEL_RUN must contain at least one model_id")
    return str(run[0])


def main() -> None:
    default_cfg = load_yaml(config.CONFIG_DIR / "default.yaml")
    preview_model = _first_model_id(config.PWO_MODEL_RUN)
    model_cfg = {**load_yaml(config.CONFIG_DIR / "model.yaml"), "run": preview_model}

    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    table = build_pwo_training_table(
        panel,
        feature_cfg=default_cfg.get("feature", {}),
        walkforward_cfg=config.WALKFORWARD,
        dataset_cfg=default_cfg.get("dataset", {}),
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        n_weight_candidates=config.PWO_N_WEIGHT_CANDIDATES,
        weight_seed=config.PWO_WEIGHT_SEED,
        include_equal_weight=config.PWO_INCLUDE_EQUAL_WEIGHT,
        dirichlet_alpha=config.PWO_DIRICHLET_ALPHA,
        show_progress=True,
    )
    selections = run_pwo_walkforward_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        model_cfg=model_cfg,
        show_progress=True,
    )

    if selections.empty:
        print("(empty — check WALKFORWARD / training-table coverage)")
        return
    print(selections.head(config.PREVIEW_ROWS).to_string(index=False))
    if len(selections) > config.PREVIEW_ROWS:
        print("\n...\n")
        print(selections.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
