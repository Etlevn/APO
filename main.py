"""
Production entry: cached panel → CPO training table → walk-forward selection → CSV.

Usage (from project root):
    python main.py

Writes one table: config.SELECTIONS_CSV (decision_date × model × strategy picks).
When ``config.HIST_MEAN_RUN`` is True, also appends non-ML ``hist_mean`` rows.
Preview scripts under script/run_*.py fetch without caching and do not write output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from data import load_etf_panel_cached
from dataset import build_training_table
from model import run_hist_mean_selection, run_walkforward_selection
from strategy import select_candidate_cfg
from util import load_yaml


def main() -> None:
    default_cfg = load_yaml(config.CONFIG_DIR / "default.yaml")
    model_yaml = load_yaml(config.CONFIG_DIR / "model.yaml")
    model_cfg = {**model_yaml, "run": config.MODEL_RUN}
    candidate_cfg = select_candidate_cfg(config.CANDIDATE, config.STRATEGY_RUN)

    panel = load_etf_panel_cached(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
        cache_dir=config.CACHE_DIR,
        cache_format=config.CACHE_FORMAT,
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
        show_progress=True,
    )

    parts: list[pd.DataFrame] = []
    ml_selections = run_walkforward_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        model_cfg=model_cfg,
        show_progress=True,
    )
    if not ml_selections.empty:
        parts.append(ml_selections)

    if config.HIST_MEAN_RUN:
        hist_selections = run_hist_mean_selection(
            table,
            walkforward_cfg=config.WALKFORWARD,
            min_train_rows=int(model_yaml.get("min_train_rows", 3)),
            show_progress=True,
        )
        if not hist_selections.empty:
            parts.append(hist_selections)

    if not parts:
        raise RuntimeError("Walk-forward selection produced no rows; check WALKFORWARD / data coverage.")

    selections = pd.concat(parts, ignore_index=True).sort_values(
        ["model_id", "position_mode", "decision_date", "strategy_id"]
    ).reset_index(drop=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selections.to_csv(config.SELECTIONS_CSV, index=False)
    print(selections.head(config.PREVIEW_ROWS).to_string(index=False))
    if len(selections) > config.PREVIEW_ROWS:
        print("\n...\n")
        print(selections.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
