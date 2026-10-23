"""
Production entry: cached panel → CPO training table → walk-forward selection → CSV.

Usage (from project root):
    python main.py

Writes:
  - config.SELECTIONS_CSV — decision_date × model × strategy picks (+ label_end)
  - config.RUN_CONFIG_JSON — snapshot of settings for reproducible eval.py

Preview scripts under script/run_*.py fetch without caching and do not write output/.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from data import load_etf_panel_cached
from dataset import build_training_table
from model import run_walkforward_selection
from strategy import select_candidate_cfg
from util import build_main_run_config, load_yaml, save_main_run_config


def main() -> None:
    default_cfg = load_yaml(config.CONFIG_DIR / "default.yaml")
    dataset_cfg = default_cfg.get("dataset", {})
    feature_cfg = default_cfg.get("feature", {})
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
        feature_cfg=feature_cfg,
        walkforward_cfg=config.WALKFORWARD,
        dataset_cfg=dataset_cfg,
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.WEIGHTS,
        position_modes=config.POSITION_RUN,
        overlay_scale=config.POSITION_OVERLAY_SCALE,
        show_progress=True,
    )

    selections = run_walkforward_selection(
        table,
        walkforward_cfg=config.WALKFORWARD,
        model_cfg=model_cfg,
        show_progress=True,
    )
    if selections.empty:
        raise RuntimeError("Walk-forward selection produced no rows; check WALKFORWARD / data coverage.")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selections.to_csv(config.SELECTIONS_CSV, index=False)
    save_main_run_config(
        config.RUN_CONFIG_JSON,
        build_main_run_config(
            tickers=config.TICKERS,
            start=config.START,
            end=config.END,
            price_col=config.PRICE_COL,
            weights=config.WEIGHTS,
            cache_format=config.CACHE_FORMAT,
            walkforward=config.WALKFORWARD,
            strategy_run=config.STRATEGY_RUN,
            position_run=config.POSITION_RUN,
            position_overlay_scale=config.POSITION_OVERLAY_SCALE,
            candidate=candidate_cfg,
            model_run=config.MODEL_RUN,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            model_cfg=model_cfg,
        ),
    )

    print(selections.head(config.PREVIEW_ROWS).to_string(index=False))
    if len(selections) > config.PREVIEW_ROWS:
        print("\n...\n")
        print(selections.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
