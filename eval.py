"""
Evaluate entry: full backtest tables only (no plotting).

Usage (from project root):
    python eval.py

Writes ``SUMMARY_CSV`` and ``PERIODS_CSV`` under output/.
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

# ============================== [CONFIG] ==============================
# Full compute from SELECTIONS_PATH → SUMMARY_CSV + PERIODS_CSV (no plot).

SELECTIONS_PATH = config.SELECTIONS_CSV
SUMMARY_CSV = config.OUTPUT_DIR / "eval_summary.csv"
PERIODS_CSV = config.OUTPUT_DIR / "eval_periods.csv"

# ======================================================================

from data import load_etf_panel_cached
from evaluate import run_evaluation


def main() -> None:
    panel = load_etf_panel_cached(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
        cache_dir=config.CACHE_DIR,
        cache_format=config.CACHE_FORMAT,
    )
    summary, periods = run_evaluation(
        panel,
        selections_path=SELECTIONS_PATH,
        walkforward_cfg=config.WALKFORWARD,
        candidate_cfg=config.CANDIDATE,
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.WEIGHTS,
        overlay_scale=config.POSITION_OVERLAY_SCALE,
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    periods.to_csv(PERIODS_CSV, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
