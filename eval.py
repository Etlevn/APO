"""
Evaluate entry: full backtest tables only (no plotting).

Usage (from project root):
    python eval.py

Writes ``SUMMARY_CSV`` and ``PERIODS_CSV`` under output/.
When ``RUN_DIAGNOSIS`` is True and selections include hist_mean, also writes
``DIAGNOSIS_CSV`` (ML vs hist_mean selection diagnosis).
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

# ============================== [CONFIG] ==============================
# Full compute from SELECTIONS_PATH → SUMMARY_CSV + PERIODS_CSV (no plot).
# Optional: RUN_DIAGNOSIS → DIAGNOSIS_CSV (ML vs hist_mean; no refit).

SELECTIONS_PATH = config.SELECTIONS_CSV
SUMMARY_CSV = config.OUTPUT_DIR / "eval_summary.csv"
PERIODS_CSV = config.OUTPUT_DIR / "eval_periods.csv"
DIAGNOSIS_CSV = config.OUTPUT_DIR / "eval_diagnosis.csv"

RUN_DIAGNOSIS: bool = True
# Diagnosis filters (lists; [] = all in selections except hist_mean for models):
DIAG_MODEL_IDS: list[str] = []
DIAG_POSITION_IDS: list[str] = []
DIAG_STRATEGY_IDS: list[str] = []

# ======================================================================

from data import load_etf_panel_cached
from evaluate import run_evaluation, run_ml_vs_hist_mean_diagnosis


def _maybe_write_diagnosis(selections_path: Path) -> None:
    if not RUN_DIAGNOSIS:
        return

    preview = pd.read_csv(selections_path, usecols=["model_id"])
    models = set(preview["model_id"].astype(str).unique())
    if "hist_mean" not in models:
        print(
            "RUN_DIAGNOSIS=True but selections have no model_id='hist_mean'; "
            "skipping eval_diagnosis.csv (enable HIST_MEAN_RUN in config.py / re-run main)."
        )
        return

    diagnosis = run_ml_vs_hist_mean_diagnosis(
        selections_path,
        model_ids=DIAG_MODEL_IDS or None,
        position_ids=DIAG_POSITION_IDS or None,
        strategy_ids=DIAG_STRATEGY_IDS or None,
    )
    if diagnosis.empty:
        print("Diagnosis produced no rows; skipping eval_diagnosis.csv.")
        return

    diagnosis.to_csv(DIAGNOSIS_CSV, index=False)
    print("\n=== ML vs hist_mean diagnosis ===\n")
    print(diagnosis.to_string(index=False))
    print(f"\nwrote {DIAGNOSIS_CSV}  rows={len(diagnosis)}")


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

    _maybe_write_diagnosis(Path(SELECTIONS_PATH))


if __name__ == "__main__":
    main()
