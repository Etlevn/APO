"""
PWO evaluate entry: backtest tables + optional equity plot.

Usage (from project root):
    python eval_pwo.py

Writes ``SUMMARY_CSV`` / ``PERIODS_CSV`` (and optional ``DIAGNOSIS_CSV``).
When ``RUN_PLOT`` is True, also writes an equity PNG (same style as plot_spo).
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
# Full compute from SELECTIONS_PATH → SUMMARY / PERIODS (+ optional diagnosis / plot).

SELECTIONS_PATH = config.PWO_SELECTIONS_CSV
SUMMARY_CSV = config.OUTPUT_DIR / "pwo_summary.csv"
PERIODS_CSV = config.OUTPUT_DIR / "pwo_periods.csv"
DIAGNOSIS_CSV = config.OUTPUT_DIR / "pwo_diagnosis.csv"

RUN_DIAGNOSIS: bool = True
DIAG_MODEL_IDS: list[str] = []  # [] = all ML models in selections

RUN_PLOT: bool = True
# Plot filters (lists; [] = all). PWO has no position/strategy dims.
SLEEVE_IDS: list[str] = ["cpo", "baseline", "market"]
MODEL_IDS: list[str] = []  # [] = all cpo models; e.g. ["hgb", "hist_mean"]
DATE_START: str | None = None
DATE_END: str | None = None

# ======================================================================

from data import load_etf_panel_cached
from evaluate import (
    run_pwo_evaluation,
    run_pwo_ml_vs_hist_mean_diagnosis,
    save_pwo_equity_plot,
)

_SLEEVE_ABBR = {"cpo": "c", "baseline": "b", "market": "m"}
_MODEL_ABBR = {"hgb": "h", "ridge": "ri", "rf": "rf", "hist_mean": "hm"}


def _abbr_token(ids: list[str], mapping: dict[str, str]) -> str:
    if not ids:
        return "a"
    parts: list[str] = []
    for item in sorted(str(x) for x in ids):
        parts.append(mapping.get(item, item[:2]))
    return "".join(parts)


def equity_png_path(*, sleeve_ids: list[str], model_ids: list[str]) -> Path:
    stem = "_".join(
        [
            "pwo_plot",
            _abbr_token(sleeve_ids, _SLEEVE_ABBR),
            _abbr_token(model_ids, _MODEL_ABBR),
        ]
    )
    return config.OUTPUT_DIR / f"{stem}.png"


def _maybe_write_diagnosis(selections_path: Path) -> None:
    if not RUN_DIAGNOSIS:
        return

    preview = pd.read_csv(selections_path, usecols=["model_id"])
    models = set(preview["model_id"].astype(str).unique())
    if "hist_mean" not in models:
        print(
            "RUN_DIAGNOSIS=True but selections have no model_id='hist_mean'; "
            "skipping diagnosis (enable PWO_HIST_MEAN_RUN / re-run main_pwo)."
        )
        return

    diagnosis = run_pwo_ml_vs_hist_mean_diagnosis(
        selections_path,
        model_ids=DIAG_MODEL_IDS or None,
    )
    if diagnosis.empty:
        print("Diagnosis produced no rows; skipping pwo_diagnosis.csv.")
        return

    diagnosis.to_csv(DIAGNOSIS_CSV, index=False)
    print("\n=== ML vs hist_mean diagnosis ===\n")
    print(diagnosis.to_string(index=False))
    print(f"\nwrote {DIAGNOSIS_CSV}  rows={len(diagnosis)}")


def _maybe_write_plot(periods: pd.DataFrame) -> None:
    if not RUN_PLOT:
        return
    out_path = equity_png_path(sleeve_ids=SLEEVE_IDS, model_ids=MODEL_IDS)
    save_pwo_equity_plot(
        periods,
        out_path,
        sleeve_ids=SLEEVE_IDS,
        model_ids=MODEL_IDS or None,
        date_start=DATE_START,
        date_end=DATE_END,
    )
    print(f"wrote {out_path}")


def main() -> None:
    panel = load_etf_panel_cached(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
        cache_dir=config.CACHE_DIR,
        cache_format=config.CACHE_FORMAT,
    )
    summary, periods = run_pwo_evaluation(
        panel,
        selections_path=SELECTIONS_PATH,
        walkforward_cfg=config.WALKFORWARD,
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.WEIGHTS,
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    periods.to_csv(PERIODS_CSV, index=False)

    print(summary.to_string(index=False))

    _maybe_write_diagnosis(Path(SELECTIONS_PATH))
    _maybe_write_plot(periods)


if __name__ == "__main__":
    main()
