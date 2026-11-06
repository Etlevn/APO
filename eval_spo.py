"""
SPO evaluate entry: backtest tables + optional equity plot.

Usage (from project root):
    python eval_spo.py

Writes ``SUMMARY_CSV`` / ``PERIODS_CSV`` (and optional ``DIAGNOSIS_CSV``).
When ``RUN_PLOT`` is True, also writes an equity PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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

SELECTIONS_PATH = config.SPO_SELECTIONS_CSV
SUMMARY_CSV = config.OUTPUT_DIR / "spo_summary.csv"
PERIODS_CSV = config.OUTPUT_DIR / "spo_periods.csv"
DIAGNOSIS_CSV = config.OUTPUT_DIR / "spo_diagnosis.csv"

RUN_DIAGNOSIS: bool = True
# Diagnosis filters (lists; [] = all in selections except hist_mean for models):
DIAG_MODEL_IDS: list[str] = []
DIAG_POSITION_IDS: list[str] = []
DIAG_STRATEGY_IDS: list[str] = []

RUN_PLOT: bool = True
# Plot filters (lists; [] = all in the periods table). Edit top → bottom:
#   1 SLEEVE_IDS:    ["cpo", "baseline", "market"]
#   2 POSITION_IDS:  ["long_short", "long_only", "market_overlay"]
#   3 STRATEGY_IDS:  ["blend", "momentum", "trend", "mean_reversion", "ma_crossover", "breakout"]
#   4 MODEL_IDS:     ["hgb", "ridge", "rf", "hist_mean"]  — cpo only
#   5 DATE_START/END: "YYYY-MM-DD" | None (= open)
# Colors: market=gray, baseline=black, hist_mean=red, other cpo=colored cycle.
# Output: spo_plot_{slv}_{pos}_{str}_{mdl}.png  (abbr; empty filter → "a")
SLEEVE_IDS: list[str] = ["cpo", "baseline", "market"]
POSITION_IDS: list[str] = ["market_overlay"]
STRATEGY_IDS: list[str] = ["blend"]
MODEL_IDS: list[str] = []  # [] = all cpo models, e.g. hgb/ridge/rf/hist_mean
DATE_START: str | None = None
DATE_END: str | None = None

# ======================================================================

from data import load_etf_panel_cached
from evaluate import run_evaluation, run_ml_vs_hist_mean_diagnosis, save_spo_equity_plot


def _print_table(frame: pd.DataFrame) -> None:
    print(frame.head(10).to_string(index=False))
    if len(frame) > 10:
        print("\n...\n")
        print(frame.tail(10).to_string(index=False))


_SLEEVE_ABBR = {"cpo": "c", "baseline": "b", "market": "m"}
_POSITION_ABBR = {"long_short": "ls", "long_only": "lo", "market_overlay": "ov"}
_STRATEGY_ABBR = {
    "blend": "bl",
    "momentum": "mo",
    "trend": "tr",
    "mean_reversion": "mr",
    "ma_crossover": "mc",
    "breakout": "bo",
}
_MODEL_ABBR = {"hgb": "h", "ridge": "ri", "rf": "rf", "hist_mean": "hm"}


def _abbr_token(ids: list[str], mapping: dict[str, str]) -> str:
    if not ids:
        return "a"
    parts: list[str] = []
    for item in sorted(str(x) for x in ids):
        parts.append(mapping.get(item, item[:2]))
    return "".join(parts)


def equity_png_path(
    *,
    sleeve_ids: list[str],
    position_ids: list[str],
    strategy_ids: list[str],
    model_ids: list[str],
) -> Path:
    stem = "_".join(
        [
            "spo_plot",
            _abbr_token(sleeve_ids, _SLEEVE_ABBR),
            _abbr_token(position_ids, _POSITION_ABBR),
            _abbr_token(strategy_ids, _STRATEGY_ABBR),
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
            "skipping diagnosis (enable SPO_HIST_MEAN_RUN in config.py / re-run main_spo)."
        )
        return

    diagnosis = run_ml_vs_hist_mean_diagnosis(
        selections_path,
        model_ids=DIAG_MODEL_IDS or None,
        position_ids=DIAG_POSITION_IDS or None,
        strategy_ids=DIAG_STRATEGY_IDS or None,
    )
    if diagnosis.empty:
        print("Diagnosis produced no rows; skipping spo_diagnosis.csv.")
        return

    diagnosis.to_csv(DIAGNOSIS_CSV, index=False)
    print("\n=== ML vs hist_mean diagnosis ===\n")
    _print_table(diagnosis)


def _maybe_write_plot(periods: pd.DataFrame) -> None:
    if not RUN_PLOT:
        return
    out_path = equity_png_path(
        sleeve_ids=SLEEVE_IDS,
        position_ids=POSITION_IDS,
        strategy_ids=STRATEGY_IDS,
        model_ids=MODEL_IDS,
    )
    save_spo_equity_plot(
        periods,
        out_path,
        sleeve_ids=SLEEVE_IDS or None,
        position_ids=POSITION_IDS or None,
        strategy_ids=STRATEGY_IDS or None,
        model_ids=MODEL_IDS or None,
        date_start=DATE_START,
        date_end=DATE_END,
    )


def main() -> None:
    panel = load_etf_panel_cached(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
        set_id=config.TICKER_SET,
        cache_dir=config.CACHE_DIR,
        cache_format=config.CACHE_FORMAT,
    )
    summary, periods = run_evaluation(
        panel,
        selections_path=SELECTIONS_PATH,
        walkforward_cfg=config.WALKFORWARD,
        candidate_cfg=config.SPO_CANDIDATE,
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=config.SPO_WEIGHTS,
        overlay_scale=config.SPO_POSITION_OVERLAY_SCALE,
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    periods.to_csv(PERIODS_CSV, index=False)

    _print_table(summary)

    _maybe_write_diagnosis(Path(SELECTIONS_PATH))
    _maybe_write_plot(periods)


if __name__ == "__main__":
    main()
