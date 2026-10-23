"""
Evaluate entry: full backtest tables only (no plotting).

Usage (from project root):
    python eval.py

Prefers ``config.RUN_CONFIG_JSON`` written by ``main.py`` for panel / weights /
walkforward / candidate / overlay settings. Falls back to current ``config.py``
only if the snapshot is missing.

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
RUN_CONFIG_PATH = config.RUN_CONFIG_JSON
SUMMARY_CSV = config.OUTPUT_DIR / "eval_summary.csv"
PERIODS_CSV = config.OUTPUT_DIR / "eval_periods.csv"

# ======================================================================

from data import load_etf_panel_cached
from evaluate import run_evaluation
from util import load_main_run_config


def _resolve_run_settings() -> dict:
    """Load main snapshot when present; otherwise use live config.py."""
    if RUN_CONFIG_PATH.is_file():
        snap = load_main_run_config(RUN_CONFIG_PATH)
        print(f"eval: using run snapshot {RUN_CONFIG_PATH}")
        return {
            "tickers": snap["tickers"],
            "start": snap["start"],
            "end": snap.get("end"),
            "price_col": snap["price_col"],
            "weights": snap.get("weights"),
            "cache_format": snap.get("cache_format", config.CACHE_FORMAT),
            "walkforward": snap["walkforward"],
            "candidate": snap["candidate"],
            "overlay_scale": float(snap["position_overlay_scale"]),
        }

    print(
        f"eval: warning — {RUN_CONFIG_PATH} not found; "
        "falling back to current config.py (re-run main.py to write a snapshot)."
    )
    return {
        "tickers": config.TICKERS,
        "start": config.START,
        "end": config.END,
        "price_col": config.PRICE_COL,
        "weights": config.WEIGHTS,
        "cache_format": config.CACHE_FORMAT,
        "walkforward": config.WALKFORWARD,
        "candidate": config.CANDIDATE,
        "overlay_scale": config.POSITION_OVERLAY_SCALE,
    }


def main() -> None:
    settings = _resolve_run_settings()
    panel = load_etf_panel_cached(
        tickers=settings["tickers"],
        start=settings["start"],
        end=settings["end"],
        cache_dir=config.CACHE_DIR,
        cache_format=settings["cache_format"],
    )
    summary, periods = run_evaluation(
        panel,
        selections_path=SELECTIONS_PATH,
        walkforward_cfg=settings["walkforward"],
        candidate_cfg=settings["candidate"],
        price_col=settings["price_col"],
        tickers=settings["tickers"],
        weights=settings["weights"],
        overlay_scale=settings["overlay_scale"],
    )

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    periods.to_csv(PERIODS_CSV, index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
