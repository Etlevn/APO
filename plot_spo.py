"""
SPO plot entry: read eval period CSV and write equity PNG only (no evaluation).

Usage (from project root):
    python plot_spo.py

Requires an existing ``PERIODS_CSV`` (typically produced earlier by ``eval_spo.py``).
Output name: spo_plot_{sleeves}_{positions}_{strategies}_{models}.png
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
# Plot-only filters (lists; [] = all in PERIODS_CSV). Edit top → bottom:
#   1 SLEEVE_IDS:    ["cpo", "baseline", "market"]     — which path roles
#   2 POSITION_IDS:  ["long_short", "long_only", "market_overlay"]
#   3 STRATEGY_IDS:  ["blend", "momentum", "trend", "mean_reversion"]
#   4 MODEL_IDS:     ["hgb", "ridge", "rf", "hist_mean"]  — cpo only
#   5 DATE_START/END: "YYYY-MM-DD" | None (= open)
# Colors: market=gray, baseline=black, hist_mean=red, other cpo=colored cycle.
# Output: spo_plot_{slv}_{pos}_{str}_{mdl}.png  (abbr; empty filter → "a")

PERIODS_CSV = config.OUTPUT_DIR / "spo_periods.csv"

SLEEVE_IDS: list[str] = ["cpo", "baseline", "market"]
POSITION_IDS: list[str] = ["market_overlay"]
STRATEGY_IDS: list[str] = ["blend"]
MODEL_IDS: list[str] = []  # [] = all cpo models in CSV, e.g. hgb/ridge/rf/hist_mean
DATE_START: str | None = None
DATE_END: str | None = None

# ======================================================================

import pandas as pd

from evaluate import save_spo_equity_plot

_SLEEVE_ABBR = {"cpo": "c", "baseline": "b", "market": "m"}
_POSITION_ABBR = {"long_short": "ls", "long_only": "lo", "market_overlay": "ov"}
_STRATEGY_ABBR = {"blend": "bl", "momentum": "mo", "trend": "tr", "mean_reversion": "mr"}
_MODEL_ABBR = {"hgb": "h", "ridge": "ri", "rf": "rf", "hist_mean": "hm"}


def _abbr_token(ids: list[str], mapping: dict[str, str]) -> str:
    """Join abbreviations in sorted id order; [] → 'a' (all)."""
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


def main() -> None:
    path = Path(PERIODS_CSV)
    if not path.is_file():
        raise FileNotFoundError(
            f"periods file not found: {path}. Run ``python eval_spo.py`` first."
        )
    periods = pd.read_csv(path)
    if periods.empty:
        raise ValueError(f"periods file is empty: {path}")

    out_path = equity_png_path(
        sleeve_ids=SLEEVE_IDS,
        position_ids=POSITION_IDS,
        strategy_ids=STRATEGY_IDS,
        model_ids=MODEL_IDS,
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_spo_equity_plot(
        periods,
        out_path,
        sleeve_ids=SLEEVE_IDS,
        position_ids=POSITION_IDS,
        strategy_ids=STRATEGY_IDS,
        model_ids=MODEL_IDS,
        date_start=DATE_START,
        date_end=DATE_END,
    )


if __name__ == "__main__":
    main()
