"""
Manual preview entry for the strategy module.

Usage (from project root):
    python script/run_3_strategy.py

Fetches from yfinance and prints a terminal preview. Does not read or write cache.
Caching is handled only by main.py.

Baseline params = first value of each CANDIDATE grid dimension.
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
from data import describe_panel, load_etf_panel
from strategy import baseline_strategy_cfg, run_strategies, select_candidate_cfg


def main() -> None:
    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    candidate_cfg = select_candidate_cfg(config.CANDIDATE, config.STRATEGY_RUN)
    strategy_cfg = baseline_strategy_cfg(candidate_cfg, price_col=config.PRICE_COL)
    signals = run_strategies(panel, strategy_cfg)

    print(signals.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(signals.tail(config.PREVIEW_ROWS).to_string(index=False))
    print()
    print(describe_panel(signals, with_dtype=False).to_string())


if __name__ == "__main__":
    main()
