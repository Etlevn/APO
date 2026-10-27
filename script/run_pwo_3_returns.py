"""
PWO preview entry for asset / fixed-weight portfolio returns.

Usage (from project root):
    python script/run_pwo_3_returns.py

Fetches from yfinance and prints a terminal preview. Does not read or write cache.
Caching is handled only by main_pwo.py.

Baseline weights = equal weight across TICKERS (no trading strategy).
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
from portfolio import build_weighted_return_panel


def main() -> None:
    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )
    # Equal-weight buy-and-hold contributions (PWO baseline engine, not SPO strategies).
    returns = build_weighted_return_panel(
        panel,
        price_col=config.PRICE_COL,
        tickers=config.TICKERS,
        weights=None,
    )

    print(returns.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(returns.tail(config.PREVIEW_ROWS).to_string(index=False))
    print()
    print(describe_panel(returns, with_dtype=False).to_string())


if __name__ == "__main__":
    main()
