"""
Manual preview entry for the data module.

Usage (from project root):
    python script/run_1_data.py

Fetches from yfinance and prints a terminal preview. Does not read or write cache.
Caching is handled only by main.py.
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


def main() -> None:
    panel = load_etf_panel(
        tickers=config.TICKERS,
        start=config.START,
        end=config.END,
    )

    print(panel.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(panel.tail(config.PREVIEW_ROWS).to_string(index=False))
    print()
    print(describe_panel(panel).to_string())


if __name__ == "__main__":
    main()
