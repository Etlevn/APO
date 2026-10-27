"""
PWO preview entry for the weight-candidate module.

Usage (from project root):
    python script/run_pwo_4_candidate.py

Builds long-only simplex weight candidates from config (equal weight + Dirichlet
samples) and prints a terminal preview. Does not fetch market data or use cache.
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
from candidate import build_weight_candidates


def main() -> None:
    candidates = build_weight_candidates(
        config.TICKERS,
        n_candidates=config.PWO_N_WEIGHT_CANDIDATES,
        seed=config.PWO_WEIGHT_SEED,
        include_equal_weight=config.PWO_INCLUDE_EQUAL_WEIGHT,
        dirichlet_alpha=config.PWO_DIRICHLET_ALPHA,
    )

    print(candidates.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(candidates.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
