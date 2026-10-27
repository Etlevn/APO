"""
SPO preview entry for the candidate module.

Usage (from project root):
    python script/run_spo_4_candidate.py

Builds the SPO parameter grid from config.CANDIDATE and prints a terminal preview.
Does not fetch market data or use cache.
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
from candidate import build_candidates
from strategy import select_candidate_cfg


def main() -> None:
    candidate_cfg = select_candidate_cfg(config.CANDIDATE, config.STRATEGY_RUN)
    candidates = build_candidates(candidate_cfg)

    print(candidates.head(config.PREVIEW_ROWS).to_string(index=False))
    print("\n...\n")
    print(candidates.tail(config.PREVIEW_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
