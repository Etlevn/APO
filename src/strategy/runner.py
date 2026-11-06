"""Run the Stage 0 baseline strategy suite with fixed parameters."""

from __future__ import annotations

import pandas as pd

from .breakout import run_breakout
from .ma_crossover import run_ma_crossover
from .mean_reversion import run_mean_reversion
from .momentum import run_momentum
from .trend import run_trend

_RUNNERS = {
    "momentum": lambda panel, cfg, price_col: run_momentum(
        panel,
        lookback=int(cfg.get("lookback", 20)),
        price_col=price_col,
    ),
    "trend": lambda panel, cfg, price_col: run_trend(
        panel,
        ma_window=int(cfg.get("ma_window", 50)),
        price_col=price_col,
    ),
    "mean_reversion": lambda panel, cfg, price_col: run_mean_reversion(
        panel,
        lookback=int(cfg.get("lookback", 20)),
        entry_z=float(cfg.get("entry_z", 1.0)),
        price_col=price_col,
    ),
    "ma_crossover": lambda panel, cfg, price_col: run_ma_crossover(
        panel,
        fast_window=int(cfg.get("fast_window", 20)),
        slow_window=int(cfg.get("slow_window", 50)),
        price_col=price_col,
    ),
    "breakout": lambda panel, cfg, price_col: run_breakout(
        panel,
        lookback=int(cfg.get("lookback", 20)),
        price_col=price_col,
    ),
}


def run_candidate(
    panel: pd.DataFrame,
    strategy_id: str,
    params: dict,
    *,
    price_col: str,
) -> pd.DataFrame:
    """Run one strategy at one parameter point."""
    try:
        runner = _RUNNERS[strategy_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported strategy_id: {strategy_id!r}") from exc
    return runner(panel, params, price_col)


def run_strategies(panel: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    """
    Execute strategies present in ``strategy_cfg`` (typically from baseline_strategy_cfg).

    Returns a long panel:
    timestamp, asset_id, strategy_id, position, asset_return, strategy_return
    """
    price_col = strategy_cfg.get("price_col", "adj_close")
    frames: list[pd.DataFrame] = []
    for strategy_id, runner in _RUNNERS.items():
        if strategy_id not in strategy_cfg:
            continue
        cfg = strategy_cfg.get(strategy_id) or {}
        if not isinstance(cfg, dict):
            raise TypeError(f"strategy_cfg[{strategy_id!r}] must be a mapping")
        frames.append(runner(panel, cfg, price_col))

    if not frames:
        raise ValueError("strategy_cfg contains no runnable strategies.")

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["timestamp", "strategy_id", "asset_id"]).reset_index(drop=True)
