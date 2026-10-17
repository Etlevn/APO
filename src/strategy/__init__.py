"""Baseline trading strategies for Stage 0."""

from .baseline import baseline_strategy_cfg
from .mean_reversion import run_mean_reversion
from .momentum import run_momentum
from .position_mode import (
    SUPPORTED_POSITION_MODES,
    apply_position_mode,
    resolve_position_ids,
)
from .runner import run_strategies
from .selection import SUPPORTED_STRATEGIES, resolve_strategy_ids, select_candidate_cfg
from .trend import run_trend

__all__ = [
    "SUPPORTED_POSITION_MODES",
    "SUPPORTED_STRATEGIES",
    "apply_position_mode",
    "baseline_strategy_cfg",
    "resolve_position_ids",
    "resolve_strategy_ids",
    "run_mean_reversion",
    "run_momentum",
    "run_strategies",
    "run_trend",
    "select_candidate_cfg",
]
