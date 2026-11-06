"""Resolve which strategies to run from config."""

from __future__ import annotations

from collections.abc import Sequence

SUPPORTED_STRATEGIES = (
    "momentum",
    "trend",
    "mean_reversion",
    "ma_crossover",
    "breakout",
)


def resolve_strategy_ids(strategy_ids: str | Sequence[str]) -> list[str]:
    """Accept one id (``\"momentum\"``) or several (``[\"momentum\", \"trend\"]``)."""
    if isinstance(strategy_ids, str):
        ids = [strategy_ids]
    else:
        ids = [str(item) for item in strategy_ids]

    if not ids:
        raise ValueError("SPO_STRATEGY_RUN must contain at least one strategy_id.")

    unknown = [strategy_id for strategy_id in ids if strategy_id not in SUPPORTED_STRATEGIES]
    if unknown:
        raise ValueError(
            f"Unsupported strategy_id(s) {unknown}; choose from {list(SUPPORTED_STRATEGIES)}"
        )
    return ids


def select_candidate_cfg(
    candidate_cfg: dict,
    strategy_ids: str | Sequence[str],
) -> dict[str, dict]:
    """Filter SPO_CANDIDATE grids to the strategies listed in SPO_STRATEGY_RUN."""
    ids = resolve_strategy_ids(strategy_ids)
    missing = [strategy_id for strategy_id in ids if strategy_id not in candidate_cfg]
    if missing:
        raise KeyError(f"SPO_STRATEGY_RUN strategies missing from SPO_CANDIDATE: {missing}")
    return {strategy_id: candidate_cfg[strategy_id] for strategy_id in ids}
