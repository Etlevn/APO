"""Resolve which strategies to run from config."""

from __future__ import annotations

from collections.abc import Sequence

SUPPORTED_STRATEGIES = ("momentum", "trend", "mean_reversion")


def resolve_strategy_ids(strategy_ids: str | Sequence[str]) -> list[str]:
    """Accept one id (``\"momentum\"``) or several (``[\"momentum\", \"trend\"]``)."""
    if isinstance(strategy_ids, str):
        ids = [strategy_ids]
    else:
        ids = [str(item) for item in strategy_ids]

    if not ids:
        raise ValueError("STRATEGY_RUN must contain at least one strategy_id.")

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
    """Filter CANDIDATE grids to the strategies listed in STRATEGY_RUN."""
    ids = resolve_strategy_ids(strategy_ids)
    missing = [strategy_id for strategy_id in ids if strategy_id not in candidate_cfg]
    if missing:
        raise KeyError(f"STRATEGY_RUN strategies missing from CANDIDATE: {missing}")
    return {strategy_id: candidate_cfg[strategy_id] for strategy_id in ids}
