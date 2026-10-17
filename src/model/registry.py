"""Build sklearn conditional-performance estimators from config."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

SUPPORTED_MODELS = ("hgb", "ridge", "rf")


def resolve_model_ids(
    model_cfg: dict,
    model_ids: str | Sequence[str] | None = None,
) -> list[str]:
    """
    Resolve which model_ids to run.

    Accepts a single id (``\"hgb\"``) or a list (``[\"hgb\", \"ridge\"]``).
    Resolution order: explicit ``model_ids`` → ``model_cfg[\"run\"]`` →
    ``model_cfg[\"candidates\"]`` (legacy).
    """
    if model_ids is None:
        if "run" not in model_cfg and "candidates" not in model_cfg:
            raise ValueError("MODEL.run must be set (one or more model_ids).")
        raw = model_cfg.get("run", model_cfg.get("candidates"))
    else:
        raw = model_ids

    if isinstance(raw, str):
        ids = [raw]
    else:
        ids = [str(item) for item in raw]

    if not ids:
        raise ValueError("MODEL.run must contain at least one model_id.")

    unknown = [model_id for model_id in ids if model_id not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(
            f"Unsupported model_id(s) {unknown}; choose from {list(SUPPORTED_MODELS)}"
        )
    return ids


def create_model(model_id: str, model_cfg: dict):
    """Instantiate a regressor; ``model_id`` in {hgb, ridge, rf}."""
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_id {model_id!r}; choose from {SUPPORTED_MODELS}")

    params = dict(model_cfg.get("params", {}).get(model_id, {}))
    random_state = model_cfg.get("random_state", 42)

    if model_id == "hgb":
        return HistGradientBoostingRegressor(random_state=random_state, **params)
    if model_id == "ridge":
        return Ridge(**params)
    return RandomForestRegressor(random_state=random_state, **params)
