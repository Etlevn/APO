"""Sample long-only asset weight candidates on the simplex."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weight_column(ticker: str) -> str:
    """Column name for a ticker weight, e.g. SPY → w_SPY."""
    return f"w_{ticker}"


def resolve_dirichlet_alpha(
    n_assets: int,
    alpha: float | list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Broadcast scalar α or validate a per-asset α vector (all entries > 0)."""
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1.")
    if isinstance(alpha, (float, int)):
        value = float(alpha)
        if value <= 0:
            raise ValueError(f"Dirichlet alpha must be > 0, got {value}.")
        return np.full(n_assets, value, dtype=float)

    arr = np.asarray(alpha, dtype=float).reshape(-1)
    if arr.size != n_assets:
        raise ValueError(
            f"Dirichlet alpha length {arr.size} must match n_assets={n_assets}."
        )
    if np.any(arr <= 0):
        raise ValueError(f"Dirichlet alpha entries must be > 0, got {arr.tolist()}.")
    return arr


def build_weight_candidates(
    tickers: list[str],
    *,
    n_candidates: int = 64,
    seed: int = 42,
    include_equal_weight: bool = True,
    dirichlet_alpha: float | list[float] | tuple[float, ...] | np.ndarray = 1.0,
) -> pd.DataFrame:
    """
    Build a finite table of long-only, fully invested weight vectors.

    Constraints per row: w_i >= 0 and sum_i w_i = 1.

    - Optional equal weight as ``weight_000`` (baseline).
    - Additional rows are i.i.d. Dirichlet(α) samples on the simplex.

    ``n_candidates`` counts random draws only (equal weight is extra when included).
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    if n_candidates < 0:
        raise ValueError("n_candidates must be >= 0.")

    tickers = [str(t) for t in tickers]
    n_assets = len(tickers)
    w_cols = [weight_column(t) for t in tickers]
    alpha = resolve_dirichlet_alpha(n_assets, dirichlet_alpha)
    rows: list[dict[str, object]] = []

    if include_equal_weight:
        share = 1.0 / n_assets
        row: dict[str, object] = {
            "candidate_id": "weight_000",
            "is_equal_weight": True,
        }
        for col in w_cols:
            row[col] = share
        rows.append(row)

    if n_candidates > 0:
        rng = np.random.default_rng(seed)
        samples = rng.dirichlet(alpha, size=n_candidates)
        start_idx = 1 if include_equal_weight else 0
        for i, weights in enumerate(samples):
            row = {
                "candidate_id": f"weight_{start_idx + i:03d}",
                "is_equal_weight": False,
            }
            for col, value in zip(w_cols, weights, strict=True):
                row[col] = float(value)
            rows.append(row)

    if not rows:
        raise ValueError("No weight candidates: enable equal weight or set n_candidates > 0.")

    frame = pd.DataFrame(rows)
    ordered = ["candidate_id", "is_equal_weight", *w_cols]
    frame = frame.loc[:, ordered]
    for col in w_cols:
        frame[col] = frame[col].astype(float)
    frame[w_cols] = frame[w_cols].div(frame[w_cols].sum(axis=1), axis=0)
    return frame.reset_index(drop=True)


def iter_weight_vectors(
    candidates: pd.DataFrame,
    tickers: list[str],
) -> list[tuple[str, dict[str, float]]]:
    """Return (candidate_id, {ticker: weight}) for each row."""
    tickers = [str(t) for t in tickers]
    out: list[tuple[str, dict[str, float]]] = []
    for row in candidates.to_dict(orient="records"):
        weights = {ticker: float(row[weight_column(ticker)]) for ticker in tickers}
        out.append((str(row["candidate_id"]), weights))
    return out
