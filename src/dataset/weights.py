"""Resolve and validate portfolio / cross-sectional asset weights."""

from __future__ import annotations


def resolve_weights(
    tickers: list[str],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Build per-ticker weights that sum to 1.

    ``weights=None`` → equal weight across ``tickers``.
    Otherwise every ticker must appear; values are renormalized to sum to 1.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")

    if weights is None:
        share = 1.0 / len(tickers)
        return {ticker: share for ticker in tickers}

    missing = [ticker for ticker in tickers if ticker not in weights]
    if missing:
        raise KeyError(f"WEIGHTS missing tickers: {missing}")
    extra = [ticker for ticker in weights if ticker not in tickers]
    if extra:
        raise KeyError(f"WEIGHTS has unknown tickers not in TICKERS: {extra}")

    total = float(sum(float(weights[ticker]) for ticker in tickers))
    if total <= 0:
        raise ValueError("WEIGHTS must sum to a positive value.")
    return {ticker: float(weights[ticker]) / total for ticker in tickers}
