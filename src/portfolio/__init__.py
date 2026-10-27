"""Fixed-weight asset returns and portfolio aggregation (shared by SPO/PWO)."""

from .returns import build_asset_returns, build_weighted_return_panel, portfolio_returns

__all__ = [
    "build_asset_returns",
    "build_weighted_return_panel",
    "portfolio_returns",
]
