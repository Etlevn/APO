"""SPO parameter candidate generation."""

from .grid import PARAM_COLUMNS, build_candidates, iter_candidate_params

__all__ = [
    "PARAM_COLUMNS",
    "build_candidates",
    "iter_candidate_params",
]
