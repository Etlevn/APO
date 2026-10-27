"""Candidate generation: SPO parameter grids and PWO weight vectors."""

from .grid import PARAM_COLUMNS, build_candidates, iter_candidate_params
from .weights import (
    build_weight_candidates,
    iter_weight_vectors,
    resolve_dirichlet_alpha,
    weight_column,
)

__all__ = [
    "PARAM_COLUMNS",
    "build_candidates",
    "build_weight_candidates",
    "iter_candidate_params",
    "iter_weight_vectors",
    "resolve_dirichlet_alpha",
    "weight_column",
]
