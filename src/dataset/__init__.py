"""CPO training-table construction (SPO parameter and PWO weight tables)."""

from .calendar import (
    build_event_dates,
    decision_dates,
    parse_span,
    sample_dates,
    sample_start_date,
    schedule_dates,
)
from .pwo_builder import build_pwo_training_table
from .spo_builder import build_spo_training_table
from .weights import resolve_weights

# Backward-compatible alias
build_training_table = build_spo_training_table

__all__ = [
    "build_event_dates",
    "build_pwo_training_table",
    "build_spo_training_table",
    "build_training_table",
    "decision_dates",
    "parse_span",
    "resolve_weights",
    "sample_dates",
    "sample_start_date",
    "schedule_dates",
]
