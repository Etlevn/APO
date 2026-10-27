"""Backtest / evaluate walk-forward selection artifacts."""

from .diagnose import run_ml_vs_hist_mean_diagnosis, run_pwo_ml_vs_hist_mean_diagnosis
from .filter import filter_selections
from .pwo_plot import save_pwo_equity_plot
from .pwo_report import load_pwo_selections, run_pwo_evaluation
from .spo_plot import save_spo_equity_plot
from .spo_report import load_selections, run_evaluation

__all__ = [
    "filter_selections",
    "load_pwo_selections",
    "load_selections",
    "run_evaluation",
    "run_ml_vs_hist_mean_diagnosis",
    "run_pwo_evaluation",
    "run_pwo_ml_vs_hist_mean_diagnosis",
    "save_pwo_equity_plot",
    "save_spo_equity_plot",
]
