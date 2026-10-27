"""User-editable project settings (data, walk-forward, SPO / PWO selection).

Feature/dataset recipes: config/default.yaml
Estimator hyperparameters: config/model.yaml

Baseline SPO strategy params are not stored separately: use the first value of
each CANDIDATE grid dimension (see strategy.baseline_strategy_cfg).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"

# =============================================================================
# MODE
# =============================================================================
# Which research track production / preview scripts target.
#   "spo" — strategy-parameter conditional optimization
#   "pwo" — ETF-weight conditional optimization (asset-layer PWO)
MODE: str = "spo"  # "spo" | "pwo"

# =============================================================================
# Shared (SPO + PWO)
# =============================================================================

# yfinance ETF daily panel
TICKERS: list[str] = ["SPY", "TLT", "GLD"]  # "SPY" | "TLT" | "GLD"
START: str = "2014-12-31"
END: str | None = "2026-09-30"  # None = download through the latest available bar; yfinance end is exclusive

CACHE_DIR: Path = PROJECT_ROOT / "data"
CACHE_FORMAT: str = "parquet"  # parquet | csv

OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Price column used by features and strategies.
PRICE_COL: str = "adj_close"

# Preview controls used by script/run_*.py
PREVIEW_ROWS: int = 10

# Walk-forward. Span choices: "1y" | "6m" | "3m" | "1m" | "1w" | "1d"
# Labeled samples start at first_decision_date - train_window (warmup for fitting).
# Prefer rebalance_freq == label_horizon (hold window matches live trade horizon).
# sample_freq may equal rebalance_freq, or be denser (more train rows).
# Day-panel time semantics: README → "Time semantics".
#
# Common fund-style presets (copy values into WALKFORWARD below):
#   Annual  rebalance: train_window=2y, sample_freq=1m, rebalance_freq=1y, label_horizon=1y
#   Semi-annual:       train_window=2y, sample_freq=1m, rebalance_freq=6m, label_horizon=6m
#   Quarterly:         train_window=2y, sample_freq=1m, rebalance_freq=3m, label_horizon=3m
#   Monthly:           train_window=2y, sample_freq=1m, rebalance_freq=1m, label_horizon=1m
#   Weekly:            train_window=1y, sample_freq=1w, rebalance_freq=1w, label_horizon=1w
# first_decision_date must satisfy: first_decision_date - train_window >= earliest panel date.
WALKFORWARD: dict[str, str] = {
    "mode": "rolling",  # "rolling" | "expanding"
    "train_window": "2y",  # fit history length; "2y" | "1y" | "6m" | "3m" | "1m" | "1w" | "1d"
    "sample_freq": "1w",  # how often to build training rows (can be denser than rebalance)
    "rebalance_freq": "1m",  # how often to live-select / rebalance; same span choices
    "label_horizon": "1m",  # forward window for label y / OOS hold; same span choices
    "first_decision_date": "2017-01-01",  # first model selection date ("YYYY-MM-DD")
}

# =============================================================================
# SPO-specific (strategy parameter optimization)
# =============================================================================

# Walk-forward selection artifacts written by main_spo.py
SELECTIONS_CSV: Path = OUTPUT_DIR / "spo_sel.csv"

# Portfolio / cross-section weights across TICKERS when aggregating SPO features
# and strategy returns (renormalized to sum 1). None = equal weight.
# Example: {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}
WEIGHTS: dict[str, float] | None = None

# Which strategies to run. One id or several.
# Choices: "momentum" | "trend" | "mean_reversion"
STRATEGY_RUN: str | list[str] = ["momentum", "trend", "mean_reversion"]

# Position / exposure mode on top of strategy signals. One id or several.
# Choices: "long_short" | "long_only" | "market_overlay"
#   long_short     — raw ±1 / 0
#   long_only      — clip shorts: 0 / +1
#   market_overlay — position = 1 + OVERLAY_SCALE * raw_signal (buy-hold + overlay)
POSITION_RUN: str | list[str] = ["long_short", "long_only", "market_overlay"]
POSITION_OVERLAY_SCALE: float = 1.0  # used only by market_overlay

# Finite SPO parameter grids (edit lists to customize the search space).
# Unused parameters for a strategy should simply be omitted.
# Only strategies listed in STRATEGY_RUN are used.
# Baseline = first value of each list (via strategy.baseline_strategy_cfg).
CANDIDATE: dict[str, dict[str, list]] = {
    "momentum": {
        "lookback": [10, 20, 40, 60],
    },
    "trend": {
        "ma_window": [20, 50, 100, 200],
    },
    "mean_reversion": {
        "lookback": [10, 20, 40],
        "entry_z": [0.5, 1.0, 1.5],
    },
}

# Which conditional models to run for SPO. One id or several.
# Choices: "hgb" | "ridge" | "rf". Hyperparameters live in config/model.yaml.
MODEL_RUN: str | list[str] = ["hgb", "ridge", "rf"]

# Non-ML historical-mean candidate selection for SPO.
# When True, main_spo.py also writes model_id="hist_mean" rows into SELECTIONS_CSV.
HIST_MEAN_RUN: bool = True

# =============================================================================
# PWO-specific (portfolio weight optimization)
# =============================================================================
# Asset-layer conditional portfolio weights over TICKERS (not strategy sleeves).
# Training interface: (E_t, w) → forward portfolio performance of buy-and-hold
# under weight vector w. Candidates and full pipeline land in later PWO steps.

# Walk-forward selection artifacts written by main_pwo.py
PWO_SELECTIONS_CSV: Path = OUTPUT_DIR / "pwo_sel.csv"

# Conditional models for PWO weight selection (may mirror MODEL_RUN).
PWO_MODEL_RUN: str | list[str] = ["hgb", "ridge", "rf"]

# When True, main_pwo.py also writes model_id="hist_mean" rows into PWO_SELECTIONS_CSV.
PWO_HIST_MEAN_RUN: bool = True

# Weight candidate sampling (long-only simplex). Equal weight is optional baseline.
PWO_INCLUDE_EQUAL_WEIGHT: bool = True

# Number of random Dirichlet draws (excludes the equal-weight row when included).
PWO_N_WEIGHT_CANDIDATES: int = 99

# RNG seed for Dirichlet sampling.
PWO_WEIGHT_SEED: int = 42

# Dirichlet concentration α. Larger α → samples closer to the mean (equal weight
# when α is uniform). Smaller α → more corner / concentrated portfolios.
#   float → same α for every ticker (default 1.0 = uniform on the simplex)
#   list[float] → one α per TICKERS entry (must match length)
PWO_DIRICHLET_ALPHA: float | list[float] = 1.0
