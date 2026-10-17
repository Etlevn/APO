# APO

**Adaptive Portfolio Optimization Using Machine Learning for Market Regime Dynamics**

APO is a modular research pipeline for **conditional portfolio optimization (CPO)**: at each decision date, market features observable at that time are used to predict the forward performance of candidate strategy-parameter choices, and the best candidate is selected out of sample. Strategy-parameter optimization (SPO) shares one data, labeling, walk-forward, and evaluation stack so modules can be developed and compared independently.

## What it does

1. **Data** — Download and cache a multi-ETF daily panel (default: SPY, TLT, GLD via yfinance).
2. **Features & labels** — Build a CPO training table: market features \(E_t\) paired with candidate parameters \(x\) and forward performance labels \(y\).
3. **Walk-forward selection** — Fit conditional models (HistGradientBoosting, Ridge, Random Forest) without look-ahead; at each rebalance date pick the candidate with the highest predicted label (\(\arg\max_x \hat F(E_t, x)\)).
4. **Position modes** — Apply long/short, long-only, or market-overlay exposure on top of strategy signals.
5. **Evaluation & plots** — Backtest CPO vs a fixed-parameter baseline vs buy-and-hold market; write summary/period CSVs and equity charts.

## Requirements

- Python ≥ 3.11
- Dependencies listed in `pyproject.toml` (pandas, numpy, scikit-learn, yfinance, matplotlib, …)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick start

Edit settings in `config.py` (tickers, walk-forward calendar, strategies, models, position modes). Feature recipes and estimator hyperparameters live under `config/`.

```bash
# 1) Cached panel → training table → walk-forward selections
python main.py
# → output/main_selections.csv

# 2) Full backtest tables (CPO / baseline / market)
python eval.py
# → output/eval_summary.csv, output/eval_periods.csv

# 3) Equity curves from eval periods (edit filters at the top of plot.py)
python plot.py
# → output/plot_*.png
```

Preview scripts under `script/` fetch data without writing the production selection CSV.

## Configuration highlights

| Setting                      | Role                                                                                           |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| `WALKFORWARD`                | Rolling/expanding mode, train length, sample / rebalance / label horizons, first decision date |
| `STRATEGY_RUN` / `CANDIDATE` | Which SPO strategies and parameter grids to search                                             |
| `MODEL_RUN`                  | Conditional estimators (`hgb`, `ridge`, `rf`)                                                  |
| `POSITION_RUN`               | `long_short`, `long_only`, `market_overlay`                                                    |
| `WEIGHTS`                    | Cross-sectional ETF weights (`None` = equal weight)                                            |

Baseline strategy parameters are the **first value** of each list in `CANDIDATE`. Prefer `rebalance_freq == label_horizon` so the hold window matches the live trade horizon; `sample_freq` may be denser to enlarge the training set.

## Repository layout

```
config.py              # User-editable run settings
config/                # YAML feature / model recipes
main.py                # Production selection pipeline
eval.py                # Backtest → summary + period CSVs
plot.py                # Equity plots from eval periods
src/
  data/                # Panel download & cache
  dataset/             # CPO training-table builder
  feature/             # Market features
  strategy/            # Momentum, trend, mean-reversion; position modes
  model/               # Walk-forward fit & selection
  evaluate/            # Paths, metrics, filters, plotting
  util/                # Shared helpers
script/                # Interactive / preview runners
output/                # Selections, eval tables, plots
data/                  # Local panel cache
```

## Methods (brief)

- **CPO interface:** \((E_t, x) \mapsto\) predicted forward performance; selection is \(\arg\max_x \hat F(E_t, x)\).
- **Sleeves in evaluation:** `cpo` (model-selected parameters), `baseline` (fixed first-grid parameters), `market` (buy-and-hold on the weighted ETF basket).

### Time semantics (day panel)

All clocks are **calendar dates** (normalized), not intraday timestamps. The pipeline keeps one stable convention; CSV products do **not** duplicate every conceptual clock as its own column.

| Conceptual clock            | Rule in code                                                                                                                                                                              | In artifacts                                                                                                                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Feature cutoff**          | Features use the latest panel row with `timestamp <= decision_date` (as-of). If no complete as-of feature vector exists, the sample is **dropped** (no look-ahead into the label window). | Not a column; only feature values are stored.                                                                                                                                                                            |
| **Decision**                | Live selection / sample anchor date.                                                                                                                                                      | `decision_date` in `main_selections.csv` and `eval_periods.csv`.                                                                                                                                                         |
| **Execution / label start** | Realized returns and hold P&L use bars strictly after the decision: `(decision_date, label_end]`.                                                                                         | Implied by that open interval; no separate `execution_time`. Hold end is `label_end` in `eval_periods.csv` (and in the in-memory training table).                                                                        |
| **Train window**            | For a live decision \(t\): rows with `decision_date < t` and `label_end <= t` (labels fully realized). Rolling mode also requires `decision_date >= t - train_window`.                    | Not written to selection CSV; `train_rows` is the count. In-memory training table has `train_start` (= sample `decision_date - train_window`) but no `train_end` column (`train_end` ≡ live \(t\) under the mask above). |

**Anti-leakage summary:** at decision \(t\), features and model inputs are observable on or before \(t\); incomplete warm-up rows are discarded rather than filled from the future; training labels must have finished by \(t\); evaluation/selection P&L starts after \(t\`.

Local directories `data/` (panel cache) and `output/` (run artifacts) are generated by the pipeline and are not part of the source tree for distribution.

## Acknowledgments

This codebase implements a research pipeline inspired by conditional portfolio optimization ideas. In particular:

1. **Chan, Ernest, et al. (2023).** _Conditional Portfolio Optimization: Using Machine Learning to Adapt Capital Allocations to Market Regimes._ Available on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4383184) (abstract 4383184). Primary reference for the CPO interface \((E_t, x) \mapsto\) predicted forward performance and regime-adaptive selection.
2. **Chan, Ernest.** Material from _Algorithmic Trading_ (book chapter excerpt circulating as _CPO From Algorithmic Trading_), illustrating regime-aware **strategy-parameter** optimization (SPO) on mean-reversion-style examples.

Any errors or deviations in this open-source implementation are the authors’ alone; please cite the original works above when referring to the underlying methodology.

## License

This project is released under the [MIT License](LICENSE).
