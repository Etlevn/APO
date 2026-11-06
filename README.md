# APO

**Adaptive Portfolio Optimization Using Machine Learning for Market Regime Dynamics**

APO is a modular research framework inspired by **conditional portfolio optimization (CPO)**. At each live decision date, a model uses only information available at that time to estimate the forward performance of a finite set of candidate decisions and selects the candidate with the highest predicted label:

$$
(E_t, x) \mapsto \widehat{F}(E_t, x),
\qquad
x_t^* = \arg\max_x \widehat{F}(E_t, x).
$$

The repository implements two independently executable research tracks on one shared data, feature, labeling, walk-forward, and evaluation stack:

- **Strategy Parameter Optimization (SPO):** \(x=p\), where the model selects parameters for each trading strategy and position mode.
- **Portfolio Weight Optimization (PWO):** \(x=w\), where the model selects a long-only ETF weight vector.

The current implementation is a research baseline for transparent, reproducible experimentation. Reported returns are gross of transaction costs and slippage; the included assets, strategies, features, models, and candidate grids are starting points rather than claims of optimality.

## Pipeline

1. **Data** — Download and cache a daily adjusted-close ETF panel from Yahoo Finance. Select a preset universe containing 3, 5, 10, 25, 50, or 100 ETFs.
2. **Features** — Build point-in-time market features observable on or before each sample date.
3. **Candidates** — Expand finite SPO parameter grids or sample PWO weight vectors.
4. **Labels** — Pair each market state and candidate with its realized performance over the following hold period.
5. **Walk-forward selection** — Fit HGB, Ridge, and Random Forest models without look-ahead and compare them with an adaptive historical-mean selector.
6. **Evaluation** — Compare model-selected paths with fixed or equal-weight baselines and the market; write period tables, summary metrics, diagnostics, and equity plots.

## Requirements

- Python 3.11 or later
- Dependencies declared in `pyproject.toml`

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## Quick start

Choose an ETF preset and edit the shared walk-forward settings in `config.py`. Strategy grids, feature recipes, model hyperparameters, and ETF universes live under `config/`.

```bash
# 1) Cached panel -> training table -> walk-forward selections
python main_spo.py   # output/spo_sel.csv
python main_pwo.py   # output/pwo_sel.csv

# 2) Backtest tables, optional diagnosis, and optional equity plot
python eval_spo.py   # output/spo_summary.csv, spo_periods.csv, ...
python eval_pwo.py   # output/pwo_summary.csv, pwo_periods.csv, ...
```

For an explanatory, in-memory walkthrough, open:

- `notebook/demo_spo.ipynb`
- `notebook/demo_pwo.ipynb`

The notebooks call the same reusable functions as the production entrypoints. They display intermediate DataFrames and do not need to write intermediate CSV files.

Preview scripts under `script/` provide smaller terminal checks for individual modules:

- Shared data preview: `run_1_data.py`
- SPO previews: `run_spo_2_feature.py` through `run_spo_6_model.py`
- PWO previews: `run_pwo_2_feature.py` through `run_pwo_6_model.py`

Preview scripts do not write production selection files. Full configured runs belong in `main_spo.py` and `main_pwo.py`.

## Configuration

### Configuration files

| File | Purpose |
| --- | --- |
| `config.py` | Runtime mode, selected ETF preset, dates, walk-forward settings, enabled strategies/models/position modes, output paths, and PWO sampling controls |
| `config/ticker.yaml` | Nested ETF presets: `etf_3`, `etf_5`, `etf_10`, `etf_25`, `etf_50`, and `etf_100` |
| `config/feature.yaml` | Shared point-in-time feature definitions and lookback windows |
| `config/strategy.yaml` | SPO strategy parameter grids; the first value of each parameter list defines the fixed baseline |
| `config/model.yaml` | Estimator hyperparameters, train-fold standardization, random seed, and minimum training rows |

### Important runtime settings

| Setting | Role |
| --- | --- |
| `TICKER_SET` | Selects one ETF universe from `config/ticker.yaml` |
| `START`, `END`, `PRICE_COL` | Data range and price field (`adj_close` by default) |
| `LABEL_METRIC` | Candidate forward label (`cumulative_return` or `mean_return`) |
| `WALKFORWARD` | Rolling/expanding mode, training length, sample frequency, rebalance frequency, label horizon, and first live decision |
| `SPO_STRATEGY_RUN` | SPO strategies enabled for the run |
| `SPO_FEATURE_COLUMNS` | Columns from `feature.yaml` used by the SPO model |
| `SPO_POSITION_RUN` | `long_short`, `long_only`, and/or `market_overlay` |
| `SPO_MODEL_RUN` | Conditional SPO estimators: `hgb`, `ridge`, and/or `rf` |
| `PWO_FEATURE_COLUMNS` | Columns from `feature.yaml` used by the PWO model |
| `PWO_MODEL_RUN` | Conditional PWO estimators: `hgb`, `ridge`, and/or `rf` |
| `PWO_N_WEIGHT_CANDIDATES` | Number of random Dirichlet draws, excluding the optional equal-weight candidate |
| `PWO_DIRICHLET_ALPHA` | Shape of the long-only weight-candidate distribution |

The default walk-forward design uses weekly training samples, monthly live decisions, a one-month forward label, and a rolling two-year training window. `sample_freq` may be denser than `rebalance_freq`; keeping `rebalance_freq == label_horizon` aligns the live holding period with the target window.

### ETF universe presets

The presets are nested so larger universes retain the smaller core sets.

| Preset | Intended role |
| --- | --- |
| `etf_3` | Minimal baseline: SPY, TLT, GLD |
| `etf_5` | Adds developed ex-US equity and US aggregate bonds |
| `etf_10` | Main compact multi-asset research universe |
| `etf_25` | Medium research universe with styles, countries, bond sleeves, silver, and the dollar |
| `etf_50` | Broader regime study with sectors and country ETFs; holdings overlap |
| `etf_100` | High-dimensional stress test with greater overlap, liquidity variation, and commodity-futures effects |

Cache filenames include the preset identifier, date range, and ticker digest so different universes do not silently reuse one another.

### SPO candidates

The current strategy file contains five elementary price-based strategies:

| Strategy | Candidate parameters | Current candidates |
| --- | --- | ---: |
| `momentum` | Return `lookback` | 4 |
| `trend` | Moving-average `ma_window` | 4 |
| `mean_reversion` | `lookback` x `entry_z` | 9 |
| `ma_crossover` | `fast_window` x `slow_window` | 9 |
| `breakout` | Channel `lookback` | 3 |

The grids currently produce 29 SPO candidates. Multi-parameter grids are expanded as Cartesian products; `fast_window < slow_window` is enforced by the strategy implementation. Nearby parameter values often create highly correlated signals, so expanding a grid should prioritize distinct time scales rather than dense spacing.

The fixed-parameter SPO baseline uses the first value in each parameter list. Reordering or expanding a list can therefore change both candidate IDs and the baseline; selection and evaluation artifacts should be regenerated after any grid change.

### PWO candidates

PWO builds an optional equal-weight vector plus reproducible Dirichlet samples. Every candidate is fully invested and long-only:

$$
w_i \ge 0,
\qquad
\sum_i w_i = 1.
$$

The flat random candidate design is a transparent baseline, not a scalable optimizer. For the 25-, 50-, or 100-asset presets, a modest number of random draws covers the simplex very sparsely. Larger-universe research should consider substantially more candidates, structured benchmark portfolios, weight caps, sparse sampling, or hierarchical allocation by asset class.

## Baselines and evaluation

Both tracks include `hist_mean`, an adaptive non-ML selector. At each live decision date it uses the same eligible rolling history as the ML models, averages realized labels by candidate, and selects the candidate with the highest historical mean. This is a stronger test of conditional ML than comparison with a static baseline alone.

Evaluation sleeves are:

- **SPO `cpo`:** model-selected strategy parameters.
- **SPO `baseline`:** fixed first-grid strategy parameters.
- **PWO `cpo`:** model-selected ETF weights.
- **PWO `baseline`:** equal-weight ETF portfolio.
- **`market`:** buy-and-hold under `SPO_WEIGHTS`; with `SPO_WEIGHTS=None`, it is equal-weight.

Summary tables report cumulative return, annualized return, annualized volatility, Sharpe ratio, maximum drawdown, and hit rate. Diagnosis tables compare ML selections with `hist_mean`, including agreement rates and realized performance when their choices differ.

## Time semantics and anti-leakage rules

All clocks are normalized calendar dates, not intraday timestamps.

| Conceptual clock | Rule | Artifact representation |
| --- | --- | --- |
| **Feature cutoff** | Use the latest complete feature vector with `timestamp <= decision_date`; drop incomplete warm-up rows | Feature values only |
| **Decision** | Sample anchor or live selection date | `decision_date` |
| **Execution / label start** | Realized returns begin strictly after the decision | Open interval `(decision_date, label_end]` |
| **Train eligibility** | For live decision `t`, require `decision_date < t` and `label_end <= t` | `train_rows` records the eligible-row count |
| **Rolling window** | In rolling mode, also require `decision_date >= t - train_window` | Applied in memory during selection |

At decision `t`, feature inputs are observable on or before `t`, training labels are fully realized by `t`, train-fold standardization is fitted only on eligible rows, and selected-path P&L begins after `t`.

## Outputs

| Command | Main outputs |
| --- | --- |
| `python main_spo.py` | `spo_sel.csv` |
| `python eval_spo.py` | `spo_summary.csv`, `spo_periods.csv`, optional `spo_diagnosis.csv`, optional `spo_plot_*.png` |
| `python main_pwo.py` | `pwo_sel.csv` |
| `python eval_pwo.py` | `pwo_summary.csv`, `pwo_periods.csv`, optional `pwo_diagnosis.csv`, optional `pwo_plot_*.png` |

All files are written under `output/`. Local `data/` caches and `output/` artifacts are generated by the pipeline and are not source inputs for distribution.

## Repository layout

```text
config.py                  # User-editable runtime settings
config/
  ticker.yaml              # ETF universe presets
  feature.yaml             # Shared feature recipes
  strategy.yaml            # SPO parameter grids
  model.yaml               # Estimator hyperparameters
notebook/
  demo_spo.ipynb           # In-memory SPO walkthrough
  demo_pwo.ipynb           # In-memory PWO walkthrough
main_spo.py                # SPO production selection pipeline
main_pwo.py                # PWO production selection pipeline
eval_spo.py                # SPO evaluation, diagnosis, and optional plot
eval_pwo.py                # PWO evaluation, diagnosis, and optional plot
script/                    # Module-level preview runners
src/
  data/                    # Panel download, cache, validation, ticker presets
  feature/                 # Point-in-time market features
  candidate/               # SPO parameter grids and PWO weight candidates
  strategy/                # Five price-based strategies and position modes
  portfolio/               # Asset and weighted-portfolio returns
  dataset/                 # SPO/PWO supervised-table builders and calendar
  model/                   # Historical mean and walk-forward ML selection
  evaluate/                # Metrics, reports, diagnostics, filters, plots
  util/                    # YAML and progress helpers
data/                      # Generated local panel cache
output/                    # Generated selections, reports, and plots
```

## Acknowledgments

This research implementation is inspired by:

1. **Chan, Ernest, et al. (2023).** *Conditional Portfolio Optimization: Using Machine Learning to Adapt Capital Allocations to Market Regimes.* [SSRN abstract 4383184](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4383184).
2. **Chan, Ernest.** Material from *Algorithmic Trading* circulated as *CPO From Algorithmic Trading*, illustrating regime-aware conditional strategy-parameter optimization.

The papers motivate the conditional candidate-performance interface. The modular SPO/PWO software architecture, experiment presets, and any implementation errors or deviations are the responsibility of this project.

## License

This project is released under the [MIT License](LICENSE).
