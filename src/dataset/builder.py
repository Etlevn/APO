"""Build CPO training rows: market features + candidate params + forward label."""

from __future__ import annotations

import numpy as np
import pandas as pd

from candidate import PARAM_COLUMNS, build_candidates, iter_candidate_params
from feature import FEATURE_COLUMNS, build_features
from strategy import (
    apply_position_mode,
    resolve_position_ids,
    run_mean_reversion,
    run_momentum,
    run_trend,
)
from util import ProgressBar

from .calendar import (
    build_event_dates,
    decision_period_map,
    label_end_for_event,
    parse_span,
)
from .weights import resolve_weights


def _run_one_candidate(
    panel: pd.DataFrame,
    strategy_id: str,
    params: dict,
    *,
    price_col: str,
) -> pd.DataFrame:
    if strategy_id == "momentum":
        return run_momentum(panel, lookback=int(params["lookback"]), price_col=price_col)
    if strategy_id == "trend":
        return run_trend(panel, ma_window=int(params["ma_window"]), price_col=price_col)
    if strategy_id == "mean_reversion":
        return run_mean_reversion(
            panel,
            lookback=int(params["lookback"]),
            entry_z=float(params["entry_z"]),
            price_col=price_col,
        )
    raise ValueError(f"Unsupported strategy_id: {strategy_id!r}")


def _portfolio_returns(
    signal_frame: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Weighted average of per-asset strategy returns by timestamp."""
    frame = signal_frame.copy()
    frame["weight"] = frame["asset_id"].map(weights)
    if frame["weight"].isna().any():
        bad = sorted(frame.loc[frame["weight"].isna(), "asset_id"].astype(str).unique())
        raise KeyError(f"signal assets missing from WEIGHTS: {bad}")
    frame["weighted_return"] = frame["strategy_return"].astype(float) * frame["weight"]
    series = frame.groupby("timestamp", sort=True)["weighted_return"].sum().sort_index()
    series.index = pd.to_datetime(series.index)
    return series


def _asof_feature_vector(
    features: pd.DataFrame,
    decision_date: pd.Timestamp,
    *,
    weights: dict[str, float],
) -> pd.Series | None:
    available = features.loc[features["timestamp"] <= decision_date]
    if available.empty:
        return None
    asof_ts = available["timestamp"].max()
    slice_ = available.loc[
        available["timestamp"] == asof_ts, ["asset_id", *FEATURE_COLUMNS]
    ].copy()
    if slice_.empty:
        return None
    slice_["weight"] = slice_["asset_id"].map(weights)
    if slice_["weight"].isna().any():
        bad = sorted(slice_.loc[slice_["weight"].isna(), "asset_id"].astype(str).unique())
        raise KeyError(f"feature assets missing from WEIGHTS: {bad}")
    # Renormalize over assets present on this date.
    total = float(slice_["weight"].sum())
    if total <= 0:
        return None
    slice_["weight"] = slice_["weight"] / total
    weighted = slice_[list(FEATURE_COLUMNS)].multiply(slice_["weight"], axis=0).sum(axis=0)
    return weighted


def _feature_for_sample(
    features: pd.DataFrame,
    sample_date: pd.Timestamp,
    *,
    weights: dict[str, float],
) -> pd.Series | None:
    """As-of sample_date only; drop the sample if warm-up features are incomplete."""
    return _asof_feature_vector(features, sample_date, weights=weights)


def _forward_label(
    returns: pd.Series,
    decision_date: pd.Timestamp,
    label_end: pd.Timestamp,
    *,
    label_metric: str,
) -> tuple[float, int] | tuple[None, int]:
    """
    Label uses realized strategy returns strictly after the decision timestamp
    through the label horizon end: (decision_date, label_end].
    """
    window = returns.loc[(returns.index > decision_date) & (returns.index <= label_end)]
    window = window.dropna()
    if window.empty:
        return None, 0
    if label_metric == "cumulative_return":
        value = float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0)
    elif label_metric == "mean_return":
        value = float(window.mean())
    else:
        raise ValueError(f"Unsupported label_metric: {label_metric!r}")
    return value, int(len(window))


def build_training_table(
    panel: pd.DataFrame,
    *,
    candidate_cfg: dict,
    feature_cfg: dict,
    walkforward_cfg: dict,
    dataset_cfg: dict | None = None,
    price_col: str | None = None,
    tickers: list[str] | None = None,
    weights: dict[str, float] | None = None,
    position_modes: str | list[str] = "long_short",
    overlay_scale: float = 1.0,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Construct completed CPO samples.

    Training rows are built on ``sample_freq`` (plus live rebalance dates).
    Live model selection uses only ``rebalance_freq`` dates (is_model_decision).

    Sample dates begin at first_decision_date - train_window, so the first model
    decision already has a full train_window of realized (E, x, y) rows.

    Each row is one (decision_date, position_mode, strategy_id, candidate_id) with:
    - cross-sectional weighted features at the sample date (E_t)
    - candidate parameters (x)
    - forward portfolio performance label (y): live rows hold to the next
      formal rebalance boundary; train-only rows use label_horizon
    """
    dataset_cfg = dataset_cfg or {}
    price_col = price_col or feature_cfg.get("price_col", "adj_close")
    label_metric = dataset_cfg.get("label_metric", "cumulative_return")
    if tickers is None:
        tickers = sorted(panel["asset_id"].astype(str).unique().tolist())
    resolved_weights = resolve_weights(tickers, weights)
    modes = resolve_position_ids(position_modes)

    candidates = build_candidates(candidate_cfg)
    features = build_features(
        panel,
        price_col=price_col,
        windows=feature_cfg.get("windows"),
    )
    features = features.dropna(subset=list(FEATURE_COLUMNS)).copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"])

    panel_ts = pd.to_datetime(panel["timestamp"])
    max_ts = panel_ts.max()
    panel_start = panel_ts.min()
    trading_days = panel_ts
    first_decision = pd.Timestamp(walkforward_cfg["first_decision_date"]).normalize()
    rebalance_freq = walkforward_cfg["rebalance_freq"]
    sample_freq = walkforward_cfg.get("sample_freq", rebalance_freq)
    label_horizon = walkforward_cfg.get("label_horizon", rebalance_freq)
    event_dates, decision_set = build_event_dates(
        first_decision_date=first_decision,
        train_window=walkforward_cfg["train_window"],
        sample_freq=sample_freq,
        rebalance_freq=rebalance_freq,
        label_horizon=label_horizon,
        max_timestamp=max_ts,
        trading_days=trading_days,
        panel_start=panel_start,
    )
    if not event_dates:
        return pd.DataFrame()

    train_offset = parse_span(walkforward_cfg["train_window"])
    decision_end_by_date = decision_period_map(
        first_decision_date=first_decision,
        rebalance_freq=rebalance_freq,
        max_timestamp=max_ts,
        trading_days=trading_days,
    )
    if set(decision_end_by_date) != decision_set:
        raise RuntimeError("decision schedule and decision-period map are inconsistent")

    candidate_items = list(iter_candidate_params(candidates))
    n_candidates = len(candidate_items)
    n_dates = len(event_dates)
    n_modes = len(modes)
    rows: list[dict[str, object]] = []
    # One tick per (mode × candidate × sample_date).
    pbar = (
        ProgressBar(n_modes * n_candidates * n_dates, desc="dataset")
        if show_progress
        else None
    )
    try:
        for position_mode in modes:
            for strategy_id, candidate_id, params in candidate_items:
                if pbar is not None:
                    pbar.set_description(f"dataset/{position_mode}/{strategy_id}")
                    pbar.set_postfix(cand=str(candidate_id))
                raw = _run_one_candidate(panel, strategy_id, params, price_col=price_col)
                signal = apply_position_mode(
                    raw,
                    position_mode,
                    overlay_scale=overlay_scale,
                )
                port_ret = _portfolio_returns(signal, resolved_weights)

                cand_row = candidates.loc[candidates["candidate_id"] == candidate_id].iloc[0]
                for sample_date in event_dates:
                    sample_date = pd.Timestamp(sample_date).normalize()
                    is_model_decision = sample_date in decision_set
                    label_end = label_end_for_event(
                        sample_date,
                        label_horizon=label_horizon,
                        trading_days=trading_days,
                        next_decision=(
                            decision_end_by_date[sample_date]
                            if is_model_decision
                            else None
                        ),
                    )
                    if label_end is None:
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    feat = _feature_for_sample(
                        features,
                        sample_date,
                        weights=resolved_weights,
                    )
                    if feat is None:
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    y, n_days = _forward_label(
                        port_ret,
                        sample_date,
                        label_end,
                        label_metric=label_metric,
                    )
                    if y is None:
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    row: dict[str, object] = {
                        "decision_date": sample_date,
                        "label_end": pd.Timestamp(label_end).normalize(),
                        "train_start": (sample_date - train_offset).normalize(),
                        "is_model_decision": is_model_decision,
                        "position_mode": position_mode,
                        "strategy_id": strategy_id,
                        "candidate_id": candidate_id,
                        "label": y,
                        "label_n_days": n_days,
                    }
                    for key in PARAM_COLUMNS:
                        row[key] = cand_row[key]
                    for col in FEATURE_COLUMNS:
                        row[col] = float(feat[col])
                    rows.append(row)
                    if pbar is not None:
                        pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    ordered = [
        "decision_date",
        "label_end",
        "train_start",
        "is_model_decision",
        "position_mode",
        "strategy_id",
        "candidate_id",
        *PARAM_COLUMNS,
        *FEATURE_COLUMNS,
        "label",
        "label_n_days",
    ]
    frame = frame.loc[:, ordered]
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    frame["label_end"] = pd.to_datetime(frame["label_end"])
    frame["train_start"] = pd.to_datetime(frame["train_start"])
    return frame.sort_values(
        ["decision_date", "position_mode", "strategy_id", "candidate_id"]
    ).reset_index(drop=True)
