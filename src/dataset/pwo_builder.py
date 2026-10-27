"""Build PWO training rows: market features + weight candidates + forward portfolio label."""

from __future__ import annotations

import numpy as np
import pandas as pd

from candidate import build_weight_candidates, iter_weight_vectors, weight_column
from feature import FEATURE_COLUMNS, build_features
from portfolio import build_asset_returns, portfolio_returns
from util import ProgressBar

from .calendar import build_event_dates, parse_span
from .weights import resolve_weights


def _asof_feature_vector(
    features: pd.DataFrame,
    decision_date: pd.Timestamp,
    *,
    feature_weights: dict[str, float],
) -> pd.Series | None:
    """Equal/fixed-weight cross-section of per-asset features as-of decision_date."""
    available = features.loc[features["timestamp"] <= decision_date]
    if available.empty:
        return None
    asof_ts = available["timestamp"].max()
    slice_ = available.loc[
        available["timestamp"] == asof_ts, ["asset_id", *FEATURE_COLUMNS]
    ].copy()
    if slice_.empty:
        return None
    if slice_[list(FEATURE_COLUMNS)].isna().any(axis=None):
        return None
    slice_["weight"] = slice_["asset_id"].map(feature_weights)
    if slice_["weight"].isna().any():
        bad = sorted(slice_.loc[slice_["weight"].isna(), "asset_id"].astype(str).unique())
        raise KeyError(f"feature assets missing from feature_weights: {bad}")
    total = float(slice_["weight"].sum())
    if total <= 0:
        return None
    slice_["weight"] = slice_["weight"] / total
    return slice_[list(FEATURE_COLUMNS)].multiply(slice_["weight"], axis=0).sum(axis=0)


def _forward_label(
    returns: pd.Series,
    decision_date: pd.Timestamp,
    label_end: pd.Timestamp,
    *,
    label_metric: str,
) -> tuple[float, int] | tuple[None, int]:
    """Label on (decision_date, label_end] of portfolio returns under candidate weights."""
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


def build_pwo_training_table(
    panel: pd.DataFrame,
    *,
    feature_cfg: dict,
    walkforward_cfg: dict,
    dataset_cfg: dict | None = None,
    price_col: str | None = None,
    tickers: list[str] | None = None,
    n_weight_candidates: int = 64,
    weight_seed: int = 42,
    include_equal_weight: bool = True,
    dirichlet_alpha: float | list[float] = 1.0,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Construct PWO samples: (E_t, w, y) with asset-layer buy-and-hold labels.

    - ``E_t``: as-of cross-sectional features (equal-weighted across tickers).
    - ``w``: long-only weight candidate (columns ``w_<TICKER>``).
    - ``y``: forward portfolio return under ``w`` over ``label_horizon``.

    Calendar semantics match SPO (``sample_freq`` / ``rebalance_freq`` / ``label_horizon``).
    """
    dataset_cfg = dataset_cfg or {}
    price_col = price_col or feature_cfg.get("price_col", "adj_close")
    label_metric = dataset_cfg.get("label_metric", "cumulative_return")
    if tickers is None:
        tickers = sorted(panel["asset_id"].astype(str).unique().tolist())
    tickers = [str(t) for t in tickers]
    feature_weights = resolve_weights(tickers, None)
    w_cols = [weight_column(t) for t in tickers]

    weight_table = build_weight_candidates(
        tickers,
        n_candidates=n_weight_candidates,
        seed=weight_seed,
        include_equal_weight=include_equal_weight,
        dirichlet_alpha=dirichlet_alpha,
    )
    weight_items = iter_weight_vectors(weight_table, tickers)

    features = build_features(
        panel,
        price_col=price_col,
        windows=feature_cfg.get("windows"),
    )
    features = features.dropna(subset=list(FEATURE_COLUMNS)).copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"])

    asset_ret = build_asset_returns(panel, price_col=price_col)

    max_ts = pd.to_datetime(panel["timestamp"]).max()
    panel_start = pd.to_datetime(panel["timestamp"]).min()
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
        panel_start=panel_start,
    )
    if not event_dates:
        return pd.DataFrame()

    train_offset = parse_span(walkforward_cfg["train_window"])
    label_offset = parse_span(label_horizon)

    feat_by_date: dict[pd.Timestamp, pd.Series | None] = {}
    for sample_date in event_dates:
        feat_by_date[sample_date] = _asof_feature_vector(
            features,
            sample_date,
            feature_weights=feature_weights,
        )

    n_candidates = len(weight_items)
    n_dates = len(event_dates)
    rows: list[dict[str, object]] = []
    pbar = ProgressBar(n_candidates * n_dates, desc="pwo_dataset") if show_progress else None
    try:
        for candidate_id, weights in weight_items:
            if pbar is not None:
                pbar.set_description("pwo_dataset")
                pbar.set_postfix(cand=str(candidate_id))
            port_ret = portfolio_returns(asset_ret, weights)
            cand_row = weight_table.loc[weight_table["candidate_id"] == candidate_id].iloc[0]
            for sample_date in event_dates:
                feat = feat_by_date[sample_date]
                if feat is None:
                    if pbar is not None:
                        pbar.update(1)
                    continue
                label_end = sample_date + label_offset
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
                    "decision_date": sample_date.normalize(),
                    "label_end": pd.Timestamp(label_end).normalize(),
                    "train_start": (sample_date - train_offset).normalize(),
                    "is_model_decision": sample_date.normalize() in decision_set,
                    "candidate_id": candidate_id,
                    "is_equal_weight": bool(cand_row["is_equal_weight"]),
                    "label": y,
                    "label_n_days": n_days,
                }
                for col in w_cols:
                    row[col] = float(cand_row[col])
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
        "candidate_id",
        "is_equal_weight",
        *w_cols,
        *FEATURE_COLUMNS,
        "label",
        "label_n_days",
    ]
    frame = frame.loc[:, ordered]
    frame["decision_date"] = pd.to_datetime(frame["decision_date"])
    frame["label_end"] = pd.to_datetime(frame["label_end"])
    frame["train_start"] = pd.to_datetime(frame["train_start"])
    return frame.sort_values(["decision_date", "candidate_id"]).reset_index(drop=True)
