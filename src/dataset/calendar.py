"""Walk-forward calendar helpers (rolling decision schedule)."""

from __future__ import annotations

import re

import pandas as pd

_SPAN_RE = re.compile(r"^(\d+)([ymwd])$", re.IGNORECASE)


def parse_span(span: str) -> pd.DateOffset:
    """Parse tokens like 1y, 6m, 3m, 1m, 1w, 1d into a DateOffset."""
    text = str(span).strip().lower()
    match = _SPAN_RE.fullmatch(text)
    if not match:
        raise ValueError(
            f"Unsupported span {span!r}; expected tokens like 1y, 6m, 3m, 1m, 1w, 1d"
        )
    amount = int(match.group(1))
    unit = match.group(2)
    if amount <= 0:
        raise ValueError(f"span amount must be positive, got {span!r}")
    if unit == "y":
        return pd.DateOffset(years=amount)
    if unit == "m":
        return pd.DateOffset(months=amount)
    if unit == "w":
        return pd.DateOffset(weeks=amount)
    return pd.DateOffset(days=amount)


def sample_start_date(
    *,
    first_decision_date: str | pd.Timestamp,
    train_window: str,
) -> pd.Timestamp:
    """Earliest labeled sample date: one train_window before the first model decision."""
    first = pd.Timestamp(first_decision_date).normalize()
    return (first - parse_span(train_window)).normalize()


def schedule_dates(
    *,
    start: str | pd.Timestamp,
    step_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
) -> list[pd.Timestamp]:
    """
    Build dates stepped by ``step_freq``.

    A date t is kept only if t + label_horizon <= max_timestamp
    (so the forward label window can fully complete).
    """
    current = pd.Timestamp(start).normalize()
    step = parse_span(step_freq)
    horizon = parse_span(label_horizon)
    max_ts = pd.Timestamp(max_timestamp).normalize()

    dates: list[pd.Timestamp] = []
    while True:
        label_end = current + horizon
        if label_end > max_ts:
            break
        dates.append(current)
        current = current + step
    return dates


def decision_dates(
    *,
    first_decision_date: str | pd.Timestamp,
    rebalance_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Live model decision / rebalance dates starting at first_decision_date."""
    return schedule_dates(
        start=first_decision_date,
        step_freq=rebalance_freq,
        label_horizon=label_horizon,
        max_timestamp=max_timestamp,
    )


def sample_dates(
    *,
    first_decision_date: str | pd.Timestamp,
    train_window: str,
    sample_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
    panel_start: str | pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """
    Labeled sample dates used to fit F(E, x), stepped by ``sample_freq``.

    Starts at first_decision_date - train_window so the first model decision
    already has a full train_window of realized labels.

    Raises ValueError if that start is earlier than the earliest panel date.
    """
    start = sample_start_date(
        first_decision_date=first_decision_date,
        train_window=train_window,
    )
    if panel_start is not None:
        earliest = pd.Timestamp(panel_start).normalize()
        if start < earliest:
            raise ValueError(
                "first_decision_date - train_window is before the earliest data date: "
                f"sample_start={start.date().isoformat()}, "
                f"earliest_data={earliest.date().isoformat()}, "
                f"first_decision_date={pd.Timestamp(first_decision_date).date().isoformat()}, "
                f"train_window={train_window!r}. "
                "Extend START / data coverage or delay first_decision_date."
            )
    return schedule_dates(
        start=start,
        step_freq=sample_freq,
        label_horizon=label_horizon,
        max_timestamp=max_timestamp,
    )


def build_event_dates(
    *,
    first_decision_date: str | pd.Timestamp,
    train_window: str,
    sample_freq: str,
    rebalance_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
    panel_start: str | pd.Timestamp | None = None,
) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    """
    Union of training-sample dates and live decision dates.

    Returns
    -------
    all_dates :
        Dates for which (E, x, y) rows should be built.
    decision_set :
        Subset used for live model selection (is_model_decision=True).
    """
    samples = sample_dates(
        first_decision_date=first_decision_date,
        train_window=train_window,
        sample_freq=sample_freq,
        label_horizon=label_horizon,
        max_timestamp=max_timestamp,
        panel_start=panel_start,
    )
    decisions = decision_dates(
        first_decision_date=first_decision_date,
        rebalance_freq=rebalance_freq,
        label_horizon=label_horizon,
        max_timestamp=max_timestamp,
    )
    decision_set = {pd.Timestamp(d).normalize() for d in decisions}
    all_dates = sorted({pd.Timestamp(d).normalize() for d in samples} | decision_set)
    return all_dates, decision_set
