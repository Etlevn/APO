"""Walk-forward calendar helpers (trading-day schedule)."""

from __future__ import annotations

import re

import pandas as pd

_SPAN_RE = re.compile(r"^(\d+)([ymwd])$", re.IGNORECASE)


def parse_span(span: str) -> pd.DateOffset:
    """Parse tokens like 1y, 6m, 3m, 1m, 1w, 1d into a DateOffset."""
    amount, unit = _parse_span_parts(span)
    if unit == "y":
        return pd.DateOffset(years=amount)
    if unit == "m":
        return pd.DateOffset(months=amount)
    if unit == "w":
        return pd.DateOffset(weeks=amount)
    return pd.DateOffset(days=amount)


def _parse_span_parts(span: str) -> tuple[int, str]:
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
    return amount, unit


def normalize_trading_days(
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> pd.DatetimeIndex:
    """Unique sorted calendar-day trading timestamps from a panel."""
    idx = pd.DatetimeIndex(pd.to_datetime(trading_days)).normalize()
    return idx.unique().sort_values()


def _snap_on_or_after(
    day: pd.Timestamp,
    trading: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    day = pd.Timestamp(day).normalize()
    pos = int(trading.searchsorted(day, side="left"))
    if pos >= len(trading):
        return None
    return pd.Timestamp(trading[pos])


def _snap_on_or_before(
    day: pd.Timestamp,
    trading: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    day = pd.Timestamp(day).normalize()
    pos = int(trading.searchsorted(day, side="right")) - 1
    if pos < 0:
        return None
    return pd.Timestamp(trading[pos])


def _decision_trading_pos(
    decision_date: pd.Timestamp,
    trading: pd.DatetimeIndex,
) -> int | None:
    """Index of ``decision_date`` on the trading calendar (must be a trading day)."""
    decision = pd.Timestamp(decision_date).normalize()
    pos = int(trading.searchsorted(decision, side="left"))
    if pos >= len(trading) or pd.Timestamp(trading[pos]) != decision:
        return None
    return pos


def label_end_date(
    decision_date: str | pd.Timestamp,
    label_horizon: str,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> pd.Timestamp | None:
    """
    Hold / label window end on the panel trading calendar.

    - ``Nd``: the N-th trading day **after** ``decision_date``.
    - ``Nw`` / ``Nm`` / ``Ny``: calendar ``decision + horizon``, then snap to the
      last trading day on or before that target.

    Returns ``None`` if the end cannot be formed (past panel) or would not be
    strictly after ``decision_date``.
    """
    trading = normalize_trading_days(trading_days)
    if trading.empty:
        return None

    decision = pd.Timestamp(decision_date).normalize()
    pos = _decision_trading_pos(decision, trading)
    if pos is None:
        return None

    amount, unit = _parse_span_parts(label_horizon)
    if unit == "d":
        end_pos = pos + amount
        if end_pos >= len(trading):
            return None
        end = pd.Timestamp(trading[end_pos])
    else:
        cal_end = (decision + parse_span(label_horizon)).normalize()
        # Do not silently shorten an incomplete calendar horizon to the last
        # available panel day.  That would leak a partial future period into a
        # row whose configured label has not fully matured yet.
        if cal_end > pd.Timestamp(trading[-1]):
            return None
        end = _snap_on_or_before(cal_end, trading)
        if end is None:
            return None

    if end <= decision:
        return None
    return end


def spans_equal(left: str, right: str) -> bool:
    """True when two span tokens match after strip/lower (e.g. ``1m`` == ``1M``)."""
    return str(left).strip().lower() == str(right).strip().lower()


def label_end_for_event(
    event_date: str | pd.Timestamp,
    *,
    label_horizon: str,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    next_decision: str | pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    """
    Resolve hold / label end for one event date.

    - When ``next_decision`` is passed (caller should do so only if
      ``rebalance_freq == label_horizon``): hold is ``(event_date, next_decision]``.
    - Otherwise (train-only rows): use ``label_end_date`` / ``label_horizon``.
    """
    if next_decision is not None:
        start = pd.Timestamp(event_date).normalize()
        end = pd.Timestamp(next_decision).normalize()
        if end <= start:
            return None
        return end
    return label_end_date(event_date, label_horizon, trading_days)


def next_decision_map(
    decision_dates: list[pd.Timestamp] | set[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each formal decision (except the last) to the following decision date."""
    ordered = sorted({pd.Timestamp(d).normalize() for d in decision_dates})
    return {ordered[i]: ordered[i + 1] for i in range(len(ordered) - 1)}


def schedule_boundaries(
    *,
    start: str | pd.Timestamp,
    step_freq: str,
    max_timestamp: pd.Timestamp,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Build rebalance boundaries without requiring a label after each one.

    The final boundary may only close the previous holding period.  It becomes
    a live decision only when a following boundary is also observed.
    """
    trading = normalize_trading_days(trading_days)
    if trading.empty:
        return []

    amount, unit = _parse_span_parts(step_freq)
    limit = min(
        pd.Timestamp(max_timestamp).normalize(),
        pd.Timestamp(trading[-1]),
    )
    start_ts = pd.Timestamp(start).normalize()
    dates: list[pd.Timestamp] = []

    if unit == "d":
        first = _snap_on_or_after(start_ts, trading)
        if first is None:
            return []
        pos = int(trading.searchsorted(first, side="left"))
        while pos < len(trading):
            current = pd.Timestamp(trading[pos])
            if current > limit:
                break
            dates.append(current)
            pos += amount
        return dates

    step = parse_span(step_freq)
    cal_target = start_ts
    seen: set[pd.Timestamp] = set()
    for _ in range(len(trading) * 4 + 8):
        current = _snap_on_or_after(cal_target, trading)
        if current is None or current > limit:
            break
        if current not in seen:
            dates.append(current)
            seen.add(current)
        next_target = (cal_target + step).normalize()
        if next_target <= cal_target:
            break
        cal_target = next_target
    return dates


def decision_period_map(
    *,
    first_decision_date: str | pd.Timestamp,
    rebalance_freq: str,
    max_timestamp: pd.Timestamp,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map each evaluable decision start to its next rebalance boundary."""
    boundaries = schedule_boundaries(
        start=first_decision_date,
        step_freq=rebalance_freq,
        max_timestamp=max_timestamp,
        trading_days=trading_days,
    )
    return next_decision_map(boundaries)


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
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """
    Build sample/decision dates on the panel trading calendar.

    - ``Nd`` steps advance ``N`` **trading** days (skips weekends/holidays).
    - ``Nw`` / ``Nm`` / ``Ny`` advance on the calendar grid from ``start``, then
      snap each target to the first trading day on or after that date.

    A date t is kept only if ``label_end_date(t)`` exists and is ``<= max_timestamp``
    (so the forward label window can fully complete on the trading calendar).
    """
    trading = normalize_trading_days(trading_days)
    if trading.empty:
        return []

    amount, unit = _parse_span_parts(step_freq)
    max_ts = pd.Timestamp(max_timestamp).normalize()
    start_ts = pd.Timestamp(start).normalize()

    def _keep(current: pd.Timestamp) -> bool | None:
        """True keep, False skip (incomplete label), None stop scheduling."""
        end = label_end_date(current, label_horizon, trading)
        if end is None:
            return None
        if end > max_ts:
            return None
        return True

    dates: list[pd.Timestamp] = []

    if unit == "d":
        first = _snap_on_or_after(start_ts, trading)
        if first is None:
            return []
        pos = int(trading.searchsorted(first, side="left"))
        while pos < len(trading):
            current = pd.Timestamp(trading[pos])
            keep = _keep(current)
            if keep is None:
                break
            dates.append(current)
            pos += amount
        return dates

    # Calendar grid (week/month/year), snap each hit onto a trading day.
    step = parse_span(step_freq)
    cal_target = start_ts
    seen: set[pd.Timestamp] = set()
    # Bound iterations: ~calendar span of the panel times a small factor.
    for _ in range(len(trading) * 4 + 8):
        current = _snap_on_or_after(cal_target, trading)
        if current is None:
            break
        keep = _keep(current)
        if keep is None:
            break
        if current not in seen:
            dates.append(current)
            seen.add(current)
        next_target = (cal_target + step).normalize()
        if next_target <= cal_target:
            break
        cal_target = next_target
    return dates


def decision_dates(
    *,
    first_decision_date: str | pd.Timestamp,
    rebalance_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Evaluable live decisions whose next rebalance boundary is observed."""
    if not spans_equal(rebalance_freq, label_horizon):
        raise ValueError(
            "Phase 1 requires rebalance_freq == label_horizon so each live "
            "decision ends at the next formal rebalance boundary."
        )
    return list(decision_period_map(
        first_decision_date=first_decision_date,
        rebalance_freq=rebalance_freq,
        max_timestamp=max_timestamp,
        trading_days=trading_days,
    ))


def sample_dates(
    *,
    first_decision_date: str | pd.Timestamp,
    train_window: str,
    sample_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
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
        trading_days=trading_days,
    )


def build_event_dates(
    *,
    first_decision_date: str | pd.Timestamp,
    train_window: str,
    sample_freq: str,
    rebalance_freq: str,
    label_horizon: str,
    max_timestamp: pd.Timestamp,
    trading_days: pd.Series | pd.DatetimeIndex | list[pd.Timestamp],
    panel_start: str | pd.Timestamp | None = None,
) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    """
    Union of training-sample dates and live decision dates.

    Returns
    -------
    all_dates :
        Dates for which (E, x, y) rows should be built (trading days only).
    decision_set :
        Subset used for live model selection (is_model_decision=True).
    """
    samples = sample_dates(
        first_decision_date=first_decision_date,
        train_window=train_window,
        sample_freq=sample_freq,
        label_horizon=label_horizon,
        max_timestamp=max_timestamp,
        trading_days=trading_days,
        panel_start=panel_start,
    )
    decisions = decision_dates(
        first_decision_date=first_decision_date,
        rebalance_freq=rebalance_freq,
        label_horizon=label_horizon,
        max_timestamp=max_timestamp,
        trading_days=trading_days,
    )
    decision_set = {pd.Timestamp(d).normalize() for d in decisions}
    all_dates = sorted({pd.Timestamp(d).normalize() for d in samples} | decision_set)
    return all_dates, decision_set
