"""Download and cache daily ETF panels via yfinance."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = (
    "timestamp",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "availability_time",
)


def cache_path_for(
    tickers: list[str],
    start: str,
    end: str | None,
    cache_dir: Path,
    cache_format: str = "parquet",
) -> Path:
    """Build a deterministic cache filename for a download request."""
    end_label = end or "latest"
    ticker_label = "_".join(tickers)
    suffix = "parquet" if cache_format == "parquet" else "csv"
    return Path(cache_dir) / f"etf_{ticker_label}_{start}_{end_label}.{suffix}"


def _as_day(series: pd.Series) -> pd.Series:
    """Normalize datetimes to calendar-day precision (midnight, no intraday time)."""
    return pd.to_datetime(series).dt.normalize()


def _finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    out = panel.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["timestamp"] = _as_day(out["timestamp"])
    out["availability_time"] = _as_day(out["availability_time"])
    return out


def _normalize_single_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may still return a single-ticker MultiIndex
        if ticker in df.columns.get_level_values(0):
            df = df[ticker].copy()
        else:
            df.columns = df.columns.get_level_values(-1)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df = df.rename_axis("timestamp").reset_index()
    if "Date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"Date": "timestamp"})

    keep = [
        c
        for c in ("timestamp", "open", "high", "low", "close", "adj_close", "volume")
        if c in df.columns
    ]
    df = df[keep].copy()
    df["asset_id"] = ticker
    # Daily bar: calendar day only; treated as available after that session's close.
    df["timestamp"] = _as_day(df["timestamp"])
    df["availability_time"] = df["timestamp"]
    return df


def _download_panel(
    tickers: list[str],
    start: str,
    end: str | None,
) -> pd.DataFrame:
    # auto_adjust=False so yfinance returns both Close and Adj Close.
    # yfinance still uses DataFrame inplace= under pandas 3.1 (PDEP-8);
    # silence only that third-party deprecation for the download call.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=pd.errors.Pandas4Warning,
            module=r"yfinance(\.|$)",
        )
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no rows for tickers={tickers}, start={start}, end={end}."
        )

    frames: list[pd.DataFrame] = []
    if len(tickers) == 1:
        frames.append(_normalize_single_ticker(raw, tickers[0]))
    else:
        for ticker in tickers:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    raise KeyError(f"Ticker {ticker!r} missing from yfinance download result.")
                piece = raw[ticker].copy()
            else:
                piece = raw.copy()
            frames.append(_normalize_single_ticker(piece, ticker))

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["timestamp", "asset_id"]).reset_index(drop=True)
    return panel.loc[:, list(REQUIRED_COLUMNS)]


def load_etf_panel(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a long-format daily ETF panel from yfinance.

    Does not read or write local cache. Use ``load_etf_panel_cached`` from main.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    return _finalize_panel(_download_panel(tickers, start, end))


def read_etf_cache(path: Path) -> pd.DataFrame:
    """Read a previously written ETF panel cache."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported cache format: {path.suffix}")
    return _finalize_panel(df)


def write_etf_cache(panel: pd.DataFrame, path: Path) -> None:
    """Write an ETF panel to local cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    panel = _finalize_panel(panel)
    if path.suffix == ".parquet":
        panel.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        panel.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported cache format: {path.suffix}")


def load_etf_panel_cached(
    tickers: list[str],
    start: str,
    end: str | None = None,
    *,
    cache_dir: Path | str = "data",
    cache_format: str = "parquet",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load panel from local cache, or download and write cache if missing.

    Intended for the production ``main`` entry only; preview ``run_*`` scripts
    should call ``load_etf_panel`` instead.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list.")
    if cache_format not in {"parquet", "csv"}:
        raise ValueError("cache_format must be 'parquet' or 'csv'.")

    path = cache_path_for(tickers, start, end, Path(cache_dir), cache_format)
    if path.exists() and not force_download:
        return read_etf_cache(path)

    panel = load_etf_panel(tickers, start, end)
    write_etf_cache(panel, path)
    return panel
