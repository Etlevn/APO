"""Train-only preprocessing for engineered features."""

from __future__ import annotations

import pandas as pd

from .builder import FEATURE_COLUMNS


class FeaturePipeline:
    """
    Standardize selected feature columns.

    fit() must use training rows only in production walk-forward runs.
    """

    def __init__(self, feature_cols: tuple[str, ...] | list[str] = FEATURE_COLUMNS):
        self.feature_cols = list(feature_cols)
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.std_ is not None

    def fit(self, frame: pd.DataFrame) -> FeaturePipeline:
        missing = [c for c in self.feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"frame missing feature columns: {missing}")
        values = frame[self.feature_cols]
        self.mean_ = values.mean(numeric_only=True)
        std = values.std(numeric_only=True, ddof=0).replace(0.0, 1.0)
        self.std_ = std.fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline.transform() called before fit().")
        missing = [c for c in self.feature_cols if c not in frame.columns]
        if missing:
            raise ValueError(f"frame missing feature columns: {missing}")

        out = frame.copy()
        out[self.feature_cols] = (out[self.feature_cols] - self.mean_) / self.std_
        return out

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)
