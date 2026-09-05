"""Custom transformers used by the preprocessing pipeline.

All of them are scikit-learn transformers, so they can sit inside a Pipeline
and be fitted on training folds only — which is how leakage is prevented.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep (or drop) named columns, tolerating columns that are already gone."""

    def __init__(self, columns: list[str] | None = None, mode: str = "keep"):
        self.columns = columns or []
        self.mode = mode

    def fit(self, X, y=None):
        frame = _as_frame(X)
        if self.mode == "keep":
            self.columns_ = [c for c in self.columns if c in frame.columns]
        else:
            self.columns_ = [c for c in frame.columns if c not in self.columns]
        return self

    def transform(self, X):
        return _as_frame(X).reindex(columns=self.columns_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class WinsorizeTransformer(BaseEstimator, TransformerMixin):
    """Clip extreme values to percentile limits learned on the training data.

    Keeps every row (unlike deletion) while stopping a handful of extremes from
    dominating a scaler or a linear fit.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        self.lower_ = frame.quantile(self.lower_quantile)
        self.upper_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        return frame.clip(lower=self.lower_, upper=self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class OutlierClipTransformer(BaseEstimator, TransformerMixin):
    """Clip to Tukey fences (Q1 − k·IQR, Q3 + k·IQR) learned on training data."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        q1, q3 = frame.quantile(0.25), frame.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        return frame.clip(lower=self.lower_, upper=self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class LogTransformer(BaseEstimator, TransformerMixin):
    """log1p transform with an automatic shift so negative values still work."""

    def __init__(self, base: str = "natural", rename: bool = False):
        self.base = base
        # Renaming to log_<col> is clearer in a report but breaks any later step
        # that addresses the column by its original name, so it is off by default.
        self.rename = rename

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        minimums = frame.min()
        # shift only where needed, so untouched columns keep their exact values
        self.shift_ = (-minimums + 1e-6).clip(lower=0)
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_) + self.shift_
        out = np.log1p(frame)
        if self.base == "log10":
            out = out / np.log(10)
        elif self.base == "log2":
            out = out / np.log(2)
        return out

    def inverse_transform(self, X):
        frame = _as_frame(X)
        if self.base == "log10":
            frame = frame * np.log(10)
        elif self.base == "log2":
            frame = frame * np.log(2)
        return np.expm1(frame) - self.shift_

    def get_feature_names_out(self, input_features=None):
        names = [f"log_{c}" for c in self.feature_names_in_] if self.rename else self.feature_names_in_
        return np.asarray(names, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with how often it occurs in the training data.

    A compact alternative to one-hot for high-cardinality columns; unseen
    categories map to 0, which correctly signals "never seen before".
    """

    def __init__(self, normalise: bool = True):
        self.normalise = normalise

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        self.maps_ = {}
        for col in frame.columns:
            counts = frame[col].astype("object").value_counts(normalize=self.normalise)
            self.maps_[col] = counts.to_dict()
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_names_in_:
            out[f"{col}_freq"] = frame[col].astype("object").map(self.maps_[col]).fillna(0.0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_freq" for c in self.feature_names_in_], dtype=object)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target (mean) encoding, fitted on the training fold only.

    Target encoding leaks badly if it sees the rows it will later score, so this
    must live inside the cross-validation pipeline — never applied to the full
    dataset up front. Smoothing pulls rare categories towards the global mean.
    """

    def __init__(self, smoothing: float = 10.0, min_samples_leaf: int = 1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        frame = _as_frame(X)
        target = pd.Series(np.asarray(y, dtype=float), index=frame.index)
        self.feature_names_in_ = list(frame.columns)
        self.global_mean_ = float(target.mean())
        self.maps_ = {}
        for col in frame.columns:
            grouped = target.groupby(frame[col].astype("object"))
            counts, means = grouped.count(), grouped.mean()
            weight = 1.0 / (1.0 + np.exp(-(counts - self.min_samples_leaf) / self.smoothing))
            self.maps_[col] = (self.global_mean_ * (1 - weight) + means * weight).to_dict()
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_names_in_:
            out[f"{col}_target_enc"] = (
                frame[col].astype("object").map(self.maps_[col]).fillna(self.global_mean_)
            )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_target_enc" for c in self.feature_names_in_], dtype=object)


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Expand a datetime column into the calendar parts models can actually use."""

    def __init__(self, features: list[str] | None = None, cyclical: bool = True,
                 reference: str | None = None):
        self.features = features or ["year", "month", "day", "dayofweek", "quarter", "is_month_end"]
        self.cyclical = cyclical
        self.reference = reference

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        parsed = frame.apply(pd.to_datetime, errors="coerce")
        self.reference_ = pd.to_datetime(self.reference) if self.reference else parsed.min().min()
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_names_in_:
            series = pd.to_datetime(frame[col], errors="coerce")
            for feature in self.features:
                if feature == "is_month_end":
                    out[f"{col}_is_month_end"] = series.dt.is_month_end.astype(float)
                elif feature == "is_weekend":
                    out[f"{col}_is_weekend"] = (series.dt.dayofweek >= 5).astype(float)
                elif feature == "days_since_reference":
                    out[f"{col}_days_since_ref"] = (series - self.reference_).dt.days.astype(float)
                else:
                    out[f"{col}_{feature}"] = getattr(series.dt, feature).astype(float)
            if self.cyclical:
                # month and weekday are circular: December is adjacent to January
                out[f"{col}_month_sin"] = np.sin(2 * np.pi * series.dt.month / 12)
                out[f"{col}_month_cos"] = np.cos(2 * np.pi * series.dt.month / 12)
                out[f"{col}_dow_sin"] = np.sin(2 * np.pi * series.dt.dayofweek / 7)
                out[f"{col}_dow_cos"] = np.cos(2 * np.pi * series.dt.dayofweek / 7)
        return out.fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        names = []
        for col in self.feature_names_in_:
            for feature in self.features:
                key = {"is_month_end": "is_month_end", "is_weekend": "is_weekend",
                       "days_since_reference": "days_since_ref"}.get(feature, feature)
                names.append(f"{col}_{key}")
            if self.cyclical:
                names += [f"{col}_month_sin", f"{col}_month_cos", f"{col}_dow_sin", f"{col}_dow_cos"]
        return np.asarray(names, dtype=object)


class TextFeatures(BaseEstimator, TransformerMixin):
    """Cheap, interpretable text summaries: length, word count, digit share, etc.

    Deliberately not TF-IDF — these features are readable in a report and do not
    explode the feature space. Use TruncatedSVD on TF-IDF when you need more.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def fit(self, X, y=None):
        self.feature_names_in_ = list(_as_frame(X).columns)
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        out = pd.DataFrame(index=frame.index)
        for col in self.feature_names_in_:
            series = frame[col].astype(str)
            if self.lowercase:
                series = series.str.lower()
            out[f"{col}_char_count"] = series.str.len().astype(float)
            out[f"{col}_word_count"] = series.str.split().str.len().astype(float)
            out[f"{col}_digit_share"] = series.str.count(r"\d") / series.str.len().clip(lower=1)
            out[f"{col}_upper_share"] = frame[col].astype(str).str.count(r"[A-Z]") / series.str.len().clip(lower=1)
            out[f"{col}_unique_word_ratio"] = series.apply(
                lambda t: len(set(t.split())) / max(len(t.split()), 1)
            )
        return out.fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        suffixes = ["char_count", "word_count", "digit_share", "upper_share", "unique_word_ratio"]
        return np.asarray([f"{c}_{s}" for c in self.feature_names_in_ for s in suffixes], dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fold categories below a frequency threshold into a single 'Other' level."""

    def __init__(self, min_frequency: float = 0.01, other_label: str = "__other__"):
        self.min_frequency = min_frequency
        self.other_label = other_label

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)
        self.keep_ = {}
        for col in frame.columns:
            shares = frame[col].astype("object").value_counts(normalize=True)
            self.keep_[col] = set(shares[shares >= self.min_frequency].index)
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_).copy()
        for col in self.feature_names_in_:
            keep = self.keep_[col]
            frame[col] = frame[col].astype("object").where(frame[col].isin(keep), self.other_label)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Pairwise products (and optionally ratios) between chosen numeric columns."""

    def __init__(self, max_features: int = 8, include_ratios: bool = False):
        self.max_features = max_features
        self.include_ratios = include_ratios

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.feature_names_in_ = list(frame.columns)[: self.max_features]
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        out = frame.copy()
        cols = self.feature_names_in_
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                out[f"{a}_x_{b}"] = frame[a] * frame[b]
                if self.include_ratios:
                    out[f"{a}_div_{b}"] = frame[a] / frame[b].replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        cols = self.feature_names_in_
        names = list(cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                names.append(f"{a}_x_{b}")
                if self.include_ratios:
                    names.append(f"{a}_div_{b}")
        return np.asarray(names, dtype=object)


class AggregationFeatures(BaseEstimator, TransformerMixin):
    """Row-wise summaries across the numeric columns (sum, mean, spread, …)."""

    def __init__(self, functions: list[str] | None = None):
        self.functions = functions or ["sum", "mean", "std", "min", "max"]

    def fit(self, X, y=None):
        self.feature_names_in_ = list(_as_frame(X).columns)
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        out = frame.copy()
        for fn in self.functions:
            out[f"row_{fn}"] = getattr(frame, fn)(axis=1)
        if "max" in self.functions and "min" in self.functions:
            out["row_range"] = frame.max(axis=1) - frame.min(axis=1)
        return out.fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_) + [f"row_{fn}" for fn in self.functions]
        if "max" in self.functions and "min" in self.functions:
            names.append("row_range")
        return np.asarray(names, dtype=object)


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Passes data through unchanged — used when a step is disabled."""

    def fit(self, X, y=None):
        self.feature_names_in_ = list(_as_frame(X).columns)
        return self

    def transform(self, X):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()
    return pd.DataFrame(np.asarray(X))


class FrameTransformer(BaseEstimator, TransformerMixin):
    """Apply an inner transformer to selected columns and keep a DataFrame out.

    Chaining raw ColumnTransformers loses column names, which then makes every
    downstream explanation ("which variable mattered?") guesswork. This wrapper
    keeps names intact end to end, which matters more here than the small
    overhead of staying in pandas.
    """

    def __init__(self, transformer: Any = None, columns: list[str] | None = None,
                 mode: str = "replace", prefix: str = ""):
        self.transformer = transformer
        self.columns = columns
        self.mode = mode          # replace | append
        self.prefix = prefix

    #: Selectors resolved against the frame as it looks *at this point* in the
    #: pipeline, so a step still targets the right columns after earlier steps
    #: have added, renamed or removed some.
    SELECTORS = {
        "__all__": lambda f: list(f.columns),
        "__numeric__": lambda f: list(f.select_dtypes(include="number").columns),
        "__non_numeric__": lambda f: list(f.select_dtypes(exclude=["number", "datetime", "datetimetz"]).columns),
        "__datetime__": lambda f: list(f.select_dtypes(include=["datetime", "datetimetz"]).columns),
    }

    def _target_columns(self, frame: pd.DataFrame) -> list[str]:
        if self.columns is None:
            return list(frame.columns)
        if isinstance(self.columns, str):
            selector = self.SELECTORS.get(self.columns)
            if selector is None:
                raise ValueError(
                    f"Unknown column selector '{self.columns}'. "
                    f"Use a list of names or one of: {', '.join(self.SELECTORS)}."
                )
            return selector(frame)
        return [c for c in self.columns if c in frame.columns]

    def fit(self, X, y=None):
        frame = _as_frame(X)
        self.columns_ = self._target_columns(frame)
        self.input_columns_ = list(frame.columns)
        if not self.columns_:
            self.fitted_transformer_ = None
            self.output_names_ = list(frame.columns)
            return self
        subset = frame[self.columns_]
        self.fitted_transformer_ = self.transformer
        try:
            self.fitted_transformer_.fit(subset, y)
        except TypeError:
            self.fitted_transformer_.fit(subset)
        self.output_names_ = self._compute_output_names(frame)
        return self

    def _compute_output_names(self, frame: pd.DataFrame) -> list[str]:
        inner = self.fitted_transformer_
        try:
            produced = [str(n) for n in inner.get_feature_names_out(self.columns_)]
        except Exception:
            produced = list(self.columns_)
        if self.prefix:
            produced = [f"{self.prefix}{n}" for n in produced]
        untouched = [c for c in frame.columns if c not in self.columns_]
        if self.mode == "append":
            return list(frame.columns) + [n for n in produced if n not in frame.columns]
        return untouched + produced

    def transform(self, X):
        frame = _as_frame(X)
        if self.fitted_transformer_ is None:
            return frame
        subset = frame.reindex(columns=self.columns_)
        result = self.fitted_transformer_.transform(subset)
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(
                np.asarray(result.todense()) if hasattr(result, "todense") else np.asarray(result),
                index=frame.index,
            )
        else:
            result = result.copy()
            result.index = frame.index
        try:
            names = [str(n) for n in self.fitted_transformer_.get_feature_names_out(self.columns_)]
        except Exception:
            names = list(result.columns) if isinstance(result, pd.DataFrame) else [
                f"{self.prefix or 'feature'}_{i}" for i in range(result.shape[1])
            ]
        if len(names) == result.shape[1]:
            result.columns = [f"{self.prefix}{n}" for n in names] if self.prefix else names
        if self.mode == "append":
            new_cols = [c for c in result.columns if c not in frame.columns]
            return pd.concat([frame, result[new_cols]], axis=1)
        untouched = frame.drop(columns=self.columns_, errors="ignore")
        return pd.concat([untouched, result], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.output_names_, dtype=object)


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Final safety net: force everything numeric and finite before a model sees it.

    Any column that is still non-numeric at this point (an encoder was skipped,
    an unexpected string appeared) becomes NaN and is then filled, rather than
    blowing up inside the estimator with an unhelpful error.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        frame = _as_frame(X)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        self.feature_names_in_ = list(frame.columns)
        self.fill_values_ = numeric.median().fillna(self.fill_value)
        return self

    def transform(self, X):
        frame = _as_frame(X).reindex(columns=self.feature_names_in_)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        return numeric.fillna(self.fill_values_).fillna(self.fill_value)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)
