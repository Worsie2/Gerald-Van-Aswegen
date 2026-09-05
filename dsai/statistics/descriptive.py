"""Descriptive statistics and correlation analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def describe_numeric(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """A fuller numeric summary than pandas' describe, in one table."""
    columns = columns or list(frame.select_dtypes(include="number").columns)
    rows = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        mean, std = float(series.mean()), float(series.std(ddof=1)) if len(series) > 1 else 0.0
        rows.append(
            {
                "variable": column,
                "n": int(series.size),
                "missing": int(frame[column].isna().sum()),
                "mean": mean,
                "median": float(series.median()),
                "mode": float(series.mode().iloc[0]) if not series.mode().empty else np.nan,
                "std": std,
                "variance": float(series.var(ddof=1)) if len(series) > 1 else 0.0,
                "cv": float(std / mean) if abs(mean) > 1e-12 else np.nan,
                "min": float(series.min()),
                "q1": float(q1),
                "q3": float(q3),
                "max": float(series.max()),
                "iqr": float(q3 - q1),
                "range": float(series.max() - series.min()),
                "skewness": float(series.skew()) if len(series) > 2 else np.nan,
                "kurtosis": float(series.kurt()) if len(series) > 3 else np.nan,
                "sem": float(series.sem()) if len(series) > 1 else np.nan,
                "ci95_lower": mean - 1.96 * float(series.sem()) if len(series) > 1 else np.nan,
                "ci95_upper": mean + 1.96 * float(series.sem()) if len(series) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def percentiles(frame: pd.DataFrame, columns: list[str] | None = None,
                points: list[float] | None = None) -> pd.DataFrame:
    points = points or [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    columns = columns or list(frame.select_dtypes(include="number").columns)
    data = frame[columns].apply(pd.to_numeric, errors="coerce")
    out = data.quantile(points).T
    out.columns = [f"p{int(p * 100)}" for p in points]
    return out.reset_index(names="variable")


def describe_categorical(frame: pd.DataFrame, columns: list[str] | None = None,
                         top_n: int = 5) -> pd.DataFrame:
    columns = columns or list(frame.select_dtypes(exclude="number").columns)
    rows = []
    for column in columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        counts = series.value_counts()
        shares = counts / counts.sum()
        # Normalised entropy: 0 = one category everywhere, 1 = perfectly even spread.
        entropy = float(-(shares * np.log(shares)).sum())
        max_entropy = float(np.log(len(counts))) if len(counts) > 1 else 1.0
        rows.append(
            {
                "variable": column,
                "n": int(series.size),
                "missing": int(frame[column].isna().sum()),
                "distinct": int(counts.size),
                "mode": str(counts.index[0]),
                "mode_count": int(counts.iloc[0]),
                "mode_share": float(shares.iloc[0]),
                "entropy": entropy,
                "normalised_entropy": entropy / max_entropy if max_entropy else 0.0,
                "top_values": ", ".join(f"{k} ({v})" for k, v in counts.head(top_n).items()),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson",
                       columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or list(frame.select_dtypes(include="number").columns)
    return frame[columns].corr(method=method)


def correlation_pairs(
    frame: pd.DataFrame,
    method: str = "pearson",
    min_abs: float = 0.0,
    columns: list[str] | None = None,
    with_pvalues: bool = True,
) -> pd.DataFrame:
    """Every pair, with the coefficient, sample size and (optionally) a p-value."""
    from scipy import stats

    columns = columns or list(frame.select_dtypes(include="number").columns)
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            pair = frame[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                continue
            x, y = pair[a].to_numpy(), pair[b].to_numpy()
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            if method == "spearman":
                coefficient, pvalue = stats.spearmanr(x, y)
            elif method == "kendall":
                coefficient, pvalue = stats.kendalltau(x, y)
            else:
                coefficient, pvalue = stats.pearsonr(x, y)
            if abs(coefficient) < min_abs:
                continue
            row = {
                "variable_1": a, "variable_2": b, "n": len(pair),
                "coefficient": float(coefficient), "method": method,
                "strength": interpret_correlation(float(coefficient)),
            }
            if with_pvalues:
                row["p_value"] = float(pvalue)
                row["significant_at_5pct"] = bool(pvalue < 0.05)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.reindex(out.coefficient.abs().sort_values(ascending=False).index) if not out.empty else out


def interpret_correlation(r: float) -> str:
    magnitude = abs(r)
    label = (
        "negligible" if magnitude < 0.1 else
        "weak" if magnitude < 0.3 else
        "moderate" if magnitude < 0.5 else
        "strong" if magnitude < 0.7 else
        "very strong"
    )
    if magnitude < 0.1:
        return label
    return f"{label} {'positive' if r > 0 else 'negative'}"


def covariance_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or list(frame.select_dtypes(include="number").columns)
    return frame[columns].cov()


def variance_inflation_factors(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """VIF per predictor, with a plain reading of what each value means."""
    columns = columns or list(frame.select_dtypes(include="number").columns)
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    data = data.loc[:, data.std(ddof=0) > 0]
    if data.shape[1] < 2 or len(data) < data.shape[1] + 2:
        return pd.DataFrame(columns=["variable", "vif", "interpretation"])

    matrix = data.to_numpy(dtype=float)
    matrix = (matrix - matrix.mean(0)) / matrix.std(0)
    rows = []
    for i, column in enumerate(data.columns):
        y = matrix[:, i]
        x = np.column_stack([np.ones(len(matrix)), np.delete(matrix, i, axis=1)])
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            residual = y - x @ beta
            ss_total = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - float(residual @ residual) / ss_total if ss_total > 0 else 0.0
            vif = 1.0 / max(1e-9, 1 - r2)
        except np.linalg.LinAlgError:
            vif = float("inf")
        rows.append({"variable": column, "vif": round(float(vif), 3), "interpretation": interpret_vif(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def interpret_vif(vif: float) -> str:
    if not np.isfinite(vif) or vif >= 100:
        return "Essentially a duplicate of other predictors — its coefficient cannot be trusted at all."
    if vif >= 10:
        return "Severe multicollinearity. The coefficient's sign and size are unreliable."
    if vif >= 5:
        return "Moderate multicollinearity. Interpret this coefficient with caution."
    if vif >= 2:
        return "Mild correlation with other predictors. Not a problem in practice."
    return "Independent of the other predictors."


def group_summary(frame: pd.DataFrame, group_column: str, value_columns: list[str] | None = None,
                  max_groups: int = 30) -> pd.DataFrame:
    """Per-group means, medians and spread — the backbone of a segment profile."""
    value_columns = value_columns or list(frame.select_dtypes(include="number").columns)
    value_columns = [c for c in value_columns if c != group_column]
    if not value_columns:
        return pd.DataFrame()
    grouped = frame.groupby(group_column, observed=True)
    if grouped.ngroups > max_groups:
        keep = frame[group_column].value_counts().head(max_groups).index
        grouped = frame[frame[group_column].isin(keep)].groupby(group_column, observed=True)

    summary = grouped[value_columns].agg(["count", "mean", "median", "std"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()


def outlier_table(frame: pd.DataFrame, columns: list[str] | None = None,
                  method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
    """Which columns have outliers, how many, and where the boundaries sit."""
    columns = columns or list(frame.select_dtypes(include="number").columns)
    rows = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        if method == "zscore":
            mean, std = series.mean(), series.std(ddof=1)
            if std == 0:
                continue
            lower, upper = mean - factor * std, mean + factor * std
        else:
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask = (series < lower) | (series > upper)
        rows.append(
            {
                "variable": column,
                "method": method,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(100.0 * float(mask.mean()), 3),
                "min_outlier": float(series[mask].min()) if mask.any() else np.nan,
                "max_outlier": float(series[mask].max()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)
