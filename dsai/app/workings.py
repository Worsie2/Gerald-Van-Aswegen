"""What each fitted preprocessing step actually learned, shown as numbers.

"Median imputation applied to 3 columns" is a claim. The median it used is a
fact, and one a reader can check against their own knowledge of the data — a
median customer spend of R4 says the column is wrong long before any model does.

Every table here is computed on the whole frame **purely to be displayed**. The
pipeline itself refits inside each cross-validation fold, so the values it uses
when scoring come from training rows only and will differ slightly. Every caller
states that where it shows these.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.engines.metrics import human_number

__all__ = ["learned_parameters"]


def learned_parameters(pipeline: Any, frame: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
    """``(title, table)`` for every step in the pipeline whose values can be shown.

    Steps whose behaviour is not a small set of numbers — KNN and iterative
    imputation, dimensionality reduction — are deliberately absent rather than
    summarised badly, because a misleading summary is worse than none.
    """
    if pipeline is None or frame is None or frame.empty:
        return []

    # Each step must be shown the frame as it will actually see it. A scaler
    # placed after a drop does not standardise the dropped column, and listing it
    # as though it did is exactly the kind of plausible-looking wrong detail that
    # makes a reader stop trusting the whole table.
    available = list(frame.columns)
    out: list[tuple[str, pd.DataFrame]] = []
    for position, step in enumerate(pipeline.active_steps, start=1):
        visible = frame[[c for c in available if c in frame.columns]]
        table = _table_for(step, visible)
        if table is not None and not table.empty:
            out.append((f"{position}. {step.spec.name} — {_scope(step)}", table))
        available = _columns_after(step, available, frame)
    return out


def _columns_after(step: Any, available: list[str], frame: pd.DataFrame) -> list[str]:
    """The columns still present after this step, tracked by name.

    Only the structural steps change the column set in a way later steps care
    about. Everything else leaves the names alone, so the tracking stays simple
    and does not pretend to be a full dry run of the pipeline.
    """
    key = step.step_key
    params = step.params or {}
    if key == "drop_columns":
        removed = set(params.get("columns") or
                      (step.columns if isinstance(step.columns, list) else []))
        return [c for c in available if c not in removed]
    if key == "drop_constant_columns":
        constant = {c for c in available
                    if c in frame.columns and frame[c].nunique(dropna=True) <= 1}
        return [c for c in available if c not in constant]
    if key == "one_hot_encode":
        encoded = set(_columns(step, frame[[c for c in available if c in frame.columns]]))
        encoded = {c for c in encoded if not pd.api.types.is_numeric_dtype(frame[c])}
        return [c for c in available if c not in encoded]
    return available


def _scope(step: Any) -> str:
    if not step.columns:
        return "all applicable columns"
    if isinstance(step.columns, str):
        return str(step.columns)
    return ", ".join(step.columns[:5]) + (f" +{len(step.columns) - 5}" if len(step.columns) > 5 else "")


def _columns(step: Any, frame: pd.DataFrame, numeric_only: bool = False) -> list[str]:
    names = step.columns if isinstance(step.columns, list) else None
    if not names:
        names = list(frame.columns)
    names = [c for c in names if c in frame.columns]
    if numeric_only:
        names = [c for c in names if pd.api.types.is_numeric_dtype(frame[c])]
    return names


def _table_for(step: Any, frame: pd.DataFrame) -> pd.DataFrame | None:
    key = step.step_key
    params = step.params or {}

    if key in ("impute_mean", "impute_median"):
        statistic = "mean" if key == "impute_mean" else "median"
        rows = []
        for column in _columns(step, frame, numeric_only=True):
            series = frame[column]
            value = series.mean() if statistic == "mean" else series.median()
            rows.append({
                "Column": column,
                f"Value it would use ({statistic})": human_number(value),
                "Gaps it fills": int(series.isna().sum()),
                "Column mean": human_number(series.mean()),
                "Column median": human_number(series.median()),
            })
        return pd.DataFrame(rows)

    if key == "impute_mode":
        rows = []
        for column in _columns(step, frame):
            modes = frame[column].mode(dropna=True)
            value = modes.iloc[0] if len(modes) else None
            share = (frame[column] == value).mean() if value is not None else 0.0
            rows.append({
                "Column": column,
                "Value it would use": str(value),
                "Gaps it fills": int(frame[column].isna().sum()),
                "Share of rows already that value": f"{share:.1%}",
            })
        return pd.DataFrame(rows)

    if key == "impute_constant":
        return pd.DataFrame([{
            "Column": column,
            "Value it inserts": str(params.get("fill_value", "__missing__")),
            "Gaps it fills": int(frame[column].isna().sum()),
        } for column in _columns(step, frame)])

    if key == "impute_by_group":
        from dsai.preprocessing.transformers import GroupImputer

        group_by = params.get("group_by") or ""
        if group_by not in frame.columns:
            return None
        columns = _columns(step, frame, numeric_only=params.get("strategy") != "most_frequent")
        imputer = GroupImputer(columns=columns, group_by=group_by,
                               strategy=params.get("strategy", "median")).fit(frame)
        rows = []
        for column, values in imputer.learned_values().items():
            for group, value in values.items():
                rows.append({"Column": column, group_by: group,
                             "Value it would use": human_number(value)})
        return pd.DataFrame(rows)

    if key in ("standard_scale", "robust_scale", "minmax_scale"):
        rows = []
        for column in _columns(step, frame, numeric_only=True):
            series = frame[column].dropna()
            if series.empty:
                continue
            rows.append({
                "Column": column,
                "Centre": human_number(series.median() if key == "robust_scale" else series.mean()),
                "Scale": human_number(
                    (series.quantile(0.75) - series.quantile(0.25)) if key == "robust_scale"
                    else (series.max() - series.min()) if key == "minmax_scale"
                    else series.std()
                ),
                "Before: min → max": f"{human_number(series.min())} → {human_number(series.max())}",
            })
        return pd.DataFrame(rows)

    if key in ("winsorize", "clip_outliers"):
        low = float(params.get("lower", params.get("lower_percentile", 0.01)) or 0.01)
        high = float(params.get("upper", params.get("upper_percentile", 0.99)) or 0.99)
        rows = []
        for column in _columns(step, frame, numeric_only=True):
            series = frame[column].dropna()
            if series.empty:
                continue
            lo, hi = series.quantile(low), series.quantile(high)
            rows.append({
                "Column": column,
                "Lower limit": human_number(lo),
                "Upper limit": human_number(hi),
                "Rows it would move": int(((series < lo) | (series > hi)).sum()),
            })
        return pd.DataFrame(rows)

    if key == "one_hot_encode":
        rows = []
        for column in _columns(step, frame):
            if pd.api.types.is_numeric_dtype(frame[column]):
                continue
            levels = frame[column].astype("string").dropna().unique()
            rows.append({
                "Column": column,
                "Levels": len(levels),
                "New columns it creates": len(levels),
                "Levels found": ", ".join(sorted(str(v) for v in levels)[:12])
                                + (" …" if len(levels) > 12 else ""),
            })
        return pd.DataFrame(rows)

    if key in ("drop_columns", "drop_constant_columns"):
        named = params.get("columns") or (step.columns if isinstance(step.columns, list) else [])
        return pd.DataFrame([{"Column removed": c} for c in named])

    if key in ("log_transform", "yeo_johnson", "box_cox", "quantile_transform"):
        rows = []
        for column in _columns(step, frame, numeric_only=True):
            series = frame[column].dropna()
            if series.empty:
                continue
            rows.append({
                "Column": column,
                "Skew before": human_number(series.skew()),
                "Skew after a log1p": human_number(np.log1p(series.clip(lower=0)).skew()),
            })
        return pd.DataFrame(rows)

    # Everything else — KNN, iterative imputation, PCA, feature selection — has no
    # small set of learned numbers that could be shown without misleading.
    return None
