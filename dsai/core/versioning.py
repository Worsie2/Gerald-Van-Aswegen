"""Dataset versions, and what actually changed between them.

"The numbers moved" is the most common and least useful thing to discover after
re-running an analysis on refreshed data. The question is always *why*, and the
answer is nearly always in the data rather than the model: rows were added, a
column changed type, a category appeared, missingness rose somewhere.

A version is identified by a fingerprint over the data itself — shape, column
names, dtypes and a hash of the values — so two loads of the same file produce
the same version, and any real change produces a different one. Nothing is
stored twice: a version keeps its summary, not its rows.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import JsonMixin

__all__ = ["DatasetVersion", "VersionDiff", "fingerprint_frame", "snapshot", "diff_versions"]


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """A stable identity for the data itself, not for the file it came from.

    Deliberately over the values as well as the shape: two files with the same
    columns and row count but different contents are different datasets, and a
    fingerprint that could not tell them apart would be worse than none.
    """
    digest = hashlib.sha256()
    digest.update(f"{frame.shape}".encode())
    for column in frame.columns:
        digest.update(str(column).encode())
        digest.update(str(frame[column].dtype).encode())
    try:
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    except Exception:
        # A column pandas cannot hash (nested objects) falls back to its text.
        digest.update(frame.head(500).astype("string").to_csv(index=False).encode())
    return digest.hexdigest()[:16]


@dataclass
class DatasetVersion(JsonMixin):
    """A summary of one state of the data. Never the data itself."""

    fingerprint: str = ""
    label: str = ""
    created_at: str = field(default_factory=lambda: _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    source: str = ""
    n_rows: int = 0
    n_columns: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing: dict[str, float] = field(default_factory=dict)
    numeric_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    n_duplicates: int = 0

    @property
    def short(self) -> str:
        return self.fingerprint[:8]


def snapshot(frame: pd.DataFrame, label: str = "", source: str = "") -> DatasetVersion:
    """Summarise a frame as a version record."""
    version = DatasetVersion(
        fingerprint=fingerprint_frame(frame),
        label=label,
        source=source,
        n_rows=len(frame),
        n_columns=frame.shape[1],
        columns=[str(c) for c in frame.columns],
        dtypes={str(c): str(frame[c].dtype) for c in frame.columns},
        missing={str(c): round(float(frame[c].isna().mean()), 4) for c in frame.columns},
        n_duplicates=int(frame.duplicated().sum()),
    )
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if clean.empty:
                continue
            version.numeric_summary[str(column)] = {
                "mean": round(float(clean.mean()), 6),
                "median": round(float(clean.median()), 6),
                "std": round(float(clean.std(ddof=1)) if len(clean) > 1 else 0.0, 6),
                "min": round(float(clean.min()), 6),
                "max": round(float(clean.max()), 6),
            }
        elif series.nunique(dropna=True) <= 200:
            version.categories[str(column)] = sorted(
                str(v) for v in series.astype("string").dropna().unique()
            )
    return version


@dataclass
class VersionDiff:
    identical: bool = False
    rows_added: int = 0
    rows_removed: int = 0
    columns_added: list[str] = field(default_factory=list)
    columns_removed: list[str] = field(default_factory=list)
    types_changed: list[dict[str, str]] = field(default_factory=list)
    missingness_changed: list[dict[str, Any]] = field(default_factory=list)
    categories_added: dict[str, list[str]] = field(default_factory=dict)
    categories_removed: dict[str, list[str]] = field(default_factory=dict)
    distributions_moved: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def table(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        if self.rows_added or self.rows_removed:
            rows.append({"Change": "Row count",
                         "Detail": f"{self.rows_added:+,} row(s)" if self.rows_added
                         else f"-{self.rows_removed:,} row(s)"})
        for column in self.columns_added:
            rows.append({"Change": "Column added", "Detail": f"`{column}`"})
        for column in self.columns_removed:
            rows.append({"Change": "Column removed",
                         "Detail": f"`{column}` — anything built on it is invalid"})
        for change in self.types_changed:
            rows.append({"Change": "Type changed",
                         "Detail": f"`{change['column']}`: {change['was']} → {change['now']}"})
        for change in self.missingness_changed:
            rows.append({"Change": "Missingness",
                         "Detail": f"`{change['column']}`: {change['was']:.1%} → {change['now']:.1%}"})
        for column, values in self.categories_added.items():
            rows.append({"Change": "New categories",
                         "Detail": f"`{column}`: {', '.join(values[:6])}"
                                   + (" …" if len(values) > 6 else "")})
        for column, values in self.categories_removed.items():
            rows.append({"Change": "Categories gone",
                         "Detail": f"`{column}`: {', '.join(values[:6])}"})
        for change in self.distributions_moved:
            rows.append({"Change": "Distribution moved",
                         "Detail": f"`{change['column']}`: mean {change['was']:,.4g} → "
                                   f"{change['now']:,.4g} ({change['shift']:+.2f} std devs)"})
        return pd.DataFrame(rows)


def diff_versions(before: DatasetVersion, after: DatasetVersion) -> VersionDiff:
    """What changed between two versions, in the order a reader should care."""
    diff = VersionDiff()
    if before.fingerprint == after.fingerprint:
        diff.identical = True
        diff.summary = "These are the same data. Any difference in results is not the dataset."
        return diff

    delta = after.n_rows - before.n_rows
    diff.rows_added = max(delta, 0)
    diff.rows_removed = max(-delta, 0)

    old_columns, new_columns = set(before.columns), set(after.columns)
    diff.columns_added = sorted(new_columns - old_columns)
    diff.columns_removed = sorted(old_columns - new_columns)

    for column in sorted(old_columns & new_columns):
        was, now = before.dtypes.get(column), after.dtypes.get(column)
        if was != now:
            diff.types_changed.append({"column": column, "was": was or "?", "now": now or "?"})

        old_missing = before.missing.get(column, 0.0)
        new_missing = after.missing.get(column, 0.0)
        if abs(new_missing - old_missing) >= 0.02:
            diff.missingness_changed.append(
                {"column": column, "was": old_missing, "now": new_missing})

        old_values = set(before.categories.get(column, []))
        new_values = set(after.categories.get(column, []))
        if old_values or new_values:
            appeared = sorted(new_values - old_values)
            gone = sorted(old_values - new_values)
            if appeared:
                diff.categories_added[column] = appeared
            if gone:
                diff.categories_removed[column] = gone

        old_stats = before.numeric_summary.get(column)
        new_stats = after.numeric_summary.get(column)
        if old_stats and new_stats:
            spread = old_stats.get("std") or 0.0
            if spread > 1e-12:
                shift = (new_stats["mean"] - old_stats["mean"]) / spread
                if abs(shift) >= 0.2:
                    diff.distributions_moved.append({
                        "column": column, "was": old_stats["mean"], "now": new_stats["mean"],
                        "shift": shift,
                    })

    diff.summary = _summarise(diff, before, after)
    return diff


def _summarise(diff: VersionDiff, before: DatasetVersion, after: DatasetVersion) -> str:
    parts: list[str] = []
    if diff.rows_added:
        parts.append(f"{diff.rows_added:,} row(s) added")
    if diff.rows_removed:
        parts.append(f"{diff.rows_removed:,} row(s) removed")
    if diff.columns_added:
        parts.append(f"{len(diff.columns_added)} column(s) added")
    if diff.columns_removed:
        parts.append(f"{len(diff.columns_removed)} column(s) removed")
    if diff.types_changed:
        parts.append(f"{len(diff.types_changed)} type change(s)")
    if diff.categories_added:
        parts.append(f"new categories in {len(diff.categories_added)} column(s)")
    if diff.distributions_moved:
        parts.append(f"{len(diff.distributions_moved)} distribution(s) moved")

    headline = (f"**{before.short} → {after.short}.** " + ", ".join(parts) + "."
                if parts else
                f"**{before.short} → {after.short}.** The values differ but the shape, types, "
                "categories and distributions are all unchanged.")

    warnings: list[str] = []
    if diff.columns_removed:
        warnings.append(
            f"`{', '.join(diff.columns_removed[:3])}` is gone. Any model trained on it cannot "
            "score this version at all."
        )
    if diff.types_changed:
        warnings.append(
            "A column changing type usually means the loader read it differently, not that the "
            "data changed. Check the source before trusting a comparison across this boundary."
        )
    if diff.categories_added:
        total = sum(len(v) for v in diff.categories_added.values())
        warnings.append(
            f"{total} category value(s) appear that were not in the earlier version. A model "
            "trained on the earlier one has learned nothing about them."
        )
    if diff.distributions_moved:
        worst = max(diff.distributions_moved, key=lambda d: abs(d["shift"]))
        warnings.append(
            f"`{worst['column']}` has moved {worst['shift']:+.1f} standard deviations. A "
            "difference in results between these versions may be the data rather than anything "
            "you changed."
        )
    return headline + ("\n\n" + "\n\n".join(warnings) if warnings else "")
