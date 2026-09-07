"""Per-column imputation choices, and what each one would actually put in the gaps.

A pipeline that says "median imputation applied to 3 columns" is a decision the
reader has to take on trust. This computes the value each choice would insert,
so the choice can be judged before it is made — and so a strategy that would
fill a skewed column with a number nobody believes is visible as such.

Nothing here fits anything for real. These previews are computed on the whole
frame purely to be *shown*; the pipeline still fits its imputers inside each
cross-validation fold, and the numbers it learns there will differ slightly.
That difference is stated wherever a preview is displayed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import SemanticType

__all__ = ["STRATEGIES", "strategies_for", "preview_fill", "steps_for_choices", "Choice"]

#: label -> (step key, what it does, which semantic types it suits)
STRATEGIES: dict[str, dict[str, Any]] = {
    "Leave as is": {
        "step": None,
        "note": "The gaps stay. Only a few models tolerate that — most will refuse to train.",
        "numeric": True, "categorical": True,
    },
    "Median of the column": {
        "step": "impute_median",
        "note": "The safe default for numeric columns: unmoved by outliers, unlike the mean.",
        "numeric": True, "categorical": False,
    },
    "Mean of the column": {
        "step": "impute_mean",
        "note": "Right for a roughly symmetric column. A skewed column drags the mean away from "
                "where most of the data actually is.",
        "numeric": True, "categorical": False,
    },
    "Median within a group": {
        "step": "impute_by_group", "strategy": "median",
        "note": "The median of the row's own group — its region, tier or segment. Keeps the "
                "variation between groups that a single column-wide number would flatten.",
        "numeric": True, "categorical": False, "needs_group": True,
    },
    "Mean within a group": {
        "step": "impute_by_group", "strategy": "mean",
        "note": "The mean of the row's own group. Same idea as above; same caution about skew.",
        "numeric": True, "categorical": False, "needs_group": True,
    },
    "Most frequent value": {
        "step": "impute_mode",
        "note": "The dominant category. Over-represents it further, which matters when the gaps "
                "are numerous.",
        "numeric": False, "categorical": True,
    },
    "Most frequent within a group": {
        "step": "impute_by_group", "strategy": "most_frequent",
        "note": "The commonest value inside the row's own group.",
        "numeric": False, "categorical": True, "needs_group": True,
    },
    "A fixed value": {
        "step": "impute_constant",
        "note": "Treats 'missing' as a level in its own right. The honest choice when the fact "
                "that a value is absent carries information.",
        "numeric": True, "categorical": True, "needs_value": True,
    },
    "From similar rows (KNN)": {
        "step": "impute_knn",
        "note": "Estimated from the most similar complete rows. Better when columns are "
                "correlated; slow on large data and sensitive to scale.",
        "numeric": True, "categorical": False,
    },
    "Modelled from the other columns (MICE)": {
        "step": "impute_iterative",
        "note": "Each column with gaps is modelled from the others, repeatedly. The most "
                "sophisticated option, and the one most able to invent structure that is not there.",
        "numeric": True, "categorical": False,
    },
    "Carry the last value forward": {
        "step": "impute_forward_fill",
        "note": "Only valid when the rows are genuinely in time order.",
        "numeric": True, "categorical": True,
    },
    "Drop the column": {
        "step": "drop_columns",
        "note": "Sometimes the right answer. Imputing a column that is mostly absent invents most "
                "of the variable.",
        "numeric": True, "categorical": True,
    },
}


class Choice:
    """One column's imputation decision, as made in the interface."""

    def __init__(self, column: str, strategy: str, group_by: str | None = None,
                 fill_value: str = "") -> None:
        self.column = column
        self.strategy = strategy
        self.group_by = group_by
        self.fill_value = fill_value

    @property
    def spec(self) -> dict[str, Any]:
        return STRATEGIES[self.strategy]

    @property
    def is_action(self) -> bool:
        return self.spec["step"] is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Choice({self.column!r}, {self.strategy!r}, group_by={self.group_by!r})"


def strategies_for(semantic_type: SemanticType) -> list[str]:
    """The strategies that make sense for this kind of column."""
    numeric = semantic_type in (SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE)
    key = "numeric" if numeric else "categorical"
    return [name for name, spec in STRATEGIES.items() if spec[key]]


def preview_fill(frame: pd.DataFrame, choice: Choice) -> tuple[str, pd.DataFrame | None]:
    """What this choice would insert: a one-line answer, and a per-group table where relevant."""
    if not choice.is_action:
        return "Nothing — the gaps remain.", None
    column = choice.column
    if column not in frame.columns:
        return "—", None
    series = frame[column]
    step = choice.spec["step"]

    if step == "drop_columns":
        return "The column is removed entirely.", None
    if step == "impute_constant":
        return f"The literal value {choice.fill_value or '__missing__'!r} in every gap.", None
    if step == "impute_forward_fill":
        return "Each gap takes the value of the row above it.", None
    if step == "impute_median":
        return _one(series.median()), None
    if step == "impute_mean":
        return _one(series.mean()), None
    if step == "impute_mode":
        modes = series.mode(dropna=True)
        return _one(modes.iloc[0] if len(modes) else None), None
    if step in ("impute_knn", "impute_iterative"):
        return ("A different value per row, estimated from the other columns — there is no single "
                "number to show."), None

    if step == "impute_by_group":
        if not choice.group_by or choice.group_by not in frame.columns:
            return "Pick a grouping column to see what this would fill in.", None
        statistic = choice.spec.get("strategy", "median")
        grouped = series.groupby(frame[choice.group_by].astype("string"), dropna=True)
        if statistic == "mean":
            values = grouped.mean()
        elif statistic == "most_frequent":
            values = grouped.apply(lambda s: s.mode(dropna=True).iloc[0]
                                   if len(s.mode(dropna=True)) else np.nan)
        else:
            values = grouped.median()
        counts = grouped.count()
        gaps = series.isna().groupby(frame[choice.group_by].astype("string")).sum()
        table = pd.DataFrame({
            choice.group_by: values.index.astype(str),
            f"{statistic} of {column}": [_number(v) for v in values.to_numpy()],
            "rows it is based on": counts.reindex(values.index).fillna(0).astype(int).to_numpy(),
            "gaps it would fill": gaps.reindex(values.index).fillna(0).astype(int).to_numpy(),
        })
        thin = table[table["rows it is based on"] < 5]
        summary = f"A different value per {choice.group_by} — {len(table)} group(s)."
        if len(thin):
            summary += (f" {len(thin)} of them rest on fewer than 5 rows, which is barely better "
                        "than a guess.")
        return summary, table
    return "—", None


def steps_for_choices(choices: list[Choice]) -> list[dict[str, Any]]:
    """Turn per-column decisions into pipeline steps, merged where identical.

    Three columns all taking the column median become one step over three
    columns, not three steps — a pipeline the reader can still follow.
    """
    merged: dict[tuple, list[str]] = {}
    extras: dict[tuple, dict[str, Any]] = {}
    for choice in choices:
        if not choice.is_action:
            continue
        spec = choice.spec
        params: dict[str, Any] = {}
        if spec["step"] == "impute_by_group":
            params = {"group_by": choice.group_by or "", "strategy": spec.get("strategy", "median")}
        elif spec["step"] == "impute_constant":
            params = {"fill_value": choice.fill_value or "__missing__"}
        signature = (spec["step"], tuple(sorted(params.items())))
        merged.setdefault(signature, []).append(choice.column)
        extras[signature] = params

    steps: list[dict[str, Any]] = []
    for (step_key, _), columns in merged.items():
        signature = (step_key, tuple(sorted(extras[(step_key, _)].items())))
        params = dict(extras[signature])
        if step_key == "drop_columns":
            # This step takes its columns as a parameter rather than a scope.
            steps.append({"step_key": step_key, "columns": None,
                          "params": {"columns": sorted(columns)}})
        else:
            steps.append({"step_key": step_key, "columns": sorted(columns), "params": params})
    return steps


def _one(value: Any) -> str:
    return f"The single value {_number(value)} in every gap."


def _number(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int, float, np.integer, np.floating)):
        from dsai.engines.metrics import human_number

        return human_number(value)
    return str(value)
