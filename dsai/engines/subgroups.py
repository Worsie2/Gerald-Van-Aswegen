"""Where the model works badly, and whether that difference is real.

An average error hides its own distribution. A model with a mean error of 10%
might be at 6% almost everywhere and 40% on one region — and the region is where
the decisions get made.

The hard part is not finding groups with worse error; on any dataset, searching
enough subgroups will find one. It is deciding which differences are worth
reporting. Two safeguards, both deliberately conservative:

* **A floor on group size.** Nothing under 30 rows is reported, because the error
  in a group of nine is noise wearing the costume of a finding.
* **A comparison against how much error varies anyway.** A group is only flagged
  when its error stands outside what the spread of the errors themselves would
  produce — not merely when it is above average, which half of all groups are.

The wording is always a *measured difference*. Naming it bias or a fairness
failure would be a conclusion this evidence cannot support, and would let the
reader skip the investigation that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import TaskType
from dsai.engines.metrics import human_number

__all__ = ["Subgroup", "SubgroupReport", "discover_subgroups", "MIN_GROUP_ROWS"]

#: Below this, a group's error says more about which rows landed in it than
#: about the model.
MIN_GROUP_ROWS = 30


@dataclass
class Subgroup:
    label: str
    columns: list[str]
    n: int
    share: float
    error: float
    ratio: float
    #: How far outside ordinary variation this sits, in standard errors.
    z: float
    notable: bool
    reading: str = ""


@dataclass
class SubgroupReport:
    metric: str = ""
    overall_error: float = 0.0
    groups: list[Subgroup] = field(default_factory=list)
    notable: list[Subgroup] = field(default_factory=list)
    verdict: str = ""
    n_considered: int = 0
    n_too_small: int = 0
    note: str = ""
    usable: bool = False

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Group": g.label,
            "Rows": f"{g.n:,}",
            "Share": f"{g.share:.1%}",
            f"{self.metric}": human_number(g.error),
            "Against overall": f"{g.ratio:.2f}×",
            "Reading": g.reading,
        } for g in self.groups])


def discover_subgroups(
    frame: pd.DataFrame,
    actual: Any,
    predicted: Any,
    task_type: TaskType,
    candidate_columns: list[str] | None = None,
    max_depth: int = 2,
    min_rows: int = MIN_GROUP_ROWS,
) -> SubgroupReport:
    """Find where prediction error is materially different, with safeguards.

    ``max_depth`` of 2 allows combinations — "Region C *and* high spend" — which
    is usually where a real weakness hides. It is capped there deliberately:
    every extra level multiplies the number of groups searched, and the more
    groups searched the more certain it becomes that one of them looks bad by
    chance alone.
    """
    report = SubgroupReport()

    try:
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
    except Exception:
        report.note = "No held-out predictions to compare."
        return report

    if len(actual) != len(predicted) or len(actual) != len(frame):
        report.note = ("Predictions and rows could not be lined up, so no subgroup breakdown "
                       "is possible.")
        return report
    if len(frame) < min_rows * 2:
        report.note = (f"Only {len(frame):,} scored rows. A subgroup breakdown needs at least "
                       f"{min_rows * 2:,} to produce groups worth comparing.")
        return report

    if task_type is TaskType.REGRESSION:
        errors = np.abs(actual.astype(float) - predicted.astype(float))
        report.metric = "mean absolute error"
    else:
        errors = (np.asarray([str(a) for a in actual])
                  != np.asarray([str(p) for p in predicted])).astype(float)
        report.metric = "error rate"

    errors = pd.Series(errors, index=frame.index)
    report.overall_error = float(errors.mean())
    spread = float(errors.std(ddof=1))
    if not np.isfinite(spread) or spread <= 0:
        report.note = "Every row has the same error, so no group differs from any other."
        return report

    columns = candidate_columns or _candidate_columns(frame)
    if not columns:
        report.note = "No categorical or bandable column to break the results down by."
        return report

    facets = _facets(frame, columns)
    seen: set[tuple] = set()

    for depth in range(1, max_depth + 1):
        for combination in _combinations(facets, depth):
            key = tuple(sorted(label for label, _ in combination))
            if key in seen:
                continue
            seen.add(key)
            mask = np.logical_and.reduce([m.to_numpy() for _, m in combination])
            n = int(mask.sum())
            report.n_considered += 1
            if n < min_rows:
                report.n_too_small += 1
                continue
            if n >= len(frame) * 0.95:
                continue                      # "everyone" is not a subgroup

            group_error = float(errors[mask].mean())
            ratio = group_error / report.overall_error if report.overall_error > 1e-12 else 1.0
            # How far this group's mean sits from the overall mean, measured in
            # standard errors of its own size. A big group needs a smaller
            # difference to be notable than a small one, which is correct.
            standard_error = spread / np.sqrt(n)
            z = (group_error - report.overall_error) / standard_error if standard_error else 0.0

            group = Subgroup(
                label=" and ".join(label for label, _ in combination),
                columns=[label.split(" ")[0] for label, _ in combination],
                n=n, share=n / len(frame), error=group_error, ratio=ratio, z=float(z),
                notable=bool(abs(z) >= 3 and (ratio >= 1.3 or ratio <= 0.75)),
            )
            group.reading = _read(group)
            report.groups.append(group)

    report.groups.sort(key=lambda g: -g.ratio)
    report.notable = [g for g in report.groups if g.notable]
    report.usable = bool(report.groups)
    report.verdict = _verdict(report, min_rows)
    return report


def _candidate_columns(frame: pd.DataFrame) -> list[str]:
    """Columns worth splitting on: real categories, and numerics cut into thirds."""
    out: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) > 8:
                out.append(column)
        elif 1 < series.nunique(dropna=True) <= 25:
            out.append(column)
    return out[:8]


def _facets(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, pd.Series]]:
    """Every single condition worth testing, as a labelled boolean mask."""
    facets: list[tuple[str, pd.Series]] = []
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            # Thirds rather than arbitrary cut-points: a band a reader can name.
            try:
                low, high = series.quantile(1 / 3), series.quantile(2 / 3)
            except Exception:
                continue
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                continue
            facets.append((f"{column} low (≤{human_number(low)})", series <= low))
            facets.append((f"{column} high (>{human_number(high)})", series > high))
        else:
            text = series.astype("string")
            for value in text.dropna().value_counts().head(8).index:
                facets.append((f"{column} = {value}", text == value))
    return facets


def _combinations(facets: list[tuple[str, pd.Series]], depth: int):
    from itertools import combinations

    for group in combinations(facets, depth):
        # Two conditions on the same column would be contradictory or redundant.
        columns = [label.split(" ")[0] for label, _ in group]
        if len(set(columns)) == len(columns):
            yield group


def _read(group: Subgroup) -> str:
    if not group.notable:
        return "in line with the overall error"
    if group.ratio > 1:
        return f"{group.ratio:.1f}× the overall error — the model does worse here"
    return f"{group.ratio:.1f}× the overall error — the model does better here"


def _verdict(report: SubgroupReport, min_rows: int) -> str:
    if not report.groups:
        return (f"No group had at least {min_rows} rows, so nothing could be compared. "
                f"{report.n_too_small} candidate group(s) were too small.")
    if not report.notable:
        return (
            f"**No group stands out.** {len(report.groups)} group(s) of at least {min_rows} rows "
            f"were compared and none differs from the overall {report.metric} by more than "
            "ordinary variation would produce. That is not a guarantee the model treats every "
            "group equally — only that nothing measurable separates the groups that were tested."
        )
    worst = max(report.notable, key=lambda g: g.ratio)
    lines = [
        f"**{len(report.notable)} group(s) differ materially.** The largest gap is "
        f"**{worst.label}** — {worst.n:,} rows, {worst.ratio:.1f}× the overall {report.metric}.",
        "",
        "This is a measured difference in prediction error, not a finding of bias or unfairness. "
        "Why the model does worse there is a question about the data and the domain, and it is "
        "yours to investigate — the difference could be genuine heterogeneity, fewer training "
        "rows in that group, or a variable that means something different within it.",
    ]
    if report.n_considered > 40:
        lines += [
            "",
            f"{report.n_considered} group(s) were searched. The more combinations tested, the "
            "more likely one looks unusual by chance — these were filtered to differences beyond "
            "three standard errors, but treat a single flagged group as a lead to check rather "
            "than a result.",
        ]
    return "\n".join(lines)
