"""Decide which charts are worth drawing.

The rule is restraint: a chart earns its place by answering a question the
numbers alone do not. Twelve histograms of near-identical distributions inform
nobody, so this module proposes a small ranked set tied to what the profile and
the analysis actually found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsai.core.schema import DatasetProfile, JsonMixin, SemanticType, TaskType


@dataclass
class ChartSpec(JsonMixin):
    """One proposed chart, with the reason it is worth drawing."""

    kind: str                       # histogram | box | scatter | bar | line | heatmap | ...
    title: str
    columns: list[str] = field(default_factory=list)
    reason: str = ""
    priority: float = 0.5
    options: dict[str, Any] = field(default_factory=dict)
    group_by: str | None = None


def recommend_charts(
    profile: DatasetProfile,
    objective: Any = None,
    run: Any = None,
    max_charts: int = 12,
) -> list[ChartSpec]:
    """Propose the charts this dataset and analysis actually warrant."""
    charts: list[ChartSpec] = []
    task = objective.task_type if objective is not None else TaskType.EXPLORATORY
    target = objective.target if objective is not None else None

    charts.extend(_quality_charts(profile))
    charts.extend(_distribution_charts(profile, target))
    charts.extend(_relationship_charts(profile, target, task))
    if run is not None:
        charts.extend(_model_charts(run, task))
        charts.extend(_segment_charts(run))
        charts.extend(_forecast_charts(run, task))

    charts.sort(key=lambda c: c.priority, reverse=True)
    return charts[:max_charts]


def _quality_charts(profile: DatasetProfile) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    missing = [n for n, c in profile.columns.items() if c.missing_pct > 0]
    if len(missing) >= 2:
        charts.append(
            ChartSpec(
                kind="missing_bar",
                title="Missing values by column",
                columns=missing,
                reason=(
                    f"{len(missing)} column(s) have gaps. Seeing which ones, and how much, decides "
                    "whether to impute, drop or investigate."
                ),
                priority=0.75 if len(missing) > 3 else 0.55,
            )
        )
    return charts


def _distribution_charts(profile: DatasetProfile, target: str | None) -> list[ChartSpec]:
    charts: list[ChartSpec] = []

    if target and target in profile.columns:
        column = profile.columns[target]
        if column.is_numeric:
            charts.append(
                ChartSpec(
                    kind="histogram",
                    title=f"Distribution of {target}",
                    columns=[target],
                    reason=(
                        "The shape of the outcome decides which models are appropriate and whether "
                        "a mean is a meaningful summary at all."
                    ),
                    priority=0.95,
                    options={"show_median": True, "bins": 40},
                )
            )
        elif column.is_categorical:
            charts.append(
                ChartSpec(
                    kind="bar",
                    title=f"Class balance of {target}",
                    columns=[target],
                    reason=(
                        "Class balance determines which metrics are honest. If one class dominates, "
                        "accuracy becomes meaningless."
                    ),
                    priority=0.95,
                )
            )

    # Only draw distributions that show something: skew, outliers or heavy tails.
    interesting = [
        (name, col) for name, col in profile.columns.items()
        if col.is_numeric and name != target
        and (abs(col.skewness or 0) > 1 or col.outlier_pct > 3)
    ]
    interesting.sort(key=lambda t: abs(t[1].skewness or 0) + t[1].outlier_pct / 10, reverse=True)
    for name, column in interesting[:3]:
        reasons = []
        if abs(column.skewness or 0) > 1:
            reasons.append(f"skewness {column.skewness:.1f}")
        if column.outlier_pct > 3:
            reasons.append(f"{column.outlier_pct:.0f}% outliers")
        charts.append(
            ChartSpec(
                kind="histogram",
                title=f"Distribution of {name}",
                columns=[name],
                reason=f"Not symmetric ({', '.join(reasons)}) — the shape changes how it should be modelled and reported.",
                priority=0.6,
                options={"show_median": True},
            )
        )

    outliery = [
        name for name, col in profile.columns.items()
        if col.is_numeric and col.outlier_pct >= 5 and name != target
    ]
    if len(outliery) >= 2:
        charts.append(
            ChartSpec(
                kind="box_multi",
                title="Outliers across numeric variables",
                columns=outliery[:8],
                reason=f"{len(outliery)} variables carry heavy tails — a box plot shows where and how far.",
                priority=0.65,
            )
        )
    return charts


def _relationship_charts(profile: DatasetProfile, target: str | None, task: TaskType) -> list[ChartSpec]:
    charts: list[ChartSpec] = []

    numeric = profile.numeric_columns
    if len(numeric) >= 4:
        charts.append(
            ChartSpec(
                kind="correlation_heatmap",
                title="Correlation between numeric variables",
                columns=numeric[:25],
                reason=(
                    "Shows redundancy and multicollinearity at a glance — which variables are "
                    "really carrying the same information."
                ),
                priority=0.8 if any(abs(r) > 0.7 for _, _, r in profile.correlation_pairs) else 0.6,
            )
        )

    if target and target in profile.columns and profile.columns[target].is_numeric:
        related = [
            (a, b, r) for a, b, r in profile.correlation_pairs
            if target in (a, b) and abs(r) >= 0.25
        ][:3]
        for a, b, r in related:
            other = b if a == target else a
            charts.append(
                ChartSpec(
                    kind="scatter",
                    title=f"{target} against {other}",
                    columns=[other, target],
                    reason=(
                        f"They correlate at r = {r:.2f}. The scatter shows whether that is a genuine "
                        "straight-line relationship or an artefact of a few extreme points."
                    ),
                    priority=0.85 - 0.05 * related.index((a, b, r)),
                    options={"trendline": True},
                )
            )

        categorical = [
            n for n, c in profile.columns.items()
            if c.semantic_type in {SemanticType.CATEGORICAL_NOMINAL, SemanticType.CATEGORICAL_ORDINAL,
                                   SemanticType.BINARY}
            and c.n_unique <= 12
        ]
        for name in categorical[:2]:
            charts.append(
                ChartSpec(
                    kind="box_by_group",
                    title=f"{target} by {name}",
                    columns=[target],
                    group_by=name,
                    reason=(
                        f"Compares the distribution of {target} across {name} — differences in spread "
                        "matter as much as differences in average."
                    ),
                    priority=0.7,
                )
            )

    if task.is_classification and target:
        numeric_features = [n for n in numeric if n != target][:2]
        for name in numeric_features:
            charts.append(
                ChartSpec(
                    kind="box_by_group",
                    title=f"{name} by {target}",
                    columns=[name],
                    group_by=target,
                    reason=f"Shows whether {name} actually separates the classes, or overlaps entirely.",
                    priority=0.7,
                )
            )

    if profile.datetime_columns and numeric:
        time_column = profile.datetime_columns[0]
        value = target if target in numeric else numeric[0]
        charts.append(
            ChartSpec(
                kind="line",
                title=f"{value} over time",
                columns=[time_column, value],
                reason="A time dimension exists — trend and seasonality are invisible in a summary table.",
                priority=0.9,
            )
        )
    return charts


def _model_charts(run: Any, task: TaskType) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    if run.best is None:
        return charts

    if run.explanation and run.explanation.importances:
        charts.append(
            ChartSpec(
                kind="feature_importance",
                title="What the model relies on",
                columns=[f.feature for f in run.explanation.importances[:15]],
                reason="Ranks the variables driving the prediction, which is what the findings rest on.",
                priority=1.0,
            )
        )

    if task is TaskType.REGRESSION:
        charts.append(
            ChartSpec(
                kind="predicted_vs_actual",
                title="Predicted against actual",
                reason=(
                    "The single most informative model chart: it shows where the model works and "
                    "where it systematically misses."
                ),
                priority=0.98,
            )
        )
        charts.append(
            ChartSpec(
                kind="residual_plot",
                title="Residuals against fitted values",
                reason=(
                    "Reveals bias and changing error variance. A funnel shape means the model is "
                    "more reliable at one end of the range."
                ),
                priority=0.85,
            )
        )
    elif task.is_classification:
        charts.append(
            ChartSpec(
                kind="confusion_matrix",
                title="Confusion matrix",
                reason="Shows which errors the model actually makes, which accuracy alone hides.",
                priority=0.98,
            )
        )
        if run.best.test_scores.get("roc_auc") is not None:
            charts.append(
                ChartSpec(
                    kind="roc_curve",
                    title="ROC curve",
                    reason="Shows the trade-off between catching positives and raising false alarms at every threshold.",
                    priority=0.85,
                )
            )
            charts.append(
                ChartSpec(
                    kind="precision_recall_curve",
                    title="Precision-recall curve",
                    reason="More honest than ROC when the positive class is rare.",
                    priority=0.8,
                )
            )

    if run.tournament and len(run.tournament.ranked) > 2:
        charts.append(
            ChartSpec(
                kind="model_comparison",
                title="Model comparison",
                reason="Shows how close the field is — a narrow spread means the choice of model barely matters.",
                priority=0.75,
            )
        )
    return charts


def _segment_charts(run: Any) -> list[ChartSpec]:
    if run.segmentation is None or not getattr(run.segmentation, "clusters", None):
        return []
    return [
        ChartSpec(
            kind="cluster_scatter",
            title="Segments in two dimensions (PCA projection)",
            reason=(
                "Shows whether the segments are genuinely separated or an arbitrary slice of one "
                "continuous population."
            ),
            priority=0.95,
        ),
        ChartSpec(
            kind="cluster_profile",
            title="What distinguishes each segment",
            reason="Compares each segment against the overall average on the variables that separate them.",
            priority=0.9,
        ),
        ChartSpec(
            kind="cluster_sizes",
            title="Segment sizes",
            reason="A segment too small to act on is not a segment.",
            priority=0.6,
        ),
    ]


def _forecast_charts(run: Any, task: TaskType) -> list[ChartSpec]:
    if task is not TaskType.TIME_SERIES_FORECAST or run.best is None:
        return []
    charts = [
        ChartSpec(
            kind="forecast",
            title="History and forecast",
            reason="The forecast is only meaningful next to the history it was fitted on.",
            priority=1.0,
            options={"show_interval": bool(run.best.extras.get("forecast_lower"))},
        )
    ]
    if run.series_analysis and run.series_analysis.get("decomposition", {}).get("supported"):
        charts.append(
            ChartSpec(
                kind="decomposition",
                title="Trend, seasonality and residual",
                reason="Separates the underlying trend from the repeating pattern and the noise.",
                priority=0.85,
            )
        )
    return charts


def explain_chart_choice(charts: list[ChartSpec]) -> str:
    lines = [f"{len(charts)} chart(s) were selected, each because it answers something specific:", ""]
    lines += [f"- **{c.title}** — {c.reason}" for c in charts]
    lines.append("")
    lines.append(
        "Charts that would only restate the summary table were deliberately not drawn."
    )
    return "\n".join(lines)
