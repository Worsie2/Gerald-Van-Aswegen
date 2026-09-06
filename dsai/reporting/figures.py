"""The charts that belong in a report, and how to get them into a file.

A report that says "annual spend rises with meters installed" and shows nothing
is asking the reader to take that on trust. The same sentence beside the scatter
is checkable in a second. So the report carries charts.

Three things travel with every figure, and none of them is optional:

* a **caption** that states what the chart shows in words, so the point survives
  a reader who skims, a screen reader, and a printer that drops the image;
* an **alt** line, one sentence, for the same reasons; and
* a **table** of the numbers behind it, so a reader who cannot see the chart —
  or does not believe it — can read the figures directly.

Static image export needs a browser (Plotly renders through one). Where that is
missing the caption and the table carry the section on their own rather than the
export failing, which is why every accessor here returns ``None`` instead of
raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dsai.core.schema import TaskType

__all__ = ["ReportFigure", "build_figures", "to_png", "to_html_div", "png_available"]


@dataclass
class ReportFigure:
    """One chart, with everything a report needs in order to place it."""

    key: str
    title: str
    #: What the chart shows, in words. Stands alone if the image never renders.
    caption: str
    #: One sentence for ``alt``/screen readers.
    alt: str
    #: The heading this figure belongs under, matched against report section names.
    after_section: str
    figure: Any = None
    #: The numbers behind the chart, so the chart is never the only way to read them.
    table: pd.DataFrame | None = None
    height: int = 420

    @property
    def has_table(self) -> bool:
        return self.table is not None and not self.table.empty


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def png_available() -> bool:
    """Whether static image export will work in this environment."""
    try:
        import plotly.graph_objects as go

        go.Figure().to_image(format="png", width=10, height=10)
        return True
    except Exception:
        return False


def to_png(figure: Any, width: int = 900, height: int = 420, scale: float = 2.0) -> bytes | None:
    """PNG bytes, or ``None`` if this machine cannot render one.

    Plotly's static export drives a headless browser. On a machine without one
    the report still has to come out, so the failure is a missing picture rather
    than a missing report.
    """
    if figure is None:
        return None
    try:
        return figure.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def to_html_div(figure: Any, include_js: bool = False) -> str | None:
    """A self-contained interactive chart, ready to drop into a report body.

    The Plotly library is inlined into the first figure rather than pulled from a
    CDN, so the report opens on a machine with no internet — which is the machine
    a report tends to be read on.
    """
    if figure is None:
        return None
    try:
        return figure.to_html(
            full_html=False,
            include_plotlyjs=True if include_js else False,
            config={"displaylogo": False, "responsive": True},
            default_height=None,
        )
    except Exception:
        return None


# --------------------------------------------------------------------------
# assembling the set
# --------------------------------------------------------------------------

def build_figures(run: Any, mode: str = "light", frame: pd.DataFrame | None = None) -> list[ReportFigure]:
    """Every chart this particular run has the material to draw.

    Each one is attempted independently: a chart that cannot be drawn is left
    out, never allowed to take the report down with it.
    """
    from dsai.viz import plots

    figures: list[ReportFigure] = []

    def attempt(builder) -> None:
        try:
            item = builder()
        except Exception:
            return
        if item is not None and item.figure is not None:
            figures.append(item)

    profile = run.profile
    objective = run.objective
    best = run.best

    # ---- data quality --------------------------------------------------
    if profile is not None and any(c.n_missing for c in profile.columns.values()):
        attempt(lambda: ReportFigure(
            key="missing",
            title="Missing values by column",
            caption=_missing_caption(profile),
            alt="Bar chart of the share of missing values in each column that has any.",
            after_section="Data quality",
            figure=plots.missing_bar(profile, mode=mode),
            table=_missing_table(profile),
            height=340,
        ))

    # ---- relationships in the raw data ---------------------------------
    # With the data to hand this is the full heatmap. Without it — a run reloaded
    # from disk, say — the profile still carries the measured pairs, so the same
    # point is made as a ranked bar rather than dropped.
    if frame is not None and profile is not None and len(profile.numeric_columns) >= 3:
        attempt(lambda: ReportFigure(
            key="correlation",
            title="How the numeric variables move together",
            caption=_correlation_caption(frame, profile),
            alt="Heatmap of pairwise correlations between the numeric columns.",
            after_section="The data",
            figure=plots.correlation_heatmap(frame, profile.numeric_columns, mode=mode),
            table=_correlation_table(frame, profile),
            height=460,
        ))
    elif profile is not None and getattr(profile, "correlation_pairs", None):
        pairs = sorted(profile.correlation_pairs, key=lambda p: -abs(p[2]))[:12]
        attempt(lambda: ReportFigure(
            key="correlation",
            title="Strongest relationships between variables",
            caption=(
                f"The {len(pairs)} strongest pairwise correlations measured in the data, strongest "
                "first. These are associations, not causes — two variables can move together because "
                "a third drives both."
            ),
            alt="Bar chart of the strongest pairwise correlations between numeric columns.",
            after_section="The data",
            figure=plots.bar([f"{a} ↔ {b}" for a, b, _ in pairs], [v for _, _, v in pairs],
                             title="Strongest relationships", mode=mode, orientation="h"),
            table=pd.DataFrame([{"Variable A": a, "Variable B": b, "Correlation": round(float(v), 3)}
                                for a, b, v in pairs]),
            height=max(300, 60 + 30 * len(pairs)),
        ))

    # ---- the tournament ------------------------------------------------
    if run.tournament is not None and getattr(run.tournament, "table", None):
        attempt(lambda: ReportFigure(
            key="model_comparison",
            title="Model comparison",
            caption=_tournament_caption(run),
            alt="Bar chart comparing every model trained, on the metric the comparison was ranked by.",
            after_section="Model comparison",
            figure=plots.model_comparison(run.tournament, mode=mode),
            table=pd.DataFrame(run.tournament.table),
            height=max(340, 60 + 34 * len(run.tournament.table)),
        ))

    # ---- what the model relies on --------------------------------------
    if run.explanation is not None and getattr(run.explanation, "importances", None):
        attempt(lambda: ReportFigure(
            key="importance",
            title="What the model relies on",
            caption=_importance_caption(run),
            alt="Bar chart of the variables the selected model relies on most, strongest first.",
            after_section="What drives the outcome",
            figure=plots.feature_importance(run.explanation, mode=mode),
            table=pd.DataFrame([
                {"Variable": i.feature, "Importance": round(float(i.importance), 4),
                 "Direction": getattr(i, "direction", "") or "—"}
                for i in run.explanation.top(15)
            ]),
            height=max(320, 60 + 30 * len(run.explanation.top(15))),
        ))

    # ---- fit diagnostics -----------------------------------------------
    if best is not None and best.task_type is TaskType.REGRESSION:
        actual = best.extras.get("holdout_actual")
        predicted = best.extras.get("holdout_predicted")
        if actual and predicted:
            attempt(lambda: ReportFigure(
                key="predicted_vs_actual",
                title="Predicted against actual",
                caption=(
                    "Each point is one held-out row: what the model predicted against what actually "
                    "happened. Points on the diagonal are exact. Systematic drift away from the line "
                    "at one end of the range is bias the headline score does not show."
                ),
                alt="Scatter plot of predicted against actual values on the held-out rows, with the "
                    "line of perfect prediction.",
                after_section="Selected model and performance",
                figure=plots.predicted_vs_actual(actual, predicted, mode=mode),
            ))
            attempt(lambda: ReportFigure(
                key="residuals",
                title="Residuals against fitted values",
                caption=(
                    "The error left over, plotted against what the model predicted. A shapeless band "
                    "around zero is what you want. A funnel or a curve means the model is missing "
                    "structure that is still in the data."
                ),
                alt="Scatter plot of residuals against fitted values.",
                after_section="Selected model and performance",
                figure=plots.residual_plot(actual, predicted, mode=mode),
            ))

    if best is not None and best.task_type.is_classification:
        if run.diagnostics.get("confusion"):
            attempt(lambda: ReportFigure(
                key="confusion",
                title="Confusion matrix",
                caption=(
                    "Rows are what actually happened, columns what the model said. The off-diagonal "
                    "cells are the two kinds of mistake, and they usually do not cost the same."
                ),
                alt="Confusion matrix of actual against predicted classes.",
                after_section="Selected model and performance",
                figure=plots.confusion_matrix(run.diagnostics["confusion"], mode=mode),
                table=pd.DataFrame(run.diagnostics.get("per_class", [])) or None,
            ))
        scores = best.extras.get("holdout_score")
        positive = best.extras.get("holdout_positive_label")
        if scores and positive is not None:
            binary = [1 if str(a) == str(positive) else 0 for a in best.extras["holdout_actual"]]
            attempt(lambda: ReportFigure(
                key="roc",
                title="ROC curve",
                caption=(
                    "How the true-positive rate trades against the false-positive rate as the decision "
                    "threshold moves. The area under it is the headline number; the shape says where "
                    "along the range the model is actually useful."
                ),
                alt="ROC curve for the selected model against the diagonal of random guessing.",
                after_section="Selected model and performance",
                figure=plots.roc_curve(binary, scores, mode=mode),
            ))
            attempt(lambda: ReportFigure(
                key="precision_recall",
                title="Precision against recall",
                caption=(
                    "More honest than ROC when the classes are unbalanced: it ignores the large, easy "
                    "negative class and shows what catching more positives costs in false alarms."
                ),
                alt="Precision-recall curve for the selected model.",
                after_section="Selected model and performance",
                figure=plots.precision_recall_curve(binary, scores, mode=mode),
            ))

    # ---- segments -------------------------------------------------------
    if run.segmentation is not None and getattr(run.segmentation, "clusters", None):
        attempt(lambda: ReportFigure(
            key="cluster_sizes",
            title="Segment sizes",
            caption=_segment_size_caption(run),
            alt="Bar chart of how many rows fall in each segment.",
            after_section="Key findings",
            figure=plots.cluster_sizes(run.segmentation, mode=mode),
            table=pd.DataFrame([
                {"Segment": c.name, "Rows": c.size, "Share": f"{c.share:.1%}"}
                for c in run.segmentation.clusters
            ]),
            height=320,
        ))
        attempt(lambda: ReportFigure(
            key="cluster_profile",
            title="What distinguishes each segment",
            caption=(
                "Each variable shown as how far the segment sits from the overall average, in standard "
                "deviations. A bar near zero means the segment is unremarkable on that variable; the "
                "long bars are what the segment is actually about."
            ),
            alt="Grouped bar chart of how far each segment sits from the overall average on each variable.",
            after_section="Key findings",
            figure=plots.cluster_profile(run.segmentation, mode=mode),
            height=440,
        ))

    # ---- time series -----------------------------------------------------
    if run.series_analysis and best is not None:
        attempt(lambda: ReportFigure(
            key="forecast",
            title="History and forecast",
            caption=_forecast_caption(run),
            alt="Line chart of the historical series continuing into the forecast, with its interval.",
            after_section="Key findings",
            figure=plots.forecast_plot(best, mode=mode,
                                       target_name=(objective.target if objective else "value") or "value"),
            height=420,
        ))
        decomposition = run.series_analysis.get("decomposition", {})
        if decomposition.get("supported"):
            attempt(lambda: ReportFigure(
                key="decomposition",
                title="Trend, seasonality and what is left",
                caption=(
                    "The series split into its long-run movement, its repeating pattern and the residual. "
                    "If the residual still has visible shape, something systematic has not been captured."
                ),
                alt="Three stacked line charts: the trend, the seasonal component and the residual.",
                after_section="Key findings",
                figure=plots.decomposition_plot(decomposition, mode=mode),
                height=520,
            ))

    return figures


# --------------------------------------------------------------------------
# captions — each one says what the chart shows, not what a chart is
# --------------------------------------------------------------------------

def _missing_caption(profile: Any) -> str:
    worst = sorted(profile.columns.values(), key=lambda c: c.missing_pct, reverse=True)[:3]
    named = ", ".join(f"{c.name} ({c.missing_pct:.1f}%)" for c in worst if c.n_missing)
    return (
        f"Columns with values missing, worst first: {named}. Where a value is missing matters as much "
        "as how many — missingness concentrated in one group biases whatever is built on top of it."
    ) if named else "No column has a missing value."


def _missing_table(profile: Any) -> pd.DataFrame:
    rows = [
        {"Column": c.name, "Missing": c.n_missing, "Share": f"{c.missing_pct:.1f}%"}
        for c in profile.columns.values() if c.n_missing
    ]
    return pd.DataFrame(sorted(rows, key=lambda r: -r["Missing"]))


def _correlation_caption(frame: pd.DataFrame, profile: Any) -> str:
    pairs = _top_correlations(frame, profile, n=3)
    if not pairs:
        return "Pairwise correlations between the numeric columns."
    described = "; ".join(f"{a} and {b} at {v:+.2f}" for a, b, v in pairs)
    return (
        f"The strongest relationships are {described}. These are associations, not causes — two "
        "variables can move together because a third drives both."
    )


def _correlation_table(frame: pd.DataFrame, profile: Any) -> pd.DataFrame:
    return pd.DataFrame([
        {"Variable A": a, "Variable B": b, "Correlation": round(v, 3)}
        for a, b, v in _top_correlations(frame, profile, n=12)
    ])


def _top_correlations(frame: pd.DataFrame, profile: Any, n: int = 5) -> list[tuple[str, str, float]]:
    columns = [c for c in profile.numeric_columns if c in frame.columns]
    if len(columns) < 2:
        return []
    matrix = frame[columns].corr(numeric_only=True)
    seen: list[tuple[str, str, float]] = []
    for i, a in enumerate(matrix.columns):
        for b in matrix.columns[i + 1:]:
            value = matrix.loc[a, b]
            if pd.notna(value):
                seen.append((a, b, float(value)))
    return sorted(seen, key=lambda t: -abs(t[2]))[:n]


def _tournament_caption(run: Any) -> str:
    from dsai.engines.metrics import human_number

    metric = run.plan.primary_metric if run.plan else "the primary metric"
    rows = run.tournament.table
    best = run.best
    lead = ""
    if best is not None and len(rows) > 1:
        scores = [r.get(metric) for r in rows if isinstance(r.get(metric), (int, float))]
        if len(scores) > 1:
            lead = (
                f" {best.model_name} came out in front at "
                f"{human_number(best.test_scores.get(metric))}, against "
                f"{human_number(sorted(scores)[len(scores) // 2])} for the middle of the field."
            )
    return (
        f"Every model trained, compared on {metric} against the same held-out rows and the same "
        f"validation split.{lead} Best-performing on this dataset under this validation strategy — "
        "not best in general."
    )


def _importance_caption(run: Any) -> str:
    top = run.explanation.top(3)
    if not top:
        return "What the selected model relies on."
    named = ", ".join(i.feature for i in top)
    method = run.explanation.method or "the model's own importance measure"
    return (
        f"The selected model leans hardest on {named}. Measured by {method}. This is what the model "
        "uses, which is not the same as what causes the outcome — a variable can be important to a "
        "model because it stands in for something else."
    )


def _segment_size_caption(run: Any) -> str:
    clusters = run.segmentation.clusters
    largest = max(clusters, key=lambda c: c.size)
    return (
        f"{len(clusters)} segments. The largest, {largest.name}, holds {largest.size:,} rows "
        f"({largest.share:.0%}). Very uneven sizes are worth a second look: one tiny segment is often "
        "outliers rather than a group worth acting on."
    )


def _forecast_caption(run: Any) -> str:
    trend = run.series_analysis.get("trend", {}).get("direction", "")
    seasonal = run.series_analysis.get("seasonality", {}).get("detected")
    parts = ["History, then the forecast, with the interval around it."]
    if trend:
        parts.append(f"The series trends {trend}.")
    parts.append("A repeating seasonal pattern was detected." if seasonal
                 else "No repeating seasonal pattern was found.")
    parts.append("The interval widens with distance because uncertainty compounds; treat the far end "
                 "as a range, not a number.")
    return " ".join(parts)
