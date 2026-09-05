"""The Insight Engine: turn analytical output into findings a person can read.

Its one discipline is keeping the evidence types separate. A finding is labelled
as observed in the data, established by a statistical test, derived from a model,
interpreted by the platform, or asserted by the user — and never presented as a
stronger kind than it is.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, EvidenceKind, Finding, Objective, TaskType,
)
from dsai.engines import metrics as M
from dsai.engines.experiment import ExperimentResult
from dsai.explain.importance import ExplanationBundle


def generate_insights(
    profile: DatasetProfile,
    objective: Objective,
    context: BusinessContext | None = None,
    frame: pd.DataFrame | None = None,
    best: ExperimentResult | None = None,
    explanation: ExplanationBundle | None = None,
    tournament: Any = None,
    diagnostics: dict[str, Any] | None = None,
    segmentation: Any = None,
    series_analysis: dict[str, Any] | None = None,
    max_findings: int = 20,
) -> list[Finding]:
    """Collect findings from every source that produced one, then rank them."""
    context = context or BusinessContext()
    findings: list[Finding] = []

    findings.extend(_data_findings(profile, frame))
    findings.extend(_relationship_findings(profile, frame))
    findings.extend(_model_findings(objective, best, tournament, explanation))
    findings.extend(_driver_findings(explanation, objective, profile))
    findings.extend(_diagnostic_findings(diagnostics))
    findings.extend(_context_findings(context, profile))
    if segmentation is not None:
        findings.extend(getattr(segmentation, "findings", []))
    findings.extend(_series_findings(series_analysis, objective))

    return _rank(findings)[:max_findings]


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

def _data_findings(profile: DatasetProfile, frame: pd.DataFrame | None) -> list[Finding]:
    out: list[Finding] = []
    out.append(
        Finding(
            title=f"The dataset holds {profile.n_rows:,} rows across {profile.n_columns} variables",
            detail=(
                f"{len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical, "
                f"{len(profile.datetime_columns)} date and {len(profile.text_columns)} text column(s). "
                f"Data-quality score {profile.quality_score}/100."
            ),
            kind=EvidenceKind.OBSERVED,
            evidence=[f"{profile.n_rows:,} rows", f"{profile.n_columns} columns"],
            confidence=Confidence.HIGH,
        )
    )

    critical = profile.issues_by_severity("critical")
    if critical:
        out.append(
            Finding(
                title=f"{len(critical)} critical data-quality issue(s) need attention",
                detail=" ".join(i.message for i in critical[:3]),
                kind=EvidenceKind.OBSERVED,
                evidence=[f"{i.code}: {i.message}" for i in critical],
                columns=sorted({c for i in critical for c in i.columns})[:10],
                confidence=Confidence.HIGH,
            )
        )

    skewed = [
        (name, col.skewness) for name, col in profile.columns.items()
        if col.skewness is not None and abs(col.skewness) > 2
    ]
    if skewed:
        skewed.sort(key=lambda t: abs(t[1]), reverse=True)
        name, skew = skewed[0]
        column = profile.columns[name]
        out.append(
            Finding(
                title=f"'{name}' is heavily concentrated at one end of its range",
                detail=(
                    f"Skewness of {skew:.2f}. The median is {column.median:,.4g} while the mean is "
                    f"{column.mean:,.4g} — a gap that size means averages will mislead anyone reading "
                    "them. Report the median, or a distribution, instead."
                ),
                kind=EvidenceKind.OBSERVED,
                evidence=[
                    f"Median {column.median:,.4g} versus mean {column.mean:,.4g}",
                    f"Range {column.minimum:,.4g} to {column.maximum:,.4g}",
                ],
                columns=[name],
                confidence=Confidence.HIGH,
            )
        )

    concentrated = [
        (name, col) for name, col in profile.columns.items()
        if col.is_categorical and col.dominant_pct > 80 and not col.is_constant
    ]
    if concentrated:
        name, column = concentrated[0]
        out.append(
            Finding(
                title=f"'{name}' is dominated by a single category",
                detail=(
                    f"{column.dominant_pct:.1f}% of rows are '{column.dominant_value}'. There is very "
                    "little variation for a model to learn from, and comparisons involving the other "
                    "categories rest on few observations."
                ),
                kind=EvidenceKind.OBSERVED,
                evidence=[f"'{column.dominant_value}': {column.dominant_pct:.1f}% of rows"],
                columns=[name],
                confidence=Confidence.HIGH,
            )
        )
    return out


def _relationship_findings(profile: DatasetProfile, frame: pd.DataFrame | None) -> list[Finding]:
    out: list[Finding] = []
    strong = [(a, b, r) for a, b, r in profile.correlation_pairs if abs(r) >= 0.6][:5]
    for a, b, r in strong[:3]:
        direction = "rise together" if r > 0 else "move in opposite directions"
        out.append(
            Finding(
                title=f"'{a}' and '{b}' are strongly related",
                detail=(
                    f"They {direction} (r = {r:.2f}), sharing {r ** 2:.0%} of their variation. "
                    "This may be one causing the other, both being driven by something else, or the "
                    "two being different measures of the same thing — the correlation alone cannot say which."
                ),
                kind=EvidenceKind.STATISTICAL,
                evidence=[f"Pearson r = {r:.3f} across {profile.n_rows:,} rows"],
                columns=[a, b],
                metrics={"r": float(r), "r_squared": float(r ** 2)},
                confidence=Confidence.HIGH if profile.n_rows >= 100 else Confidence.MODERATE,
                caveats=["Correlation is not causation."],
            )
        )

    nonlinear = [r for r in profile.relationships if r.kind == "nonlinearity"][:2]
    for relationship in nonlinear:
        out.append(
            Finding(
                title=f"The link between '{relationship.columns[0]}' and '{relationship.columns[1]}' is curved, not straight",
                detail=(
                    relationship.message
                    + " A straight-line model will systematically miss part of this relationship."
                ),
                kind=EvidenceKind.STATISTICAL,
                evidence=[f"{relationship.method}: gap of {relationship.statistic:.3f}"],
                columns=relationship.columns,
                confidence=Confidence.MODERATE,
            )
        )
    return out


def _model_findings(
    objective: Objective,
    best: ExperimentResult | None,
    tournament: Any,
    explanation: ExplanationBundle | None,
) -> list[Finding]:
    out: list[Finding] = []
    if best is None:
        return out

    metric = M.default_metrics(objective.task_type)
    metric = metric[0] if metric else "r2"
    score = best.primary(metric)
    if score is None:
        return out

    out.append(
        Finding(
            title=f"{best.model_name} predicts '{objective.target}' with {M.METRIC_LABELS.get(metric, metric)} of {M.format_metric(metric, score)}",
            detail=(
                f"{M.explain_metric(metric)} This figure comes from {best.score_source} data, not from "
                f"the rows the model was trained on. "
                + _performance_verdict(objective.task_type, metric, score)
            ),
            kind=EvidenceKind.MODEL,
            evidence=[
                f"{M.METRIC_LABELS.get(metric, metric)} = {M.format_metric(metric, score)} ({best.score_source(metric)})",
                f"Trained on {best.n_train:,} rows, tested on {best.n_test:,}",
                f"{best.validation.n_splits} validation folds" if best.validation.n_splits else "single hold-out split",
            ],
            metrics={metric: float(score)},
            confidence=Confidence.HIGH if best.validation.n_splits >= 3 else Confidence.MODERATE,
            caveats=best.warnings,
        )
    )

    if tournament is not None and getattr(tournament, "ranked", None):
        ranked = tournament.ranked
        transparent = [r for r in ranked if r.interpretability == "transparent"]
        complex_models = [r for r in ranked if r.interpretability != "transparent"]
        if transparent and complex_models:
            simple_best = transparent[0]
            complex_best = complex_models[0]
            gap = abs((simple_best.primary_score or 0) - (complex_best.primary_score or 0))
            reference = abs(complex_best.primary_score or 1) or 1
            if gap / reference < 0.05:
                out.append(
                    Finding(
                        title="A simple model does the job as well as a complex one",
                        detail=(
                            f"{simple_best.model_name} scores within {gap / reference:.1%} of "
                            f"{complex_best.model_name}. The extra complexity is not buying accuracy, "
                            "so the interpretable model is the better choice unless something else argues otherwise."
                        ),
                        kind=EvidenceKind.MODEL,
                        evidence=[
                            f"{simple_best.model_name}: {M.format_metric(tournament.primary_metric, simple_best.primary_score)}",
                            f"{complex_best.model_name}: {M.format_metric(tournament.primary_metric, complex_best.primary_score)}",
                        ],
                        confidence=Confidence.HIGH,
                    )
                )
    return out


def _performance_verdict(task_type: TaskType, metric: str, score: float) -> str:
    if metric == "r2":
        if score < 0:
            return "The model performs worse than simply predicting the average — it is not usable."
        if score < 0.3:
            return (
                "Most of the variation is still unexplained. The measured variables do not capture "
                "what actually drives this outcome."
            )
        if score < 0.6:
            return "A moderate share of the variation is explained — useful for direction, not for precise prediction."
        return "A large share of the variation is explained, which is strong for behavioural data."
    if metric in {"roc_auc", "pr_auc"}:
        if score < 0.6:
            return "Barely better than guessing — the features carry little signal for this outcome."
        if score < 0.75:
            return "Useful for prioritisation, but too imprecise for individual decisions."
        return "Strong discrimination between the classes."
    if metric == "mase":
        return (
            "Below 1 means it beats simply repeating the last observed value."
            if score < 1 else
            "At or above 1 this forecast is no better than repeating the last value."
        )
    return ""


def _driver_findings(
    explanation: ExplanationBundle | None,
    objective: Objective,
    profile: DatasetProfile,
) -> list[Finding]:
    if explanation is None or not explanation.importances:
        return []
    out: list[Finding] = []
    top = explanation.importances[:5]
    total = sum(abs(f.importance) for f in explanation.importances) or 1.0

    out.append(
        Finding(
            title=f"'{top[0].feature}' is the strongest driver of '{objective.target}'",
            detail=(
                (top[0].interpretation or
                 f"It accounts for roughly {abs(top[0].importance) / total:.0%} of the model's total "
                 f"reliance on any variable.")
                + f" Method: {explanation.method}."
            ),
            kind=EvidenceKind.MODEL,
            evidence=[
                f"{f.feature}: importance {f.importance:,.4g}"
                + (f" ({f.direction})" if f.direction else "")
                for f in top
            ],
            columns=[f.feature for f in top],
            confidence=explanation.confidence,
            caveats=explanation.caveats + [
                "Importance describes what the model used, not what causes the outcome."
            ],
        )
    )

    negligible = [f for f in explanation.importances if abs(f.importance) < 0.01 * abs(top[0].importance)]
    if len(negligible) >= 3:
        out.append(
            Finding(
                title=f"{len(negligible)} variable(s) contribute almost nothing",
                detail=(
                    "Dropping them would make the model simpler and cheaper to maintain with little "
                    "or no loss of accuracy. Worth testing before you commit to collecting them."
                ),
                kind=EvidenceKind.MODEL,
                evidence=[f.feature for f in negligible[:10]],
                columns=[f.feature for f in negligible[:10]],
                confidence=Confidence.MODERATE,
            )
        )

    significant = [
        c for c in explanation.coefficients
        if not c.get("is_intercept") and c.get("p_value") is not None
    ]
    if significant:
        reliable = [c for c in significant if c.get("significant_at_5pct")]
        unreliable = [c for c in significant if not c.get("significant_at_5pct")]
        if reliable:
            out.append(
                Finding(
                    title=f"{len(reliable)} of {len(significant)} coefficients are statistically distinguishable from zero",
                    detail=(
                        "The remaining "
                        f"{len(unreliable)} could plausibly have no effect at all — their apparent "
                        "influence is within what sampling noise would produce."
                    ),
                    kind=EvidenceKind.STATISTICAL,
                    evidence=[
                        f"{c['term']}: {c['coefficient']:,.4g} (p = {c['p_value']:.4f}, "
                        f"95% CI {c.get('ci_lower', float('nan')):,.4g} to {c.get('ci_upper', float('nan')):,.4g})"
                        for c in reliable[:6]
                    ],
                    columns=[c["term"] for c in reliable[:6]],
                    confidence=Confidence.HIGH,
                )
            )
    return out


def _diagnostic_findings(diagnostics: dict[str, Any] | None) -> list[Finding]:
    if not diagnostics or not diagnostics.get("usable"):
        return []
    issues = diagnostics.get("issues", [])
    if not issues:
        return [
            Finding(
                title="The model's errors show no systematic pattern",
                detail=(
                    "Residuals are centred on zero with no relationship to the predicted value. "
                    "The model is not systematically over- or under-predicting any part of the range."
                ),
                kind=EvidenceKind.STATISTICAL,
                evidence=[diagnostics.get("interpretation", "")],
                confidence=Confidence.HIGH,
            )
        ]
    return [
        Finding(
            title=f"{len(issues)} issue(s) in how the model's errors are distributed",
            detail=" ".join(issues[:2]),
            kind=EvidenceKind.STATISTICAL,
            evidence=issues,
            confidence=Confidence.HIGH,
        )
    ]


def _series_findings(series_analysis: dict[str, Any] | None, objective: Objective) -> list[Finding]:
    if not series_analysis:
        return []
    out: list[Finding] = []
    trend = series_analysis.get("trend", {})
    if trend.get("significant"):
        out.append(
            Finding(
                title=f"'{objective.target}' has a clear {trend['direction']} trend",
                detail=trend["interpretation"],
                kind=EvidenceKind.STATISTICAL,
                evidence=[
                    f"Rank correlation with time = {trend['rank_correlation']:.3f} (p = {trend['p_value']:.2e})",
                    f"Change of {trend['total_change']:,.4g} across the series"
                    + (f" ({trend['pct_change_over_series']:+.1f}%)" if trend.get("pct_change_over_series") else ""),
                ],
                confidence=Confidence.HIGH,
            )
        )
    seasonality = series_analysis.get("seasonality", {})
    if seasonality.get("detected"):
        out.append(
            Finding(
                title=f"A repeating pattern of length {seasonality['period']} is present",
                detail=seasonality["interpretation"],
                kind=EvidenceKind.STATISTICAL,
                evidence=[
                    f"Explains {seasonality['strength']:.1%} of the variation",
                    f"Swing of {seasonality['amplitude']:,.4g} between peak and trough",
                ],
                confidence=Confidence.HIGH,
            )
        )
    breaks = series_analysis.get("structural_breaks", {})
    if breaks.get("detected"):
        out.append(
            Finding(
                title=f"The level shifts partway through the series (around {breaks['label']})",
                detail=breaks["interpretation"],
                kind=EvidenceKind.STATISTICAL,
                evidence=[
                    f"Mean before: {breaks['mean_before']:,.4g}",
                    f"Mean after: {breaks['mean_after']:,.4g}",
                ],
                confidence=Confidence.MODERATE,
            )
        )
    return out


def _context_findings(context: BusinessContext, profile: DatasetProfile) -> list[Finding]:
    """Record the user's assumptions as assumptions — never as measurements."""
    out: list[Finding] = []
    if context.is_empty:
        return out
    if context.description:
        out.append(
            Finding(
                title="What you told the platform this data represents",
                detail=context.description,
                kind=EvidenceKind.USER_ASSUMPTION,
                evidence=["Supplied by you; not verified against the data."],
                confidence=Confidence.MODERATE,
            )
        )
    for assumption in context.assumptions[:3]:
        out.append(
            Finding(
                title="Assumption you provided",
                detail=assumption,
                kind=EvidenceKind.USER_ASSUMPTION,
                evidence=["Your assumption. Everything downstream inherits it."],
                confidence=Confidence.LOW,
            )
        )
    for limitation in context.limitations[:3]:
        out.append(
            Finding(
                title="Limitation you flagged",
                detail=limitation,
                kind=EvidenceKind.USER_ASSUMPTION,
                evidence=["Stated by you; carried into every conclusion below."],
                confidence=Confidence.MODERATE,
            )
        )
    return out


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

_KIND_WEIGHT = {
    EvidenceKind.STATISTICAL: 1.0,
    EvidenceKind.OBSERVED: 0.9,
    EvidenceKind.MODEL: 0.85,
    EvidenceKind.USER_ASSUMPTION: 0.4,
    EvidenceKind.INTERPRETATION: 0.5,
}
_CONFIDENCE_WEIGHT = {
    Confidence.HIGH: 1.0,
    Confidence.MODERATE: 0.7,
    Confidence.LOW: 0.4,
    Confidence.SPECULATIVE: 0.2,
}


def _rank(findings: list[Finding]) -> list[Finding]:
    def score(finding: Finding) -> float:
        base = _KIND_WEIGHT.get(finding.kind, 0.5) * _CONFIDENCE_WEIGHT.get(finding.confidence, 0.5)
        return base + 0.05 * min(len(finding.evidence), 4)

    return sorted(findings, key=score, reverse=True)


def group_by_evidence(findings: list[Finding]) -> dict[str, list[Finding]]:
    """Group findings so the reader can see what is measured versus interpreted."""
    labels = {
        EvidenceKind.OBSERVED: "Observed in the data",
        EvidenceKind.STATISTICAL: "Established by a statistical test",
        EvidenceKind.MODEL: "Derived from a model",
        EvidenceKind.INTERPRETATION: "Platform interpretation",
        EvidenceKind.USER_ASSUMPTION: "Your assumptions (not verified)",
    }
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(labels.get(finding.kind, "Other"), []).append(finding)
    return grouped
