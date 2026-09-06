"""The Recommendation Engine: turn findings into things to actually do.

Every recommendation must trace back to a specific analytical output. A
recommendation with no evidence attached is an opinion, and this engine does not
produce those — if the evidence is not there, the recommendation is not made.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, EvidenceKind, Finding, Objective,
    Recommendation, TaskType,
)
from dsai.engines import metrics as M
from dsai.engines.metrics import human_number
from dsai.engines.experiment import ExperimentResult
from dsai.engines.selfcheck import SelfCheckReport, validate_recommendation
from dsai.explain.importance import ExplanationBundle


def generate_recommendations(
    profile: DatasetProfile,
    objective: Objective,
    findings: list[Finding],
    context: BusinessContext | None = None,
    best: ExperimentResult | None = None,
    tournament: Any = None,
    explanation: ExplanationBundle | None = None,
    segmentation: Any = None,
    series_analysis: dict[str, Any] | None = None,
    self_check: SelfCheckReport | None = None,
    frame: pd.DataFrame | None = None,
) -> list[Recommendation]:
    """Produce evidence-linked recommendations across four categories."""
    context = context or BusinessContext()
    recommendations: list[Recommendation] = []

    recommendations.extend(_data_quality_recommendations(profile))
    recommendations.extend(_action_recommendations(objective, explanation, best, context, frame, profile))
    recommendations.extend(_segment_recommendations(segmentation, context, frame))
    recommendations.extend(_forecast_recommendations(series_analysis, best, objective))
    recommendations.extend(_model_recommendations(tournament, best, objective))
    recommendations.extend(_risk_recommendations(profile, best, findings))
    recommendations.extend(_next_analysis_recommendations(profile, objective, best, explanation, context))

    if self_check is not None:
        recommendations = [validate_recommendation(r, self_check) for r in recommendations]

    return _rank(recommendations)


# --------------------------------------------------------------------------
# categories
# --------------------------------------------------------------------------

def _data_quality_recommendations(profile: DatasetProfile) -> list[Recommendation]:
    out: list[Recommendation] = []
    for issue in profile.issues_by_severity("critical")[:4]:
        out.append(
            Recommendation(
                action=issue.suggested_action or f"Resolve: {issue.message}",
                reason=issue.message,
                evidence=[f"Affects: {', '.join(issue.columns[:6])}" if issue.columns else issue.message],
                confidence=Confidence.HIGH,
                expected_impact="Every conclusion drawn from this data inherits the problem until it is fixed.",
                category="data_quality",
                traceable_to=[f"quality_issue:{issue.code}"],
            )
        )
    heavy_missing = [
        (name, col.missing_pct) for name, col in profile.columns.items() if col.missing_pct > 40
    ]
    if heavy_missing:
        names = ", ".join(f"{n} ({p:.0f}%)" for n, p in heavy_missing[:5])
        out.append(
            Recommendation(
                action="Decide deliberately what to do about the heavily incomplete columns",
                reason=(
                    f"{len(heavy_missing)} column(s) are more than 40% empty: {names}. Imputing that "
                    "much means most of the column is invented; dropping it means losing whatever "
                    "signal it carries. Neither is automatically right."
                ),
                evidence=[f"'{n}': {p:.1f}% missing" for n, p in heavy_missing[:6]],
                confidence=Confidence.HIGH,
                expected_impact="Removes a source of results that look solid but rest largely on filled-in values.",
                category="data_quality",
                traceable_to=["profile:missingness"],
            )
        )
    return out


def _action_recommendations(
    objective: Objective,
    explanation: ExplanationBundle | None,
    best: ExperimentResult | None,
    context: BusinessContext,
    frame: pd.DataFrame | None,
    profile: DatasetProfile,
) -> list[Recommendation]:
    if explanation is None or not explanation.importances or best is None:
        return []
    out: list[Recommendation] = []
    currency = context.currency or "ZAR"
    top = explanation.importances[:3]

    for feature in top[:2]:
        column = profile.columns.get(feature.feature)
        evidence = [
            f"Importance rank {feature.rank} of {len(explanation.importances)} ({explanation.method})",
            f"Model: {best.model_name}",
        ]
        if column and column.is_numeric and column.mean is not None:
            evidence.append(
                f"'{feature.feature}' ranges {human_number(column.minimum)} to {human_number(column.maximum)}, "
                f"median {human_number(column.median)}"
            )
        direction_text = ""
        if feature.direction:
            direction_text = (
                f" Higher {feature.feature} is associated with a "
                f"{'higher' if feature.direction == 'increases' else 'lower'} {objective.target}."
            )
        out.append(
            Recommendation(
                action=f"Focus effort on '{feature.feature}' when trying to move '{objective.target}'",
                reason=(
                    f"It is the {'strongest' if feature.rank == 1 else _ordinal(feature.rank) + ' strongest'} "
                    f"variable in the model of {objective.target}.{direction_text}"
                ),
                evidence=evidence,
                confidence=Confidence.MODERATE,
                expected_impact=(
                    "Unknown in size. The model shows association, so the size of any real effect "
                    "from intervening has to be measured by a test, not read off this analysis."
                ),
                caveats=[
                    "This is an association, not an established cause. If something else drives both "
                    f"{feature.feature} and {objective.target}, changing the first will do nothing.",
                    "Confirm the variable is one you can actually influence — some strong predictors "
                    "(age, tenure, historical totals) are not levers.",
                ],
                category="action",
                traceable_to=[f"explanation:{feature.feature}", f"experiment:{best.id}"],
            )
        )

    negligible = [
        f for f in explanation.importances
        if abs(f.importance) < 0.01 * abs(explanation.importances[0].importance)
    ]
    if len(negligible) >= 3:
        out.append(
            Recommendation(
                action=f"Test dropping {len(negligible)} low-contribution variable(s)",
                reason=(
                    "They contribute under 1% of the top variable's importance. A simpler model is "
                    "cheaper to run, easier to explain, and less likely to break when one input goes missing."
                ),
                evidence=[f.feature for f in negligible[:10]],
                confidence=Confidence.MODERATE,
                expected_impact="Likely no measurable accuracy loss — but re-run the comparison to confirm before committing.",
                category="further_analysis",
                traceable_to=["explanation:low_importance"],
            )
        )
    return out


def _segment_recommendations(segmentation: Any, context: BusinessContext,
                             frame: pd.DataFrame | None) -> list[Recommendation]:
    if segmentation is None or not getattr(segmentation, "clusters", None):
        return []
    out: list[Recommendation] = []
    currency = context.currency or "ZAR"
    real = [c for c in segmentation.clusters if not c.is_noise and c.defining_features]
    if not real:
        return out

    def value_score(cluster) -> float:
        for feature in cluster.defining_features:
            if any(token in feature["feature"].lower()
                   for token in ("spend", "revenue", "value", "sales", "profit", "amount", "income")):
                return feature["std_deviations"]
        return -np.inf

    valuable = max(real, key=value_score)
    if value_score(valuable) > 0.5:
        driver = next(
            f for f in valuable.defining_features
            if any(t in f["feature"].lower() for t in ("spend", "revenue", "value", "sales", "profit", "amount", "income"))
        )
        out.append(
            Recommendation(
                action=f"Prioritise {valuable.name} for high-value treatment",
                reason=(
                    f"This segment averages {driver['cluster_mean']:,.2f} on {driver['feature']} against "
                    f"{driver['overall_mean']:,.2f} across all rows — "
                    f"{abs(driver.get('pct_difference') or 0):.0f}% above average — while making up only "
                    f"{valuable.share:.1%} of the base."
                ),
                evidence=[
                    f"Segment size: {valuable.size:,} rows ({valuable.share:.1%})",
                    f"{driver['feature']}: {driver['cluster_mean']:,.2f} versus {driver['overall_mean']:,.2f} overall",
                ] + [
                    f"{f['feature']}: {human_number(f['cluster_mean'])} ({f['direction']} than average)"
                    for f in valuable.defining_features[1:3]
                ],
                confidence=Confidence.MODERATE if valuable.size >= 30 else Confidence.LOW,
                expected_impact=(
                    f"This group already accounts for a disproportionate share of {driver['feature']}. "
                    "Retention effort here protects more value per customer than the same effort elsewhere."
                ),
                caveats=[
                    "Segments describe this dataset. Re-clustering on new data can shift the boundaries.",
                    "High current value does not mean high growth potential — those are different questions.",
                ],
                category="action",
                traceable_to=[f"cluster:{valuable.label}"],
            )
        )

    largest = max(real, key=lambda c: c.size)
    if largest.label != valuable.label and largest.share > 0.3:
        out.append(
            Recommendation(
                action=f"Design the default experience around {largest.name}",
                reason=(
                    f"It holds {largest.size:,} rows ({largest.share:.1%}) — the plurality of the base. "
                    "Whatever suits this group is what most people will encounter."
                ),
                evidence=[largest.description],
                confidence=Confidence.MODERATE,
                expected_impact="Affects the majority of the population by definition.",
                category="action",
                traceable_to=[f"cluster:{largest.label}"],
            )
        )

    if getattr(segmentation, "warnings", None):
        out.append(
            Recommendation(
                action="Treat the segmentation as provisional",
                reason=" ".join(segmentation.warnings[:2]),
                evidence=segmentation.warnings,
                confidence=Confidence.HIGH,
                expected_impact="Prevents building a strategy on segment boundaries that will not hold up.",
                category="risk",
                traceable_to=["clustering:warnings"],
            )
        )
    return out


def _forecast_recommendations(series_analysis, best: ExperimentResult | None,
                              objective: Objective) -> list[Recommendation]:
    if objective.task_type is not TaskType.TIME_SERIES_FORECAST or best is None:
        return []
    out: list[Recommendation] = []
    forecast = best.extras.get("forecast")
    mase = best.test_scores.get("mase")

    if forecast:
        horizon = best.extras.get("horizon", len(forecast))
        history = best.extras.get("history_values") or []
        recent = float(np.mean(history[-min(len(history), horizon):])) if history else None
        projected = float(np.mean(forecast))
        change = ((projected - recent) / abs(recent) * 100) if recent else None
        evidence = [
            f"Forecast over the next {horizon} period(s): {human_number(projected)} on average",
            f"Model: {best.model_name}",
        ]
        if mase is not None:
            evidence.append(
                f"MASE {mase:.2f} on the hold-out window "
                f"({'beats' if mase < 1 else 'does not beat'} a naive last-value forecast)"
            )
        if best.extras.get("forecast_lower"):
            evidence.append(
                f"95% interval on the first period: {human_number(best.extras['forecast_lower'][0])} to "
                f"{human_number(best.extras['forecast_upper'][0])}"
            )
        out.append(
            Recommendation(
                action=(
                    f"Plan for {objective.target} averaging {human_number(projected)} over the next "
                    f"{horizon} period(s)"
                    + (f", about {change:+.1f}% against the most recent comparable window" if change is not None else "")
                ),
                reason=f"Projected by {best.model_name}, back-tested on data it had not seen.",
                evidence=evidence,
                confidence=(
                    Confidence.MODERATE if mase is not None and mase < 0.8 else
                    Confidence.LOW if mase is not None and mase < 1.0 else Confidence.SPECULATIVE
                ),
                expected_impact="Gives a defensible planning number with a stated error range.",
                caveats=[
                    "A forecast assumes the future behaves like the past. Any change in pricing, "
                    "competition, regulation or the market invalidates it.",
                    "Uncertainty grows with the horizon — the last period is far less reliable than the first.",
                ] + ([
                    "This model does not beat a naive last-value forecast. Use the naive figure and "
                    "spend the effort on better data instead."
                ] if mase is not None and mase >= 1 else []),
                category="action",
                traceable_to=[f"experiment:{best.id}"],
            )
        )

    if series_analysis:
        breaks = series_analysis.get("structural_breaks", {})
        if breaks.get("detected"):
            out.append(
                Recommendation(
                    action=f"Refit the forecast using only data after {breaks['label']}",
                    reason=(
                        f"The level shifts from {human_number(breaks['mean_before'])} to {human_number(breaks['mean_after'])} "
                        "at that point. Training across the break averages two different regimes and "
                        "will under-predict the current one."
                    ),
                    evidence=[breaks["interpretation"]],
                    confidence=Confidence.MODERATE,
                    expected_impact="Usually a substantial accuracy gain when a genuine regime change has occurred.",
                    category="further_analysis",
                    traceable_to=["timeseries:structural_break"],
                )
            )
        gaps = series_analysis.get("gaps", {})
        if gaps.get("missing_periods", 0) > 0:
            out.append(
                Recommendation(
                    action="Fill or resample the missing periods before forecasting again",
                    reason=gaps["interpretation"],
                    evidence=[f"{gaps['missing_periods']} missing period(s) across {gaps['n_gaps']} gap(s)"],
                    confidence=Confidence.HIGH,
                    expected_impact="Seasonality estimates become reliable; the forecast horizon becomes unambiguous.",
                    category="data_quality",
                    traceable_to=["timeseries:gaps"],
                )
            )
    return out


def _model_recommendations(tournament: Any, best: ExperimentResult | None,
                           objective: Objective) -> list[Recommendation]:
    if tournament is None or not getattr(tournament, "recommended", None):
        return []
    out: list[Recommendation] = []
    recommended = tournament.recommended
    metric = tournament.primary_metric

    out.append(
        Recommendation(
            action=f"Use {recommended.model_name} as the working model",
            reason=(
                f"Best-performing model on this dataset under the chosen validation strategy — "
                f"{M.METRIC_LABELS.get(metric, metric)} of "
                f"{M.format_metric(metric, recommended.primary_score)} ({recommended.score_source}) — "
                "once generalisation, stability, interpretability and training cost are weighed together."
            ),
            evidence=[
                f"{M.METRIC_LABELS.get(metric, metric)} = {M.format_metric(metric, recommended.primary_score)}",
                f"Composite score {recommended.composite}",
            ] + (
                [f"{recommended.lift_over_baseline:+.1f}% against the naive baseline"]
                if recommended.lift_over_baseline is not None else []
            ),
            confidence=Confidence.MODERATE,
            expected_impact="Sets the operating point for any decision that depends on these predictions.",
            caveats=(
                ["This is the best model *for this dataset under this validation strategy* — not the "
                 "best model in general. New data can change the ranking."]
                + recommended.concerns
            ),
            category="action",
            traceable_to=[f"tournament:{recommended.result_id}"],
        )
    )

    simplest = getattr(tournament, "simplest_acceptable", None)
    if simplest is not None and simplest.result_id != recommended.result_id:
        out.append(
            Recommendation(
                action=f"Consider {simplest.model_name} if the analysis must be explained or audited",
                reason=(
                    f"It scores within 5% of the recommendation on {M.METRIC_LABELS.get(metric, metric)} "
                    "while being far easier to justify to a non-technical audience or a regulator."
                ),
                evidence=[
                    f"{M.METRIC_LABELS.get(metric, metric)} = {M.format_metric(metric, simplest.primary_score)}",
                    f"Interpretability: {simplest.interpretability}",
                ],
                confidence=Confidence.MODERATE,
                expected_impact="Small accuracy cost in exchange for a model whose reasoning can be shown.",
                category="action",
                traceable_to=[f"tournament:{simplest.result_id}"],
            )
        )
    return out


def _risk_recommendations(profile: DatasetProfile, best: ExperimentResult | None,
                          findings: list[Finding]) -> list[Recommendation]:
    out: list[Recommendation] = []
    if best is not None and best.overfitting_gap is not None and best.overfitting_gap > 0.2:
        out.append(
            Recommendation(
                action="Do not quote the training-set performance figures",
                reason=(
                    f"Training performance exceeds held-out performance by {best.overfitting_gap:.3f}. "
                    "The training number describes memorisation, not capability."
                ),
                evidence=[w for w in best.warnings if "Overfit" in w] or [f"Gap: {best.overfitting_gap:.3f}"],
                confidence=Confidence.HIGH,
                expected_impact="Prevents committing to a level of accuracy the model will not deliver in production.",
                category="risk",
                traceable_to=[f"experiment:{best.id}"],
            )
        )

    if profile.leakage_suspects:
        out.append(
            Recommendation(
                action="Verify that every predictor is knowable before the outcome occurs",
                reason=(
                    f"{len(profile.leakage_suspects)} column(s) look like they may encode the answer. "
                    "A model built on those will score brilliantly in testing and fail completely in use."
                ),
                evidence=[f"{s['column']}: {s['reason']}" for s in profile.leakage_suspects[:5]],
                confidence=Confidence.HIGH,
                expected_impact="Distinguishes a model that predicts from one that is reading the answer.",
                category="risk",
                traceable_to=["profile:leakage"],
            )
        )

    if profile.n_rows < 100:
        out.append(
            Recommendation(
                action=f"Collect more data before acting on this — {profile.n_rows} rows is not enough",
                reason=(
                    "At this sample size, estimates swing widely between samples. Anything found here "
                    "is a hypothesis, not a result."
                ),
                evidence=[f"{profile.n_rows} rows, {profile.n_columns} columns"],
                confidence=Confidence.HIGH,
                expected_impact="More data is the highest-value next step by a wide margin.",
                category="risk",
                traceable_to=["profile:sample_size"],
            )
        )
    return out


def _next_analysis_recommendations(
    profile: DatasetProfile,
    objective: Objective,
    best: ExperimentResult | None,
    explanation: ExplanationBundle | None,
    context: BusinessContext,
) -> list[Recommendation]:
    out: list[Recommendation] = []

    if best is not None and objective.task_type.is_supervised:
        metric = M.default_metrics(objective.task_type)
        metric = metric[0] if metric else "r2"
        score = best.primary(metric)
        weak = (
            (metric == "r2" and score is not None and score < 0.3)
            or (metric in {"roc_auc", "pr_auc"} and score is not None and score < 0.65)
        )
        if weak:
            out.append(
                Recommendation(
                    action="Add explanatory variables rather than trying more algorithms",
                    reason=(
                        f"Every model tried lands around {M.format_metric(metric, score)}. When the whole "
                        "field performs similarly and poorly, the limit is the information in the "
                        "features, not the choice of algorithm."
                    ),
                    evidence=[f"Best {M.METRIC_LABELS.get(metric, metric)}: {M.format_metric(metric, score)}"],
                    confidence=Confidence.HIGH,
                    expected_impact="The only realistic route to better performance on this problem.",
                    category="further_analysis",
                    traceable_to=[f"experiment:{best.id}"],
                )
            )

    if profile.datetime_columns and objective.task_type is not TaskType.TIME_SERIES_FORECAST:
        out.append(
            Recommendation(
                action=f"Look at how this changes over time using '{profile.datetime_columns[0]}'",
                reason=(
                    "The data carries a time dimension that the current analysis treats as just another "
                    "column. Trend, seasonality and regime changes are invisible to it."
                ),
                evidence=[f"Date column available: {profile.datetime_columns[0]}"],
                confidence=Confidence.MODERATE,
                expected_impact="Often reveals that a stable-looking average is hiding a clear trend.",
                category="further_analysis",
                traceable_to=["profile:datetime"],
            )
        )

    if objective.task_type is not TaskType.CLUSTERING and len(profile.numeric_columns) >= 3 and profile.n_rows >= 100:
        out.append(
            Recommendation(
                action="Segment the population and check whether the relationships hold within each group",
                reason=(
                    "A single model fitted across the whole population can hide opposite effects in "
                    "different groups — the average of two opposing trends is a flat line."
                ),
                evidence=[f"{len(profile.numeric_columns)} numeric variables across {profile.n_rows:,} rows"],
                confidence=Confidence.MODERATE,
                expected_impact="Frequently the difference between a weak overall model and two strong group-specific ones.",
                category="further_analysis",
                traceable_to=["profile:structure"],
            )
        )

    if explanation is not None and explanation.importances:
        top = explanation.importances[0].feature
        out.append(
            Recommendation(
                action=f"Test whether '{top}' actually causes the change, rather than merely tracking it",
                reason=(
                    "It is the strongest predictor in the model, which makes it the highest-value "
                    "thing to establish causality for. A controlled test or a natural experiment can "
                    "do that; this data cannot."
                ),
                evidence=[f"'{top}' ranks first on {explanation.method}"],
                confidence=Confidence.HIGH,
                expected_impact="Converts a correlation into something you can safely act on.",
                category="further_analysis",
                traceable_to=[f"explanation:{top}"],
            )
        )
    return out


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th — small thing, but '2th strongest' undermines the whole report."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_CATEGORY_PRIORITY = {"risk": 4, "data_quality": 3, "action": 2, "further_analysis": 1}
_CONFIDENCE_PRIORITY = {
    Confidence.HIGH: 3, Confidence.MODERATE: 2, Confidence.LOW: 1, Confidence.SPECULATIVE: 0,
}


def _rank(recommendations: list[Recommendation]) -> list[Recommendation]:
    return sorted(
        recommendations,
        key=lambda r: (
            _CATEGORY_PRIORITY.get(r.category, 0) * 10
            + _CONFIDENCE_PRIORITY.get(r.confidence, 0)
            + min(len(r.evidence), 3) * 0.1
        ),
        reverse=True,
    )


def group_by_category(recommendations: list[Recommendation]) -> dict[str, list[Recommendation]]:
    labels = {
        "risk": "Risks to address first",
        "data_quality": "Data quality",
        "action": "Actions to take",
        "further_analysis": "What to investigate next",
    }
    grouped: dict[str, list[Recommendation]] = {}
    for recommendation in recommendations:
        grouped.setdefault(labels.get(recommendation.category, "Other"), []).append(recommendation)
    return grouped
