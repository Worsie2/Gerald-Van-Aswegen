"""Validation gate: sanity-check a conclusion before it is presented.

Every finding and recommendation passes through here first. The checks are the
questions a careful analyst asks before saying anything out loud — is the sample
big enough, could one outlier be driving this, is it leakage, is it causal? —
and each one either downgrades the confidence or attaches a caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dsai.core.schema import (
    Confidence, DatasetProfile, EvidenceKind, Finding, JsonMixin, Objective, Recommendation, TaskType,
)
from dsai.engines.experiment import ExperimentResult

CONFIDENCE_ORDER = [Confidence.SPECULATIVE, Confidence.LOW, Confidence.MODERATE, Confidence.HIGH]


def downgrade(confidence: Confidence, steps: int = 1) -> Confidence:
    index = CONFIDENCE_ORDER.index(confidence)
    return CONFIDENCE_ORDER[max(0, index - steps)]


@dataclass
class Check(JsonMixin):
    question: str
    passed: bool
    detail: str = ""
    severity: str = "warning"     # blocking | warning | note
    downgrade_steps: int = 0


@dataclass
class SelfCheckReport(JsonMixin):
    checks: list[Check] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    confidence_adjustment: int = 0

    @property
    def passed(self) -> bool:
        return not self.blocking

    def apply(self, confidence: Confidence) -> Confidence:
        return downgrade(confidence, self.confidence_adjustment) if self.confidence_adjustment else confidence

    def summary(self) -> str:
        failed = [c for c in self.checks if not c.passed]
        if not failed:
            return f"All {len(self.checks)} validation checks passed."
        return (
            f"{len(failed)} of {len(self.checks)} validation checks raised something: "
            + "; ".join(c.question for c in failed)
        )


def check_analysis(
    profile: DatasetProfile,
    objective: Objective,
    results: list[ExperimentResult] | None = None,
    best: ExperimentResult | None = None,
    primary_metric: str = "",
) -> SelfCheckReport:
    """Run the standard battery of checks against a completed analysis."""
    report = SelfCheckReport()
    results = results or []

    _check_sample_size(report, profile, objective)
    _check_dimensionality(report, profile, objective)
    _check_leakage(report, profile, best)
    _check_class_balance(report, profile, objective, best)
    _check_overfitting(report, best)
    _check_stability(report, best, primary_metric)
    _check_baseline(report, results, best, primary_metric)
    _check_outlier_influence(report, profile, objective)
    _check_data_quality(report, profile)
    _check_causality(report, objective)
    _check_temporal_validity(report, profile, objective)

    for check in report.checks:
        if not check.passed:
            report.confidence_adjustment += check.downgrade_steps
            if check.severity == "blocking":
                report.blocking.append(check.detail)
            else:
                report.caveats.append(check.detail)
    # Capped at two: a three-step downgrade lands every finding on "speculative"
    # regardless of where it started, which destroys the gradient the levels exist for.
    report.confidence_adjustment = min(report.confidence_adjustment, 2)
    return report


def _add(report: SelfCheckReport, question: str, passed: bool, detail: str = "",
         severity: str = "warning", steps: int = 1) -> None:
    report.checks.append(
        Check(question=question, passed=passed, detail=detail, severity=severity,
              downgrade_steps=0 if passed else steps)
    )


def _check_sample_size(report, profile: DatasetProfile, objective: Objective) -> None:
    n = profile.n_rows
    minimum = 30 if not objective.task_type.is_supervised else 50
    _add(
        report, "Is the dataset large enough to support a conclusion?",
        n >= minimum,
        (
            f"Only {n} rows. Below about {minimum} observations, estimates swing widely between "
            "samples and almost nothing can be established with confidence. Treat any result as a "
            "hypothesis worth collecting more data to test."
        ),
        severity="blocking" if n < 20 else "warning",
        steps=2 if n < 30 else 1,
    )


def _check_dimensionality(report, profile: DatasetProfile, objective: Objective) -> None:
    n_features = len(objective.features) if objective.features else len(profile.modelling_columns)
    ratio = n_features / max(profile.n_rows, 1)
    _add(
        report, "Are there enough rows per variable?",
        ratio <= 0.1,
        (
            f"{n_features} variables against {profile.n_rows} rows (1 variable per "
            f"{profile.n_rows / max(n_features, 1):.1f} rows). With this ratio, some variables will "
            "appear predictive by chance alone. Regularisation and out-of-sample validation are "
            "carrying more weight than usual here."
        ),
        steps=1,
    )


def _check_leakage(report, profile: DatasetProfile, best: ExperimentResult | None) -> None:
    critical = [s for s in profile.leakage_suspects if s.get("severity") == "critical"]
    suspicious_score = False
    if best is not None:
        for metric in ("r2", "roc_auc", "f1", "accuracy"):
            value = best.primary(metric)
            if value is not None and value > 0.98:
                suspicious_score = True
                break
    passed = not critical and not suspicious_score
    detail_parts = []
    if critical:
        detail_parts.append(
            f"{len(critical)} column(s) correlate almost perfectly with the target: "
            + ", ".join(s["column"] for s in critical[:5]) + "."
        )
    if suspicious_score:
        detail_parts.append(
            "The model scores near-perfectly. In real data that almost always means a variable is "
            "restating the answer rather than predicting it."
        )
    _add(
        report, "Could the model be seeing the answer (leakage)?",
        passed,
        " ".join(detail_parts) + " Confirm every predictor is genuinely known before the outcome occurs.",
        severity="blocking" if critical and suspicious_score else "warning",
        steps=2,
    )


def _check_class_balance(report, profile: DatasetProfile, objective: Objective, best) -> None:
    if not objective.task_type.is_classification or not objective.target:
        report.checks.append(Check("Is the target balanced enough to model?", True, severity="note"))
        return
    column = profile.columns.get(objective.target)
    if column is None or not column.top_values:
        return
    counts = list(column.top_values.values())
    total = sum(counts)
    minority = min(counts) / total if total else 1.0
    _add(
        report, "Is the target balanced enough to model?",
        minority >= 0.05,
        (
            f"The smallest class is {minority:.1%} of the data. Accuracy is meaningless at this "
            f"balance — a model predicting the majority class every time would score "
            f"{1 - minority:.1%}. Judge this on precision, recall and PR-AUC instead."
        ),
        steps=1,
    )


def _check_overfitting(report, best: ExperimentResult | None) -> None:
    if best is None or best.overfitting_gap is None:
        return
    gap = best.overfitting_gap
    _add(
        report, "Is the model overfitting?",
        gap <= 0.15,
        (
            f"Training performance exceeds held-out performance by {gap:.3f}. The model has learned "
            "detail specific to the training rows. The held-out figure is the honest one; the "
            "training figure should not be quoted."
        ),
        steps=1 if gap <= 0.3 else 2,
    )


def _check_stability(report, best: ExperimentResult | None, primary_metric: str) -> None:
    if best is None or not primary_metric or primary_metric not in best.validation.mean_scores:
        return
    cv = best.validation.stability(primary_metric)
    _add(
        report, "Is the result stable across folds?",
        np.isfinite(cv) and cv <= 0.25,
        (
            f"The score varies by {cv:.0%} between cross-validation folds "
            f"({best.validation.mean_scores[primary_metric]:.3f} ± "
            f"{best.validation.std_scores.get(primary_metric, 0):.3f}). Quoting a single number would "
            "overstate how well this is pinned down — report the range."
        ),
        steps=1,
    )


def _check_baseline(report, results: list[ExperimentResult], best, primary_metric: str) -> None:
    from dsai.engines import metrics as M
    from dsai.registry.base import REGISTRY

    if not results or not primary_metric or best is None:
        return
    baseline = None
    for r in results:
        try:
            if REGISTRY.get(r.model_key).baseline:
                baseline = r
                break
        except KeyError:
            continue
    if baseline is None:
        return
    baseline_score = baseline.primary(primary_metric)
    best_score = best.primary(primary_metric)
    if baseline_score is None or best_score is None:
        return
    beats = M.is_better(primary_metric, best_score, baseline_score)
    _add(
        report, "Does the model beat a naive baseline?",
        beats,
        (
            f"The recommended model scores {M.format_metric(primary_metric, best_score)} against "
            f"{M.format_metric(primary_metric, baseline_score)} for a model that ignores every "
            "predictor. Nothing here is adding value over the simplest possible rule — the problem "
            "framing or the features need to change, not the algorithm."
        ),
        severity="blocking",
        steps=3,
    )


def _check_outlier_influence(report, profile: DatasetProfile, objective: Objective) -> None:
    if not objective.target or objective.target not in profile.columns:
        return
    column = profile.columns[objective.target]
    outlier_pct = column.outlier_pct or 0.0
    # A skewed but genuine distribution (spend, income, claim size) routinely shows
    # 5-10% IQR outliers. Only an unusual concentration should cost confidence.
    _add(
        report, "Could a handful of extreme values be driving this?",
        outlier_pct < 10.0,
        (
            f"{column.n_outliers_iqr} outlier(s) ({outlier_pct:.1f}%) in the target '{objective.target}'. "
            "Re-run with those rows excluded: if the conclusion changes, it is a statement about "
            "those few rows rather than about the population."
        ),
        steps=1,
    )
    if 5.0 <= outlier_pct < 10.0:
        report.caveats.append(
            f"'{objective.target}' has a long tail ({outlier_pct:.1f}% of values sit outside the IQR "
            "fences). That is normal for monetary and count data, but it means the mean is not a "
            "typical value — report medians alongside averages."
        )


def _check_data_quality(report, profile: DatasetProfile) -> None:
    critical = profile.issues_by_severity("critical")
    _add(
        report, "Are there unresolved data-quality problems?",
        not critical,
        (
            f"{len(critical)} critical data-quality issue(s) remain: "
            + "; ".join(i.message for i in critical[:3])
            + ". These propagate straight into the conclusions."
        ),
        steps=1,
    )


def _check_causality(report, objective: Objective) -> None:
    """Always fires. The point is that the caveat is never quietly dropped."""
    is_observational = objective.source != "experiment"
    report.checks.append(
        Check(
            question="Is the relationship causal, or only associational?",
            passed=False if is_observational else True,
            detail=(
                "This is observational data, so every relationship found is an association. A third "
                "variable, reverse causation or selection into the sample would each produce the same "
                "pattern. Acting on these findings assumes the relationship holds under intervention, "
                "which only a controlled test can establish."
            ),
            severity="note",
            downgrade_steps=0,
        )
    )
    if is_observational:
        report.caveats.append(report.checks[-1].detail)


def _check_temporal_validity(report, profile: DatasetProfile, objective: Objective) -> None:
    if objective.task_type is not TaskType.TIME_SERIES_FORECAST:
        if profile.datetime_columns and objective.task_type.is_supervised:
            # Advisory only: a date column may be an attribute of an entity (signup
            # date on a customer row) rather than an ordering of observations. The
            # platform cannot tell which, so it raises the question without
            # penalising a conclusion that may be perfectly sound.
            report.checks.append(
                Check(
                    question="Was time respected in the validation split?",
                    passed=False,
                    detail=(
                        f"The data has a date column ('{profile.datetime_columns[0]}') but was validated "
                        "with random folds. That is correct if each row is a separate entity and the "
                        "date is just one of its attributes. It is wrong if the rows are observations "
                        "over time and you intend to predict forward — in that case random folds let "
                        "the model train on the future to predict the past, and the score is inflated. "
                        "Only you can say which this is."
                    ),
                    severity="note",
                    downgrade_steps=0,
                )
            )
            report.caveats.append(report.checks[-1].detail)
        return
    if objective.time_column and objective.time_column in profile.columns:
        column = profile.columns[objective.time_column]
        _add(
            report, "Is the time series complete?",
            not column.n_missing_periods,
            (
                f"About {column.n_missing_periods} period(s) are missing from '{objective.time_column}'. "
                "Gaps distort seasonality estimates and make the forecast horizon ambiguous."
            ),
            steps=1,
        )


def validate_recommendation(
    recommendation: Recommendation,
    report: SelfCheckReport,
) -> Recommendation:
    """Downgrade a recommendation's confidence and attach the checks' caveats."""
    recommendation.confidence = report.apply(recommendation.confidence)
    for caveat in report.caveats:
        if caveat not in recommendation.caveats:
            recommendation.caveats.append(caveat)
    if report.blocking:
        recommendation.confidence = Confidence.SPECULATIVE
        recommendation.caveats.insert(
            0,
            "This recommendation did not pass validation: "
            + "; ".join(report.blocking)
            + " Resolve that before acting on it.",
        )
    return recommendation


def validate_finding(finding: Finding, report: SelfCheckReport) -> Finding:
    finding.confidence = report.apply(finding.confidence)
    if finding.kind in {EvidenceKind.MODEL, EvidenceKind.INTERPRETATION}:
        for caveat in report.caveats:
            if caveat not in finding.caveats:
                finding.caveats.append(caveat)
    return finding
