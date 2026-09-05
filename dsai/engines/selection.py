"""The Model Tournament: rank finished experiments and recommend one.

The highest score does not win automatically. A model that scores 0.91 while
overfitting badly and swinging 30% between folds is worse, in practice, than one
scoring 0.88 steadily — and a transparent model that matches a black box makes
the black box pointless.

The engine therefore returns four named choices rather than a single verdict:
recommended, highest-performance, simplest-acceptable and a runner-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import Confidence, Decision, JsonMixin, TaskType
from dsai.engines import metrics as M
from dsai.engines.experiment import ExperimentResult
from dsai.registry.base import REGISTRY, Interpretability, ModelRegistry

#: Local alias so the normaliser does not have to reach into the metrics module
#: for every candidate.
LOWER_IS_BETTER_LOCAL = M.LOWER_IS_BETTER

# How the composite score is weighted. Exposed so the weighting is auditable
# rather than buried in a formula.
DEFAULT_WEIGHTS = {
    "performance": 0.50,
    "generalisation": 0.20,   # penalty for the train-to-held-out gap
    "stability": 0.15,        # penalty for score variance across folds
    "interpretability": 0.10,
    "efficiency": 0.05,       # training time
}


@dataclass
class RankedModel(JsonMixin):
    result_id: str
    model_key: str
    model_name: str
    primary_metric: str
    primary_score: float | None
    score_source: str = ""
    composite: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    beats_baseline: bool | None = None
    lift_over_baseline: float | None = None
    concerns: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    interpretability: str = ""
    training_time_s: float = 0.0


@dataclass
class TournamentOutcome(JsonMixin):
    """The full result of comparing a field of models."""

    primary_metric: str = ""
    task_type: TaskType = TaskType.EXPLORATORY
    ranked: list[RankedModel] = field(default_factory=list)
    recommended: RankedModel | None = None
    highest_performance: RankedModel | None = None
    simplest_acceptable: RankedModel | None = None
    alternative: RankedModel | None = None
    baseline: RankedModel | None = None
    decisions: list[Decision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    table: list[dict[str, Any]] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def summary(self) -> str:
        if not self.recommended:
            return "No model produced a usable result."
        best = self.recommended
        return (
            f"Recommended: {best.model_name}. "
            f"{M.METRIC_LABELS.get(self.primary_metric, self.primary_metric)} = "
            f"{M.format_metric(self.primary_metric, best.primary_score)} ({best.score_source})."
        )


def run_tournament(
    results: list[ExperimentResult],
    primary_metric: str,
    task_type: TaskType,
    interpretability_need: str = "moderate",
    weights: dict[str, float] | None = None,
    registry: ModelRegistry | None = None,
    display_metrics: list[str] | None = None,
) -> TournamentOutcome:
    """Score, rank and choose. Returns the reasoning as well as the ranking."""
    registry = registry or REGISTRY
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    outcome = TournamentOutcome(primary_metric=primary_metric, task_type=task_type, weights=weights)

    usable = [r for r in results if r.status == "success" and r.primary(primary_metric) is not None]
    failed = [r for r in results if r.status != "success"]
    for r in failed:
        outcome.warnings.append(f"{r.model_name} did not complete: {r.error or 'unknown error'}")

    if not usable:
        outcome.warnings.append(
            f"No model produced a usable {primary_metric}. Check the data-quality issues and the "
            "failure messages above before drawing any conclusion."
        )
        return outcome

    baseline_result = _find_baseline(usable, registry)
    baseline_score = baseline_result.primary(primary_metric) if baseline_result else None

    field_scores = [r.primary(primary_metric) for r in usable]
    field_scores = [s for s in field_scores if s is not None and np.isfinite(s)]
    ranked = [
        _rank_one(r, primary_metric, baseline_score, weights, interpretability_need, registry, field_scores)
        for r in usable
    ]
    ranked.sort(key=lambda x: x.composite, reverse=True)
    for i, item in enumerate(ranked, 1):
        item.rank = i
    outcome.ranked = ranked

    outcome.table = _build_table(usable, ranked, primary_metric, task_type, display_metrics)
    outcome.baseline = next((r for r in ranked if r.result_id == (baseline_result.id if baseline_result else None)), None)

    non_baseline = [r for r in ranked if r.result_id != (baseline_result.id if baseline_result else None)]
    pool = non_baseline or ranked

    outcome.recommended = pool[0]
    outcome.highest_performance = max(
        pool, key=lambda r: (r.primary_score if M.is_better(primary_metric, r.primary_score, None) else 0)
        if primary_metric not in M.LOWER_IS_BETTER else -(r.primary_score or float("inf"))
    )
    outcome.simplest_acceptable = _simplest_acceptable(pool, primary_metric, registry)
    outcome.alternative = next(
        (r for r in pool[1:] if r.model_key != outcome.recommended.model_key), None
    )

    _record_decisions(outcome, usable, ranked, primary_metric, baseline_result, registry)
    _record_warnings(outcome, usable, ranked, primary_metric, baseline_score)
    return outcome


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _rank_one(
    result: ExperimentResult,
    primary_metric: str,
    baseline_score: float | None,
    weights: dict[str, float],
    interpretability_need: str,
    registry: ModelRegistry,
    field_scores: list[float] | None = None,
) -> RankedModel:
    score = result.primary(primary_metric)
    ranked = RankedModel(
        result_id=result.id,
        model_key=result.model_key,
        model_name=result.model_name,
        primary_metric=primary_metric,
        primary_score=score,
        score_source=result.score_source(primary_metric),
        interpretability=result.interpretability,
        training_time_s=result.training_time_s,
        concerns=list(result.warnings),
    )

    performance = _normalise_performance(primary_metric, score, result.task_type, field_scores)
    generalisation = _generalisation_component(result)
    stability = _stability_component(result, primary_metric)
    interpretability = _interpretability_component(result, interpretability_need)
    efficiency = _efficiency_component(result)

    ranked.components = {
        "performance": round(performance, 4),
        "generalisation": round(generalisation, 4),
        "stability": round(stability, 4),
        "interpretability": round(interpretability, 4),
        "efficiency": round(efficiency, 4),
    }
    ranked.composite = round(
        sum(weights[k] * v for k, v in ranked.components.items() if k in weights), 4
    )

    if baseline_score is not None and score is not None:
        ranked.beats_baseline = M.is_better(primary_metric, score, baseline_score)
        if abs(baseline_score) > 1e-9:
            improvement = (baseline_score - score) if primary_metric in M.LOWER_IS_BETTER else (score - baseline_score)
            ranked.lift_over_baseline = round(float(improvement / abs(baseline_score)) * 100, 1)

    if generalisation > 0.9:
        ranked.strengths.append("training and held-out performance agree closely")
    if stability > 0.9:
        ranked.strengths.append("scores barely move between folds")
    if result.interpretability == "transparent":
        ranked.strengths.append("the mechanism can be read directly")
    if ranked.beats_baseline is False:
        ranked.concerns.append("does not beat the naive baseline — it is not adding value")
    return ranked


def _normalise_performance(
    metric: str,
    score: float | None,
    task_type: TaskType,
    field_scores: list[float] | None = None,
) -> float:
    """Map a raw metric onto 0-1, so different metrics can share one composite.

    Bounded metrics (R², AUC, F1) map directly. Unbounded ones (RMSE, MAE) have
    no absolute scale, so they are normalised against the rest of the field —
    the best model in this tournament scores 1, the worst 0.
    """
    if score is None or not np.isfinite(score):
        return 0.0
    if metric in {"r2", "explained_variance"}:
        return float(np.clip(score, 0.0, 1.0))
    if metric in {"roc_auc", "pr_auc", "f1", "f1_macro", "accuracy", "balanced_accuracy",
                  "precision", "recall", "silhouette"}:
        return float(np.clip((score + 1) / 2 if metric == "silhouette" else score, 0.0, 1.0))
    if metric == "mcc":
        return float(np.clip((score + 1) / 2, 0.0, 1.0))
    if metric == "mase":
        # 0 is perfect, 1 is "no better than naive"; beyond 2 it is simply bad.
        return float(np.clip(1.0 - score / 2.0, 0.0, 1.0))
    if metric in {"mape", "smape"}:
        return float(np.clip(1.0 - score / 100.0, 0.0, 1.0))
    # Unbounded error metrics: rank within this tournament's field.
    if not field_scores or len(field_scores) < 2:
        return 0.5
    low, high = min(field_scores), max(field_scores)
    if high - low < 1e-12:
        return 0.5
    normalised = (score - low) / (high - low)
    # Lower is better for these, so invert.
    return float(np.clip(1.0 - normalised if metric in LOWER_IS_BETTER_LOCAL else normalised, 0.0, 1.0))


def _generalisation_component(result: ExperimentResult) -> float:
    gap = result.overfitting_gap
    if gap is None:
        return 0.7  # unknown, so neither rewarded nor punished
    return float(np.clip(1.0 - max(gap, 0.0) * 2.0, 0.0, 1.0))


def _stability_component(result: ExperimentResult, metric: str) -> float:
    if not result.validation.fold_scores.get(metric):
        return 0.6
    cv = result.validation.stability(metric)
    if not np.isfinite(cv):
        return 0.3
    return float(np.clip(1.0 - cv * 2.0, 0.0, 1.0))


def _interpretability_component(result: ExperimentResult, need: str) -> float:
    base = {"transparent": 1.0, "moderate": 0.6, "opaque": 0.2}.get(result.interpretability, 0.5)
    if need == "critical":
        return base ** 0.5
    if need == "low":
        return 0.5 + base * 0.5
    return base


def _efficiency_component(result: ExperimentResult) -> float:
    seconds = max(result.training_time_s, 1e-3)
    return float(np.clip(1.0 - np.log10(seconds + 1) / 3.0, 0.0, 1.0))


def _find_baseline(results: list[ExperimentResult], registry: ModelRegistry) -> ExperimentResult | None:
    for r in results:
        try:
            if registry.get(r.model_key).baseline:
                return r
        except KeyError:
            continue
    return None


def _simplest_acceptable(
    ranked: list[RankedModel],
    primary_metric: str,
    registry: ModelRegistry,
    tolerance: float = 0.05,
) -> RankedModel | None:
    """The most interpretable model within `tolerance` of the best performance.

    This is the question most analyses should ask and rarely do: what is the
    plainest model I can defend, given how little accuracy it actually costs?
    """
    scored = [r for r in ranked if r.primary_score is not None and np.isfinite(r.primary_score)]
    if not scored:
        return None
    best = max(scored, key=lambda r: -r.primary_score if primary_metric in M.LOWER_IS_BETTER else r.primary_score)
    best_score = best.primary_score
    if best_score is None or abs(best_score) < 1e-12:
        return best

    acceptable = []
    for r in scored:
        if primary_metric in M.LOWER_IS_BETTER:
            within = r.primary_score <= best_score * (1 + tolerance)
        else:
            within = r.primary_score >= best_score * (1 - tolerance)
        if within:
            acceptable.append(r)
    if not acceptable:
        return best

    def simplicity(item: RankedModel) -> tuple[int, float]:
        try:
            spec = registry.get(item.model_key)
            return (spec.interpretability.rank, -spec.cost.rank)
        except KeyError:
            return (0, 0.0)

    return max(acceptable, key=simplicity)


# --------------------------------------------------------------------------
# presentation and reasoning
# --------------------------------------------------------------------------

def _build_table(
    results: list[ExperimentResult],
    ranked: list[RankedModel],
    primary_metric: str,
    task_type: TaskType,
    display_metrics: list[str] | None,
) -> list[dict[str, Any]]:
    columns = display_metrics or M.default_metrics(task_type)
    if primary_metric not in columns:
        columns = [primary_metric] + columns
    by_id = {r.id: r for r in results}
    rows = []
    for item in ranked:
        result = by_id[item.result_id]
        row: dict[str, Any] = {
            "Rank": item.rank,
            "Model": item.model_name,
            "Composite": item.composite,
        }
        for metric in columns:
            value = result.primary(metric)
            row[M.METRIC_LABELS.get(metric, metric)] = round(value, 4) if value is not None and np.isfinite(value) else None
        row["Training time (s)"] = round(result.training_time_s, 3)
        row["Interpretability"] = result.interpretability
        row["Overfit gap"] = result.overfitting_gap
        row["Score from"] = item.score_source
        row["Warnings"] = len(result.warnings)
        rows.append(row)
    return rows


def _record_decisions(
    outcome: TournamentOutcome,
    results: list[ExperimentResult],
    ranked: list[RankedModel],
    primary_metric: str,
    baseline: ExperimentResult | None,
    registry: ModelRegistry,
) -> None:
    best = outcome.recommended
    if best is None:
        return
    label = M.METRIC_LABELS.get(primary_metric, primary_metric)

    evidence = [
        f"{label} = {M.format_metric(primary_metric, best.primary_score)} ({best.score_source})",
        f"Composite score {best.composite} from "
        + ", ".join(f"{k} {v:.2f}" for k, v in best.components.items()),
    ]
    if best.lift_over_baseline is not None:
        evidence.append(f"{best.lift_over_baseline:+.1f}% against the naive baseline")
    if best.strengths:
        evidence.append("Strengths: " + "; ".join(best.strengths))

    rejected = []
    for item in ranked[1:6]:
        if item.result_id == best.result_id:
            continue
        if item.concerns:
            reason = item.concerns[0]
        elif not _materially_better(primary_metric, best.primary_score, item.primary_score):
            reason = (
                f"scores the same as the recommendation on {label} to within a rounding difference; "
                "the recommendation wins on the other criteria, and either would be defensible"
            )
        else:
            reason = (
                f"{label} {M.format_metric(primary_metric, item.primary_score)} versus "
                f"{M.format_metric(primary_metric, best.primary_score)} for the recommended model"
            )
        rejected.append({"option": item.model_name, "reason": reason})

    outcome.decisions.append(
        Decision(
            stage="model_recommendation",
            decision=f"Recommend {best.model_name}",
            reason=_recommendation_reason(best, outcome, registry),
            evidence=evidence,
            rejected=rejected,
            confidence=_recommendation_confidence(best, ranked),
        )
    )

    if outcome.simplest_acceptable and outcome.simplest_acceptable.result_id != best.result_id:
        simple = outcome.simplest_acceptable
        outcome.decisions.append(
            Decision(
                stage="model_recommendation",
                decision=f"Simplest acceptable alternative: {simple.model_name}",
                reason=(
                    f"It lands within 5% of the best {label} while being easier to explain and defend. "
                    "If the analysis has to be justified to a non-technical audience or a regulator, "
                    "that trade is usually worth making."
                ),
                evidence=[f"{label} = {M.format_metric(primary_metric, simple.primary_score)}",
                          f"Interpretability: {simple.interpretability}"],
                confidence=Confidence.MODERATE,
            )
        )

    if (
        outcome.highest_performance
        and outcome.highest_performance.result_id != best.result_id
        and _materially_better(outcome.primary_metric, outcome.highest_performance.primary_score, best.primary_score)
    ):
        top = outcome.highest_performance
        outcome.decisions.append(
            Decision(
                stage="model_recommendation",
                decision=f"Highest raw score: {top.model_name} — not recommended",
                reason=(
                    "It has the best headline number but loses on the other criteria: "
                    + ("; ".join(top.concerns[:2]) if top.concerns else
                       "weaker generalisation, stability or interpretability once those are weighed in.")
                ),
                evidence=[f"{label} = {M.format_metric(primary_metric, top.primary_score)}",
                          f"Composite {top.composite} versus {best.composite} for the recommendation"],
                confidence=Confidence.MODERATE,
            )
        )


def _materially_better(metric: str, candidate: float | None, reference: float | None,
                       relative_threshold: float = 0.01) -> bool:
    """Is the difference big enough to be worth a sentence, or is it rounding?

    Presenting a 0.0002 gap as a trade-off would be dishonest precision.
    """
    if candidate is None or reference is None:
        return False
    if not (np.isfinite(candidate) and np.isfinite(reference)):
        return False
    scale = max(abs(reference), 1e-9)
    difference = (reference - candidate) if metric in M.LOWER_IS_BETTER else (candidate - reference)
    return bool(difference / scale > relative_threshold)


def _recommendation_reason(best: RankedModel, outcome: TournamentOutcome, registry: ModelRegistry) -> str:
    label = M.METRIC_LABELS.get(outcome.primary_metric, outcome.primary_metric)
    try:
        spec = registry.get(best.model_key)
        traits = spec.advantages[0].lower() if spec.advantages else "a good fit for this data"
    except KeyError:
        traits = "a good fit for this data"
    parts = [
        f"Best-performing model for this dataset under the chosen validation strategy, not best in general.",
        f"{label} of {M.format_metric(outcome.primary_metric, best.primary_score)} ({best.score_source}), "
        f"and it wins on the weighted comparison once generalisation, stability, interpretability and "
        f"training cost are taken into account, not on the raw score alone.",
        f"It also {traits}.",
    ]
    if best.beats_baseline is False:
        parts.append(
            "Note that it does not beat the naive baseline, which means none of these models is "
            "currently adding value over the simplest possible rule."
        )
    return " ".join(parts)


def _recommendation_confidence(best: RankedModel, ranked: list[RankedModel]) -> Confidence:
    if best.concerns:
        return Confidence.LOW if len(best.concerns) > 1 else Confidence.MODERATE
    if len(ranked) > 1:
        margin = best.composite - ranked[1].composite
        if margin < 0.02:
            return Confidence.MODERATE
    if best.components.get("stability", 0) > 0.85 and best.components.get("generalisation", 0) > 0.85:
        return Confidence.HIGH
    return Confidence.MODERATE


def _record_warnings(
    outcome: TournamentOutcome,
    results: list[ExperimentResult],
    ranked: list[RankedModel],
    primary_metric: str,
    baseline_score: float | None,
) -> None:
    if baseline_score is not None:
        beaten = [r for r in ranked if r.beats_baseline]
        if not beaten:
            outcome.warnings.append(
                "No model beat the naive baseline. Either the features carry no signal for this "
                "target, the target is close to random, or the problem is framed wrongly. Adding "
                "more models will not fix any of those."
            )
        elif len(beaten) == 1 and len(ranked) > 3:
            outcome.warnings.append(
                "Only one model beat the baseline, and by a narrow margin. Treat that as weak "
                "evidence rather than a result."
            )

    if len(ranked) > 1:
        top_scores = [r.primary_score for r in ranked[:3] if r.primary_score is not None]
        if len(top_scores) > 1 and np.std(top_scores) < 1e-6:
            outcome.warnings.append(
                "The top models score identically. That usually means the signal is dominated by "
                "one or two features that every model finds — check for leakage before celebrating."
            )

    overfitters = [r for r in results if r.overfitting_gap is not None and r.overfitting_gap > 0.2]
    if len(overfitters) >= max(2, len(results) // 2):
        outcome.warnings.append(
            f"{len(overfitters)} of {len(results)} models overfit noticeably. That is usually a "
            "sample-size or feature-count problem, not a model problem."
        )


def tournament_frame(outcome: TournamentOutcome) -> pd.DataFrame:
    """The tournament as a DataFrame, for display or export."""
    return pd.DataFrame(outcome.table) if outcome.table else pd.DataFrame()
