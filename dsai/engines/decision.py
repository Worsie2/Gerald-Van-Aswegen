"""The Decision Engine: which analytical approaches suit *this* problem.

It queries the model registry with the constraints the data imposes, then scores
the survivors on how well their metadata matches the situation — dataset size,
dimensionality, missingness, linearity, class balance, interpretability needs and
compute budget. It never claims a model is best; it produces a ranked shortlist
with reasons, which the experiment engine then actually tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, Decision, JsonMixin, Objective,
    SemanticType, TaskType,
)
from dsai.registry.base import REGISTRY, Cost, Interpretability, ModelRegistry, ModelSpec


@dataclass
class Constraints(JsonMixin):
    """What the user (or the environment) will and will not accept."""

    max_models: int = 10
    time_budget: str = "balanced"          # fast | balanced | thorough
    interpretability_need: str = "moderate"  # critical | high | moderate | low
    prefer_families: list[str] = field(default_factory=list)
    exclude_models: list[str] = field(default_factory=list)
    include_models: list[str] = field(default_factory=list)
    allow_expensive: bool = True
    require_probabilities: bool = False

    @property
    def max_cost(self) -> Cost:
        return {"fast": Cost.LOW, "balanced": Cost.HIGH, "thorough": Cost.VERY_HIGH}.get(
            self.time_budget, Cost.HIGH
        )

    @property
    def min_interpretability(self) -> Interpretability | None:
        if self.interpretability_need == "critical":
            return Interpretability.TRANSPARENT
        if self.interpretability_need == "high":
            return Interpretability.MODERATE
        return None


@dataclass
class ModelCandidate(JsonMixin):
    spec_key: str
    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    category: str = ""
    family: str = ""
    interpretability: str = ""
    cost: str = ""


@dataclass
class AnalysisPlan(JsonMixin):
    """The engine's answer: what to run, in what order, and why."""

    objective: Objective
    candidates: list[ModelCandidate] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    validation_strategy: dict[str, Any] = field(default_factory=dict)
    primary_metric: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def model_keys(self) -> list[str]:
        return [c.spec_key for c in self.candidates]


# --------------------------------------------------------------------------
# dataset situation summary
# --------------------------------------------------------------------------

@dataclass
class Situation(JsonMixin):
    """The measured facts the model-selection rules actually key off."""

    n_rows: int = 0
    n_features: int = 0
    tiny: bool = False
    small: bool = False
    large: bool = False
    very_large: bool = False
    wide: bool = False
    has_missing: bool = False
    missing_share: float = 0.0
    has_categorical: bool = False
    high_cardinality: bool = False
    nonlinear_evidence: bool = False
    multicollinear: bool = False
    outlier_heavy: bool = False
    imbalanced: bool = False
    severely_imbalanced: bool = False
    n_classes: int = 0
    minority_share: float = 1.0
    temporal: bool = False
    noisy: bool = False
    # target characteristics — these decide whether a GLM's link function is legal
    target_min: float | None = None
    target_is_count: bool = False
    target_strictly_positive: bool = False
    target_skew: float | None = None


def assess_situation(
    profile: DatasetProfile,
    objective: Objective,
    frame: Any = None,
) -> Situation:
    features = objective.features or [c for c in profile.modelling_columns if c != objective.target]
    n_features = len(features)
    n_rows = profile.n_rows

    missing_values = sum(profile.columns[c].n_missing for c in features if c in profile.columns)
    total_cells = max(n_rows * max(n_features, 1), 1)

    situation = Situation(
        n_rows=n_rows,
        n_features=n_features,
        tiny=n_rows < 50,
        small=n_rows < 500,
        large=n_rows >= 50_000,
        very_large=n_rows >= 500_000,
        wide=n_features > max(20, n_rows / 10),
        missing_share=round(missing_values / total_cells, 4),
        has_missing=missing_values > 0,
        temporal=bool(profile.datetime_columns),
    )
    situation.has_categorical = any(
        profile.columns[c].is_categorical for c in features if c in profile.columns
    )
    situation.high_cardinality = any(
        profile.columns[c].semantic_type is SemanticType.HIGH_CARDINALITY_CATEGORICAL
        for c in features if c in profile.columns
    )
    situation.nonlinear_evidence = any(r.kind == "nonlinearity" for r in profile.relationships)
    situation.multicollinear = any(v >= 10 for v in profile.multicollinearity.values())
    outliery = [c for c in features if c in profile.columns and profile.columns[c].outlier_pct >= 5]
    situation.outlier_heavy = len(outliery) >= max(1, n_features // 3)

    target_col = profile.columns.get(objective.target) if objective.target else None
    if target_col is not None and target_col.is_numeric:
        situation.target_min = target_col.minimum
        situation.target_skew = target_col.skewness
        situation.target_strictly_positive = bool(target_col.minimum is not None and target_col.minimum > 0)
        situation.target_is_count = bool(
            target_col.semantic_type is SemanticType.NUMERIC_DISCRETE
            and (target_col.minimum or 0) >= 0
        )

    if objective.task_type.is_classification and objective.target and frame is not None:
        from dsai.core.profiler import class_imbalance

        balance = class_imbalance(frame[objective.target])
        situation.imbalanced = bool(balance.get("imbalanced"))
        situation.severely_imbalanced = bool(balance.get("severely_imbalanced"))
        situation.n_classes = int(balance.get("n_classes", 0))
        situation.minority_share = float(balance.get("minority_share", 1.0))
    return situation


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

def plan_analysis(
    profile: DatasetProfile,
    objective: Objective,
    context: BusinessContext | None = None,
    constraints: Constraints | None = None,
    registry: ModelRegistry | None = None,
    frame: Any = None,
) -> AnalysisPlan:
    """Produce a ranked, reasoned shortlist of approaches for one objective."""
    context = context or BusinessContext()
    constraints = constraints or Constraints()
    registry = registry or REGISTRY

    situation = assess_situation(profile, objective, frame)
    plan = AnalysisPlan(objective=objective)
    plan.primary_metric = choose_primary_metric(objective, situation, context)
    plan.validation_strategy = choose_validation(objective, situation, profile)

    _record_feasibility_warnings(plan, situation, objective, profile)

    pool = _candidate_pool(registry, objective, situation, constraints)
    if not pool:
        plan.warnings.append(
            f"No registered model fits a {objective.task_type.value} problem under the current "
            "constraints. Loosen the interpretability or time-budget setting."
        )
        return plan

    scored = [_score_model(spec, situation, objective, constraints, context) for spec in pool]
    scored.sort(key=lambda c: c.score, reverse=True)

    forced = [
        _score_model(registry.get(k), situation, objective, constraints, context)
        for k in constraints.include_models if k in registry
    ]
    selected = _diversify(scored, constraints.max_models, forced)
    plan.candidates = selected

    _record_selection_decisions(plan, selected, scored, situation, objective, constraints)
    return plan


def _candidate_pool(
    registry: ModelRegistry,
    objective: Objective,
    situation: Situation,
    constraints: Constraints,
) -> list[ModelSpec]:
    pool = registry.find(
        task_type=objective.task_type,
        max_cost=constraints.max_cost if not constraints.allow_expensive else None,
        min_interpretability=constraints.min_interpretability,
        n_rows=situation.n_rows,
        n_features=situation.n_features,
        require_proba=constraints.require_probabilities,
        only_available=True,
    )
    excluded = set(constraints.exclude_models)
    return [s for s in pool if s.key not in excluded]


def _score_model(
    spec: ModelSpec,
    situation: Situation,
    objective: Objective,
    constraints: Constraints,
    context: BusinessContext,
) -> ModelCandidate:
    """Score a model on fit-to-situation. Positive reasons, negative concerns."""
    score = 0.5
    reasons: list[str] = []
    concerns: list[str] = []

    # --- sample size -------------------------------------------------------
    if situation.tiny:
        if spec.family in {"linear", "statistical", "probabilistic"} or spec.baseline:
            score += 0.20
            reasons.append(f"only {situation.n_rows} rows, and simple models are the honest choice at that size")
        if spec.family in {"neural", "ensemble"} and not spec.baseline:
            score -= 0.25
            concerns.append("needs far more rows than are available to estimate reliably")
    elif situation.small:
        if spec.cost.rank >= Cost.HIGH.rank and spec.family == "neural":
            score -= 0.15
            concerns.append("data-hungry model on a small dataset")
        if spec.family in {"linear", "tree", "probabilistic"}:
            score += 0.08
            reasons.append("well suited to a modest sample size")
    else:
        if spec.family in {"ensemble"}:
            score += 0.15
            reasons.append("enough rows for an ensemble to pay off")
        if spec.baseline:
            score -= 0.05

    if situation.very_large and spec.cost.rank >= Cost.HIGH.rank:
        score -= 0.20
        concerns.append(f"expensive to train on {situation.n_rows:,} rows")
    if situation.large and "fast" in spec.tags:
        score += 0.10
        reasons.append("designed for large datasets")

    # --- dimensionality ----------------------------------------------------
    if situation.wide:
        if spec.handles_high_dimensionality:
            score += 0.20
            reasons.append(f"handles {situation.n_features} features against {situation.n_rows} rows")
        elif spec.family in {"distance", "kernel"}:
            score -= 0.20
            concerns.append("distance-based methods degrade badly in high dimensions")

    # --- missing data ------------------------------------------------------
    if situation.has_missing and spec.handles_missing:
        score += 0.12
        reasons.append("handles missing values natively, without imputing a value that was never observed")

    # --- categoricals ------------------------------------------------------
    if situation.high_cardinality and spec.handles_categorical:
        score += 0.10
        reasons.append("handles high-cardinality categories directly")

    # --- shape of the relationship ----------------------------------------
    if situation.nonlinear_evidence:
        if spec.nonlinear:
            score += 0.18
            reasons.append("the data shows nonlinear relationships, which this model can capture")
        else:
            score -= 0.12
            concerns.append("assumes straight-line relationships, but the data suggests curvature")
    else:
        if not spec.nonlinear and spec.interpretability is Interpretability.TRANSPARENT:
            score += 0.08
            reasons.append("no strong nonlinearity detected, so a simpler model is defensible")

    # --- collinearity ------------------------------------------------------
    if situation.multicollinear:
        if "regularised" in spec.tags or spec.family in {"ensemble", "tree"}:
            score += 0.15
            reasons.append("copes with strongly correlated predictors")
        elif spec.supports_coefficients and spec.family in {"linear", "statistical"} and "regularised" not in spec.tags:
            score -= 0.15
            concerns.append("collinearity makes unregularised coefficients unstable and hard to trust")

    # --- outliers ----------------------------------------------------------
    if situation.outlier_heavy:
        if spec.robust_to_outliers:
            score += 0.12
            reasons.append("resistant to the heavy tails in this data")
        elif spec.family in {"linear", "statistical"}:
            score -= 0.10
            concerns.append("sensitive to the outliers present")

    # --- class imbalance ---------------------------------------------------
    if situation.severely_imbalanced:
        if any(h.name in {"class_weight", "scale_pos_weight"} for h in spec.hyperparameters):
            score += 0.15
            reasons.append(f"supports class weighting, which matters at {situation.minority_share:.1%} minority share")
        elif not spec.baseline:
            score -= 0.08
            concerns.append("no built-in handling for severe class imbalance")

    # --- interpretability --------------------------------------------------
    need = constraints.interpretability_need
    style = (context.analysis_style or "").lower()
    explanatory = style.startswith("explan") or need in {"critical", "high"}
    if explanatory:
        score += 0.10 * (spec.interpretability.rank - 2)
        if spec.interpretability is Interpretability.TRANSPARENT:
            reasons.append("its mechanism can be read directly, which the stated objective requires")
        elif spec.interpretability is Interpretability.OPAQUE:
            concerns.append("needs post-hoc explanation before anyone can act on it")
    elif need == "low":
        score += 0.05 if spec.interpretability is Interpretability.OPAQUE else 0.0

    # --- compute budget ----------------------------------------------------
    if constraints.time_budget == "fast" and spec.cost.rank >= Cost.HIGH.rank:
        score -= 0.20
        concerns.append("too slow for the chosen time budget")
    if constraints.time_budget == "thorough" and "high_accuracy" in spec.tags:
        score += 0.10
        reasons.append("among the strongest performers when there is time to tune it")

    # --- does the target actually satisfy this model's link function? -------
    if objective.task_type is TaskType.REGRESSION and situation.target_min is not None:
        if spec.key in {"gamma_regression", "tweedie_regression"} and not situation.target_strictly_positive:
            score -= 0.45
            concerns.append(f"requires a strictly positive target, but the minimum here is {situation.target_min:g}")
        if spec.key == "poisson_regression":
            if (situation.target_min or 0) < 0:
                score -= 0.45
                concerns.append("Poisson regression cannot model a target with negative values")
            elif not situation.target_is_count:
                score -= 0.25
                concerns.append("the target is continuous, not a count, so the Poisson likelihood is the wrong assumption")
            else:
                score += 0.20
                reasons.append("the target is a non-negative count, which is exactly what this likelihood is for")
        if spec.key == "gamma_regression" and situation.target_strictly_positive and (situation.target_skew or 0) > 1:
            score += 0.15
            reasons.append("the target is positive and right-skewed, which suits a Gamma likelihood")

    # --- user family preference --------------------------------------------
    if constraints.prefer_families and spec.family in constraints.prefer_families:
        score += 0.15
        reasons.append("matches the model family you asked for")

    # --- baselines always earn a place --------------------------------------
    if spec.baseline:
        score = max(score, 0.45)
        reasons.append("included as a benchmark so you can see what the real models are worth")

    return ModelCandidate(
        spec_key=spec.key,
        name=spec.name,
        score=round(min(1.0, max(0.0, score)), 4),
        reasons=reasons,
        concerns=concerns,
        category=spec.category,
        family=spec.family,
        interpretability=spec.interpretability.value,
        cost=spec.cost.value,
    )


def _diversify(
    scored: list[ModelCandidate],
    max_models: int,
    forced: list[ModelCandidate] | None = None,
) -> list[ModelCandidate]:
    """Pick a shortlist that spans model families rather than five flavours of boosting.

    A tournament of near-identical models tells you nothing about whether the
    problem needs complexity at all.
    """
    selected: list[ModelCandidate] = list(forced or [])
    seen = {c.spec_key for c in selected}
    family_counts: dict[str, int] = {}
    for c in selected:
        family_counts[c.family] = family_counts.get(c.family, 0) + 1

    cap = 3 if max_models >= 8 else 2
    for candidate in scored:
        if len(selected) >= max_models:
            break
        if candidate.spec_key in seen:
            continue
        if family_counts.get(candidate.family, 0) >= cap and not candidate.spec_key.startswith("dummy"):
            continue
        selected.append(candidate)
        seen.add(candidate.spec_key)
        family_counts[candidate.family] = family_counts.get(candidate.family, 0) + 1

    # top up if family caps left room unused
    if len(selected) < max_models:
        for candidate in scored:
            if len(selected) >= max_models:
                break
            if candidate.spec_key not in seen:
                selected.append(candidate)
                seen.add(candidate.spec_key)

    # always keep a baseline in the field
    if not any(c.spec_key.startswith("dummy") or c.spec_key in {"naive_forecast"} for c in selected):
        for candidate in scored:
            if candidate.spec_key.startswith("dummy") or candidate.spec_key == "naive_forecast":
                selected.append(candidate)
                break
    return selected


# --------------------------------------------------------------------------
# validation and metric choice
# --------------------------------------------------------------------------

def choose_validation(objective: Objective, situation: Situation, profile: DatasetProfile) -> dict[str, Any]:
    """Pick a validation scheme that suits the data, and say why."""
    task = objective.task_type

    if task is TaskType.TIME_SERIES_FORECAST:
        n_splits = 3 if situation.n_rows < 100 else 5
        return {
            "strategy": "time_series_split",
            "n_splits": n_splits,
            "reason": (
                "Time-ordered data must be validated forward in time. Random folds would train on "
                "the future to predict the past, which inflates the score and cannot happen in production."
            ),
            "shuffle": False,
        }

    if not task.is_supervised:
        return {
            "strategy": "none",
            "reason": (
                "This is an unsupervised task, so there is no held-out label to score against. "
                "Quality is judged by internal validity measures and by whether the result is usable."
            ),
        }

    if situation.tiny:
        return {
            "strategy": "repeated_stratified_kfold" if task.is_classification else "repeated_kfold",
            "n_splits": min(5, max(2, situation.n_rows // 10)),
            "n_repeats": 5,
            "reason": (
                f"With only {situation.n_rows} rows a single split is mostly luck. Repeating the "
                "cross-validation gives an estimate of how much the score itself varies."
            ),
            "shuffle": True,
            "random_state": 42,
        }

    if task.is_classification:
        n_splits = 5 if situation.n_rows < 10_000 else 5
        if situation.severely_imbalanced and situation.minority_share * situation.n_rows < n_splits * 5:
            n_splits = max(2, int(situation.minority_share * situation.n_rows // 5))
        return {
            "strategy": "stratified_kfold",
            "n_splits": max(2, n_splits),
            "shuffle": True,
            "random_state": 42,
            "reason": (
                "Stratified folds keep the class proportions the same in every fold; with imbalanced "
                "data, unstratified folds can end up with almost none of the minority class."
            ),
        }

    return {
        "strategy": "kfold",
        "n_splits": 5 if situation.n_rows >= 200 else 3,
        "shuffle": True,
        "random_state": 42,
        "reason": "Standard k-fold cross-validation: every row is used for both training and testing, across folds.",
    }


def choose_primary_metric(objective: Objective, situation: Situation, context: BusinessContext) -> str:
    task = objective.task_type
    if task is TaskType.REGRESSION:
        return "rmse" if not situation.outlier_heavy else "mae"
    if task is TaskType.BINARY_CLASSIFICATION:
        if situation.severely_imbalanced:
            return "pr_auc"
        if situation.imbalanced:
            return "roc_auc"
        return "f1"
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return "balanced_accuracy" if situation.imbalanced else "f1_macro"
    if task is TaskType.TIME_SERIES_FORECAST:
        return "mase"
    if task is TaskType.CLUSTERING:
        return "silhouette"
    if task is TaskType.ANOMALY_DETECTION:
        return "anomaly_rate"
    return "n/a"


# --------------------------------------------------------------------------
# decision logging
# --------------------------------------------------------------------------

def _record_feasibility_warnings(plan: AnalysisPlan, situation: Situation,
                                 objective: Objective, profile: DatasetProfile) -> None:
    if situation.tiny:
        plan.warnings.append(
            f"Only {situation.n_rows} rows. Any model here is fitting noise as much as signal — "
            "treat every result as indicative, not conclusive."
        )
    if situation.wide:
        plan.warnings.append(
            f"{situation.n_features} features against {situation.n_rows} rows. Some variables will look "
            "predictive purely by chance; regularisation and honest validation matter more than model choice."
        )
    if situation.severely_imbalanced:
        plan.warnings.append(
            f"The minority class is only {situation.minority_share:.1%} of the data. Accuracy is "
            "meaningless here — a model predicting the majority class every time would score "
            f"{1 - situation.minority_share:.1%}."
        )
    critical = profile.issues_by_severity("critical")
    for issue in critical:
        plan.warnings.append(f"Data quality: {issue.message}")
    if objective.task_type is TaskType.TIME_SERIES_FORECAST and objective.time_column:
        col = profile.columns.get(objective.time_column)
        if col and col.n_missing_periods:
            plan.warnings.append(
                f"'{objective.time_column}' has roughly {col.n_missing_periods} missing period(s). "
                "Gaps distort seasonality estimates — fill or resample them before forecasting."
            )


def _record_selection_decisions(
    plan: AnalysisPlan,
    selected: list[ModelCandidate],
    all_scored: list[ModelCandidate],
    situation: Situation,
    objective: Objective,
    constraints: Constraints,
) -> None:
    plan.decisions.append(
        Decision(
            stage="problem_framing",
            decision=f"Treat this as {objective.task_type.value.replace('_', ' ')}"
                     + (f" with '{objective.target}' as the target" if objective.target else ""),
            reason=objective.rationale,
            evidence=[
                f"{situation.n_rows:,} rows, {situation.n_features} candidate features",
                f"Objective source: {objective.source.replace('_', ' ')}",
            ],
            confidence=objective.confidence,
        )
    )
    plan.decisions.append(
        Decision(
            stage="validation",
            decision=f"Validate with {plan.validation_strategy.get('strategy', 'n/a').replace('_', ' ')}",
            reason=plan.validation_strategy.get("reason", ""),
            evidence=[f"{plan.validation_strategy.get('n_splits', '-')} splits"],
            confidence=Confidence.HIGH,
        )
    )
    plan.decisions.append(
        Decision(
            stage="metric",
            decision=f"Rank primarily on {plan.primary_metric}",
            reason=_metric_reason(plan.primary_metric, situation),
            confidence=Confidence.HIGH,
        )
    )

    selected_keys = {c.spec_key for c in selected}
    rejected = [
        {"option": c.name, "reason": c.concerns[0] if c.concerns else "scored below the shortlist cut-off"}
        for c in all_scored if c.spec_key not in selected_keys
    ][:8]
    plan.decisions.append(
        Decision(
            stage="model_selection",
            decision=f"Shortlist {len(selected)} model(s) to train and compare",
            reason=(
                "The shortlist spans different model families deliberately. If a simple model matches "
                "a complex one, that is itself the finding — and none of these are recommended until "
                "they have actually been validated on this data."
            ),
            evidence=[f"{c.name}: {c.reasons[0] if c.reasons else 'general-purpose fit'}" for c in selected[:8]],
            rejected=rejected,
            confidence=Confidence.MODERATE,
        )
    )


def _metric_reason(metric: str, situation: Situation) -> str:
    return {
        "rmse": "Squared error penalises large misses hardest, which is usually what a business cares about; it is in the target's own units.",
        "mae": "Mean absolute error is chosen over RMSE because the outliers here would let a few extreme rows dictate the ranking.",
        "roc_auc": "AUC measures ranking quality across every threshold, so it is not distorted by the class imbalance present.",
        "pr_auc": "With a severely imbalanced target, precision-recall AUC reflects performance on the rare class that actually matters; ROC AUC would look flattering.",
        "f1": "F1 balances precision and recall, which matters when both false positives and false negatives cost something.",
        "f1_macro": "Macro-averaged F1 weights every class equally, so small classes are not drowned out.",
        "balanced_accuracy": "Balanced accuracy averages recall across classes, so the majority class cannot carry the score.",
        "mase": "MASE compares the forecast against a naive baseline, so a score above 1 means the model is worse than doing nothing.",
        "silhouette": "Silhouette measures how well separated the clusters are, without needing ground-truth labels.",
        "anomaly_rate": "There are no labels, so the flag rate plus manual review is the honest measure.",
    }.get(metric, "Chosen as the most appropriate summary measure for this problem type.")
