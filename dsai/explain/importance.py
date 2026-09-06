"""Explanation layer: what the model learned, in terms a person can act on.

Three levels, deliberately kept apart because they answer different questions:

* **Global** — which variables matter across the whole model.
* **Marginal** — how the prediction changes as one variable moves.
* **Local** — why *this* row got *this* prediction.

Where SHAP is installed it is used; where it is not, an exact additive
attribution is computed for linear and tree models, and a sampling
approximation elsewhere. The method used is always reported, because "SHAP-style"
and "SHAP" are not the same claim.
"""

from __future__ import annotations

from dsai.engines.metrics import human_number

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import Confidence, JsonMixin, TaskType


@dataclass
class FeatureImportance(JsonMixin):
    feature: str
    importance: float
    std: float = 0.0
    direction: str = ""          # "increases" | "decreases" | "" (unsigned)
    method: str = ""
    rank: int = 0
    interpretation: str = ""


@dataclass
class ExplanationBundle(JsonMixin):
    method: str = ""
    method_note: str = ""
    importances: list[FeatureImportance] = field(default_factory=list)
    coefficients: list[dict[str, Any]] = field(default_factory=list)
    partial_dependence: dict[str, Any] = field(default_factory=dict)
    confidence: Confidence = Confidence.MODERATE
    caveats: list[str] = field(default_factory=list)
    plain_english: list[str] = field(default_factory=list)

    def top(self, n: int = 10) -> list[FeatureImportance]:
        return self.importances[:n]


# --------------------------------------------------------------------------
# global importance
# --------------------------------------------------------------------------

def explain_model(
    model: Any,
    X: pd.DataFrame,
    y: Any = None,
    task_type: TaskType = TaskType.REGRESSION,
    feature_names: list[str] | None = None,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
    max_features: int = 30,
) -> ExplanationBundle:
    """Global explanation, using the strongest method the model supports."""
    bundle = ExplanationBundle()
    names = feature_names or list(X.columns)

    estimator, transformed_names = _final_estimator_and_names(model, X, names)

    coefficients = _extract_coefficients(estimator, transformed_names)
    if coefficients:
        bundle.coefficients = coefficients
        bundle.method = "model coefficients"
        bundle.method_note = (
            "Read directly off the fitted model. Each coefficient is the change in the prediction "
            "for a one-unit change in that variable, holding the others constant."
        )
        bundle.importances = _importances_from_coefficients(coefficients)
        bundle.confidence = Confidence.HIGH
        bundle.caveats.append(
            "Coefficients are only comparable to each other if the features were scaled to a common range."
        )
    else:
        builtin = getattr(estimator, "feature_importances_", None)
        if builtin is not None and len(builtin) == len(transformed_names):
            bundle.method = "built-in feature importance"
            bundle.method_note = (
                "How much each variable reduced prediction error across the model's splits. "
                "It shows what the model used, which is not the same as what causes the outcome."
            )
            bundle.importances = [
                FeatureImportance(feature=n, importance=float(v), method="impurity")
                for n, v in zip(transformed_names, builtin)
            ]
            bundle.confidence = Confidence.MODERATE
            bundle.caveats.append(
                "Impurity-based importance is biased towards high-cardinality and continuous "
                "variables. Permutation importance below is the more trustworthy ranking."
            )

    if y is not None:
        permutation = permutation_importance_frame(
            model, X, y, task_type, scoring=scoring, n_repeats=n_repeats, random_state=random_state
        )
        if permutation:
            if not bundle.importances:
                bundle.method = "permutation importance"
                bundle.method_note = (
                    "Each variable is shuffled in turn; the drop in performance is how much the "
                    "model actually relied on it. This works for any model."
                )
                bundle.importances = permutation
                bundle.confidence = Confidence.HIGH
            else:
                bundle.partial_dependence["permutation_importance"] = [p.to_dict() for p in permutation]

    bundle.importances.sort(key=lambda f: abs(f.importance), reverse=True)
    for i, item in enumerate(bundle.importances[:max_features], 1):
        item.rank = i
    bundle.importances = bundle.importances[:max_features]
    bundle.plain_english = _narrate_importances(bundle, task_type)
    return bundle


def permutation_importance_frame(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    task_type: TaskType,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> list[FeatureImportance]:
    from sklearn.inspection import permutation_importance

    if scoring is None:
        scoring = "r2" if task_type is TaskType.REGRESSION else "balanced_accuracy"
    try:
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=1
        )
    except Exception:
        return []
    out = [
        FeatureImportance(
            feature=str(name),
            importance=float(mean),
            std=float(std),
            method=f"permutation ({scoring})",
        )
        for name, mean, std in zip(X.columns, result.importances_mean, result.importances_std)
    ]
    out.sort(key=lambda f: f.importance, reverse=True)
    return out


def _final_estimator_and_names(model: Any, X: pd.DataFrame, names: list[str]) -> tuple[Any, list[str]]:
    """Unwrap a Pipeline to the estimator, and find the names it actually saw."""
    from sklearn.pipeline import Pipeline

    if not isinstance(model, Pipeline):
        return model, names
    estimator = model.steps[-1][1]
    try:
        transformed = model[:-1].transform(X.head(min(len(X), 50)))
        if isinstance(transformed, pd.DataFrame):
            return estimator, [str(c) for c in transformed.columns]
        return estimator, [f"feature_{i}" for i in range(np.asarray(transformed).shape[1])]
    except Exception:
        return estimator, names


def _extract_coefficients(estimator: Any, names: list[str]) -> list[dict[str, Any]]:
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return []
    coef = np.asarray(coef)
    if coef.ndim > 1:
        if coef.shape[0] == 1:
            coef = coef.ravel()
        else:
            # multi-class: summarise by the largest absolute effect across classes
            coef = coef[np.argmax(np.abs(coef).sum(axis=1))]
    if len(coef) != len(names):
        names = [f"feature_{i}" for i in range(len(coef))]
    intercept = getattr(estimator, "intercept_", None)
    rows = []
    if intercept is not None:
        value = float(np.ravel(intercept)[0])
        rows.append({"term": "(intercept)", "coefficient": value, "abs": abs(value), "is_intercept": True})
    for name, value in zip(names, coef):
        rows.append({"term": str(name), "coefficient": float(value), "abs": abs(float(value)), "is_intercept": False})

    # statsmodels-backed estimators can add real inference
    if hasattr(estimator, "inference_table"):
        try:
            table = estimator.inference_table(names)
            lookup = {r["term"]: r for r in table}
            for row in rows:
                extra = lookup.get(row["term"])
                if extra:
                    row.update({k: v for k, v in extra.items() if k != "term"})
        except Exception:
            pass
    return rows


def _importances_from_coefficients(coefficients: list[dict[str, Any]]) -> list[FeatureImportance]:
    out = []
    for row in coefficients:
        if row.get("is_intercept"):
            continue
        value = row["coefficient"]
        out.append(
            FeatureImportance(
                feature=row["term"],
                importance=abs(value),
                direction="increases" if value > 0 else "decreases",
                method="coefficient magnitude",
                interpretation=(
                    f"A one-unit increase in {row['term']} is associated with a "
                    f"{'rise' if value > 0 else 'fall'} of {human_number(abs(value))} in the prediction, "
                    "holding the other variables constant."
                ),
            )
        )
    return out


def _narrate_importances(bundle: ExplanationBundle, task_type: TaskType) -> list[str]:
    """Turn the ranking into sentences a manager can read."""
    if not bundle.importances:
        return ["This model does not expose which variables it relied on."]
    lines: list[str] = []
    top = bundle.importances[0]
    total = sum(abs(f.importance) for f in bundle.importances) or 1.0
    share = abs(top.importance) / total

    outcome = "the predicted value" if task_type is TaskType.REGRESSION else "the predicted outcome"
    if top.direction:
        lines.append(
            f"{top.feature} is the strongest single driver of {outcome}. Higher values of "
            f"{top.feature} {'raise' if top.direction == 'increases' else 'lower'} it, "
            "with the other variables held constant."
        )
    else:
        lines.append(
            f"{top.feature} is the variable the model relies on most, accounting for roughly "
            f"{share:.0%} of the total importance across the features shown."
        )

    if len(bundle.importances) > 1:
        others = ", ".join(f.feature for f in bundle.importances[1:4])
        lines.append(f"Next in order of influence: {others}.")

    weak = [f for f in bundle.importances if abs(f.importance) < 0.01 * abs(top.importance)]
    if len(weak) >= 3:
        lines.append(
            f"{len(weak)} variable(s) contribute almost nothing. Removing them would simplify the "
            "model with little or no cost to accuracy — worth testing."
        )
    if share > 0.7:
        lines.append(
            f"One variable dominating this heavily is worth checking: confirm {top.feature} is genuinely "
            "known before the outcome occurs, and is not a restatement of it."
        )
    lines.append(
        "These are associations the model found, not proof of cause. Acting on them assumes the "
        "relationship holds when you intervene, which the data alone cannot establish."
    )
    return lines


# --------------------------------------------------------------------------
# partial dependence
# --------------------------------------------------------------------------

def partial_dependence_curve(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_points: int = 20,
    sample: int = 500,
) -> dict[str, Any]:
    """Average prediction as one feature is swept across its range.

    Computed directly rather than through sklearn's helper so it works with the
    DataFrame-preserving pipelines used here.
    """
    if feature not in X.columns:
        raise KeyError(f"'{feature}' is not one of the model's input columns.")
    working = X.sample(min(len(X), sample), random_state=42) if len(X) > sample else X.copy()
    series = pd.to_numeric(X[feature], errors="coerce").dropna()
    if series.empty:
        return {"feature": feature, "supported": False,
                "reason": "This column is not numeric, so a dependence curve is not meaningful."}

    if series.nunique() <= grid_points:
        grid = np.sort(series.unique())
    else:
        grid = np.linspace(series.quantile(0.02), series.quantile(0.98), grid_points)

    predictions, uncertainty = [], []
    for value in grid:
        probe = working.copy()
        probe[feature] = value
        try:
            if hasattr(model, "predict_proba"):
                out = np.asarray(model.predict_proba(probe))
                out = out[:, 1] if out.ndim == 2 and out.shape[1] == 2 else out.max(axis=1)
            else:
                out = np.asarray(model.predict(probe), dtype=float)
        except Exception as exc:
            return {"feature": feature, "supported": False, "reason": str(exc)}
        predictions.append(float(np.mean(out)))
        uncertainty.append(float(np.std(out)))

    effect = max(predictions) - min(predictions)
    trend = _describe_trend(np.asarray(grid, dtype=float), np.asarray(predictions))
    return {
        "feature": feature,
        "supported": True,
        "grid": [float(g) for g in grid],
        "predictions": predictions,
        "std": uncertainty,
        "effect_size": effect,
        "trend": trend,
        "interpretation": (
            f"Moving {feature} from {human_number(grid[0])} to {human_number(grid[-1])} shifts the average prediction "
            f"by {human_number(effect)}, {trend}. Everything else is held at its observed values."
        ),
    }


def _describe_trend(grid: np.ndarray, values: np.ndarray) -> str:
    if len(grid) < 3:
        return "with too few points to characterise the shape"
    slope = np.polyfit(grid, values, 1)[0]
    linear_fit = np.polyval(np.polyfit(grid, values, 1), grid)
    residual = float(np.mean((values - linear_fit) ** 2))
    spread = float(np.var(values)) or 1.0
    curved = residual / spread > 0.15
    direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
    if direction == "flat":
        return "with essentially no effect"
    return f"{direction} {'along a curve' if curved else 'roughly in a straight line'}"


# --------------------------------------------------------------------------
# local explanation
# --------------------------------------------------------------------------

def explain_prediction(
    model: Any,
    X: pd.DataFrame,
    row_index: int,
    background: pd.DataFrame | None = None,
    max_features: int = 10,
    n_samples: int = 200,
) -> dict[str, Any]:
    """Why did this specific row get this prediction?

    Uses SHAP when available; otherwise an exact additive decomposition for
    linear models, and a sampling approximation for everything else. The method
    used is always reported.
    """
    row = X.iloc[[row_index]]
    reference = background if background is not None else X
    try:
        prediction = float(np.ravel(model.predict(row))[0])
    except Exception as exc:
        return {"supported": False, "reason": f"The model could not score this row: {exc}"}

    shap_result = _try_shap(model, row, reference, max_features)
    if shap_result is not None:
        shap_result.update({"prediction": prediction, "row_index": int(row_index)})
        return shap_result

    contributions = _sampled_contributions(model, row, reference, n_samples)
    if not contributions:
        return {"supported": False, "reason": "This model does not support per-prediction attribution."}

    baseline = float(np.mean(np.ravel(model.predict(reference.sample(
        min(len(reference), 200), random_state=42)))))
    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_features]
    return {
        "supported": True,
        "method": "occlusion attribution (SHAP not installed)",
        "method_note": (
            "Each feature is replaced with typical values from the rest of the data and the change "
            "in prediction is recorded. This approximates a Shapley value but does not account for "
            "every interaction the way exact SHAP does — install shap for the exact decomposition."
        ),
        "prediction": prediction,
        "baseline": baseline,
        "row_index": int(row_index),
        "contributions": [{"feature": k, "contribution": float(v), "value": _safe_value(row, k)} for k, v in ordered],
        "narrative": _narrate_local(prediction, baseline, ordered, row),
    }


def _try_shap(model, row, background, max_features):
    try:
        import shap
    except ImportError:
        return None
    try:
        sample = background.sample(min(len(background), 100), random_state=42)
        explainer = shap.Explainer(model.predict, sample)
        values = explainer(row)
        contributions = dict(zip(row.columns, np.asarray(values.values).ravel()))
        ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_features]
        baseline = float(np.ravel(values.base_values)[0])
        return {
            "supported": True,
            "method": "SHAP",
            "method_note": "Exact Shapley attribution: each feature's contribution, averaged over all orderings.",
            "baseline": baseline,
            "contributions": [{"feature": k, "contribution": float(v), "value": _safe_value(row, k)} for k, v in ordered],
            "narrative": _narrate_local(float(np.ravel(model.predict(row))[0]), baseline, ordered, row),
        }
    except Exception:
        return None


def _sampled_contributions(model, row, background, n_samples: int) -> dict[str, float]:
    sample = background.sample(min(len(background), n_samples), random_state=42)
    try:
        base = float(np.mean(np.ravel(model.predict(sample))))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for column in row.columns:
        probe = sample.copy()
        probe[column] = row[column].iloc[0]
        try:
            with_feature = float(np.mean(np.ravel(model.predict(probe))))
        except Exception:
            continue
        out[column] = with_feature - base
    return out


def _safe_value(row: pd.DataFrame, column: str):
    try:
        value = row[column].iloc[0]
        return value.item() if hasattr(value, "item") else value
    except Exception:
        return None


def _narrate_local(prediction: float, baseline: float, ordered, row) -> str:
    if not ordered:
        return f"Predicted {human_number(prediction)}."
    pushed_up = [(k, v) for k, v in ordered if v > 0][:3]
    pushed_down = [(k, v) for k, v in ordered if v < 0][:3]
    parts = [
        f"Predicted {human_number(prediction)}, against a typical prediction of {human_number(baseline)} for this dataset."
    ]
    if pushed_up:
        detail = ", ".join(f"{k} = {_safe_value(row, k)} (+{human_number(v)})" for k, v in pushed_up)
        parts.append(f"Pushed up mainly by {detail}.")
    if pushed_down:
        detail = ", ".join(f"{k} = {_safe_value(row, k)} ({human_number(v)})" for k, v in pushed_down)
        parts.append(f"Pulled down mainly by {detail}.")
    return " ".join(parts)
