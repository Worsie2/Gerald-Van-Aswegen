"""Metric computation, with the metric set adapting to the problem type."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import TaskType

#: Metrics where a *lower* value is better.
LOWER_IS_BETTER = {
    "rmse", "mae", "mape", "smape", "mase", "log_loss", "brier",
    "davies_bouldin", "reconstruction_error", "aic", "bic", "stress",
    "poisson_deviance", "gamma_deviance", "max_error", "median_absolute_error",
}

METRIC_LABELS = {
    "r2": "R²",
    "rmse": "RMSE",
    "mae": "MAE",
    "mape": "MAPE %",
    "smape": "sMAPE %",
    "mase": "MASE",
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC",
    "f1": "F1",
    "f1_macro": "F1 (macro)",
    "mcc": "Matthews corr.",
    "balanced_accuracy": "Balanced accuracy",
    "silhouette": "Silhouette",
    "calinski_harabasz": "Calinski-Harabasz",
    "davies_bouldin": "Davies-Bouldin",
}

METRIC_EXPLANATIONS = {
    "r2": "Share of the variation in the target that the model explains. 0 means no better than predicting the average; negative means worse than that.",
    "rmse": "Typical prediction error, in the target's own units, with big misses weighted heavily.",
    "mae": "Average size of the prediction error, in the target's own units. Easier to explain than RMSE and less swayed by outliers.",
    "mape": "Average error as a percentage of the actual value. Blows up when actual values are near zero.",
    "smape": "Symmetric percentage error — better behaved than MAPE around zero.",
    "mase": "Error relative to a naive forecast. Below 1 means the model beats simply repeating the last value; above 1 means it does not.",
    "accuracy": "Share of predictions that were correct. Misleading when the classes are imbalanced.",
    "balanced_accuracy": "Average recall across the classes, so a rare class counts as much as a common one.",
    "precision": "Of the cases flagged positive, the share that really were. Matters when acting on a false positive is costly.",
    "recall": "Of the real positives, the share the model caught. Matters when missing a positive is costly.",
    "f1": "Harmonic mean of precision and recall — a single number when both errors matter.",
    "roc_auc": "Probability the model ranks a random positive above a random negative. 0.5 is a coin flip.",
    "pr_auc": "Area under the precision-recall curve. The honest headline metric for rare positives.",
    "log_loss": "Penalises confident wrong probabilities. Lower is better; it rewards calibration, not just ranking.",
    "mcc": "Correlation between predictions and truth, robust to imbalance. Ranges from -1 to 1.",
    "silhouette": "How much closer each point sits to its own cluster than the next nearest, from -1 to 1. Above about 0.5 is well-separated.",
    "calinski_harabasz": "Ratio of between-cluster to within-cluster spread. Higher is better; only comparable within one dataset.",
    "davies_bouldin": "Average similarity between each cluster and its closest neighbour. Lower is better.",
}


def is_better(metric: str, a: float, b: float) -> bool:
    """True when score `a` is better than score `b` for this metric."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return False
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return True
    return a < b if metric in LOWER_IS_BETTER else a > b


def sklearn_scoring_name(metric: str) -> str:
    """Map our metric names onto scikit-learn scorer strings."""
    return {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "mape": "neg_mean_absolute_percentage_error",
        "r2": "r2",
        "explained_variance": "explained_variance",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "log_loss": "neg_log_loss",
        "mcc": "matthews_corrcoef",
    }.get(metric, metric)


# --------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------

def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error, median_absolute_error, r2_score,
    )

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {}

    out = {
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "max_error": float(np.max(np.abs(y_true - y_pred))),
    }
    non_zero = np.abs(y_true) > 1e-9
    if non_zero.any():
        out["mape"] = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)
    denominator = np.abs(y_true) + np.abs(y_pred)
    valid = denominator > 1e-9
    if valid.any():
        out["smape"] = float(np.mean(2 * np.abs(y_pred[valid] - y_true[valid]) / denominator[valid]) * 100)
    variance = float(np.var(y_true))
    out["explained_variance"] = float(1 - np.var(y_true - y_pred) / variance) if variance > 0 else float("nan")
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    labels: list[Any] | None = None,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
        f1_score, log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = labels if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()), key=str)
    binary = len(classes) == 2

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    average = "binary" if binary else "macro"
    kwargs: dict[str, Any] = {"zero_division": 0}
    if binary:
        kwargs["pos_label"] = classes[-1]
    out["precision"] = float(precision_score(y_true, y_pred, average=average, **kwargs))
    out["recall"] = float(recall_score(y_true, y_pred, average=average, **kwargs))
    out["f1"] = float(f1_score(y_true, y_pred, average=average, **kwargs))
    if not binary:
        out["f1_macro"] = out["f1"]
        out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    if y_proba is not None:
        proba = np.asarray(y_proba)
        try:
            if binary:
                positive = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
                out["roc_auc"] = float(roc_auc_score((y_true == classes[-1]).astype(int), positive))
                out["pr_auc"] = float(average_precision_score((y_true == classes[-1]).astype(int), positive))
                out["brier"] = float(np.mean(((y_true == classes[-1]).astype(float) - positive) ** 2))
            elif proba.ndim == 2 and proba.shape[1] == len(classes):
                out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
            if proba.ndim == 2:
                out["log_loss"] = float(log_loss(y_true, proba, labels=list(classes)))
        except (ValueError, IndexError):
            pass  # a fold with a single class present, or shape mismatch — skip silently
    return out


def confusion_details(y_true: Any, y_pred: Any, labels: list[Any] | None = None) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix

    classes = labels if labels is not None else sorted(set(np.asarray(y_true).tolist()), key=str)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    out: dict[str, Any] = {
        "labels": [str(c) for c in classes],
        "matrix": matrix.tolist(),
    }
    if len(classes) == 2:
        tn, fp, fn, tp = matrix.ravel()
        out.update(
            {
                "true_negative": int(tn), "false_positive": int(fp),
                "false_negative": int(fn), "true_positive": int(tp),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
            }
        )
    return out


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def clustering_metrics(X: Any, labels: Any) -> dict[str, float]:
    from sklearn.metrics import (
        calinski_harabasz_score, davies_bouldin_score, silhouette_score,
    )

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    core = labels != -1  # DBSCAN-style noise points are excluded from the scores
    unique = np.unique(labels[core])
    out: dict[str, float] = {
        "n_clusters": int(len(unique)),
        "n_noise": int((~core).sum()),
        "noise_share": float((~core).mean()),
    }
    if len(unique) < 2 or core.sum() < 3:
        out.update({"silhouette": float("nan"), "calinski_harabasz": float("nan"),
                    "davies_bouldin": float("nan")})
        return out
    try:
        out["silhouette"] = float(silhouette_score(X[core], labels[core]))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X[core], labels[core]))
        out["davies_bouldin"] = float(davies_bouldin_score(X[core], labels[core]))
    except ValueError:
        pass
    sizes = pd.Series(labels[core]).value_counts()
    out["smallest_cluster"] = int(sizes.min())
    out["largest_cluster"] = int(sizes.max())
    out["size_imbalance_ratio"] = float(sizes.max() / max(sizes.min(), 1))
    return out


# --------------------------------------------------------------------------
# forecasting
# --------------------------------------------------------------------------

def forecast_metrics(
    y_true: Any,
    y_pred: Any,
    y_train: Any = None,
    seasonal_period: int = 1,
) -> dict[str, float]:
    """Forecast accuracy, including MASE against a naive in-sample benchmark."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    length = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:length], y_pred[:length]

    out = regression_metrics(y_true, y_pred)
    if y_train is not None and len(np.asarray(y_train)) > seasonal_period:
        train = np.asarray(y_train, dtype=float).ravel()
        naive_error = np.mean(np.abs(train[seasonal_period:] - train[:-seasonal_period]))
        if naive_error > 1e-12:
            out["mase"] = float(np.mean(np.abs(y_true - y_pred)) / naive_error)
    direction_true = np.sign(np.diff(y_true))
    direction_pred = np.sign(np.diff(y_pred))
    if direction_true.size:
        out["directional_accuracy"] = float(np.mean(direction_true == direction_pred))
    return out


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def compute_metrics(task_type: TaskType, **kwargs: Any) -> dict[str, float]:
    if task_type is TaskType.REGRESSION:
        return regression_metrics(kwargs["y_true"], kwargs["y_pred"])
    if task_type.is_classification:
        return classification_metrics(
            kwargs["y_true"], kwargs["y_pred"], kwargs.get("y_proba"), kwargs.get("labels")
        )
    if task_type is TaskType.CLUSTERING:
        return clustering_metrics(kwargs["X"], kwargs["labels"])
    if task_type is TaskType.TIME_SERIES_FORECAST:
        return forecast_metrics(
            kwargs["y_true"], kwargs["y_pred"], kwargs.get("y_train"), kwargs.get("seasonal_period", 1)
        )
    return {}


def default_metrics(task_type: TaskType) -> list[str]:
    """The columns the model tournament shows for each problem type."""
    return {
        TaskType.REGRESSION: ["r2", "rmse", "mae", "mape"],
        TaskType.BINARY_CLASSIFICATION: ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "mcc"],
        TaskType.MULTICLASS_CLASSIFICATION: ["balanced_accuracy", "f1_macro", "accuracy", "mcc"],
        TaskType.TIME_SERIES_FORECAST: ["mase", "rmse", "mae", "mape", "smape"],
        TaskType.CLUSTERING: ["silhouette", "calinski_harabasz", "davies_bouldin", "n_clusters"],
        TaskType.ANOMALY_DETECTION: ["n_anomalies", "anomaly_rate"],
    }.get(task_type, [])


def explain_metric(metric: str) -> str:
    return METRIC_EXPLANATIONS.get(metric, f"'{metric}' is a model evaluation measure.")


def format_metric(metric: str, value: float | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if metric in {"mape", "smape"}:
        return f"{value:,.1f}%"
    if metric in {"n_clusters", "n_noise", "n_anomalies", "smallest_cluster", "largest_cluster"}:
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
