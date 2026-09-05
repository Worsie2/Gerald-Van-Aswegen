"""Model diagnostics: does the fitted model actually satisfy its own assumptions?

Reporting R² without looking at the residuals is how a badly-specified model
gets shipped. These checks are run automatically for every regression and
classification result the platform produces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import JsonMixin


def regression_diagnostics(y_true: Any, y_pred: Any, n_features: int = 1) -> dict[str, Any]:
    """Residual analysis: independence, constant variance, normality, influence."""
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    residuals = y_true - y_pred
    n = len(residuals)
    if n < 8:
        return {"usable": False, "reason": f"Only {n} residuals — too few to diagnose."}

    std = float(np.std(residuals, ddof=1))
    standardised = residuals / std if std > 0 else residuals
    out: dict[str, Any] = {
        "usable": True,
        "n": n,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": std,
        "residual_skew": float(pd.Series(residuals).skew()),
        "residual_kurtosis": float(pd.Series(residuals).kurt()),
        "issues": [],
        "checks": {},
    }

    # Systematic bias: residuals should average out to zero.
    scale = max(float(np.std(y_true, ddof=1)), 1e-12)
    if abs(out["residual_mean"]) > 0.1 * scale:
        out["issues"].append(
            f"Residuals average {out['residual_mean']:,.4g} rather than zero — the model is "
            "systematically biased, consistently over- or under-predicting."
        )

    # Normality of residuals (matters for prediction intervals, not for point accuracy).
    if 8 <= n <= 5000:
        statistic, p_value = stats.shapiro(residuals)
        out["checks"]["normality"] = {"test": "Shapiro-Wilk", "statistic": float(statistic), "p_value": float(p_value)}
        if p_value < 0.05:
            out["issues"].append(
                "Residuals are not normally distributed. Point predictions are still valid, but "
                "confidence and prediction intervals built on a normal assumption will be wrong."
            )
    else:
        statistic, p_value = stats.normaltest(residuals)
        out["checks"]["normality"] = {"test": "D'Agostino-Pearson", "statistic": float(statistic), "p_value": float(p_value)}

    # Heteroscedasticity: correlation between |residual| and fitted value.
    if np.std(y_pred) > 0:
        correlation, p_value = stats.spearmanr(np.abs(residuals), y_pred)
        out["checks"]["heteroscedasticity"] = {
            "test": "Spearman correlation of |residual| with fitted value",
            "correlation": float(correlation), "p_value": float(p_value),
        }
        if p_value < 0.05 and abs(correlation) > 0.2:
            direction = "grows" if correlation > 0 else "shrinks"
            out["issues"].append(
                f"Error size {direction} with the predicted value (ρ = {correlation:.2f}). The model "
                "is more reliable at one end of the range than the other — say so when reporting."
            )

    # Autocorrelation (Durbin-Watson) — only meaningful if the rows have an order.
    if n > 10:
        dw = float(np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2))
        out["checks"]["durbin_watson"] = {
            "statistic": dw,
            "reading": (
                "positive autocorrelation" if dw < 1.5 else
                "negative autocorrelation" if dw > 2.5 else "no autocorrelation"
            ),
        }
        if dw < 1.5 or dw > 2.5:
            out["issues"].append(
                f"Durbin-Watson = {dw:.2f}: consecutive residuals are correlated. If the rows have a "
                "time or spatial order, the standard errors are understated and the model is missing structure."
            )

    influential = int(np.sum(np.abs(standardised) > 3))
    out["n_large_residuals"] = influential
    out["large_residual_share"] = round(influential / n, 4)
    if influential:
        out["issues"].append(
            f"{influential} observation(s) sit more than 3 standard deviations from the fit. "
            "These are the rows the model understands least — look at them individually."
        )
        out["largest_errors"] = _largest_errors(y_true, y_pred, residuals, top_n=10)

    out["interpretation"] = (
        "Residual checks passed: no systematic bias, no obvious pattern in the errors."
        if not out["issues"] else
        f"{len(out['issues'])} residual issue(s) found — see below. These do not necessarily "
        "invalidate the predictions, but they do limit what can be claimed from them."
    )
    return out


def _largest_errors(y_true, y_pred, residuals, top_n: int = 10) -> list[dict[str, float]]:
    order = np.argsort(np.abs(residuals))[::-1][:top_n]
    return [
        {
            "position": int(i),
            "actual": float(y_true[i]),
            "predicted": float(y_pred[i]),
            "residual": float(residuals[i]),
        }
        for i in order
    ]


def classification_diagnostics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Confusion structure, per-class performance, calibration and error patterns."""
    from dsai.engines.metrics import confusion_details

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = labels or sorted(set(y_true.tolist()), key=str)
    out: dict[str, Any] = {
        "usable": True,
        "n": len(y_true),
        "confusion": confusion_details(y_true, y_pred, classes),
        "issues": [],
        "per_class": [],
    }

    for label in classes:
        actual = y_true == label
        predicted = y_pred == label
        support = int(actual.sum())
        tp = int((actual & predicted).sum())
        precision = tp / max(int(predicted.sum()), 1)
        recall = tp / max(support, 1)
        out["per_class"].append(
            {
                "class": str(label),
                "support": support,
                "predicted_count": int(predicted.sum()),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
            }
        )

    for row in out["per_class"]:
        if row["support"] > 0 and row["recall"] < 0.3:
            out["issues"].append(
                f"Class '{row['class']}' is caught only {row['recall']:.0%} of the time "
                f"({row['support']} cases in the data). The model is effectively ignoring it."
            )
        if row["predicted_count"] == 0 and row["support"] > 0:
            out["issues"].append(
                f"Class '{row['class']}' is never predicted at all, despite appearing "
                f"{row['support']} times. Try class weighting or resampling."
            )

    if y_proba is not None:
        calibration = calibration_check(y_true, y_proba, classes)
        out["calibration"] = calibration
        if calibration.get("poorly_calibrated"):
            out["issues"].append(
                "Predicted probabilities are poorly calibrated: when the model says 70%, the real "
                "rate is materially different. Ranking is still usable; the probabilities are not."
            )

    out["interpretation"] = (
        "No structural problems found in the classification errors."
        if not out["issues"] else f"{len(out['issues'])} issue(s) found in how errors are distributed."
    )
    return out


def calibration_check(y_true: Any, y_proba: Any, classes: list[Any], n_bins: int = 10) -> dict[str, Any]:
    """Do predicted probabilities match observed frequencies?"""
    proba = np.asarray(y_proba)
    if proba.ndim != 2 or proba.shape[1] != 2 or len(classes) != 2:
        return {"supported": False, "reason": "Calibration is checked for binary problems only."}

    positive = proba[:, 1]
    actual = (np.asarray(y_true) == classes[-1]).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows, gaps = [], []
    for i in range(n_bins):
        mask = (positive >= bins[i]) & (positive < bins[i + 1] if i < n_bins - 1 else positive <= bins[i + 1])
        if mask.sum() < 5:
            continue
        predicted_rate = float(positive[mask].mean())
        observed_rate = float(actual[mask].mean())
        rows.append(
            {
                "bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": round(predicted_rate, 4),
                "observed_rate": round(observed_rate, 4),
                "gap": round(observed_rate - predicted_rate, 4),
            }
        )
        gaps.append(abs(observed_rate - predicted_rate) * mask.sum())

    total = sum(r["n"] for r in rows) or 1
    expected_calibration_error = sum(gaps) / total
    return {
        "supported": True,
        "bins": rows,
        "expected_calibration_error": round(float(expected_calibration_error), 4),
        "poorly_calibrated": bool(expected_calibration_error > 0.1),
        "interpretation": (
            f"Average gap between predicted probability and observed rate: "
            f"{expected_calibration_error:.1%}. "
            + ("Probabilities can be read at face value." if expected_calibration_error <= 0.1
               else "Treat these as scores for ranking, not as true probabilities.")
        ),
    }


def learning_curve_data(
    model: Any,
    X: pd.DataFrame,
    y: Any,
    scoring: str = "r2",
    cv: int = 3,
    train_sizes: list[float] | None = None,
) -> dict[str, Any]:
    """Does more data help, or is the model already at its ceiling?

    The answer decides whether the next step is "collect more data" or
    "change the model / features" — a genuinely useful thing to know.
    """
    from sklearn.model_selection import learning_curve

    sizes = train_sizes or [0.2, 0.4, 0.6, 0.8, 1.0]
    try:
        absolute, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=sizes, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan
        )
    except Exception as exc:
        return {"supported": False, "reason": str(exc)}

    train_mean = np.nanmean(train_scores, axis=1)
    test_mean = np.nanmean(test_scores, axis=1)
    gap = float(train_mean[-1] - test_mean[-1])
    improvement = float(test_mean[-1] - test_mean[0])
    recent_gain = float(test_mean[-1] - test_mean[-2]) if len(test_mean) > 1 else 0.0

    if recent_gain > 0.02:
        verdict = (
            "Still improving with more data. Collecting more rows is likely the highest-value next step."
        )
    elif gap > 0.15:
        verdict = (
            "Held-out performance has plateaued while training performance stays high — the model is "
            "overfitting. Simplify it or regularise it rather than gathering more data."
        )
    else:
        verdict = (
            "Performance has plateaued and train and test agree. More of the same data will not help; "
            "better features or a different framing would."
        )

    return {
        "supported": True,
        "train_sizes": [int(s) for s in absolute],
        "train_scores": [float(v) for v in train_mean],
        "test_scores": [float(v) for v in test_mean],
        "final_gap": round(gap, 4),
        "total_improvement": round(improvement, 4),
        "verdict": verdict,
        "scoring": scoring,
    }
