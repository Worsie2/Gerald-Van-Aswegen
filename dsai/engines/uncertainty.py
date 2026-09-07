"""How wrong a single prediction is likely to be, and when not to make one.

A point prediction is a number with the uncertainty deleted. "R126,450" invites
a decision that "R118,000 to R135,000" would not, and the difference between
those two statements is usually the whole question.

Two mechanisms here, and both are deliberately conservative:

**Prediction intervals** by split conformal prediction. The model is scored on
rows it did not train on; the distribution of those errors is the interval. It
assumes only that new rows resemble the calibration rows — no distributional
assumption, no assumption that the model is correct. The cost is that the
interval is the same width everywhere, which is honest rather than clever: a
model with no notion of where it is less certain should not be made to pretend
it has one.

**Abstention.** A model asked about a row unlike anything it trained on returns a
number with the same confidence as any other. Refusing is more useful than
guessing, so a row can be marked *abstained* — and an abstention is never
silently replaced with a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import TaskType

__all__ = ["IntervalModel", "AbstentionRule", "AbstentionResult", "fit_conformal",
           "assess_rows", "RELIABLE", "CAUTION", "ABSTAINED"]

RELIABLE = "reliable"
CAUTION = "caution"
ABSTAINED = "abstained"


@dataclass
class IntervalModel:
    """A conformal interval fitted on held-out residuals."""

    #: The half-width added either side of a prediction.
    half_width: float = 0.0
    coverage: float = 0.9
    n_calibration: int = 0
    method: str = "split conformal"
    usable: bool = False
    note: str = ""

    def interval(self, prediction: float) -> tuple[float, float]:
        return (prediction - self.half_width, prediction + self.half_width)

    def apply(self, predictions: Any) -> pd.DataFrame:
        values = np.asarray(predictions, dtype=float)
        return pd.DataFrame({
            "prediction": values,
            "lower": values - self.half_width,
            "upper": values + self.half_width,
        })

    def explain(self, units: str = "") -> str:
        if not self.usable:
            return self.note
        suffix = f" {units}" if units else ""
        return (
            f"Intervals cover **{self.coverage:.0%}** of outcomes, computed by {self.method} on "
            f"{self.n_calibration:,} held-out rows: the model was scored on rows it had not seen, "
            f"and {self.coverage:.0%} of those errors were within ±{self.half_width:,.4g}{suffix}. "
            "The width is the same for every row — this method knows how wrong the model usually "
            "is, not where it is less sure, and widening some intervals and not others would be "
            "inventing a confidence the model does not have."
        )


def fit_conformal(actual: Any, predicted: Any, coverage: float = 0.9) -> IntervalModel:
    """Fit a split-conformal interval from held-out predictions.

    Uses the finite-sample corrected quantile ⌈(n+1)·coverage⌉/n, which is what
    gives the coverage guarantee rather than merely approximating it.
    """
    model = IntervalModel(coverage=coverage)
    try:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError):
        model.note = "Prediction intervals need numeric outcomes."
        return model

    mask = np.isfinite(actual) & np.isfinite(predicted)
    residuals = np.abs(actual[mask] - predicted[mask])
    n = residuals.size
    model.n_calibration = int(n)

    if n < 20:
        model.note = (
            f"Only {n} held-out row(s) available. Below about 20 the interval would be a guess "
            "about a guess, so none is offered rather than one nobody should rely on."
        )
        return model

    rank = int(np.ceil((n + 1) * coverage))
    if rank > n:
        model.note = (
            f"{coverage:.0%} coverage cannot be guaranteed from {n} calibration rows — the "
            "required quantile falls outside the data. More held-out rows, or a lower coverage "
            "level, would be needed."
        )
        return model

    model.half_width = float(np.sort(residuals)[rank - 1])
    model.usable = True
    return model


@dataclass
class AbstentionRule:
    """One reason a prediction should not be trusted for a particular row."""

    name: str
    reason: str
    severity: str = CAUTION        # caution | abstained


@dataclass
class AbstentionResult:
    status: pd.Series = field(default_factory=pd.Series)
    reasons: pd.Series = field(default_factory=pd.Series)
    counts: dict[str, int] = field(default_factory=dict)
    rules_applied: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def n_abstained(self) -> int:
        return int(self.counts.get(ABSTAINED, 0))

    def summary(self) -> str:
        total = sum(self.counts.values())
        if not total:
            return "No rows were assessed."
        parts = [f"{self.counts.get(RELIABLE, 0):,} reliable"]
        if self.counts.get(CAUTION):
            parts.append(f"{self.counts[CAUTION]:,} to use with caution")
        if self.counts.get(ABSTAINED):
            parts.append(f"{self.counts[ABSTAINED]:,} refused")
        return " · ".join(parts)


def assess_rows(
    new_frame: pd.DataFrame,
    training_frame: pd.DataFrame | None,
    required: list[str],
    contract: Any = None,
    probabilities: Any = None,
    min_confidence: float = 0.55,
) -> AbstentionResult:
    """Decide, per row, whether a prediction on it can be trusted.

    Every rule is a statement about the *row*, not the model's output — a model
    is equally confident about a row it understands and one it has never seen
    anything like, so its own confidence cannot be the only test.
    """
    result = AbstentionResult()
    if new_frame.empty:
        result.note = "No rows to assess."
        return result

    index = new_frame.index
    status = pd.Series(RELIABLE, index=index, dtype=object)
    reasons = pd.Series("", index=index, dtype=object)

    def mark(mask: pd.Series, level: str, reason: str, rule: str) -> None:
        mask = mask.reindex(index, fill_value=False).astype(bool)
        if not mask.any():
            return
        result.rules_applied.append(rule)
        # Abstention wins over caution: a row failing two tests is refused.
        promote = mask & ((status != ABSTAINED) if level == ABSTAINED else (status == RELIABLE))
        status.loc[promote] = level
        reasons.loc[mask] = (reasons.loc[mask].str.rstrip("; ") + "; " + reason).str.lstrip("; ")

    present = [c for c in required if c in new_frame.columns]

    # 1. Nothing to go on.
    if present:
        missing_share = new_frame[present].isna().mean(axis=1)
        mark(missing_share >= 0.5, ABSTAINED,
             "half or more of the columns the model needs are empty on this row",
             "severe missingness")
        mark((missing_share > 0.2) & (missing_share < 0.5), CAUTION,
             "several of the columns the model needs are empty",
             "partial missingness")

    # 2. A category the model has never seen. It has learned nothing about it.
    known: dict[str, set] = {}
    if contract is not None and getattr(contract, "categories", None):
        known = {k: set(v) for k, v in contract.categories.items()}
    elif training_frame is not None:
        for column in present:
            if not pd.api.types.is_numeric_dtype(new_frame[column]):
                known[column] = set(training_frame[column].astype("string").dropna().unique()) \
                    if column in training_frame.columns else set()
    for column, values in known.items():
        if column not in new_frame.columns or not values:
            continue
        arriving = new_frame[column].astype("string")
        unseen = arriving.notna() & ~arriving.isin(values)
        mark(unseen, CAUTION, f"`{column}` holds a value the model never saw in training",
             "unseen category")

    # 3. Outside the range the model was trained on. Extrapolation, not prediction.
    if training_frame is not None:
        for column in present:
            if not pd.api.types.is_numeric_dtype(new_frame[column]):
                continue
            if column not in training_frame.columns:
                continue
            trained = training_frame[column].dropna()
            if trained.empty:
                continue
            low, high = trained.quantile(0.001), trained.quantile(0.999)
            span = high - low
            if span <= 0:
                continue
            values = pd.to_numeric(new_frame[column], errors="coerce")
            far = (values < low - span) | (values > high + span)
            near = ((values < low) | (values > high)) & ~far
            mark(far, ABSTAINED,
                 f"`{column}` is far outside anything in the training data",
                 "outside training support")
            mark(near, CAUTION, f"`{column}` sits beyond the training range",
                 "edge of training range")

    # 4. The model's own uncertainty, where it has one worth reading.
    if probabilities is not None:
        try:
            confidence = pd.Series(np.max(np.asarray(probabilities, dtype=float), axis=1),
                                   index=index)
            mark(confidence < min_confidence, CAUTION,
                 f"the model's own confidence is below {min_confidence:.0%}",
                 "low model confidence")
        except Exception:
            pass

    result.status = status
    result.reasons = reasons
    result.counts = {level: int((status == level).sum())
                     for level in (RELIABLE, CAUTION, ABSTAINED)}
    result.rules_applied = sorted(set(result.rules_applied))
    if not result.rules_applied:
        result.note = ("Every row resembles the training data on every test applied. That is not "
                       "a guarantee the predictions are right — only that nothing about these "
                       "rows makes them obviously unanswerable.")
    return result
