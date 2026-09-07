"""Where to draw the line, given what each kind of mistake actually costs.

A classifier produces a probability; a decision needs a cut-off. The default of
0.5 is not a considered choice — it is what you get when nobody states what the
two mistakes cost. Where a false negative costs twenty times a false positive,
0.5 is simply wrong, and no amount of model accuracy fixes it.

So the threshold is chosen from stated costs, and the answer is explicitly
conditional on them: change the costs and the recommended threshold changes.
Nothing here is universally optimal, and the wording never says it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["CostModel", "ThresholdPoint", "ThresholdAnalysis", "analyse_thresholds"]


@dataclass
class CostModel:
    """What each outcome is worth, in the user's own currency.

    Costs are positive numbers meaning "this much worse than nothing". A benefit
    is a negative cost, which is how a correctly caught positive earns its keep.
    """

    false_positive: float = 1.0
    false_negative: float = 1.0
    #: What acting on a positive costs regardless of whether it was right.
    intervention: float = 0.0
    #: What a correctly caught positive is worth, as a negative cost.
    true_positive_benefit: float = 0.0
    currency: str = "ZAR"

    @property
    def ratio(self) -> float:
        return (self.false_negative / self.false_positive) if self.false_positive else float("inf")

    def describe(self) -> str:
        if self.false_positive == self.false_negative:
            return ("Both kinds of mistake cost the same, which is the assumption behind a 0.5 "
                    "threshold. If that is not true here, say what each actually costs.")
        worse = "missing a positive" if self.ratio > 1 else "a false alarm"
        return (f"{worse.capitalize()} is stated as **{max(self.ratio, 1 / self.ratio):.1f}×** "
                "more costly than the other. That ratio, not the model, is what moves the "
                "threshold.")


@dataclass
class ThresholdPoint:
    threshold: float
    predicted_positive: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    total_cost: float
    precision: float
    recall: float
    f1: float

    @property
    def caught_share(self) -> float:
        actual_positive = self.true_positive + self.false_negative
        return self.true_positive / actual_positive if actual_positive else 0.0


@dataclass
class ThresholdAnalysis:
    points: list[ThresholdPoint] = field(default_factory=list)
    recommended: ThresholdPoint | None = None
    default: ThresholdPoint | None = None
    costs: CostModel = field(default_factory=CostModel)
    n: int = 0
    positive_label: str = ""
    note: str = ""
    usable: bool = False

    def table(self) -> pd.DataFrame:
        rows = []
        for point in self.points:
            rows.append({
                "Threshold": f"{point.threshold:.2f}",
                "Flagged": f"{point.predicted_positive:,}",
                "Caught": f"{point.true_positive:,}",
                "Missed": f"{point.false_negative:,}",
                "False alarms": f"{point.false_positive:,}",
                "Precision": f"{point.precision:.1%}",
                "Recall": f"{point.recall:.1%}",
                f"Expected cost ({self.costs.currency})": f"{point.total_cost:,.0f}",
            })
        return pd.DataFrame(rows)

    def verdict(self) -> str:
        if not self.usable or self.recommended is None or self.default is None:
            return self.note
        saving = self.default.total_cost - self.recommended.total_cost
        currency = self.costs.currency
        lines = [
            f"**At the costs you stated, the cheapest threshold is "
            f"{self.recommended.threshold:.2f}**, not the default 0.5.",
            "",
            f"It flags {self.recommended.predicted_positive:,} of {self.n:,} rows, catches "
            f"{self.recommended.caught_share:.0%} of actual positives, and raises "
            f"{self.recommended.false_positive:,} false alarm(s) — an expected cost of "
            f"{currency} {self.recommended.total_cost:,.0f} against "
            f"{currency} {self.default.total_cost:,.0f} at 0.5"
            + (f", a saving of {currency} {saving:,.0f}." if saving > 0
               else ". The default is already the cheapest here."),
            "",
            "**This depends entirely on the costs you supplied.** Change them and the "
            "recommendation changes. It is not a universally optimal threshold, and it is not a "
            "statement about the model — the model is the same at every threshold.",
        ]
        return "\n".join(lines)


def analyse_thresholds(
    y_true: Any,
    scores: Any,
    positive_label: Any = 1,
    costs: CostModel | None = None,
    steps: int = 19,
) -> ThresholdAnalysis:
    """Sweep the threshold and cost every outcome at each one."""
    costs = costs or CostModel()
    analysis = ThresholdAnalysis(costs=costs, positive_label=str(positive_label))

    try:
        scores = np.asarray(scores, dtype=float).ravel()
        actual = np.asarray([1 if str(v) == str(positive_label) else 0 for v in y_true])
    except (TypeError, ValueError):
        analysis.note = "Threshold analysis needs a probability or score per row."
        return analysis

    mask = np.isfinite(scores)
    scores, actual = scores[mask], actual[mask]
    analysis.n = int(scores.size)

    if analysis.n < 30:
        analysis.note = (f"Only {analysis.n} row(s) with a score. Too few to choose a threshold "
                         "from — the counts in each cell would be noise.")
        return analysis
    if actual.sum() == 0 or actual.sum() == analysis.n:
        analysis.note = ("Every row has the same outcome, so no threshold separates anything.")
        return analysis

    for threshold in np.linspace(0.05, 0.95, steps):
        analysis.points.append(_score_at(actual, scores, float(threshold), costs))

    analysis.recommended = min(analysis.points, key=lambda p: p.total_cost)
    analysis.default = min(analysis.points, key=lambda p: abs(p.threshold - 0.5))
    analysis.usable = True
    return analysis


def _score_at(actual: np.ndarray, scores: np.ndarray, threshold: float,
              costs: CostModel) -> ThresholdPoint:
    predicted = (scores >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (actual == 1)))
    fp = int(np.sum((predicted == 1) & (actual == 0)))
    tn = int(np.sum((predicted == 0) & (actual == 0)))
    fn = int(np.sum((predicted == 0) & (actual == 1)))

    total = (
        fp * costs.false_positive
        + fn * costs.false_negative
        + (tp + fp) * costs.intervention
        - tp * costs.true_positive_benefit
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ThresholdPoint(
        threshold=threshold, predicted_positive=tp + fp,
        true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
        total_cost=float(total), precision=precision, recall=recall, f1=f1,
    )
