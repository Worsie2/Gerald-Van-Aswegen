"""How much weight this analysis can carry, and exactly what is holding it up.

A single number for "confidence" is a lie of precision: it invites the reader to
treat an analyst's summary as a probability. What a reader actually needs is the
list — what supports the conclusion, what weakens it, and what nothing in the
data can settle either way.

So this produces three things:

* **Evidence strength**, 0-100, explicitly *not* a statistical confidence level.
  It is a weighted count of checks that passed and failed, and it is never shown
  without the list it was computed from.
* **Assumption debt**, 0-100, the unresolved analytical risk — the things that
  would have to be true for the conclusion to hold, that have not been shown.
* **Known unknowns**, the questions this design cannot answer at all. These do
  not move either score, because they are not deficiencies to be fixed; they are
  the boundary of what the method can do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsai.core.schema import TaskType

__all__ = ["TrustAssessment", "assess_trust", "Factor"]


@dataclass
class Factor:
    """One thing that raises or lowers how much weight the result can carry."""

    statement: str
    supporting: bool
    #: How much it moves the score. Kept small and explicit so the arithmetic is
    #: checkable rather than a black box.
    weight: int = 10
    detail: str = ""


@dataclass
class TrustAssessment:
    score: int = 0
    verdict: str = ""
    supporting: list[Factor] = field(default_factory=list)
    reducing: list[Factor] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    assumption_debt: int = 0
    debt_items: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)

    @property
    def band(self) -> str:
        if self.blocking:
            return "not usable"
        if self.score >= 75:
            return "strong"
        if self.score >= 55:
            return "moderate"
        if self.score >= 35:
            return "weak"
        return "very weak"

    @property
    def debt_band(self) -> str:
        if self.assumption_debt >= 60:
            return "high"
        if self.assumption_debt >= 30:
            return "moderate"
        return "low"

    def render(self) -> str:
        lines = [f"Evidence strength {self.score}/100 — {self.band}.", "", self.verdict, ""]
        if self.supporting:
            lines += ["Supporting:", ""] + [f"- {f.statement}" for f in self.supporting] + [""]
        if self.reducing:
            lines += ["Reducing confidence:", ""] + [f"- {f.statement}" for f in self.reducing] + [""]
        if self.unknowns:
            lines += ["What this analysis cannot tell you:", ""] + [f"- {u}" for u in self.unknowns]
        return "\n".join(lines)


#: The self-check questions, and what each is worth to the evidence-strength
#: score. Weighted by how much a failure actually undermines a conclusion: a
#: model that cannot beat doing nothing is worthless, while a note about
#: causality is a permanent property of observational data rather than a fault.
_CHECK_WEIGHTS = {
    "Does the model beat a naive baseline?": 20,
    "Could the model be seeing the answer (leakage)?": 18,
    "Is the model overfitting?": 14,
    "Is the result stable across folds?": 12,
    "Is the dataset large enough to support a conclusion?": 10,
    "Are there enough rows per variable?": 8,
    "Is the target balanced enough to model?": 8,
    "Could a handful of extreme values be driving this?": 6,
    "Are there unresolved data-quality problems?": 6,
    "Was time respected in the validation split?": 8,
    "Is the relationship causal, or only associational?": 0,
}
_DEFAULT_WEIGHT = 6


def assess_trust(run: Any) -> TrustAssessment:
    """Read the completed run and say how much weight it can carry."""
    assessment = TrustAssessment()

    if run.self_check is None:
        assessment.score = 0
        assessment.verdict = "This analysis has not been validated, so nothing can be said about how far to trust it."
        return assessment

    assessment.blocking = list(run.self_check.blocking)

    earned = 0
    available = 0
    for check in run.self_check.checks:
        weight = _CHECK_WEIGHTS.get(check.question, _DEFAULT_WEIGHT)
        if weight == 0:
            continue                       # advisory, not a deficiency
        available += weight
        factor = Factor(statement=_phrase(check), supporting=check.passed,
                        weight=weight, detail=check.detail)
        if check.passed:
            earned += weight
            assessment.supporting.append(factor)
        else:
            assessment.reducing.append(factor)

    assessment.score = int(round(100 * earned / available)) if available else 0

    # Things that are true of the design rather than of this run, and cost
    # confidence regardless of how well the model performed.
    _add_design_factors(run, assessment)

    assessment.reducing.sort(key=lambda f: -f.weight)
    assessment.supporting.sort(key=lambda f: -f.weight)

    assessment.unknowns = _known_unknowns(run)
    assessment.assumption_debt, assessment.debt_items = _assumption_debt(run, assessment)
    assessment.verdict = _verdict(run, assessment)
    return assessment


def _phrase(check: Any) -> str:
    """Turn a yes/no question into a statement of what was found."""
    answers = {
        "Does the model beat a naive baseline?": ("Beats doing nothing", "Does not beat doing nothing"),
        "Could the model be seeing the answer (leakage)?": ("No leakage detected", "A column may be giving away the answer"),
        "Is the model overfitting?": ("Generalises to held-out rows", "Overfits the training rows"),
        "Is the result stable across folds?": ("Stable across folds", "Score swings between folds"),
        "Is the dataset large enough to support a conclusion?": ("Enough rows to conclude something", "Few rows for a firm conclusion"),
        "Are there enough rows per variable?": ("Enough rows per variable", "Too many variables for the rows available"),
        "Is the target balanced enough to model?": ("Target balanced enough to model", "Target is badly imbalanced"),
        "Could a handful of extreme values be driving this?": ("Not driven by a few extreme values", "A few extreme values may be driving this"),
        "Are there unresolved data-quality problems?": ("No unresolved quality problems", "Unresolved data-quality problems"),
        "Was time respected in the validation split?": ("Time respected in validation", "Time may not have been respected in validation"),
    }
    pair = answers.get(check.question)
    if pair:
        return pair[0] if check.passed else pair[1]
    return check.question


def _add_design_factors(run: Any, assessment: TrustAssessment) -> None:
    """Properties of the study design, not of the model."""
    objective = run.objective
    if objective is not None and objective.task_type is not TaskType.TIME_SERIES_FORECAST:
        assessment.reducing.append(Factor(
            "Observational data — associations only, not causes", supporting=False, weight=0,
            detail="Nothing in this design can separate a relationship from a third variable "
                   "driving both, and no amount of data will change that.",
        ))
    if run.profile is not None:
        missing = sum(c.n_missing for c in run.profile.columns.values())
        cells = max(run.profile.n_rows * run.profile.n_columns, 1)
        share = missing / cells
        if share > 0.05:
            assessment.reducing.append(Factor(
                f"{share:.0%} of all cells were missing and had to be filled in",
                supporting=False, weight=0,
                detail="Every imputed value is an estimate the model then treats as data.",
            ))
        elif missing == 0:
            assessment.supporting.append(Factor(
                "No missing values anywhere in the dataset", supporting=True, weight=0))
    if run.best is not None and run.best.validation.n_splits >= 5:
        assessment.supporting.append(Factor(
            f"Cross-validated over {run.best.validation.n_splits} folds", supporting=True, weight=0))


def _known_unknowns(run: Any) -> list[str]:
    """What this analysis cannot answer, whatever its numbers look like.

    Not deficiencies. Stating them is the difference between a result that knows
    its own boundary and one that invites the reader to walk past it.
    """
    target = run.objective.target if run.objective else "the outcome"
    unknowns = [
        f"Whether anything here *causes* {target} to change. This is observational data; an "
        "intervention could produce a different result entirely.",
        "Whether a variable nobody measured explains the relationship. The strongest predictor "
        "in a dataset is sometimes a stand-in for something absent from it.",
        "Whether these rows represent the wider population, or only whoever happened to be in "
        "this file.",
        "Whether the relationship holds into the future. Everything here describes the period "
        "the data covers.",
    ]
    if run.objective and run.objective.task_type is TaskType.CLUSTERING:
        unknowns.append(
            "Whether these segments are real groups or a convenient partition of a continuum. "
            "Clustering always returns clusters, including from noise."
        )
    if run.profile is not None and run.profile.n_rows < 500:
        unknowns.append(
            f"How much of this would survive a second sample. With {run.profile.n_rows:,} rows, "
            "estimates move noticeably between samples."
        )
    return unknowns


def _assumption_debt(run: Any, assessment: TrustAssessment) -> tuple[int, list[str]]:
    """Unresolved analytical risk — what would have to be true, and has not been shown."""
    items: list[str] = []
    debt = 0

    for factor in assessment.reducing:
        if factor.weight:
            items.append(factor.statement)
            debt += factor.weight

    if run.context is not None:
        for assumption in run.context.assumptions or []:
            items.append(f"Your assumption, unverified: {assumption}")
            debt += 6
        if run.context.is_empty:
            items.append("No business context was given, so nothing the platform concluded could "
                         "be checked against what you know")
            debt += 10

    if run.pipeline is not None:
        imputers = [s for s in run.pipeline.active_steps if s.step_key.startswith("impute")]
        if imputers:
            items.append(f"{len(imputers)} imputation step(s) — every filled gap is an estimate "
                         "treated afterwards as data")
            debt += 5

    if run.profile is not None and run.profile.leakage_suspects:
        items.append(f"{len(run.profile.leakage_suspects)} column(s) flagged as possibly giving "
                     "away the answer and not yet ruled out")
        debt += 12

    return min(100, debt), items


def _verdict(run: Any, assessment: TrustAssessment) -> str:
    """One sentence a reader can act on, with the boundary stated in the same breath."""
    if assessment.blocking:
        return ("**Not usable as it stands.** " + " ".join(assessment.blocking)
                + " Fix that before acting on anything here.")
    band = assessment.band
    if band == "strong":
        return ("**Useful evidence, not proof of causation.** The result holds up under the checks "
                "that were run, and the relationships it describes are associations.")
    if band == "moderate":
        return ("**Worth acting on with a check in place.** Enough held up to be useful, with "
                "real caveats listed below. Test on a small group before rolling anything out.")
    if band == "weak":
        return ("**Treat as a lead, not a finding.** Too much did not hold up for this to carry a "
                "decision on its own. Worth investigating, not worth acting on.")
    return ("**Not enough here to conclude anything.** Most of the validation checks raised "
            "something. The honest next step is better data, not a different model.")
