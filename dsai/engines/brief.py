"""The plan, in plain English, with a verdict on whether it is worth running.

Expensive work should not start before someone has read what it will do. This
turns the analysis plan into something a non-technical reader can approve or
reject: the question, the data, the method, the constraints, and a straight
answer to "can this data defensibly answer this question at all?".

The verdict is deliberately three-valued. "Suitable" and "cannot answer" are
easy; the useful case is the middle one, where the analysis will produce numbers
that look fine and rest on something the reader should know about first.

The **data contract** is the machine-readable half of the same idea: the
requirements this analysis has of its data, recorded with the run so the same
requirements can be checked again when new data arrives to be scored. A model is
only valid on data that meets the contract it was trained under.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dsai.core.schema import JsonMixin, TaskType

__all__ = ["AnalysisBrief", "DataContract", "ContractCheck", "build_brief", "build_contract",
           "check_contract"]

SUITABLE = "suitable"
CAUTION = "caution"
UNSUITABLE = "unsuitable"


@dataclass
class AnalysisBrief(JsonMixin):
    """What is about to run, written for someone who has to approve it."""

    question: str = ""
    dataset: str = ""
    n_rows: int = 0
    n_columns: int = 0
    time_period: str = ""
    quality_score: float = 0.0
    task: str = ""
    target: str | None = None
    predictors: list[str] = field(default_factory=list)
    validation: str = ""
    folds: int = 0
    primary_metric: str = ""
    constraints: list[str] = field(default_factory=list)
    verdict: str = SUITABLE
    verdict_reasons: list[str] = field(default_factory=list)
    n_models: int = 0

    @property
    def verdict_label(self) -> str:
        return {
            SUITABLE: "Suitable",
            CAUTION: "Proceed with caution",
            UNSUITABLE: "Cannot defensibly answer this",
        }[self.verdict]

    def render(self) -> str:
        lines = [
            f"**Question.** {self.question}", "",
            f"**Dataset.** {self.dataset} — {self.n_rows:,} rows, {self.n_columns} columns"
            + (f", {self.time_period}" if self.time_period else "")
            + f". Data quality {self.quality_score:.0f}/100.", "",
            f"**Proposed task.** {self.task}"
            + (f", predicting `{self.target}`" if self.target else ""), "",
            f"**Predictors.** {len(self.predictors)} column(s).", "",
            f"**Validation.** {self.validation}"
            + (f", {self.folds} folds" if self.folds else "")
            + f", ranked on {self.primary_metric}.", "",
        ]
        if self.constraints:
            lines += ["**Constraints.**", ""] + [f"- {c}" for c in self.constraints] + [""]
        lines += [f"**Assessment.** {self.verdict_label}.", ""]
        lines += [f"- {r}" for r in self.verdict_reasons]
        return "\n".join(lines)


def _plain_question(objective: Any, target: str | None) -> str:
    """The objective as a sentence, not a task-type enum."""
    task = objective.task_type
    if task is TaskType.REGRESSION:
        return f"What predicts {target}, and how much of it can be explained?"
    if task is TaskType.BINARY_CLASSIFICATION:
        return f"Which rows are likely to be {target}, and what distinguishes them?"
    if task is TaskType.MULTICLASS_CLASSIFICATION:
        return f"Which category of {target} does each row belong to, and why?"
    if task is TaskType.CLUSTERING:
        return "Are there natural groups in this data, and what makes each one distinct?"
    if task is TaskType.TIME_SERIES_FORECAST:
        return f"What is {target} likely to do next, and how confident can that be?"
    if task is TaskType.ANOMALY_DETECTION:
        return "Which rows do not look like the rest, and what makes them unusual?"
    if task is TaskType.ASSOCIATION_RULES:
        return "What tends to occur together?"
    if task is TaskType.HYPOTHESIS_TESTING:
        return "Is the difference between these groups real, or could it be chance?"
    return "What does this data contain, and what is worth looking at?"


def build_brief(profile: Any, objective: Any, plan: Any, context: Any = None,
                frame: pd.DataFrame | None = None) -> AnalysisBrief:
    """Assemble the brief from the plan that is about to run."""
    target = objective.target
    brief = AnalysisBrief(
        question=_plain_question(objective, target or "the outcome"),
        dataset=profile.name if getattr(profile, "name", "") else "this dataset",
        n_rows=profile.n_rows,
        n_columns=profile.n_columns,
        quality_score=profile.quality_score,
        task=objective.task_type.value.replace("_", " "),
        target=target,
        predictors=[c for c in profile.columns if c != target],
        validation=str(plan.validation_strategy.get("strategy", "")).replace("_", " "),
        folds=int(plan.validation_strategy.get("n_splits") or 0),
        primary_metric=plan.primary_metric,
        n_models=len(plan.candidates),
    )

    if frame is not None and profile.datetime_columns:
        column = profile.datetime_columns[0]
        try:
            series = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not series.empty:
                brief.time_period = (f"{series.min():%b %Y} to {series.max():%b %Y}"
                                     f" (by {column})")
        except Exception:
            pass

    brief.constraints = _constraints(profile, objective, context)
    brief.verdict, brief.verdict_reasons = _verdict(profile, objective, plan)
    return brief


def _constraints(profile: Any, objective: Any, context: Any) -> list[str]:
    out: list[str] = []
    if objective.task_type is not TaskType.TIME_SERIES_FORECAST:
        out.append("Observational data — whatever is found will be an association, not a cause.")
    missing = sum(c.n_missing for c in profile.columns.values())
    if missing:
        cells = max(profile.n_rows * profile.n_columns, 1)
        out.append(f"{missing:,} missing values ({missing / cells:.1%} of all cells) will be "
                   "filled in by the pipeline.")
    if profile.leakage_suspects:
        out.append(f"{len(profile.leakage_suspects)} column(s) look like they may give away the "
                   "answer and are worth ruling out first.")
    if context is not None:
        for assumption in (context.assumptions or [])[:4]:
            out.append(f"Your assumption, not checked: {assumption}")
        for limitation in (context.limitations or [])[:4]:
            out.append(f"You flagged: {limitation}")
        if context.exclude_variables:
            out.append(f"{len(context.exclude_variables)} variable(s) excluded at your request.")
    return out


def _verdict(profile: Any, objective: Any, plan: Any) -> tuple[str, list[str]]:
    """Can this data defensibly answer this question?"""
    reasons: list[str] = []
    verdict = SUITABLE

    n = profile.n_rows
    supervised = objective.task_type.is_supervised
    floor = 50 if supervised else 30

    if n < floor:
        verdict = UNSUITABLE
        reasons.append(f"Only {n:,} rows. Below about {floor}, an estimate says more about which "
                       "rows happened to be sampled than about the world.")
    elif n < floor * 4:
        verdict = CAUTION
        reasons.append(f"{n:,} rows is workable but thin. Expect results to move noticeably on a "
                       "second sample.")
    else:
        reasons.append(f"{n:,} rows is enough to support a conclusion.")

    if supervised and not objective.target:
        verdict = UNSUITABLE
        reasons.append("No target variable is set, and this kind of analysis cannot run without one.")
    elif supervised and objective.target in profile.columns:
        column = profile.columns[objective.target]
        if column.missing_pct > 30:
            verdict = UNSUITABLE if column.missing_pct > 60 else CAUTION
            reasons.append(f"The target `{objective.target}` is {column.missing_pct:.0f}% missing. "
                           "Those rows cannot be used, so the analysis runs on the rest — which "
                           "may not resemble the whole.")
        elif column.n_unique <= 1:
            verdict = UNSUITABLE
            reasons.append(f"`{objective.target}` holds the same value in every row, so there is "
                           "nothing to predict.")

    predictors = max(profile.n_columns - 1, 1)
    if supervised and n / predictors < 10:
        verdict = CAUTION if verdict == SUITABLE else verdict
        reasons.append(f"{n:,} rows across {predictors} predictors is roughly "
                       f"{n / predictors:.0f} rows each. A model can fit noise at that ratio.")

    if profile.leakage_suspects:
        verdict = CAUTION if verdict == SUITABLE else verdict
        reasons.append(f"{len(profile.leakage_suspects)} column(s) may give away the answer. If "
                       "they do, the result will look excellent and mean nothing.")

    critical = profile.issues_by_severity("critical")
    if critical:
        verdict = CAUTION if verdict == SUITABLE else verdict
        reasons.append(f"{len(critical)} critical data-quality issue(s) are unresolved: "
                       + "; ".join(i.message for i in critical[:2]))

    if not plan.candidates:
        verdict = UNSUITABLE
        reasons.append("No algorithm in the registry suits this data and this question.")
    elif verdict == SUITABLE:
        reasons.append(f"{len(plan.candidates)} suitable algorithm(s) will be compared under "
                       f"{plan.validation_strategy.get('strategy', 'validation')}.")
    return verdict, reasons


# --------------------------------------------------------------------------
# the data contract
# --------------------------------------------------------------------------

@dataclass
class ContractCheck(JsonMixin):
    requirement: str
    passed: bool
    detail: str = ""
    severity: str = "warning"      # blocking | warning | note


@dataclass
class DataContract(JsonMixin):
    """What this analysis requires of its data, recorded so it can be re-checked.

    Stored with the run and applied again when new rows arrive to be scored: a
    model is valid only on data meeting the contract it was trained under, and
    without a written contract that comparison is done from memory or not at all.
    """

    target: str | None = None
    required_columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    max_missing_share: dict[str, float] = field(default_factory=dict)
    min_rows: int = 0
    time_column: str | None = None
    checks: list[ContractCheck] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return not [c for c in self.checks if not c.passed and c.severity == "blocking"]

    def summary(self) -> str:
        failed = [c for c in self.checks if not c.passed]
        if not failed:
            return f"All {len(self.checks)} requirements met."
        blocking = [c for c in failed if c.severity == "blocking"]
        if blocking:
            return (f"{len(blocking)} requirement(s) not met, which stops the analysis: "
                    + "; ".join(c.requirement for c in blocking))
        return f"{len(failed)} requirement(s) raised something worth knowing."


def build_contract(profile: Any, objective: Any, frame: pd.DataFrame | None = None) -> DataContract:
    """Write the contract from the data the analysis is about to run on."""
    target = objective.target
    required = [name for name in profile.columns if name != target]

    contract = DataContract(
        target=target,
        required_columns=required,
        column_types={name: column.semantic_type.value for name, column in profile.columns.items()},
        max_missing_share={name: round(column.missing_pct / 100, 4)
                           for name, column in profile.columns.items() if column.n_missing},
        min_rows=max(30, int(profile.n_rows * 0.01)),
        time_column=objective.time_column,
    )

    if frame is not None:
        for name, column in profile.columns.items():
            if column.semantic_type.value.startswith("categorical") or column.semantic_type.value == "binary":
                if name in frame.columns and column.n_unique <= 200:
                    values = frame[name].astype("string").dropna().unique()
                    contract.categories[name] = sorted(str(v) for v in values)

    contract.checks = check_contract(contract, profile, objective, frame)
    return contract


def check_contract(contract: DataContract, profile: Any, objective: Any,
                   frame: pd.DataFrame | None = None) -> list[ContractCheck]:
    """Check the contract against a profile. Used before modelling and at scoring."""
    checks: list[ContractCheck] = []
    supervised = objective.task_type.is_supervised

    if supervised:
        present = bool(contract.target and contract.target in profile.columns)
        checks.append(ContractCheck(
            "Target exists", present,
            f"`{contract.target}`" if present else "no target column is set",
            severity="blocking",
        ))
        if present:
            column = profile.columns[contract.target]
            checks.append(ContractCheck(
                "Target missingness within tolerance", column.missing_pct <= 30,
                f"{column.missing_pct:.1f}% missing",
                severity="blocking" if column.missing_pct > 60 else "warning",
            ))

    missing_required = [c for c in contract.required_columns if c not in profile.columns]
    checks.append(ContractCheck(
        "Required predictors present", not missing_required,
        "all present" if not missing_required
        else f"absent: {', '.join(missing_required[:6])}",
        severity="blocking",
    ))

    checks.append(ContractCheck(
        "Enough rows", profile.n_rows >= contract.min_rows,
        f"{profile.n_rows:,} rows (contract requires at least {contract.min_rows:,})",
        severity="blocking",
    ))

    checks.append(ContractCheck(
        "Duplicate rows within tolerance", profile.duplicate_pct <= 5,
        f"{profile.n_duplicate_rows:,} duplicate row(s), {profile.duplicate_pct:.2f}%",
    ))

    checks.append(ContractCheck(
        "No unresolved leakage suspects", not profile.leakage_suspects,
        "none flagged" if not profile.leakage_suspects
        else ", ".join(s["column"] for s in profile.leakage_suspects[:5]),
    ))

    if objective.task_type is TaskType.TIME_SERIES_FORECAST:
        ordered = bool(contract.time_column and contract.time_column in profile.columns)
        checks.append(ContractCheck(
            "Time ordering available", ordered,
            f"`{contract.time_column}`" if ordered else "no time column identified",
            severity="blocking",
        ))

    if objective.task_type.is_classification and contract.target in profile.columns:
        classes = profile.columns[contract.target].n_unique
        checks.append(ContractCheck(
            "Enough classes to model", classes >= 2,
            f"{classes} distinct value(s) in the target",
            severity="blocking",
        ))

    thin = [name for name, values in contract.categories.items() if len(values) > 50]
    if thin:
        checks.append(ContractCheck(
            "Categories manageable", False,
            f"{', '.join(thin[:4])} carry more than 50 levels, which one-hot encoding will "
            "turn into a very wide matrix",
        ))
    elif contract.categories:
        checks.append(ContractCheck("Categories manageable", True,
                                    f"{len(contract.categories)} categorical column(s)"))
    return checks
