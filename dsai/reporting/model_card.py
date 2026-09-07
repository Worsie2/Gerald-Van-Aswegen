"""A model card: what this model is for, and what it is not for.

A trained model outlives the conversation that produced it. Six months later
someone finds the file and has to decide whether to use it — on what data, for
what decision, with what caveats. Without a card that decision is made from
guesswork.

The section that matters most is **inappropriate uses**. Every model card format
lists what a model is good at; the failure mode in practice is a model used
somewhere nobody considered, and the only defence is having written down where
it should not go.

Assembled from the run rather than authored, so the card cannot drift away from
what the model actually is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dsai.core.schema import TaskType
from dsai.engines import metrics as M
from dsai.engines.metrics import human_number

__all__ = ["ModelCard", "build_model_card"]


@dataclass
class ModelCard:
    model_name: str = ""
    model_key: str = ""
    purpose: str = ""
    target: str | None = None
    task: str = ""
    dataset: str = ""
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0
    validation: str = ""
    metrics: list[tuple[str, str, str]] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    appropriate_uses: list[str] = field(default_factory=list)
    inappropriate_uses: list[str] = field(default_factory=list)
    subgroups: list[dict[str, Any]] = field(default_factory=list)
    calibration: str = ""
    drift_sensitivity: str = ""
    fingerprint: str = ""
    created_at: str = ""
    environment: dict[str, str] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"# Model card — {self.model_name}", "",
            f"*{self.purpose}*", "",
            "## What it is",
            "",
            f"- **Task**: {self.task}",
            f"- **Predicts**: `{self.target}`" if self.target else "- **Predicts**: no target (unsupervised)",
            f"- **Trained on**: {self.dataset} — {self.n_train:,} rows, "
            f"{self.n_features} feature(s) after preprocessing",
            f"- **Held out**: {self.n_test:,} rows, scored once",
            f"- **Validation**: {self.validation}",
            f"- **Fingerprint**: `{self.fingerprint}`",
            f"- **Built**: {self.created_at}",
            "",
        ]
        if self.metrics:
            lines += ["## How well it does", "",
                      "| Metric | Value | Measured on |", "| --- | ---: | --- |"]
            lines += [f"| {name} | {value} | {source} |" for name, value, source in self.metrics]
            lines += ["", "Performance is on this dataset under this validation strategy. It is "
                      "not a guarantee of performance anywhere else.", ""]

        for heading, items in [
            ("## Appropriate uses", self.appropriate_uses),
            ("## Inappropriate uses", self.inappropriate_uses),
            ("## Assumptions it makes", self.assumptions),
            ("## Known limitations", self.limitations),
        ]:
            if items:
                lines += [heading, ""] + [f"- {item}" for item in items] + [""]

        if self.preprocessing:
            lines += ["## Preprocessing it expects", ""]
            lines += [f"{i + 1}. {step}" for i, step in enumerate(self.preprocessing)]
            lines += ["", "New data must go through the same pipeline. The transformations are "
                      "the ones fitted during training, not refitted on the new rows.", ""]

        if self.hyperparameters:
            lines += ["## Hyper-parameters", "", "| Parameter | Value |", "| --- | --- |"]
            lines += [f"| `{k}` | `{v}` |" for k, v in sorted(self.hyperparameters.items())
                      if v is not None]
            lines.append("")

        if self.subgroups:
            lines += ["## Performance by subgroup", "",
                      "Where the model does better and worse. A difference here is a measured "
                      "difference, not a judgement about fairness — investigating why is the "
                      "reader's job.", "",
                      "| Group | Rows | Score | Against overall |", "| --- | ---: | ---: | --- |"]
            for row in self.subgroups:
                lines.append(f"| {row['group']} | {row['n']:,} | {row['score']} | {row['reading']} |")
            lines.append("")

        if self.calibration:
            lines += ["## Probability calibration", "", self.calibration, ""]
        if self.drift_sensitivity:
            lines += ["## Sensitivity to changing data", "", self.drift_sensitivity, ""]
        if self.environment:
            lines += ["## Built with", "", "| Component | Version |", "| --- | --- |"]
            lines += [f"| {k} | {v} |" for k, v in sorted(self.environment.items())]
        return "\n".join(lines)


def build_model_card(run: Any, result: Any = None, frame: Any = None) -> ModelCard:
    """Assemble the card for a model in a completed run."""
    result = result or run.best
    if result is None:
        return ModelCard(model_name="none", purpose="No model completed in this run.")

    objective = run.objective
    task = result.task_type
    card = ModelCard(
        model_name=result.model_name,
        model_key=result.model_key,
        target=result.target,
        task=task.value.replace("_", " "),
        dataset=run.dataset_name,
        n_train=result.n_train,
        n_test=result.n_test,
        n_features=result.n_features_out,
        hyperparameters=dict(result.hyperparameters or {}),
        preprocessing=list(result.preprocessing or []),
        created_at=run.created_at,
        environment=dict(run.environment or {}),
    )
    card.purpose = _purpose(result, objective)

    if run.plan is not None:
        strategy = run.plan.validation_strategy
        card.validation = (f"{str(strategy.get('strategy', '')).replace('_', ' ')}"
                           f", {strategy.get('n_splits', '—')} folds")
    for metric in M.default_metrics(task):
        value = result.primary(metric)
        if value is None:
            continue
        card.metrics.append((M.METRIC_LABELS.get(metric, metric), human_number(value),
                             result.score_source(metric)))

    try:
        from dsai.repro.provenance import build_manifest

        card.fingerprint = build_manifest(run).fingerprint()
    except Exception:
        card.fingerprint = ""

    spec = _spec_for(result.model_key)
    if spec is not None:
        card.assumptions = list(spec.assumptions or [])
        card.limitations = list(spec.limitations or [])
        # The registry stores these as terse fragments ("multicollinearity",
        # "wide datasets"). Read as a list of uses they are ungrammatical, so
        # they are made into statements rather than dropped.
        card.appropriate_uses = [
            item if item[:1].isupper() and item.endswith(".")
            else f"Data characterised by {item[0].lower()}{item[1:]}."
            for item in (spec.good_for or [])
        ]

    card.limitations += _run_limitations(run, result)
    card.appropriate_uses += _appropriate(run, result)
    card.inappropriate_uses = _inappropriate(run, result)
    card.calibration = _calibration(run, task)
    card.drift_sensitivity = _drift_sensitivity(run, result)
    card.subgroups = _subgroups(run, result, frame)
    return card


def _spec_for(key: str):
    try:
        from dsai.registry.base import REGISTRY

        return REGISTRY.get(key)
    except Exception:
        return None


def _purpose(result: Any, objective: Any) -> str:
    if result.target:
        return (f"Predicts `{result.target}` from {len(result.features)} column(s), for "
                f"{objective.task_type.value.replace('_', ' ') if objective else 'analysis'}.")
    return "Finds structure in the data without a target variable."


def _run_limitations(run: Any, result: Any) -> list[str]:
    out: list[str] = []
    if result.overfitting_gap is not None and result.overfitting_gap > 0.1:
        out.append(f"Scores {result.overfitting_gap:.0%} better on training rows than held-out "
                   "ones — it has partly memorised the training data.")
    metric = run.plan.primary_metric if run.plan else ""
    spread = result.validation.std_scores.get(metric)
    mean = result.validation.mean_scores.get(metric)
    if spread and mean and abs(mean) > 1e-9 and spread / abs(mean) > 0.15:
        out.append(f"The score varies {spread / abs(mean):.0%} between cross-validation folds, so "
                   "any single figure for its performance is approximate.")
    if run.profile is not None:
        missing = sum(c.n_missing for c in run.profile.columns.values())
        if missing:
            out.append(f"Trained on data with {missing:,} imputed values. Those estimates are "
                       "baked into what it learned.")
        if run.profile.n_rows < 500:
            out.append(f"Trained on only {run.profile.n_rows:,} rows.")
    if run.self_check is not None:
        out += [c.detail for c in run.self_check.checks
                if not c.passed and c.severity == "blocking" and c.detail]
    return out


def _appropriate(run: Any, result: Any) -> list[str]:
    out = [
        f"Scoring new rows drawn from the same population as {run.dataset_name}, carrying the "
        "same columns.",
        "Ranking or prioritising — comparing rows against each other, where the ordering matters "
        "more than the exact number.",
    ]
    if result.task_type is TaskType.REGRESSION:
        out.append("Estimating a magnitude where an error of the size reported above is tolerable.")
    if result.task_type.is_classification:
        out.append("Flagging rows for human review, where a person makes the final decision.")
    return out


def _inappropriate(run: Any, result: Any) -> list[str]:
    """Where this model should not go. The section that earns the card its keep."""
    out = [
        "Deciding what to change. This model describes association, not cause — its predictors "
        "moving together with the outcome does not mean acting on them moves the outcome.",
        "Data from a different population, period or process from the training data. Predictions "
        "on unfamiliar rows come back just as confidently and mean nothing.",
        "Any decision about an individual person that is taken without a human in the loop, "
        "particularly where it affects their access to something.",
    ]
    if result.task_type.is_classification:
        out.append("Treating its probabilities as calibrated likelihoods without checking the "
                   "calibration curve first.")
    if run.profile is not None and run.profile.leakage_suspects:
        out.append(f"Any use at all until the {len(run.profile.leakage_suspects)} leakage "
                   "suspect(s) flagged in the data have been ruled out.")
    if result.n_train < 500:
        out.append("High-stakes decisions. It was trained on too few rows for that.")
    return out


def _calibration(run: Any, task: TaskType) -> str:
    if not task.is_classification:
        return ""
    calibration = (run.diagnostics or {}).get("calibration", {})
    if not calibration.get("supported"):
        return ("This model does not produce probabilities that can be checked for calibration, "
                "so its scores should be read as rankings rather than likelihoods.")
    return calibration.get("interpretation", "")


def _drift_sensitivity(run: Any, result: Any) -> str:
    if run.explanation is None or not run.explanation.importances:
        return ""
    top = run.explanation.top(3)
    named = ", ".join(f"`{item.feature}`" for item in top)
    return (f"Leans most on {named}. A shift in the distribution of those columns will move its "
            "predictions more than a shift anywhere else — they are what to watch when scoring "
            "new data.")


def _subgroups(run: Any, result: Any, frame: Any) -> list[dict[str, Any]]:
    """Where the model does better and worse, across the categorical columns.

    Only reported where a group has enough rows to say anything: a group of nine
    with a bad score is noise, and presenting it as a finding invites someone to
    act on nothing.
    """
    import numpy as np
    import pandas as pd

    if frame is None or run.profile is None or result.target is None:
        return []
    actual = result.extras.get("holdout_actual")
    # Regression keeps numbers under one key, classification keeps labels under
    # another; a subgroup breakdown needs whichever this model produced.
    predicted = (result.extras.get("holdout_predicted")
                 or result.extras.get("holdout_predicted_labels"))
    if not actual or not predicted or len(actual) != len(predicted):
        return []

    index = result.extras.get("holdout_index")
    if index is None or len(index) != len(actual):
        return []
    try:
        held = frame.loc[list(index)]
    except Exception:
        return []

    errors = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) \
        if result.task_type is TaskType.REGRESSION else \
        (np.asarray([str(a) for a in actual]) != np.asarray([str(p) for p in predicted])).astype(float)
    overall = float(np.mean(errors))
    label = "mean absolute error" if result.task_type is TaskType.REGRESSION else "error rate"

    rows: list[dict[str, Any]] = []
    for column in run.profile.categorical_columns[:4]:
        if column not in held.columns:
            continue
        groups = pd.Series(errors, index=held.index).groupby(held[column].astype("string"))
        for name, values in groups:
            if len(values) < 30:
                continue
            score = float(values.mean())
            ratio = score / overall if overall > 1e-12 else 1.0
            reading = ("in line with overall" if 0.8 <= ratio <= 1.25
                       else f"{ratio:.1f}× the overall {label}" if ratio > 1.25
                       else f"better than overall ({ratio:.1f}×)")
            rows.append({"group": f"{column} = {name}", "n": len(values),
                         "score": human_number(score), "reading": reading, "ratio": ratio})
    rows.sort(key=lambda r: -r["ratio"])
    return rows[:12]
