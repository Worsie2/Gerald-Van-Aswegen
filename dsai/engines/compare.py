"""Two runs, side by side, and what actually changed between them.

Changing one preprocessing choice and re-running tells you nothing unless you
can see what it did. This diffs two completed runs — what was asked, what was
done, and what came out — and states which differences matter.

The awkward case is deliberate: two runs whose scores are not comparable, because
they answered different questions or were validated differently. Reporting a
score difference between those is worse than reporting nothing, so it is refused
by name rather than quietly presented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dsai.engines import metrics as M
from dsai.engines.metrics import human_number

__all__ = ["RunDiff", "compare_runs", "comparable"]


@dataclass
class RunDiff:
    comparable: bool = True
    reason: str = ""
    setup: pd.DataFrame = field(default_factory=pd.DataFrame)
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    preprocessing: pd.DataFrame = field(default_factory=pd.DataFrame)
    findings: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict: str = ""
    caveats: list[str] = field(default_factory=list)


def comparable(left: Any, right: Any) -> tuple[bool, str]:
    """Whether a score difference between these two runs means anything."""
    if left.objective is None or right.objective is None:
        return False, "One of these runs has no objective recorded."
    if left.objective.task_type is not right.objective.task_type:
        return False, (
            f"These answer different kinds of question — "
            f"{left.objective.task_type.value.replace('_', ' ')} against "
            f"{right.objective.task_type.value.replace('_', ' ')}. Their scores are not on the "
            "same scale and comparing them would be meaningless."
        )
    if left.objective.target != right.objective.target:
        return False, (
            f"These predict different things — `{left.objective.target}` against "
            f"`{right.objective.target}`. A score against one target says nothing about the other."
        )
    if left.dataset_name != right.dataset_name:
        return True, (
            "These ran on different datasets. The setup and preprocessing can still be compared, "
            "but a score difference may be the data rather than anything you changed."
        )
    return True, ""


def compare_runs(left: Any, right: Any) -> RunDiff:
    """Diff two completed runs. ``left`` is the earlier one by convention."""
    diff = RunDiff()
    diff.comparable, diff.reason = comparable(left, right)

    diff.setup = _setup_table(left, right)
    diff.preprocessing = _preprocessing_table(left, right)
    diff.findings = _findings_table(left, right)

    if diff.comparable and left.best is not None and right.best is not None:
        diff.scores = _score_table(left, right)
        diff.verdict, extra = _verdict(left, right)
        diff.caveats.extend(extra)
    elif not diff.comparable:
        diff.verdict = diff.reason
    else:
        diff.verdict = "At least one of these runs produced no usable model, so there is nothing to compare on score."

    if diff.reason and diff.comparable:
        diff.caveats.append(diff.reason)
    return diff


def _row(label: str, left: Any, right: Any) -> dict[str, Any]:
    left_text = "—" if left in (None, "") else str(left)
    right_text = "—" if right in (None, "") else str(right)
    return {
        "": label,
        "Run A": left_text,
        "Run B": right_text,
        "Changed": "yes" if left_text != right_text else "",
    }


def _setup_table(left: Any, right: Any) -> pd.DataFrame:
    rows = [
        _row("Dataset", left.dataset_name, right.dataset_name),
        _row("Rows", f"{left.profile.n_rows:,}" if left.profile else None,
             f"{right.profile.n_rows:,}" if right.profile else None),
        _row("Columns", left.profile.n_columns if left.profile else None,
             right.profile.n_columns if right.profile else None),
        _row("Question", left.objective.task_type.value.replace("_", " ") if left.objective else None,
             right.objective.task_type.value.replace("_", " ") if right.objective else None),
        _row("Target", left.objective.target if left.objective else None,
             right.objective.target if right.objective else None),
        _row("Validation",
             left.plan.validation_strategy.get("strategy", "").replace("_", " ") if left.plan else None,
             right.plan.validation_strategy.get("strategy", "").replace("_", " ") if right.plan else None),
        _row("Folds", left.plan.validation_strategy.get("n_splits") if left.plan else None,
             right.plan.validation_strategy.get("n_splits") if right.plan else None),
        _row("Ranked on", left.plan.primary_metric if left.plan else None,
             right.plan.primary_metric if right.plan else None),
        _row("Hold-out share",
             f"{left.settings.test_size:.0%}" if left.settings else None,
             f"{right.settings.test_size:.0%}" if right.settings else None),
        _row("Random seed", left.settings.random_state if left.settings else None,
             right.settings.random_state if right.settings else None),
        _row("Models trained", len(left.results), len(right.results)),
        _row("Selected model", left.best.model_name if left.best else None,
             right.best.model_name if right.best else None),
        _row("Took", f"{left.duration_s:.1f}s", f"{right.duration_s:.1f}s"),
    ]
    return pd.DataFrame(rows)


def _score_table(left: Any, right: Any) -> pd.DataFrame:
    metrics = M.default_metrics(left.best.task_type)
    rows = []
    for metric in metrics:
        a, b = left.best.primary(metric), right.best.primary(metric)
        if a is None and b is None:
            continue
        change = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            change = ("better" if M.is_better(metric, b, a)
                      else "worse" if M.is_better(metric, a, b) else "unchanged")
        rows.append({
            "Metric": M.METRIC_LABELS.get(metric, metric),
            "Run A": human_number(a),
            "Run B": human_number(b),
            "Difference": human_number(b - a) if isinstance(a, (int, float))
                          and isinstance(b, (int, float)) else "—",
            "B is": change,
        })
    rows.append({
        "Metric": "Train-to-test gap",
        "Run A": human_number(left.best.overfitting_gap),
        "Run B": human_number(right.best.overfitting_gap),
        "Difference": human_number((right.best.overfitting_gap or 0) - (left.best.overfitting_gap or 0)),
        "B is": ("better" if (right.best.overfitting_gap or 0) < (left.best.overfitting_gap or 0)
                 else "worse" if (right.best.overfitting_gap or 0) > (left.best.overfitting_gap or 0)
                 else "unchanged"),
    })
    return pd.DataFrame(rows)


def _preprocessing_table(left: Any, right: Any) -> pd.DataFrame:
    def steps(run: Any) -> list[str]:
        if run.pipeline is None:
            return []
        return [step.describe() for step in run.pipeline.active_steps]

    a, b = steps(left), steps(right)
    rows = []
    for step in a:
        rows.append({"Step": step, "Run A": "yes", "Run B": "yes" if step in b else "—",
                     "Changed": "" if step in b else "removed in B"})
    for step in b:
        if step not in a:
            rows.append({"Step": step, "Run A": "—", "Run B": "yes", "Changed": "added in B"})
    return pd.DataFrame(rows)


def _findings_table(left: Any, right: Any) -> pd.DataFrame:
    a = {f.title: f for f in left.findings}
    b = {f.title: f for f in right.findings}
    rows = []
    for title in a:
        rows.append({
            "Finding": title,
            "Run A": a[title].confidence.value,
            "Run B": b[title].confidence.value if title in b else "—",
            "Changed": "" if title in b else "gone in B",
        })
    for title in b:
        if title not in a:
            rows.append({"Finding": title, "Run A": "—", "Run B": b[title].confidence.value,
                         "Changed": "new in B"})
    return pd.DataFrame(rows)


def _verdict(left: Any, right: Any) -> tuple[str, list[str]]:
    """Whether B beat A, and whether the difference is big enough to believe."""
    metric = right.plan.primary_metric if right.plan else ""
    a, b = left.best.primary(metric), right.best.primary(metric)
    caveats: list[str] = []
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "The two runs have no metric in common to compare on.", caveats

    better = M.is_better(metric, b, a)
    change = abs(b - a)
    reference = max(abs(a), 1e-12)
    relative = change / reference

    # The fold-to-fold spread is the natural yardstick: a difference smaller than
    # the wobble between folds of a single run is not a difference.
    noise = max(
        left.best.validation.std_scores.get(metric, 0.0),
        right.best.validation.std_scores.get(metric, 0.0),
    )
    if noise and change < noise:
        caveats.append(
            f"The difference of {human_number(change)} is smaller than the fold-to-fold spread "
            f"within a single run ({human_number(noise)}). Re-running either one with a different "
            "seed could reverse it — treat these two as tied."
        )
        return (
            f"**No real difference.** Run B scores {human_number(b)} against Run A's "
            f"{human_number(a)} on {metric}, but that gap is inside the noise."
        ), caveats

    direction = "better" if better else "worse"
    if relative < 0.01:
        caveats.append(
            "A change under 1% rarely survives contact with new data. Worth confirming on a "
            "different sample before acting on it."
        )
    return (
        f"**Run B is {direction}.** {human_number(b)} against {human_number(a)} on {metric} — "
        f"a change of {human_number(change)} ({relative:.1%}). "
        f"Selected model: {right.best.model_name} against {left.best.model_name}."
    ), caveats
