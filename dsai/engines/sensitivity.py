"""Does the conclusion survive reasonable changes to how the analysis was done?

Every analysis embeds choices nobody argued about: which imputation, whether to
clip outliers, how many folds, which seed, which model family. Each was
defensible. The question this answers is whether the *result* depended on them.

A conclusion that holds across twenty reasonable specifications is worth acting
on. One that flips when the outlier treatment changes is a finding about the
outlier treatment, and the reader has to be told which choice is carrying it.

Three deliberate constraints:

* **Only reasonable specifications.** Every variation here is a choice a
  competent analyst might have made. Searching over unreasonable ones and
  reporting that the result "survived" would be theatre.
* **The same data, the same target, the same question.** Only the method varies.
* **Direction first, magnitude second.** Whether the top driver stays the top
  driver, and whether the sign holds, matters more than whether R² moved by
  0.02 — and it is what a reader will act on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dsai.core.schema import TaskType
from dsai.engines.metrics import human_number

__all__ = ["Specification", "SpecResult", "SensitivityReport", "run_sensitivity",
           "build_specifications"]


@dataclass
class Specification:
    """One defensible alternative way of running the same analysis."""

    id: str
    label: str
    #: What was varied, so a reader can see which lever this pulls.
    dimension: str
    #: Why a competent analyst might have chosen this instead.
    rationale: str = ""
    model_key: str | None = None
    aggressiveness: str = "standard"
    random_state: int = 42
    n_splits: int | None = None
    drop_columns: list[str] = field(default_factory=list)
    imputation: str | None = None       # median | mean | knn
    outlier_treatment: str | None = None  # none | winsorize | clip
    is_baseline: bool = False


@dataclass
class SpecResult:
    spec: Specification
    ok: bool = True
    error: str = ""
    score: float | None = None
    metric: str = ""
    top_driver: str = ""
    driver_direction: str = ""
    beats_baseline: bool | None = None
    elapsed_s: float = 0.0


@dataclass
class SensitivityReport:
    metric: str = ""
    baseline: SpecResult | None = None
    results: list[SpecResult] = field(default_factory=list)
    verdict: str = ""
    stable: bool = True
    agreement: float = 0.0
    n_ran: int = 0
    n_failed: int = 0
    unstable_dimensions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> list[SpecResult]:
        return [r for r in self.results if r.ok and r.score is not None]

    def table(self) -> pd.DataFrame:
        """Every specification and what it produced."""
        rows = []
        for result in self.results:
            rows.append({
                "Specification": result.spec.label,
                "Varied": result.spec.dimension,
                self.metric or "score": human_number(result.score) if result.ok else "—",
                "Top driver": result.top_driver or "—",
                "Direction": result.driver_direction or "—",
                "Status": "ok" if result.ok else f"failed: {result.error[:60]}",
            })
        return pd.DataFrame(rows)

    def spread(self) -> tuple[float | None, float | None]:
        scores = [r.score for r in self.succeeded if r.score is not None]
        return (min(scores), max(scores)) if scores else (None, None)


def build_specifications(run: Any, limit: int = 20) -> list[Specification]:
    """The alternative specifications worth trying for this particular run.

    Built from what the analysis actually did, so the variations are relevant:
    there is no point varying the outlier treatment on a dataset with no
    outliers, or the imputation on one with no gaps.
    """
    profile, objective, plan = run.profile, run.objective, run.plan
    specs: list[Specification] = [Specification(
        id="baseline", label="As run", dimension="—",
        rationale="The analysis exactly as it was performed.",
        model_key=run.best.model_key if run.best else None, is_baseline=True,
    )]

    # -- the model family ----------------------------------------------------
    # The single most consequential choice, and the one a reader is most likely
    # to suspect. Take the top few candidates that are not the winner.
    if plan is not None:
        # The baseline ignores every predictor by design, so it is not an
        # alternative specification — including it manufactures a disagreement
        # that says nothing about whether the conclusion is stable.
        others = [c for c in plan.candidates
                  if c.family != "baseline"
                  and (not run.best or c.spec_key != run.best.model_key)][:4]
        for candidate in others:
            specs.append(Specification(
                id=f"model_{candidate.spec_key}", label=f"{candidate.name} instead",
                dimension="model family",
                rationale=f"{candidate.family.replace('_', ' ')} rather than "
                          f"{run.best.family.replace('_', ' ') if run.best else 'the winner'}.",
                model_key=candidate.spec_key,
            ))

    # -- preprocessing depth -------------------------------------------------
    for depth in ("minimal", "thorough"):
        specs.append(Specification(
            id=f"prep_{depth}", label=f"{depth.capitalize()} preprocessing",
            dimension="preprocessing depth",
            rationale=("Only what is strictly necessary." if depth == "minimal"
                       else "Transform distributions and reduce dimensions as well."),
            model_key=run.best.model_key if run.best else None,
            aggressiveness=depth,
        ))

    # -- imputation, only where anything is missing --------------------------
    if profile is not None and any(c.n_missing for c in profile.columns.values()):
        for strategy in ("mean", "knn"):
            specs.append(Specification(
                id=f"impute_{strategy}", label=f"{strategy.upper()} imputation" if strategy == "knn"
                else "Mean imputation",
                dimension="imputation",
                rationale="A different, equally defensible way of filling the gaps.",
                model_key=run.best.model_key if run.best else None,
                imputation=strategy,
            ))

    # -- outliers, only where there are any ----------------------------------
    outlier_columns = [c for c in (profile.columns.values() if profile else [])
                       if (c.outlier_pct or 0) > 1]
    if outlier_columns:
        for treatment in ("winsorize", "none"):
            specs.append(Specification(
                id=f"outliers_{treatment}",
                label=("Winsorised extremes" if treatment == "winsorize"
                       else "Extremes left alone"),
                dimension="outlier treatment",
                rationale=("Clip the tails to the 1st and 99th percentile."
                           if treatment == "winsorize"
                           else "Keep every extreme value as measured."),
                model_key=run.best.model_key if run.best else None,
                outlier_treatment=treatment,
            ))

    # -- the seed and the fold count -----------------------------------------
    for seed in (7, 2024):
        specs.append(Specification(
            id=f"seed_{seed}", label=f"Random seed {seed}", dimension="random seed",
            rationale="The same method on a different split of the same rows.",
            model_key=run.best.model_key if run.best else None, random_state=seed,
        ))
    current_folds = int((plan.validation_strategy.get("n_splits") or 5) if plan else 5)
    for folds in (3, 10):
        if folds != current_folds:
            specs.append(Specification(
                id=f"folds_{folds}", label=f"{folds}-fold cross-validation",
                dimension="fold count",
                rationale="More folds means less training data per fold but more estimates.",
                model_key=run.best.model_key if run.best else None, n_splits=folds,
            ))

    # -- dropping a questionable predictor -----------------------------------
    # The most informative variation of all: if the conclusion rests on a column
    # somebody has doubts about, that is exactly what a reader needs told.
    suspects = [s["column"] for s in (profile.leakage_suspects if profile else [])][:2]
    if run.explanation is not None and run.explanation.importances:
        top = run.explanation.importances[0].feature
        base = top.split("_")[0]
        for column in (profile.columns if profile else []):
            if column != objective.target and (column == top or base == column):
                suspects.append(column)
                break
    for column in dict.fromkeys(suspects):
        specs.append(Specification(
            id=f"drop_{column}", label=f"Without `{column}`", dimension="questionable predictor",
            rationale=f"Whether the conclusion survives without {column}.",
            model_key=run.best.model_key if run.best else None, drop_columns=[column],
        ))

    return specs[:limit]


def run_sensitivity(run: Any, frame: pd.DataFrame, specs: list[Specification] | None = None,
                    registry: Any = None, progress: Any = None) -> SensitivityReport:
    """Run each specification and report whether the conclusion held.

    ``progress`` is called with ``(index, total, label)`` so a UI can show named
    work rather than a spinner.
    """
    from dsai.engines.experiment import ExperimentConfig, ExperimentEngine
    from dsai.preprocessing.recommender import recommend_pipeline
    from dsai.registry.base import REGISTRY

    registry = registry or REGISTRY
    specs = specs or build_specifications(run)
    metric = run.plan.primary_metric if run.plan else ""
    report = SensitivityReport(metric=metric)

    for index, spec in enumerate(specs):
        if progress is not None:
            progress(index, len(specs), spec.label)
        started = time.perf_counter()
        result = SpecResult(spec=spec, metric=metric)
        try:
            _run_one(run, frame, spec, registry, result, metric)
        except Exception as exc:
            result.ok = False
            result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed_s = round(time.perf_counter() - started, 2)
        report.results.append(result)
        if spec.is_baseline:
            report.baseline = result

    report.n_ran = len(report.succeeded)
    report.n_failed = sum(1 for r in report.results if not r.ok)
    _judge(report, list(frame.columns))
    return report


def _run_one(run: Any, frame: pd.DataFrame, spec: Specification, registry: Any,
             result: SpecResult, metric: str) -> None:
    from dsai.engines.experiment import ExperimentConfig, ExperimentEngine
    from dsai.preprocessing.recommender import recommend_pipeline

    objective = run.objective
    working = frame.drop(columns=[c for c in spec.drop_columns if c in frame.columns],
                         errors="ignore")

    model_key = spec.model_key or (run.best.model_key if run.best else None)
    if model_key is None:
        raise ValueError("No model to run this specification with.")
    model_spec = registry.get(model_key)

    from dsai.core.profiler import profile_dataset

    profile, typed = profile_dataset(working, name=run.dataset_name, target=objective.target)
    pipeline, _ = recommend_pipeline(
        profile, objective.task_type, objective.target, model_spec, run.context,
        aggressiveness=spec.aggressiveness,
    )
    _apply_variations(pipeline, spec, profile, objective)

    validation = dict(run.plan.validation_strategy) if run.plan else {}
    if spec.n_splits:
        validation["n_splits"] = spec.n_splits
    config = ExperimentConfig(
        test_size=run.settings.test_size,
        random_state=spec.random_state,
        cross_validate=True,
        validation=validation,
        scoring_metrics=[metric] if metric else [],
    )
    engine = ExperimentEngine(typed, objective, profile, registry, config)
    outcome = engine.run_model(model_key, pipeline)

    if outcome.status != "success":
        raise RuntimeError(outcome.error or "the model did not complete")
    result.score = outcome.primary(metric)
    result.top_driver, result.driver_direction = _top_driver(engine, outcome, typed, objective)


def _apply_variations(pipeline: Any, spec: Specification, profile: Any, objective: Any) -> None:
    """Swap the imputation or outlier steps for the ones this specification asks for."""
    if spec.imputation:
        for step in list(pipeline.steps):
            if step.step_key in ("impute_median", "impute_mean", "impute_knn"):
                pipeline.remove(step.id, by="sensitivity")
        numeric = [n for n, c in profile.columns.items()
                   if c.semantic_type.value.startswith("numeric") and n != objective.target]
        if numeric:
            key = {"mean": "impute_mean", "median": "impute_median", "knn": "impute_knn"}[
                spec.imputation]
            pipeline.add(key, columns=numeric, position=0, by="sensitivity")

    if spec.outlier_treatment:
        for step in list(pipeline.steps):
            if step.step_key in ("winsorize", "clip_iqr", "remove_outlier_rows"):
                pipeline.remove(step.id, by="sensitivity")
        if spec.outlier_treatment == "winsorize":
            numeric = [n for n, c in profile.columns.items()
                       if c.semantic_type.value.startswith("numeric") and n != objective.target]
            if numeric:
                pipeline.add("winsorize", columns=numeric, by="sensitivity")


def _top_driver(engine: Any, outcome: Any, frame: pd.DataFrame,
                objective: Any) -> tuple[str, str]:
    """What this specification says matters most — the part a reader acts on."""
    from dsai.explain.importance import explain_model

    try:
        model = engine.fitted_model(outcome.id)
        if model is None:
            return "", ""
        features = [c for c in outcome.features if c in frame.columns]
        sample = frame[features].head(400)
        target = frame[objective.target].head(400) if objective.target in frame.columns else None
        bundle = explain_model(model, sample, target, objective.task_type)
        if not bundle.importances:
            return "", ""
        top = bundle.importances[0]
        return top.feature, (top.direction or "")
    except Exception:
        return "", ""


def _base_feature(name: str, columns: list[str] | None = None) -> str:
    """The source column a feature came from, before encoding.

    One-hot encoding turns `service_tier` into `service_tier_premium`, and two
    specifications naming those are not disagreeing — they are naming the same
    variable either side of a transformation. Counting that as instability
    would report a false negative on every dataset with a categorical driver.
    """
    if columns:
        matches = [c for c in columns if name == c or name.startswith(f"{c}_")]
        if matches:
            return max(matches, key=len)
    return name.split("_")[0] if "_" in name else name


def _judge(report: SensitivityReport, columns: list[str] | None = None) -> None:
    """Did the conclusion survive, and if not, which lever broke it?"""
    succeeded = report.succeeded
    if len(succeeded) < 3:
        report.stable = False
        report.verdict = (
            f"Only {len(succeeded)} of {len(report.results)} specification(s) completed, which is "
            "too few to say whether the conclusion is stable. Treat the original result as "
            "unchecked rather than confirmed."
        )
        return

    baseline = report.baseline
    reference_driver = (baseline.top_driver if baseline and baseline.ok else
                        succeeded[0].top_driver)
    reference_base = _base_feature(reference_driver, columns)

    # Agreement on *what matters*, which is what a reader acts on — not on the
    # third decimal place of a metric, and not on whether a categorical driver
    # is named before or after encoding.
    with_driver = [r for r in succeeded if r.top_driver]
    agreeing = [r for r in with_driver
                if _base_feature(r.top_driver, columns) == reference_base]
    report.agreement = len(agreeing) / len(with_driver) if with_driver else 0.0

    disagreeing = [r for r in with_driver if r not in agreeing]
    report.unstable_dimensions = sorted({r.spec.dimension for r in disagreeing
                                         if r.spec.dimension != "—"})

    low, high = report.spread()
    spread_note = ""
    if low is not None and high is not None:
        spread_note = (f" Across them {report.metric} ranged from {human_number(low)} to "
                       f"{human_number(high)}.")

    if not with_driver:
        report.stable = False
        report.verdict = (
            f"{len(succeeded)} specification(s) ran, but none produced an explanation to compare, "
            "so there is nothing to check the conclusion against." + spread_note
        )
        return

    if report.agreement >= 0.8:
        report.stable = True
        report.verdict = (
            f"**Stable.** `{reference_driver}` came out as the strongest driver in "
            f"{len(agreeing)} of {len(with_driver)} reasonable specifications."
            + spread_note
            + " The conclusion does not depend on the analytical choices that were varied."
        )
    elif report.agreement >= 0.5:
        report.stable = False
        report.verdict = (
            f"**Partly sensitive.** `{reference_driver}` led in only {len(agreeing)} of "
            f"{len(with_driver)} specifications."
            + spread_note
            + " The conclusion holds more often than not, but it is carried in part by "
            + ("choices about " + ", ".join(report.unstable_dimensions)
               if report.unstable_dimensions else "the analytical choices made")
            + " — which a reader should be told."
        )
    else:
        report.stable = False
        report.verdict = (
            f"**Sensitive.** `{reference_driver}` led in only {len(agreeing)} of "
            f"{len(with_driver)} specifications; the answer changes with "
            + (", ".join(report.unstable_dimensions) if report.unstable_dimensions
               else "the analytical choices made")
            + "." + spread_note
            + " This is a finding about the method as much as about the data. Do not act on the "
            "original conclusion without deciding which specification is right and saying why."
        )

    if report.n_failed:
        report.notes.append(
            f"{report.n_failed} specification(s) failed to run and are excluded from the count. "
            "A specification that cannot run is not evidence either way."
        )
    if len(with_driver) < len(succeeded):
        report.notes.append(
            f"{len(succeeded) - len(with_driver)} specification(s) produced no explanation, so "
            "only their scores could be compared."
        )
