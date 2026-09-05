"""The AI Data Scientist: the orchestration layer that runs the whole workflow.

Upload → understand → context → quality → preprocess → choose methods →
run models → validate → compare → explain → insights → recommendations.

Three modes share this one engine:

* ``automatic``  — run everything end to end.
* ``assisted``   — stop after planning and hand the plan back for approval.
* ``manual``     — the caller drives each stage itself, using the same components.

Nothing here is a shortcut around the individual engines; the orchestrator only
sequences them and records what happened.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from dsai.core.profiler import apply_user_overrides, profile_dataset
from dsai.core.schema import (
    BusinessContext, Confidence, DatasetProfile, Decision, Finding, JsonMixin, Objective,
    Recommendation, TaskType, TraceEvent,
)
from dsai.dataio.loaders import DataSource
from dsai.engines import metrics as M
from dsai.engines.association import mine_associations
from dsai.engines.clustering_profile import profile_clusters, suggest_cluster_count
from dsai.engines.decision import AnalysisPlan, Constraints, plan_analysis
from dsai.engines.experiment import ExperimentConfig, ExperimentEngine, ExperimentResult
from dsai.engines.insight import generate_insights
from dsai.engines.objective import detect_objectives
from dsai.engines.recommend import generate_recommendations
from dsai.engines.selection import TournamentOutcome, run_tournament
from dsai.engines.selfcheck import SelfCheckReport, check_analysis
from dsai.engines.trace import AnalysisTrace
from dsai.explain.diagnostics import classification_diagnostics, regression_diagnostics
from dsai.explain.importance import ExplanationBundle, explain_model
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.preprocessing.recommender import recommend_pipeline
from dsai.registry.base import REGISTRY, ModelRegistry, load_builtin_models
from dsai.statistics.timeseries import analyse_series


@dataclass
class RunSettings(JsonMixin):
    mode: str = "automatic"                 # automatic | assisted | manual
    max_models: int = 8
    time_budget: str = "balanced"           # fast | balanced | thorough
    interpretability_need: str = "moderate" # critical | high | moderate | low
    tune_hyperparameters: bool = False
    test_size: float = 0.2
    random_state: int = 42
    preprocessing_aggressiveness: str = "standard"
    explain: bool = True
    max_objectives: int = 3
    run_secondary_objectives: bool = False
    exclude_models: list[str] = field(default_factory=list)
    include_models: list[str] = field(default_factory=list)


@dataclass
class AnalysisRun(JsonMixin):
    """Everything one end-to-end analysis produced. Serialisable and replayable."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = ""
    dataset_name: str = ""
    source: DataSource | None = None
    settings: RunSettings = field(default_factory=RunSettings)
    context: BusinessContext = field(default_factory=BusinessContext)

    profile: DatasetProfile | None = None
    objectives: list[Objective] = field(default_factory=list)
    objective: Objective | None = None
    plan: AnalysisPlan | None = None
    pipeline: PreprocessingPipeline | None = None
    results: list[ExperimentResult] = field(default_factory=list)
    tournament: TournamentOutcome | None = None
    best: ExperimentResult | None = None
    explanation: ExplanationBundle | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    segmentation: Any = None
    series_analysis: dict[str, Any] = field(default_factory=dict)
    associations: Any = None

    findings: list[Finding] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    self_check: SelfCheckReport | None = None
    trace: AnalysisTrace = field(default_factory=AnalysisTrace)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    duration_s: float = 0.0
    status: str = "pending"                 # pending | planned | complete | failed

    def summary(self) -> str:
        parts = []
        if self.profile:
            parts.append(f"{self.profile.n_rows:,} rows × {self.profile.n_columns} columns")
        if self.objective:
            parts.append(f"framed as {self.objective.task_type.value.replace('_', ' ')}")
        if self.tournament and self.tournament.recommended:
            parts.append(f"recommended model: {self.tournament.recommended.model_name}")
        parts.append(f"{len(self.findings)} finding(s), {len(self.recommendations)} recommendation(s)")
        return "; ".join(parts) + "."


class AIDataScientist:
    """The orchestration layer. Owns the sequence, not the analysis itself."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        trace_callback: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.registry = registry or load_builtin_models()
        self.trace_callback = trace_callback
        self._engines: dict[str, ExperimentEngine] = {}

    # -- stage 1-2: understand the data --------------------------------------
    def understand(
        self,
        frame: pd.DataFrame,
        name: str = "dataset",
        context: BusinessContext | None = None,
        source: DataSource | None = None,
        run: AnalysisRun | None = None,
    ) -> AnalysisRun:
        """Profile the data and propose objectives. Runs before anything is decided."""
        run = run or AnalysisRun()
        run.created_at = run.created_at or time.strftime("%Y-%m-%d %H:%M:%S")
        run.dataset_name = name
        run.source = source
        run.context = context or BusinessContext()
        run.trace.callback = self.trace_callback

        with run.trace.start("Loading dataset") as step:
            step.update(f"{len(frame):,} rows, {frame.shape[1]} columns")

        with run.trace.start("Profiling variables and data quality"):
            target_hint = run.context.target_variable
            profile, typed = profile_dataset(frame, name=name, target=target_hint)
            profile = apply_user_overrides(profile, run.context)
            run.profile = profile
            self._typed_frame = typed

        run.trace.add(
            "Dataset understood",
            detail=(
                f"{len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical, "
                f"{len(profile.datetime_columns)} date, {len(profile.text_columns)} text; "
                f"quality score {profile.quality_score}/100"
            ),
        )
        for issue in profile.issues_by_severity("critical"):
            run.trace.add(f"Data quality: {issue.message}", status="warning")
            run.warnings.append(issue.message)

        with run.trace.start("Identifying analytical objectives"):
            run.objectives = detect_objectives(profile, run.context, max_objectives=5)
            run.objective = run.objectives[0] if run.objectives else None
        if run.objective:
            run.trace.add(
                f"Problem framed as {run.objective.task_type.value.replace('_', ' ')}"
                + (f" on '{run.objective.target}'" if run.objective.target else ""),
                detail=run.objective.rationale,
            )
        return run

    # -- stage 3: plan --------------------------------------------------------
    def plan(self, run: AnalysisRun, objective: Objective | None = None,
             settings: RunSettings | None = None) -> AnalysisRun:
        """Decide the approach: methods, preprocessing, validation, metric."""
        if run.profile is None:
            raise RuntimeError("Call understand() before plan().")
        run.settings = settings or run.settings
        run.objective = objective or run.objective
        if run.objective is None:
            raise RuntimeError("No analytical objective is set.")

        constraints = Constraints(
            max_models=run.settings.max_models,
            time_budget=run.settings.time_budget,
            interpretability_need=run.settings.interpretability_need,
            exclude_models=run.settings.exclude_models,
            include_models=run.settings.include_models,
        )
        with run.trace.start("Selecting candidate methods") as step:
            run.plan = plan_analysis(
                run.profile, run.objective, run.context, constraints,
                self.registry, frame=self._typed_frame,
            )
            step.update(f"{len(run.plan.candidates)} candidate(s) shortlisted")
        run.decisions.extend(run.plan.decisions)
        for warning in run.plan.warnings:
            run.trace.add(warning, status="warning")
            run.warnings.append(warning)

        with run.trace.start("Designing the preprocessing pipeline") as step:
            model_spec = None
            if run.plan.candidates:
                try:
                    model_spec = self.registry.get(run.plan.candidates[0].spec_key)
                except KeyError:
                    model_spec = None
            pipeline, decisions = recommend_pipeline(
                run.profile, run.objective.task_type, run.objective.target, model_spec,
                run.context, aggressiveness=run.settings.preprocessing_aggressiveness,
            )
            run.pipeline = pipeline
            run.decisions.extend(decisions)
            step.update(f"{len(pipeline.active_steps)} step(s)")

        run.status = "planned"
        return run

    # -- stage 4-6: execute ---------------------------------------------------
    def execute(self, run: AnalysisRun) -> AnalysisRun:
        """Train, validate, compare, explain."""
        if run.plan is None or run.objective is None:
            raise RuntimeError("Call plan() before execute().")
        task = run.objective.task_type

        config = ExperimentConfig(
            test_size=run.settings.test_size,
            random_state=run.settings.random_state,
            validation=run.plan.validation_strategy,
            scoring_metrics=[run.plan.primary_metric],
            tune=run.settings.tune_hyperparameters,
        )
        engine = ExperimentEngine(self._typed_frame, run.objective, run.profile, self.registry, config)
        self._engines[run.id] = engine
        run.environment = engine.environment()

        if task is TaskType.ASSOCIATION_RULES:
            return self._execute_association(run)
        if task is TaskType.TIME_SERIES_FORECAST:
            self._prepare_series(run)

        pipelines = self._build_pipelines(run)
        model_keys = run.plan.model_keys()

        with run.trace.start(f"Training and validating {len(model_keys)} model(s)") as step:
            for i, key in enumerate(model_keys, 1):
                spec_name = self.registry.get(key).name if key in self.registry else key
                step.update(f"{i}/{len(model_keys)}: {spec_name}")
                result = engine.run_model(key, pipelines.get(key))
                run.results.append(result)
                status = "done" if result.status == "success" else "failed"
                detail = (
                    M.format_metric(run.plan.primary_metric, result.primary(run.plan.primary_metric))
                    if result.status == "success" else result.error[:120]
                )
                run.trace.add(
                    f"  {spec_name}: {run.plan.primary_metric} = {detail}"
                    if status == "done" else f"  {spec_name} failed: {detail}",
                    status=status,
                    elapsed_s=result.training_time_s,
                )
            step.update(f"{sum(1 for r in run.results if r.status == 'success')}/{len(model_keys)} succeeded")

        with run.trace.start("Comparing models") as step:
            run.tournament = run_tournament(
                run.results, run.plan.primary_metric, task,
                interpretability_need=run.settings.interpretability_need,
                registry=self.registry,
            )
            run.decisions.extend(run.tournament.decisions)
            for warning in run.tournament.warnings:
                run.warnings.append(warning)
            if run.tournament.recommended:
                best_id = run.tournament.recommended.result_id
                run.best = next((r for r in run.results if r.id == best_id), None)
                step.update(f"recommended: {run.tournament.recommended.model_name}")
            else:
                step.update("no model produced a usable result")

        if task is TaskType.CLUSTERING and run.best is not None:
            self._profile_segments(run)
        if run.settings.explain and run.best is not None and task.is_supervised and task is not TaskType.TIME_SERIES_FORECAST:
            self._explain(run, engine)
        return run

    # -- stage 7-8: interpret -------------------------------------------------
    def interpret(self, run: AnalysisRun) -> AnalysisRun:
        """Findings, validation checks, then recommendations."""
        with run.trace.start("Validating the conclusions"):
            run.self_check = check_analysis(
                run.profile, run.objective, run.results, run.best,
                run.plan.primary_metric if run.plan else "",
            )
        failed = [c for c in run.self_check.checks if not c.passed]
        if failed:
            run.trace.add(
                f"{len(failed)} validation check(s) raised something",
                status="warning",
                detail="; ".join(c.question for c in failed[:3]),
            )

        with run.trace.start("Generating findings") as step:
            run.findings = generate_insights(
                run.profile, run.objective, run.context, self._typed_frame,
                run.best, run.explanation, run.tournament, run.diagnostics,
                run.segmentation, run.series_analysis,
            )
            step.update(f"{len(run.findings)} finding(s)")

        with run.trace.start("Generating recommendations") as step:
            run.recommendations = generate_recommendations(
                run.profile, run.objective, run.findings, run.context, run.best,
                run.tournament, run.explanation, run.segmentation, run.series_analysis,
                run.self_check, self._typed_frame,
            )
            if run.associations is not None:
                run.recommendations = list(run.associations.recommendations) + run.recommendations
            step.update(f"{len(run.recommendations)} recommendation(s)")

        run.status = "complete"
        run.duration_s = run.trace.total_seconds
        run.trace.add("Analysis complete", detail=run.summary())
        return run

    # -- the whole thing ------------------------------------------------------
    def analyse(
        self,
        frame: pd.DataFrame,
        name: str = "dataset",
        context: BusinessContext | None = None,
        settings: RunSettings | None = None,
        source: DataSource | None = None,
        objective: Objective | None = None,
    ) -> AnalysisRun:
        """Run the full workflow. In assisted mode it stops after planning."""
        settings = settings or RunSettings()
        run = AnalysisRun(settings=settings)
        try:
            self.understand(frame, name, context, source, run)
            self.plan(run, objective, settings)
            if settings.mode == "assisted":
                run.trace.add(
                    "Waiting for your approval",
                    status="skipped",
                    detail="Review the plan, adjust anything, then call execute() and interpret().",
                )
                return run
            self.execute(run)
            self.interpret(run)
        except Exception as exc:
            run.status = "failed"
            run.trace.add("Analysis failed", status="failed", detail=f"{type(exc).__name__}: {exc}")
            run.warnings.append(f"{type(exc).__name__}: {exc}")
            raise
        return run

    # -- helpers --------------------------------------------------------------
    def _build_pipelines(self, run: AnalysisRun) -> dict[str, PreprocessingPipeline]:
        """One pipeline per model — preprocessing needs differ by algorithm."""
        pipelines: dict[str, PreprocessingPipeline] = {}
        for key in run.plan.model_keys():
            try:
                spec = self.registry.get(key)
            except KeyError:
                continue
            if run.objective.task_type is TaskType.TIME_SERIES_FORECAST:
                continue  # forecasters take the raw series
            pipeline, _ = recommend_pipeline(
                run.profile, run.objective.task_type, run.objective.target, spec,
                run.context, aggressiveness=run.settings.preprocessing_aggressiveness,
            )
            pipelines[key] = pipeline
        return pipelines

    def _prepare_series(self, run: AnalysisRun) -> None:
        objective = run.objective
        if not objective.target:
            return
        with run.trace.start("Analysing the time series") as step:
            frame = self._typed_frame
            if objective.time_column and objective.time_column in frame.columns:
                working = frame[[objective.time_column, objective.target]].dropna()
                working[objective.time_column] = pd.to_datetime(working[objective.time_column], errors="coerce")
                working = working.dropna().sort_values(objective.time_column)
                series = working.set_index(objective.time_column)[objective.target].astype(float)
                if series.index.has_duplicates:
                    series = series.groupby(level=0).mean()
            else:
                series = frame[objective.target].dropna().astype(float).reset_index(drop=True)
            run.series_analysis = analyse_series(series)
            step.update(
                f"trend: {run.series_analysis.get('trend', {}).get('direction', 'n/a')}, "
                f"stationarity: {run.series_analysis.get('stationarity', {}).get('verdict', 'n/a')}"
            )
        for issue in run.series_analysis.get("issues", []):
            run.trace.add(issue[:160], status="warning")
            run.warnings.append(issue)
        period = run.series_analysis.get("seasonal_period")
        if period:
            run.objective.extras["seasonal_period"] = period

    def _profile_segments(self, run: AnalysisRun) -> None:
        labels = run.best.extras.get("labels")
        index = run.best.extras.get("row_index")
        if labels is None:
            return
        with run.trace.start("Profiling the segments") as step:
            frame = self._typed_frame
            subset = frame.loc[index] if index is not None else frame
            feature_columns = [c for c in (run.objective.features or run.profile.modelling_columns)
                               if c in subset.columns]
            run.segmentation = profile_clusters(
                subset, labels, run.profile, feature_columns, currency=run.context.currency
            )
            step.update(f"{run.segmentation.n_clusters} segment(s) described")
        for warning in getattr(run.segmentation, "warnings", []):
            run.trace.add(warning[:160], status="warning")
            run.warnings.append(warning)

    def _explain(self, run: AnalysisRun, engine: ExperimentEngine) -> None:
        model = engine.fitted_model(run.best.id)
        if model is None:
            return
        with run.trace.start("Explaining the model") as step:
            X, y, _ = engine.prepare(run.pipeline)
            X_train, X_test, y_train, y_test = engine._split(X, y)
            sample = X_test if len(X_test) >= 20 else X
            sample_y = y_test if len(X_test) >= 20 else y
            run.explanation = explain_model(
                model, sample, sample_y, run.objective.task_type,
                scoring="r2" if run.objective.task_type is TaskType.REGRESSION else "balanced_accuracy",
                n_repeats=5,
            )
            step.update(f"{run.explanation.method}; top driver: "
                        f"{run.explanation.importances[0].feature if run.explanation.importances else 'n/a'}")

        with run.trace.start("Running model diagnostics"):
            predictions = model.predict(sample)
            if run.objective.task_type is TaskType.REGRESSION:
                run.diagnostics = regression_diagnostics(sample_y, predictions, len(X.columns))
            else:
                proba = model.predict_proba(sample) if hasattr(model, "predict_proba") else None
                run.diagnostics = classification_diagnostics(
                    sample_y, predictions, proba, sorted(pd.unique(y))
                )
        for issue in run.diagnostics.get("issues", []):
            run.trace.add(issue[:160], status="warning")

    def _execute_association(self, run: AnalysisRun) -> AnalysisRun:
        extras = run.objective.extras
        with run.trace.start("Mining association rules") as step:
            run.associations = mine_associations(
                self._typed_frame,
                transaction_column=extras.get("transaction_column"),
                item_column=extras.get("item_column"),
                min_support=extras.get("min_support", 0.05),
                min_confidence=extras.get("min_confidence", 0.3),
            )
            step.update(f"{len(run.associations.rules)} rule(s)")
        for warning in run.associations.warnings:
            run.trace.add(warning[:160], status="warning")
            run.warnings.append(warning)
        run.findings.extend(run.associations.findings)
        return run

    def fitted_model(self, run: AnalysisRun, result_id: str | None = None):
        engine = self._engines.get(run.id)
        if engine is None:
            return None
        return engine.fitted_model(result_id or (run.best.id if run.best else ""))

    def engine_for(self, run: AnalysisRun) -> ExperimentEngine | None:
        return self._engines.get(run.id)
