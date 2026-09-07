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
    BusinessContext, Confidence, DatasetProfile, Decision, EvidenceKind, Finding, JsonMixin,
    Objective, Recommendation, TaskType, TraceEvent,
)
from dsai.dataio.loaders import DataSource
from dsai.engines import metrics as M
from dsai.engines.metrics import human_number
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
    #: Hypothesis-test outcomes, when the objective was hypothesis testing.
    tests: list[dict[str, Any]] = field(default_factory=list)
    #: Descriptive output, when the objective was exploratory.
    exploration: dict[str, Any] = field(default_factory=dict)
    #: The k sweep, when the number of segments was not fixed by the user.
    cluster_sweep: dict[str, Any] = field(default_factory=dict)
    #: Flagged rows with a reason each, when the objective was anomaly detection.
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    #: The plan as a reviewable brief, with a verdict on whether this data can
    #: defensibly answer this question. Written before the expensive work.
    brief: Any = None
    #: What this analysis requires of its data, re-checked when new rows arrive.
    contract: Any = None
    #: How much weight the result can carry, and what is holding it up.
    trust: Any = None
    #: Every claim's supporting evidence, with identifiers that survive export.
    ledger: Any = None

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

        # The brief and the contract are written before the expensive work, so
        # a reader can decide whether it is worth running — and so the same
        # requirements can be checked again when new data arrives to be scored.
        with run.trace.start("Writing the analysis brief") as step:
            from dsai.engines.brief import build_brief, build_contract

            run.brief = build_brief(run.profile, run.objective, run.plan, run.context,
                                    self._typed_frame)
            run.contract = build_contract(run.profile, run.objective, self._typed_frame)
            step.update(f"{run.brief.verdict_label.lower()}; "
                        f"{len(run.contract.checks)} data requirement(s) checked")
        if run.brief.verdict == "unsuitable":
            for reason in run.brief.verdict_reasons:
                run.trace.add(reason, status="warning")
        for check in run.contract.checks:
            if not check.passed:
                run.trace.add(f"Data contract: {check.requirement} — {check.detail}",
                              status="warning")

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
        if task is TaskType.HYPOTHESIS_TESTING:
            return self._execute_hypothesis_tests(run)
        if task is TaskType.EXPLORATORY:
            return self._execute_exploratory(run)
        if task is TaskType.DIMENSIONALITY_REDUCTION:
            return self._execute_dimensionality(run)
        if task is TaskType.TIME_SERIES_FORECAST:
            self._prepare_series(run)

        if task is TaskType.CLUSTERING and not run.objective.n_clusters:
            self._suggest_cluster_count(run)

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
        if task is TaskType.ANOMALY_DETECTION and run.best is not None:
            self._profile_anomalies(run)
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
            # Anything the execution stage already established — hypothesis
            # tests, association rules, variance retained — is kept and the
            # profile-level findings are added to it.
            established = list(run.findings)
            generated = generate_insights(
                run.profile, run.objective, run.context, self._typed_frame,
                run.best, run.explanation, run.tournament, run.diagnostics,
                run.segmentation, run.series_analysis,
            )
            seen = {f.title for f in established}
            run.findings = established + [f for f in generated if f.title not in seen]
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
        # Assembled last, from the findings and checks the engines produced, so
        # they cannot disagree with what the report says.
        with run.trace.start("Assessing how far to trust this") as step:
            from dsai.core.evidence import build_ledger
            from dsai.engines.trust import assess_trust

            run.trust = assess_trust(run)
            run.ledger = build_ledger(run)
            step.update(f"evidence strength {run.trust.score}/100 ({run.trust.band}); "
                        f"{len(run.ledger.items)} evidence item(s)")

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

    def _suggest_cluster_count(self, run: AnalysisRun) -> None:
        """Sweep k and record how many internal measures agree on the answer.

        No single measure settles the number of clusters. Agreement between four
        of them is a real signal; disagreement means the data has no sharp
        structure and the choice belongs to the user, not to a metric.
        """
        from sklearn.preprocessing import StandardScaler

        features = [c for c in (run.objective.features or run.profile.modelling_columns)
                    if c in run.profile.numeric_columns]
        if len(features) < 2:
            return
        matrix = self._typed_frame[features].apply(pd.to_numeric, errors="coerce").dropna()
        if len(matrix) < 20:
            return

        with run.trace.start("Choosing the number of segments") as step:
            scaled = StandardScaler().fit_transform(matrix)
            run.cluster_sweep = suggest_cluster_count(scaled)
            if run.cluster_sweep.get("supported"):
                k = run.cluster_sweep["recommended_k"]
                run.objective.n_clusters = k
                step.update(f"k = {k} ({run.cluster_sweep['agreement']} agree)")
                run.decisions.append(
                    Decision(
                        stage="clustering",
                        decision=f"Use {k} segments",
                        reason=run.cluster_sweep["interpretation"],
                        evidence=[
                            f"Elbow: {run.cluster_sweep['elbow_k']}",
                            f"Best silhouette: {run.cluster_sweep['best_silhouette_k']}",
                            f"Best Calinski-Harabasz: {run.cluster_sweep['best_calinski_k']}",
                            f"Best Davies-Bouldin: {run.cluster_sweep['best_davies_bouldin_k']}",
                        ],
                        confidence={
                            "high": Confidence.HIGH, "moderate": Confidence.MODERATE,
                        }.get(run.cluster_sweep["confidence"], Confidence.LOW),
                    )
                )
            else:
                step.update("could not sweep")

    def _profile_anomalies(self, run: AnalysisRun) -> None:
        """Turn anomaly flags into rows a person can review, with a reason each."""
        best = run.best
        flags = best.extras.get("flags")
        index = best.extras.get("row_index")
        if not flags or index is None:
            return

        with run.trace.start("Explaining the flagged rows") as step:
            frame = self._typed_frame.loc[index]
            scores = best.extras.get("scores")
            flagged = [i for i, f in enumerate(flags) if f == -1]
            if scores:
                flagged.sort(key=lambda i: scores[i])
            numeric = [c for c in best.features if c in frame.columns
                       and c in run.profile.numeric_columns]
            means = frame[numeric].mean() if numeric else None
            spread = frame[numeric].std(ddof=0).replace(0, 1.0) if numeric else None

            rows = []
            for position in flagged[:200]:
                row = frame.iloc[position]
                reasons = []
                if numeric:
                    deviation = ((row[numeric] - means) / spread).astype(float)
                    for column in deviation.abs().sort_values(ascending=False).head(3).index:
                        value = float(deviation[column])
                        if abs(value) < 1.5:
                            continue
                        reasons.append(
                            f"{column} is {abs(value):.1f} SD "
                            f"{'above' if value > 0 else 'below'} average "
                            f"({human_number(row[column])} vs {human_number(means[column])})"
                        )
                rows.append({
                    "row": index[position],
                    "score": round(float(scores[position]), 4) if scores else None,
                    "why": "; ".join(reasons) or "unusual in combination rather than on any one variable",
                })
            run.anomalies = rows
            step.update(f"{len(rows)} row(s) described")

        if rows:
            run.findings.append(
                Finding(
                    title=f"{int(best.test_scores.get('n_anomalies', 0))} row(s) look unlike the rest",
                    detail=(
                        f"{best.test_scores.get('anomaly_rate', 0):.1%} of rows were flagged by "
                        f"{best.model_name}. Unusual is not the same as wrong — these are the rows "
                        "worth a human looking at, not rows to delete."
                    ),
                    kind=EvidenceKind.MODEL,
                    evidence=[r["why"] for r in rows[:4]],
                    confidence=Confidence.MODERATE,
                    caveats=[
                        "There are no labels here, so 'anomalous' means 'unlike the rest of this "
                        "data'. Whether a flagged row is actually a problem is your judgement.",
                    ],
                )
            )

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
            # Kept so residual and predicted-vs-actual charts can be drawn later
            # without refitting or re-splitting. The index goes with them so
            # errors can be joined back to the rows they came from — which is
            # what subgroup performance needs.
            run.best.extras["holdout_index"] = [
                v.item() if hasattr(v, "item") else v for v in sample.index
            ]
            if run.objective.task_type is TaskType.REGRESSION:
                run.best.extras["holdout_actual"] = [float(v) for v in np.asarray(sample_y, dtype=float)]
            else:
                run.best.extras["holdout_actual"] = [str(v) for v in np.asarray(sample_y)]
            run.best.extras["holdout_predicted"] = [float(v) for v in np.asarray(predictions, dtype=float)] \
                if run.objective.task_type is TaskType.REGRESSION else None
            if run.objective.task_type.is_classification:
                run.best.extras["holdout_predicted_labels"] = [str(v) for v in np.asarray(predictions)]
                if hasattr(model, "predict_proba"):
                    try:
                        proba_values = np.asarray(model.predict_proba(sample))
                        if proba_values.ndim == 2 and proba_values.shape[1] == 2:
                            run.best.extras["holdout_score"] = [float(v) for v in proba_values[:, 1]]
                            run.best.extras["holdout_positive_label"] = str(sorted(pd.unique(y))[-1])
                    except Exception:
                        pass
            if run.objective.task_type is TaskType.REGRESSION:
                run.diagnostics = regression_diagnostics(sample_y, predictions, len(X.columns))
            else:
                proba = model.predict_proba(sample) if hasattr(model, "predict_proba") else None
                run.diagnostics = classification_diagnostics(
                    sample_y, predictions, proba, sorted(pd.unique(y))
                )
        for issue in run.diagnostics.get("issues", []):
            run.trace.add(issue[:160], status="warning")

    def _execute_dimensionality(self, run: AnalysisRun) -> AnalysisRun:
        """Project the numeric columns down and report what each component carries.

        Unlike the supervised path there is no held-out score to compare against;
        the useful questions are how much variance survives and what the
        components are made of.
        """
        from sklearn.preprocessing import StandardScaler

        objective = run.objective
        features = [c for c in (objective.features or run.profile.numeric_columns)
                    if c in self._typed_frame.columns]
        features = [c for c in features if c in run.profile.numeric_columns]
        if len(features) < 2:
            run.warnings.append(
                "Dimensionality reduction needs at least two numeric variables; "
                f"{len(features)} were available."
            )
            run.trace.add("Not enough numeric variables to reduce", status="failed")
            return run

        matrix = self._typed_frame[features].apply(pd.to_numeric, errors="coerce").dropna()
        with run.trace.start(f"Reducing {len(features)} numeric variable(s)") as step:
            scaled = StandardScaler().fit_transform(matrix)
            results = []
            for key in run.plan.model_keys():
                try:
                    spec = self.registry.get(key)
                except KeyError:
                    continue
                result = ExperimentResult(
                    model_key=key, model_name=spec.name, task_type=TaskType.DIMENSIONALITY_REDUCTION,
                    features=features, n_train=len(matrix),
                    interpretability=spec.interpretability.value, cost=spec.cost.value,
                    family=spec.family, random_seed=run.settings.random_state,
                )
                started = time.perf_counter()
                try:
                    reducer = spec.build(n_components=min(2, len(features)))
                    projected = reducer.fit_transform(scaled)
                    result.extras["projection"] = np.asarray(projected)[:5000].tolist()
                    result.extras["row_index"] = list(matrix.index[:5000])
                    result.n_features_out = int(np.asarray(projected).shape[1])
                    ratio = getattr(reducer, "explained_variance_ratio_", None)
                    if ratio is not None:
                        result.test_scores["explained_variance_ratio"] = float(np.sum(ratio))
                        result.extras["per_component"] = [float(v) for v in ratio]
                    loadings = getattr(reducer, "components_", None)
                    if loadings is not None:
                        result.extras["loadings"] = {
                            f"component_{i + 1}": {
                                features[j]: round(float(loadings[i][j]), 4)
                                for j in range(min(len(features), loadings.shape[1]))
                            }
                            for i in range(min(len(loadings), 4))
                        }
                    result.hyperparameters = spec.default_params()
                except Exception as exc:
                    result.status = "failed"
                    result.error = f"{type(exc).__name__}: {exc}"
                result.training_time_s = round(time.perf_counter() - started, 4)
                results.append(result)
                run.trace.add(
                    f"  {spec.name}: "
                    + (f"{result.test_scores.get('explained_variance_ratio', float('nan')):.1%} "
                       "of variance retained" if result.status == "success" else result.error[:90]),
                    status="done" if result.status == "success" else "failed",
                )
            run.results = results
            step.update(f"{sum(1 for r in results if r.status == 'success')}/{len(results)} succeeded")

        succeeded = [r for r in results if r.status == "success"]
        if succeeded:
            # Prefer a method that can report how much variance it kept; a
            # projection you cannot quantify is harder to justify.
            run.best = max(
                succeeded,
                key=lambda r: (
                    "explained_variance_ratio" in r.test_scores,
                    r.test_scores.get("explained_variance_ratio", 0.0),
                ),
            )
            retained = run.best.test_scores.get("explained_variance_ratio")
            if retained is None:
                run.findings.append(
                    Finding(
                        title=f"{run.best.model_name} projected {len(features)} variables to "
                              f"{run.best.n_features_out} dimensions",
                        detail=(
                            "This method does not report how much of the original variation it "
                            "kept, because it optimises local neighbourhood structure rather than "
                            "variance. Use the projection to look for grouping; do not read "
                            "distances, cluster sizes or axis values off it, and do not feed it "
                            "into a model as if it were a faithful summary."
                        ),
                        kind=EvidenceKind.MODEL,
                        evidence=[f"{run.best.model_name}",
                                  f"{len(features)} variables → {run.best.n_features_out} dimensions"],
                        columns=features,
                        confidence=Confidence.MODERATE,
                        caveats=[
                            "Manifold projections are for visualisation. A different random seed "
                            "or parameter setting produces a different picture of the same data.",
                        ],
                    )
                )
            if retained is not None:
                run.findings.append(
                    Finding(
                        title=f"{run.best.n_features_out} component(s) retain "
                              f"{retained:.0%} of the variation in {len(features)} variables",
                        detail=(
                            "The components are blends of the original variables, so this buys "
                            "a compact representation at the cost of being able to name what drives "
                            "what. Use it to handle collinearity or to visualise structure, not to "
                            "explain the outcome."
                        ),
                        kind=EvidenceKind.MODEL,
                        evidence=[f"{run.best.model_name}"] + [
                            f"component_{i + 1}: {v:.1%}"
                            for i, v in enumerate(run.best.extras.get("per_component", [])[:4])
                        ],
                        columns=features,
                        confidence=Confidence.HIGH,
                    )
                )
        return run

    def _execute_hypothesis_tests(self, run: AnalysisRun) -> AnalysisRun:
        """Compare a numeric measure across the levels of each categorical variable.

        Assumptions are checked first, the non-parametric equivalent is run as a
        cross-check, and the p-values are corrected for the number of tests —
        without which twenty comparisons at 0.05 produce a false positive about
        two-thirds of the time.
        """
        from dsai.statistics import tests as T

        objective = run.objective
        frame = self._typed_frame
        value = objective.target or (run.profile.numeric_columns[0]
                                     if run.profile.numeric_columns else None)
        if value is None:
            run.warnings.append("Hypothesis testing needs a numeric measure to compare.")
            run.trace.add("No numeric variable to test", status="failed")
            return run

        groups = [
            c for c in (objective.features or run.profile.categorical_columns)
            if c in run.profile.columns
            and run.profile.columns[c].is_categorical
            and 2 <= run.profile.columns[c].n_unique <= 20
        ]
        if not groups:
            run.warnings.append(
                "No categorical variable with between 2 and 20 levels was available to group by."
            )
            run.trace.add("No grouping variable available", status="failed")
            return run

        outcomes = []
        with run.trace.start(f"Comparing '{value}' across {len(groups)} grouping variable(s)") as step:
            for column in groups:
                n_levels = int(frame[column].nunique())
                if n_levels == 2:
                    levels = list(frame[column].dropna().unique())[:2]
                    a = frame.loc[frame[column] == levels[0], value]
                    b = frame.loc[frame[column] == levels[1], value]
                    primary = T.t_test(a, b, names=(str(levels[0]), str(levels[1])))
                    secondary = T.mann_whitney(a, b, names=(str(levels[0]), str(levels[1])))
                else:
                    primary = T.anova(frame, value, column)
                    secondary = T.kruskal_wallis(frame, value, column)
                outcomes.append({"group": column, "primary": primary, "secondary": secondary})
                run.trace.add(
                    f"  {value} by {column}: p = {primary.p_value:.4g}"
                    + (" (significant)" if primary.significant else ""),
                )
            step.update(f"{len(outcomes)} comparison(s)")

        correction = T.multiple_comparison_correction(
            [o["primary"].p_value for o in outcomes], method="holm"
        )
        for outcome, adjusted, still in zip(outcomes, correction["adjusted"], correction["significant"]):
            outcome["adjusted_p"] = adjusted
            outcome["significant_after_correction"] = still
        run.tests = outcomes

        for outcome in outcomes:
            primary, secondary = outcome["primary"], outcome["secondary"]
            agrees = primary.significant == secondary.significant
            run.findings.append(
                Finding(
                    title=(
                        f"'{value}' differs across '{outcome['group']}'"
                        if outcome["significant_after_correction"]
                        else f"No reliable difference in '{value}' across '{outcome['group']}'"
                    ),
                    detail=primary.conclusion + " " + primary.practical_note,
                    kind=EvidenceKind.STATISTICAL,
                    evidence=[
                        f"{primary.test}: statistic {primary.statistic:.4f}, p = {primary.p_value:.4g}",
                        f"Adjusted for {correction['n_tests']} comparison(s): p = {outcome['adjusted_p']:.4g}",
                        f"Effect size ({primary.effect_size_name}) = {primary.effect_size:.3f} "
                        f"— {primary.effect_interpretation}",
                        f"Non-parametric cross-check ({secondary.test}) "
                        + ("agrees" if agrees else "disagrees"),
                    ],
                    columns=[value, outcome["group"]],
                    confidence=(
                        Confidence.HIGH if outcome["significant_after_correction"] and agrees
                        else Confidence.MODERATE if agrees else Confidence.LOW
                    ),
                    caveats=(primary.assumption_warnings or [])
                    + ([] if agrees else [
                        "The parametric and non-parametric tests disagree, which usually means an "
                        "assumption is violated. Trust the non-parametric result."
                    ]),
                )
            )
        if correction["n_tests"] > 1:
            run.warnings.append(correction["note"])
        return run

    def _execute_exploratory(self, run: AnalysisRun) -> AnalysisRun:
        """Describe the data and the relationships in it, without fitting a model."""
        from dsai.statistics.descriptive import correlation_pairs, outlier_table

        frame = self._typed_frame
        with run.trace.start("Describing distributions and relationships") as step:
            numeric = run.profile.numeric_columns
            pairs = correlation_pairs(frame, columns=numeric, min_abs=0.25) if len(numeric) >= 2 \
                else pd.DataFrame()
            outliers = outlier_table(frame, numeric) if numeric else pd.DataFrame()
            run.exploration = {
                "correlations": pairs.head(25).to_dict("records") if not pairs.empty else [],
                "outliers": outliers.to_dict("records") if not outliers.empty else [],
            }
            step.update(
                f"{len(run.exploration['correlations'])} notable relationship(s)"
            )

        if len(numeric) >= 2 and not run.exploration["correlations"]:
            run.findings.append(
                Finding(
                    title="No numeric variables move together to any useful degree",
                    detail=(
                        f"Across {len(numeric)} numeric variable(s), no pair correlates above 0.25. "
                        "They carry largely independent information, which is good for modelling — "
                        "there is little redundancy — but it also means no single numeric variable "
                        "stands in for another. If you expected a relationship here, it may be "
                        "nonlinear, or it may run through a categorical variable instead."
                    ),
                    kind=EvidenceKind.STATISTICAL,
                    evidence=[f"{len(numeric)} numeric variables tested pairwise",
                              "Strongest absolute correlation below 0.25"],
                    columns=numeric,
                    confidence=Confidence.HIGH,
                )
            )

        heavy = [r for r in run.exploration["outliers"] if r.get("pct_outliers", 0) >= 5][:3]
        for row in heavy:
            run.findings.append(
                Finding(
                    title=f"'{row['variable']}' has a long tail",
                    detail=(
                        f"{row['n_outliers']} value(s) — {row['pct_outliers']:.1f}% — sit outside "
                        f"the {human_number(row['lower_bound'])} to {human_number(row['upper_bound'])} range that the "
                        "interquartile fences mark as typical. That is normal for money and counts, "
                        "but it means the average is not a typical value."
                    ),
                    kind=EvidenceKind.OBSERVED,
                    evidence=[
                        f"Largest value: {human_number(row['max_outlier'])}",
                        f"Typical range: {human_number(row['lower_bound'])} to {human_number(row['upper_bound'])}",
                    ],
                    columns=[row["variable"]],
                    confidence=Confidence.HIGH,
                )
            )

        for row in run.exploration["correlations"][:5]:
            run.findings.append(
                Finding(
                    title=f"'{row['variable_1']}' and '{row['variable_2']}' move together",
                    detail=(
                        f"{row['strength'].capitalize()} relationship (r = {row['coefficient']:.3f}) "
                        f"across {row['n']:,} rows. They share "
                        f"{row['coefficient'] ** 2:.0%} of their variation."
                    ),
                    kind=EvidenceKind.STATISTICAL,
                    evidence=[
                        f"Pearson r = {row['coefficient']:.4f}",
                        f"p = {row.get('p_value', float('nan')):.4g}",
                        f"n = {row['n']:,}",
                    ],
                    columns=[row["variable_1"], row["variable_2"]],
                    confidence=Confidence.HIGH if row.get("significant_at_5pct") else Confidence.LOW,
                    caveats=["Correlation is not causation."],
                )
            )
        return run

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
