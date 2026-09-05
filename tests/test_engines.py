"""The engines: objective detection, model selection, experimentation, ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.core.profiler import profile_dataset
from dsai.core.schema import BusinessContext, Confidence, Objective, TaskType
from dsai.engines.decision import Constraints, assess_situation, plan_analysis
from dsai.engines.experiment import ExperimentConfig, ExperimentEngine
from dsai.engines.metrics import is_better, regression_metrics
from dsai.engines.objective import detect_objectives
from dsai.engines.selection import run_tournament
from dsai.preprocessing.recommender import recommend_pipeline
from dsai.registry.base import REGISTRY


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("predict annual spend for each customer", TaskType.REGRESSION),
    ("segment these customers into five groups", TaskType.CLUSTERING),
    ("find unusual customers", TaskType.ANOMALY_DETECTION),
    ("which customers are likely to churn", TaskType.BINARY_CLASSIFICATION),
])
def test_stated_objective_wins(question, expected, regression_frame):
    frame = regression_frame.copy()
    frame["churned"] = np.random.default_rng(0).choice([0, 1], len(frame))
    profile, _ = profile_dataset(frame)
    objectives = detect_objectives(profile, BusinessContext(stated_objective=question))
    assert objectives[0].task_type is expected
    assert objectives[0].source.startswith("user")


def test_objective_falls_back_to_the_data_when_nothing_is_stated(regression_frame):
    profile, _ = profile_dataset(regression_frame)
    objectives = detect_objectives(profile)
    assert objectives
    assert all(o.source == "ai_detected" for o in objectives)
    assert any(o.target == "annual_spend" for o in objectives)


def test_cluster_count_is_extracted(regression_frame):
    profile, _ = profile_dataset(regression_frame)
    objective = detect_objectives(profile, BusinessContext(stated_objective="create five customer segments"))[0]
    assert objective.n_clusters == 5


def test_a_stated_task_the_target_cannot_support_is_corrected(regression_frame):
    """Asking to 'classify' a continuous target must become regression."""
    profile, _ = profile_dataset(regression_frame)
    context = BusinessContext(stated_objective="classify customers by annual spend",
                              target_variable="annual_spend")
    objective = detect_objectives(profile, context)[0]
    assert objective.task_type is TaskType.REGRESSION


# --------------------------------------------------------------------------
# decision engine
# --------------------------------------------------------------------------

def test_situation_reflects_the_data(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    situation = assess_situation(profile, objective, typed)
    assert situation.n_rows == len(regression_frame)
    assert situation.has_categorical
    assert not situation.tiny


def test_plan_spans_model_families_and_keeps_a_baseline(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    plan = plan_analysis(profile, objective, None, Constraints(max_models=8), frame=typed)

    families = {c.family for c in plan.candidates}
    assert len(families) >= 3, "the shortlist must not be five flavours of one family"
    assert any(REGISTRY.get(c.spec_key).baseline for c in plan.candidates), "a baseline must always run"


def test_interpretability_constraint_is_respected(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    plan = plan_analysis(profile, objective, None,
                         Constraints(max_models=6, interpretability_need="critical"), frame=typed)
    assert all(c.interpretability == "transparent" for c in plan.candidates)


def test_time_series_never_gets_random_folds(series_frame):
    profile, typed = profile_dataset(series_frame, target="sales")
    objective = Objective(task_type=TaskType.TIME_SERIES_FORECAST, target="sales", time_column="month")
    plan = plan_analysis(profile, objective, None, Constraints(), frame=typed)
    assert plan.validation_strategy["strategy"] == "time_series_split"
    assert "future" in plan.validation_strategy["reason"].lower()


def test_severe_imbalance_changes_the_metric():
    rng = np.random.default_rng(3)
    n = 800
    frame = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    frame["rare_event"] = (rng.random(n) < 0.02).astype(int)
    profile, typed = profile_dataset(frame, target="rare_event")
    objective = Objective(task_type=TaskType.BINARY_CLASSIFICATION, target="rare_event")
    plan = plan_analysis(profile, objective, None, Constraints(), frame=typed)
    assert plan.primary_metric == "pr_auc"
    assert any("minority" in w.lower() or "imbalance" in w.lower() for w in plan.warnings)


def test_glm_is_deprioritised_when_the_target_breaks_its_assumption():
    """Gamma regression needs a strictly positive target."""
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"x": rng.normal(size=400), "z": rng.normal(size=400)})
    frame["balance"] = frame.x * 100 + rng.normal(0, 50, 400)  # spans negative values
    profile, typed = profile_dataset(frame, target="balance")
    objective = Objective(task_type=TaskType.REGRESSION, target="balance")
    plan = plan_analysis(profile, objective, None, Constraints(max_models=20), frame=typed)
    gamma = next((c for c in plan.candidates if c.spec_key == "gamma_regression"), None)
    if gamma is not None:
        assert any("positive" in concern for concern in gamma.concerns)


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def test_regression_experiment_learns_the_relationship(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    pipeline, _ = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend",
                                     REGISTRY.get("ridge"))
    result = engine.run_model("ridge", pipeline)

    assert result.status == "success"
    assert result.primary("r2") > 0.5, "the relationship in this fixture is strong and linear"
    assert result.validation.n_splits >= 2
    assert result.n_train > result.n_test > 0


def test_baseline_scores_about_zero(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    result = engine.run_model("dummy_regressor")
    assert result.status == "success"
    assert result.primary("r2") < 0.05


def test_overfitting_is_flagged():
    """A deep tree on tiny noisy data must be reported as overfitting."""
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(60, 8)), columns=[f"x{i}" for i in range(8)])
    frame["y"] = rng.normal(size=60)  # pure noise: nothing to learn
    profile, typed = profile_dataset(frame, target="y")
    objective = Objective(task_type=TaskType.REGRESSION, target="y")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    result = engine.run_model("decision_tree_regressor", hyperparameters={"max_depth": None})
    assert result.overfitting_gap is not None and result.overfitting_gap > 0.2
    assert any("overfit" in w.lower() for w in result.warnings)


def test_classification_experiment_and_probabilities(classification_frame):
    profile, typed = profile_dataset(classification_frame, target="churned")
    objective = Objective(task_type=TaskType.BINARY_CLASSIFICATION, target="churned")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    pipeline, _ = recommend_pipeline(profile, TaskType.BINARY_CLASSIFICATION, "churned",
                                     REGISTRY.get("logistic_regression"))
    result = engine.run_model("logistic_regression", pipeline)
    assert result.status == "success"
    assert result.primary("roc_auc") > 0.6


def test_clustering_experiment_finds_the_planted_groups(cluster_frame):
    profile, typed = profile_dataset(cluster_frame)
    objective = Objective(task_type=TaskType.CLUSTERING, features=list(cluster_frame.columns))
    engine = ExperimentEngine(typed, objective, profile, REGISTRY,
                              ExperimentConfig(cross_validate=False))
    pipeline, _ = recommend_pipeline(profile, TaskType.CLUSTERING, None, REGISTRY.get("kmeans"))
    result = engine.run_model("kmeans", pipeline, {"n_clusters": 3})
    assert result.status == "success"
    assert result.test_scores["n_clusters"] == 3
    assert result.test_scores["silhouette"] > 0.4, "the fixture's groups are well separated"


def test_forecast_experiment_beats_the_naive_baseline(series_frame):
    profile, typed = profile_dataset(series_frame, target="sales")
    objective = Objective(task_type=TaskType.TIME_SERIES_FORECAST, target="sales",
                          time_column="month", horizon=12)
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    result = engine.run_model("holt_winters")
    assert result.status == "success"
    assert result.test_scores["mase"] < 1.0, "Holt-Winters should beat naive on a seasonal trend"
    assert result.extras["forecast"]


def test_a_failed_model_does_not_stop_the_run(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="nonexistent_column")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    result = engine.run_model("ridge")
    assert result.status == "failed"
    assert result.error


def test_unencoded_categoricals_produce_a_useful_error(regression_frame):
    """Without a pipeline the failure must name the problem, not surface a
    scikit-learn dtype error from three layers down."""
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    result = engine.run_model("ridge")  # no pipeline supplied
    assert result.status == "failed"
    assert "needs numeric input" in result.error
    assert "recommend_pipeline" in result.error


# --------------------------------------------------------------------------
# tournament
# --------------------------------------------------------------------------

def test_tournament_ranks_and_explains(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    for key in ["ridge", "random_forest_regressor", "dummy_regressor"]:
        pipeline, _ = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend", REGISTRY.get(key))
        engine.run_model(key, pipeline)

    outcome = run_tournament(engine.results, "r2", TaskType.REGRESSION)
    assert outcome.recommended is not None
    assert outcome.baseline is not None
    assert outcome.recommended.model_key != "dummy_regressor"
    assert all(np.isfinite(r.composite) for r in outcome.ranked), "composite scores must never be NaN"
    assert outcome.decisions
    assert "not best in general" in outcome.decisions[0].reason


def test_tournament_composite_is_finite_for_unbounded_metrics(regression_frame):
    """RMSE has no natural ceiling; it must still produce a comparable score."""
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    objective = Objective(task_type=TaskType.REGRESSION, target="annual_spend")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    for key in ["ridge", "lasso", "dummy_regressor"]:
        pipeline, _ = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend", REGISTRY.get(key))
        engine.run_model(key, pipeline)
    outcome = run_tournament(engine.results, "rmse", TaskType.REGRESSION)
    assert all(np.isfinite(r.composite) for r in outcome.ranked)
    # the baseline has the worst RMSE, so it must not be recommended
    assert outcome.recommended.model_key != "dummy_regressor"


def test_tournament_warns_when_nothing_beats_the_baseline():
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    frame["y"] = rng.normal(size=300)  # no signal at all
    profile, typed = profile_dataset(frame, target="y")
    objective = Objective(task_type=TaskType.REGRESSION, target="y")
    engine = ExperimentEngine(typed, objective, profile, REGISTRY, ExperimentConfig())
    for key in ["ridge", "random_forest_regressor", "dummy_regressor"]:
        pipeline, _ = recommend_pipeline(profile, TaskType.REGRESSION, "y", REGISTRY.get(key))
        engine.run_model(key, pipeline)
    outcome = run_tournament(engine.results, "r2", TaskType.REGRESSION)
    assert any("baseline" in w.lower() for w in outcome.warnings)


def test_metric_direction_is_handled():
    assert is_better("r2", 0.8, 0.5)
    assert is_better("rmse", 100, 200)
    assert not is_better("rmse", 200, 100)
