"""Preprocessing must be editable, replayable, and structurally leak-free."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.core.profiler import profile_dataset
from dsai.core.schema import TaskType
from dsai.preprocessing.pipeline import PreprocessingPipeline
from dsai.preprocessing.recommender import recommend_pipeline
from dsai.preprocessing.steps import STEPS
from dsai.registry.base import REGISTRY


def test_pipeline_is_editable_and_reversible():
    pipeline = PreprocessingPipeline("test")
    first = pipeline.add("impute_median", columns=["a"])
    second = pipeline.add("standard_scale", columns=["a"])
    assert [s.id for s in pipeline.steps] == [first.id, second.id]

    pipeline.move(second.id, 0)
    assert [s.id for s in pipeline.steps] == [second.id, first.id]

    assert pipeline.undo()
    assert [s.id for s in pipeline.steps] == [first.id, second.id]
    assert pipeline.redo()
    assert [s.id for s in pipeline.steps] == [second.id, first.id]

    pipeline.set_enabled(first.id, False)
    assert len(pipeline.active_steps) == 1
    assert pipeline.remove(second.id)
    assert len(pipeline.steps) == 1


def test_pipeline_round_trips_through_json():
    pipeline = PreprocessingPipeline("original")
    pipeline.add("impute_median", columns=["a", "b"])
    pipeline.add("one_hot_encode", columns=["c"], drop="first")
    restored = PreprocessingPipeline.from_dict(pipeline.to_dict())
    assert [s.step_key for s in restored.steps] == [s.step_key for s in pipeline.steps]
    assert restored.steps[1].params["drop"] == "first"


def test_column_names_survive_the_whole_pipeline():
    """Explanations name variables, so names must not be lost in transit."""
    frame = pd.DataFrame({
        "amount": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "grade": ["a", "b", "a", "c", "b", "a"],
    })
    pipeline = PreprocessingPipeline("named")
    pipeline.add("impute_median", columns=["amount"])
    pipeline.add("standard_scale", columns=["amount"])
    pipeline.add("one_hot_encode", columns=["grade"])
    result = pipeline.build_sklearn_pipeline().fit_transform(frame)
    assert isinstance(result, pd.DataFrame)
    assert "amount" in result.columns
    assert any(c.startswith("grade_") for c in result.columns)


def test_row_steps_are_separated_from_fitted_steps():
    pipeline = PreprocessingPipeline("mixed")
    pipeline.add("drop_duplicates")
    pipeline.add("standard_scale", columns=["a"])
    assert [s.step_key for s in pipeline.row_steps] == ["drop_duplicates"]
    assert [s.step_key for s in pipeline.column_steps] == ["standard_scale"]

    report = pipeline.leakage_report()
    assert "Standardise (z-score)" in report["fitted_inside_cross_validation"]
    assert "Remove duplicate rows" in report["applied_before_split"]


def test_target_encoding_is_never_marked_leakage_safe():
    """Target encoding fitted outside the fold leaks badly; the metadata must say so."""
    assert STEPS.get("target_encode").leakage_safe is False
    assert STEPS.get("target_encode").needs_target is True


def test_target_encoder_only_sees_training_rows():
    """Fitted on a training fold, unseen categories fall back to the global mean.

    If it had seen the held-out rows it would return their own means instead.
    """
    from dsai.preprocessing.transformers import TargetEncoder

    train = pd.DataFrame({"city": ["a", "a", "b", "b"]})
    target = pd.Series([10.0, 12.0, 100.0, 102.0])
    encoder = TargetEncoder(smoothing=0.1).fit(train, target)

    unseen = pd.DataFrame({"city": ["z"]})
    encoded = encoder.transform(unseen).iloc[0, 0]
    assert encoded == pytest.approx(float(target.mean()))


def test_recommendation_adapts_to_the_model(regression_frame):
    profile, _ = profile_dataset(regression_frame, target="annual_spend")

    forest, _ = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend",
                                   REGISTRY.get("random_forest_regressor"))
    knn, _ = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend",
                                REGISTRY.get("knn_regressor"))

    forest_steps = {s.step_key for s in forest.active_steps}
    knn_steps = {s.step_key for s in knn.active_steps}

    scaling = {"standard_scale", "robust_scale", "minmax_scale"}
    assert not (forest_steps & scaling), "a tree model does not need scaling"
    assert knn_steps & scaling, "a distance-based model does"


def test_recommendation_drops_identifiers_and_constants(messy_frame):
    profile, _ = profile_dataset(messy_frame, target="value")
    pipeline, decisions = recommend_pipeline(profile, TaskType.REGRESSION, "value")
    dropped = {c for step in pipeline.active_steps if step.step_key == "drop_columns"
               for c in (step.columns or [])}
    assert "constant_column" in dropped
    assert any("Drop" in d.decision for d in decisions)


def test_every_decision_carries_a_reason(regression_frame):
    profile, _ = profile_dataset(regression_frame, target="annual_spend")
    _, decisions = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend",
                                      REGISTRY.get("knn_regressor"))
    assert decisions
    for decision in decisions:
        assert decision.reason.strip(), f"'{decision.decision}' was recorded without a reason"


def test_preview_reports_each_step(regression_frame):
    profile, typed = profile_dataset(regression_frame, target="annual_spend")
    pipeline, _ = recommend_pipeline(profile, TaskType.REGRESSION, "annual_spend",
                                     REGISTRY.get("knn_regressor"))
    preview = pipeline.preview(typed.drop(columns=["annual_spend"]), typed["annual_spend"])
    assert preview["stages"]
    assert all(stage.get("status") != "failed" for stage in preview["stages"])
    assert preview["final_shape"][1] > 0
