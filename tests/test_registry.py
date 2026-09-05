"""The registry is the plugin architecture — nothing may hard-code a model list."""

from __future__ import annotations

import pytest

from dsai.core.schema import TaskType
from dsai.registry.base import Cost, Interpretability, ModelRegistry, ModelSpec, REGISTRY


def test_registry_covers_every_task_type():
    stats = REGISTRY.stats()
    assert stats["total"] >= 100, "the platform is meant to support a large library"
    for task in [TaskType.REGRESSION, TaskType.BINARY_CLASSIFICATION, TaskType.CLUSTERING,
                 TaskType.TIME_SERIES_FORECAST, TaskType.ANOMALY_DETECTION,
                 TaskType.DIMENSIONALITY_REDUCTION, TaskType.ASSOCIATION_RULES]:
        assert REGISTRY.find(task_type=task, only_available=True), f"nothing available for {task.value}"


def test_every_spec_documents_itself():
    """Metadata is what the decision engine reasons over, so it cannot be empty."""
    for spec in REGISTRY.all():
        assert spec.name and spec.category and spec.task_types, f"{spec.key} is under-specified"
        assert spec.advantages, f"{spec.key} lists no advantages"
        assert spec.limitations, f"{spec.key} lists no limitations — every model has some"


def test_filtering_respects_constraints():
    transparent = REGISTRY.find(task_type=TaskType.REGRESSION,
                                min_interpretability=Interpretability.TRANSPARENT)
    assert transparent
    assert all(s.interpretability is Interpretability.TRANSPARENT for s in transparent)

    cheap = REGISTRY.find(task_type=TaskType.REGRESSION, max_cost=Cost.LOW)
    assert all(s.cost.rank <= Cost.LOW.rank for s in cheap)

    tiny_data = REGISTRY.find(task_type=TaskType.REGRESSION, n_rows=25)
    assert all(s.min_rows <= 25 for s in tiny_data)


def test_a_new_model_can_be_registered_without_touching_the_platform():
    from dsai.registry._helpers import hp, lazy

    registry = ModelRegistry()
    registry.register(ModelSpec(
        key="third_party_model",
        name="Third-Party Model",
        category="regression",
        family="ensemble",
        task_types=[TaskType.REGRESSION],
        builder=lazy("sklearn.ensemble:RandomForestRegressor"),
        advantages=["provided by a plugin"],
        limitations=["hypothetical"],
        hyperparameters=[hp("n_estimators", "int", 50, 10, 200)],
    ))
    spec = registry.get("third_party_model")
    assert spec.is_available()
    model = spec.build()
    assert model.n_estimators == 50
    assert registry.find(task_type=TaskType.REGRESSION)


def test_builders_are_lazy():
    """Specs for uninstalled libraries must sit in the registry harmlessly."""
    for key in ("catboost_regressor", "umap"):
        if key in REGISTRY:
            spec = REGISTRY.get(key)
            assert isinstance(spec.is_available(), bool)  # must not raise


def test_search_space_is_usable_for_tuning():
    spec = REGISTRY.get("random_forest_regressor")
    space = spec.search_space(n=3)
    assert "n_estimators" in space
    assert all(isinstance(v, list) and v for v in space.values())
    assert "n_jobs" not in space, "fixed parameters must not be searched"


def test_duplicate_registration_is_refused():
    registry = ModelRegistry()
    spec = ModelSpec(key="x", name="X", category="regression", task_types=[TaskType.REGRESSION])
    registry.register(spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)
    registry.register(spec, replace=True)  # explicit replacement is fine
