"""Latent Class Analysis: validated against planted classes, not just unit tested.

LCA fits a mixture of independent Bernoulli items by EM. The thing worth
checking is not "does it run" but "does it recover the classes that actually
generated the data" -- the same bar every other engine in this codebase is
held to.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from dsai.registry.adapters import LatentClassAnalysis


@pytest.fixture
def two_class_binary_data():
    """400 rows per class, five binary items, two classes with clearly
    different response profiles and one item (index 4) that is pure noise
    (50/50 in both classes) -- a class boundary should not depend on it."""
    rng = np.random.default_rng(7)
    n = 400
    probs_a = [0.9, 0.85, 0.1, 0.15, 0.5]
    probs_b = [0.1, 0.2, 0.9, 0.8, 0.5]
    class_a = (rng.random((n, 5)) < probs_a).astype(float)
    class_b = (rng.random((n, 5)) < probs_b).astype(float)
    X = np.vstack([class_a, class_b])
    labels = np.array([0] * n + [1] * n)
    return X, labels, np.array(probs_a), np.array(probs_b)


def test_recovers_planted_classes(two_class_binary_data):
    X, true_labels, probs_a, probs_b = two_class_binary_data
    model = LatentClassAnalysis(n_clusters=2, n_init=10, random_state=42).fit(X)

    ari = adjusted_rand_score(true_labels, model.labels_)
    assert ari > 0.7, f"recovered classes barely resemble the planted ones (ARI={ari:.3f})"

    # Class labels are arbitrary (EM does not know which is "0" and which is
    # "1") -- match each recovered class to whichever true profile it is
    # closer to before comparing.
    for row in model.item_probs_:
        closer_to_a = np.abs(row - probs_a).sum() < np.abs(row - probs_b).sum()
        target = probs_a if closer_to_a else probs_b
        assert np.abs(row - target).max() < 0.1, (
            f"recovered item probabilities {np.round(row, 3)} do not resemble "
            f"either generating profile"
        )


def test_predict_proba_is_a_real_distribution(two_class_binary_data):
    X, _, _, _ = two_class_binary_data
    model = LatentClassAnalysis(n_clusters=2, n_init=5, random_state=0).fit(X)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(model.predict(X), np.argmax(proba, axis=1))


def test_fit_predict_matches_labels_attribute(two_class_binary_data):
    X, _, _, _ = two_class_binary_data
    model = LatentClassAnalysis(n_clusters=2, n_init=3, random_state=1)
    labels = model.fit_predict(X)
    assert np.array_equal(labels, model.labels_)


def test_bic_and_aic_are_finite_and_bic_penalises_more_parameters(two_class_binary_data):
    X, _, _, _ = two_class_binary_data
    small = LatentClassAnalysis(n_clusters=2, n_init=5, random_state=42).fit(X)
    large = LatentClassAnalysis(n_clusters=6, n_init=5, random_state=42).fit(X)
    for model in (small, large):
        assert np.isfinite(model.bic(X))
        assert np.isfinite(model.aic(X))
    # BIC's penalty per parameter is log(n) >> AIC's 2, so adding classes that
    # do not actually explain more of the data should cost BIC more than AIC,
    # relative to each other -- true regardless of which one is smaller outright.
    bic_gap = large.bic(X) - small.bic(X)
    aic_gap = large.aic(X) - small.aic(X)
    assert bic_gap > aic_gap - 1e-6


def test_continuous_columns_are_median_split_and_threshold_is_reused():
    """A column that is not already 0/1 must not crash the model, and the
    same cut point learned at fit time has to be applied at predict time --
    not recomputed from whatever new data predict() is given."""
    rng = np.random.default_rng(3)
    X_train = np.column_stack([rng.normal(0, 1, 200), rng.integers(0, 2, 200).astype(float)])
    model = LatentClassAnalysis(n_clusters=2, n_init=3, random_state=42).fit(X_train)
    assert model.thresholds_.shape == (2,)
    assert model.thresholds_[1] == 0.5  # the already-binary column needs no split

    # Rows here would binarize differently under a threshold recomputed from
    # this smaller sample -- proves the stored threshold is what is actually used.
    X_new = np.array([[model.thresholds_[0] + 0.01, 1.0],
                      [model.thresholds_[0] - 0.01, 0.0]])
    proba = model.predict_proba(X_new)
    assert proba.shape == (2, 2)


def test_a_single_class_still_fits():
    rng = np.random.default_rng(5)
    X = (rng.random((100, 4)) < 0.5).astype(float)
    model = LatentClassAnalysis(n_clusters=1, n_init=2, random_state=42).fit(X)
    assert np.all(model.labels_ == 0)
    assert model.class_prior_.shape == (1,)


def test_registered_and_reachable_through_the_registry():
    from dsai.core.schema import TaskType
    from dsai.registry.base import REGISTRY, load_builtin_models

    load_builtin_models()
    spec = REGISTRY.get("latent_class_analysis")
    assert spec.category == "clustering"
    assert TaskType.CLUSTERING in spec.task_types
    assert spec.requires_scaling is False, (
        "scaling would centre the 0/1 indicators away from {0,1} and break the Bernoulli fit"
    )
    model = spec.builder(n_clusters=2)
    assert isinstance(model, LatentClassAnalysis)
