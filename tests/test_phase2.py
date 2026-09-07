"""Robustness, uncertainty, abstention, thresholds, subgroups, versions, review.

The shared obligation across all of it: never manufacture precision. Several
tests below check that a tool refuses to answer rather than answering badly —
a conformal interval on 3 rows, a threshold on 20, a subgroup of 9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.core.schema import TaskType
from dsai.core.versioning import diff_versions, fingerprint_frame, snapshot
from dsai.engines.sensitivity import _base_feature, build_specifications
from dsai.engines.subgroups import MIN_GROUP_ROWS, discover_subgroups
from dsai.engines.threshold import CostModel, analyse_thresholds
from dsai.engines.uncertainty import ABSTAINED, CAUTION, RELIABLE, assess_rows, fit_conformal


# --------------------------------------------------------------------------
# conformal prediction intervals
# --------------------------------------------------------------------------

def test_conformal_intervals_achieve_their_stated_coverage():
    """The whole point of conformal prediction is that the guarantee is real."""
    rng = np.random.default_rng(0)
    actual = rng.normal(100, 15, 500)
    predicted = actual + rng.normal(0, 10, 500)
    model = fit_conformal(actual, predicted, coverage=0.9)
    assert model.usable

    fresh_actual = rng.normal(100, 15, 20_000)
    fresh_predicted = fresh_actual + rng.normal(0, 10, 20_000)
    covered = np.mean(np.abs(fresh_actual - fresh_predicted) <= model.half_width)
    assert 0.87 <= covered <= 0.93


def test_higher_coverage_means_a_wider_interval():
    rng = np.random.default_rng(1)
    actual = rng.normal(0, 1, 500)
    predicted = actual + rng.normal(0, 1, 500)
    narrow = fit_conformal(actual, predicted, 0.8)
    wide = fit_conformal(actual, predicted, 0.99)
    assert wide.half_width > narrow.half_width


def test_conformal_refuses_rather_than_guessing_from_too_few_rows():
    model = fit_conformal([1, 2, 3, 4], [1, 2, 3, 4])
    assert not model.usable
    assert "guess about a guess" in model.note


def test_conformal_refuses_impossible_coverage():
    """99% coverage from 25 rows needs a quantile outside the data."""
    rng = np.random.default_rng(2)
    actual = rng.normal(0, 1, 25)
    model = fit_conformal(actual, actual + rng.normal(0, 1, 25), coverage=0.99)
    assert not model.usable
    assert "cannot be guaranteed" in model.note


def test_interval_explanation_admits_its_own_limitation():
    rng = np.random.default_rng(3)
    actual = rng.normal(0, 1, 300)
    model = fit_conformal(actual, actual + rng.normal(0, 1, 300))
    text = model.explain()
    assert "same for every row" in text
    assert "not where it is less sure" in text


# --------------------------------------------------------------------------
# abstention
# --------------------------------------------------------------------------

@pytest.fixture
def training():
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "region": rng.choice(["A", "B"], 200),
        "size": rng.normal(50, 5, 200),
    })


def test_a_row_far_outside_training_is_refused(training):
    new = pd.DataFrame({"region": ["A"], "size": [100_000.0]})
    result = assess_rows(new, training, ["region", "size"])
    assert result.status.iloc[0] == ABSTAINED
    assert "far outside" in result.reasons.iloc[0]


def test_an_unseen_category_is_caution_not_refusal(training):
    """The model has learned nothing about it, but the row is not nonsense."""
    new = pd.DataFrame({"region": ["Z"], "size": [50.0]})
    result = assess_rows(new, training, ["region", "size"])
    assert result.status.iloc[0] == CAUTION


def test_a_mostly_empty_row_is_refused(training):
    new = pd.DataFrame({"region": [None], "size": [np.nan]})
    result = assess_rows(new, training, ["region", "size"])
    assert result.status.iloc[0] == ABSTAINED


def test_an_ordinary_row_is_reliable(training):
    new = pd.DataFrame({"region": ["A"], "size": [50.0]})
    result = assess_rows(new, training, ["region", "size"])
    assert result.status.iloc[0] == RELIABLE


def test_refusal_outranks_caution(training):
    """A row failing two tests is refused, not merely cautioned."""
    new = pd.DataFrame({"region": ["Z"], "size": [100_000.0]})
    result = assess_rows(new, training, ["region", "size"])
    assert result.status.iloc[0] == ABSTAINED


def test_low_model_confidence_is_flagged(training):
    new = pd.DataFrame({"region": ["A", "A"], "size": [50.0, 51.0]})
    proba = np.array([[0.5, 0.5], [0.95, 0.05]])
    result = assess_rows(new, training, ["region", "size"], probabilities=proba)
    assert result.status.iloc[0] == CAUTION
    assert result.status.iloc[1] == RELIABLE


def test_abstention_counts_are_reported(training):
    new = pd.DataFrame({"region": ["A", "Z", None], "size": [50.0, 50.0, np.nan]})
    result = assess_rows(new, training, ["region", "size"])
    assert sum(result.counts.values()) == 3
    assert result.n_abstained >= 1
    assert "refused" in result.summary()


def test_scoring_flags_abstained_rows_rather_than_dropping_them(regression_run):
    """Never silently replace a refusal with a guess — or with nothing."""
    from dsai.engines.scoring import score_new_data

    run, typed = regression_run
    engine = getattr(run, "_engine", None)
    if engine is None:
        pytest.skip("no fitted model on this fixture")
    model = engine.fitted_model(run)

    new = typed.drop(columns=[run.best.target], errors="ignore").head(60).copy()
    numeric = [c for c in new.columns if pd.api.types.is_numeric_dtype(new[c])][0]
    new.loc[new.index[:5], numeric] = 1e12

    result = score_new_data(model, new, run.best, training_frame=typed,
                            contract=run.contract)
    assert result.n_scored == 60           # nothing dropped
    assert "reliability" in result.predictions.columns
    assert (result.predictions["reliability"] == ABSTAINED).sum() >= 5
    assert any("abstained" in c for c in result.caveats)


# --------------------------------------------------------------------------
# thresholds and cost
# --------------------------------------------------------------------------

@pytest.fixture
def scored():
    rng = np.random.default_rng(5)
    n = 2000
    actual = (rng.uniform(0, 1, n) < 0.2).astype(int)
    scores = np.clip(actual * 0.5 + rng.normal(0.25, 0.2, n), 0, 1)
    return actual, scores


def test_the_threshold_follows_the_cost_ratio(scored):
    actual, scores = scored
    expensive_misses = analyse_thresholds(actual, scores, 1,
                                          CostModel(false_positive=1, false_negative=50))
    expensive_alarms = analyse_thresholds(actual, scores, 1,
                                          CostModel(false_positive=50, false_negative=1))
    # Costly misses should push the threshold down (flag more), not up.
    assert expensive_misses.recommended.threshold < expensive_alarms.recommended.threshold
    assert expensive_misses.recommended.predicted_positive > \
           expensive_alarms.recommended.predicted_positive


def test_the_recommendation_is_never_called_universally_optimal(scored):
    actual, scores = scored
    analysis = analyse_thresholds(actual, scores, 1, CostModel(false_negative=10))
    verdict = analysis.verdict()
    assert "not a universally optimal threshold" in verdict
    assert "depends entirely on the costs you supplied" in verdict


def test_the_recommended_threshold_is_actually_the_cheapest(scored):
    actual, scores = scored
    analysis = analyse_thresholds(actual, scores, 1, CostModel(false_negative=7))
    assert analysis.recommended.total_cost == min(p.total_cost for p in analysis.points)


def test_threshold_refuses_on_too_few_rows():
    analysis = analyse_thresholds([0, 1] * 5, [0.1, 0.9] * 5, 1)
    assert not analysis.usable
    assert "Too few" in analysis.note


def test_threshold_refuses_when_every_outcome_is_the_same():
    analysis = analyse_thresholds([1] * 100, np.linspace(0, 1, 100), 1)
    assert not analysis.usable
    assert "same outcome" in analysis.note


def test_equal_costs_are_named_as_the_assumption_behind_0_5():
    assert "0.5 threshold" in CostModel(1, 1).describe()


# --------------------------------------------------------------------------
# subgroup discovery
# --------------------------------------------------------------------------

def test_subgroup_discovery_finds_a_planted_weakness():
    rng = np.random.default_rng(6)
    n = 1200
    frame = pd.DataFrame({
        "region": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "spend": rng.lognormal(10, 1, n),
    })
    actual = rng.normal(100, 20, n)
    predicted = actual + rng.normal(0, 8, n)
    bad = (frame.region == "C") & (frame.spend > frame.spend.quantile(2 / 3))
    predicted[bad.to_numpy()] += rng.normal(0, 40, int(bad.sum()))

    report = discover_subgroups(frame, actual, predicted, TaskType.REGRESSION)
    assert report.usable
    assert report.notable
    worst = max(report.notable, key=lambda g: g.ratio)
    assert "region = C" in worst.label
    assert worst.ratio > 2


def test_no_group_smaller_than_the_floor_is_reported():
    rng = np.random.default_rng(7)
    n = 400
    frame = pd.DataFrame({"tiny": ["rare"] * 5 + ["common"] * (n - 5)})
    actual = rng.normal(0, 1, n)
    report = discover_subgroups(frame, actual, actual + rng.normal(0, 1, n),
                                TaskType.REGRESSION)
    assert all(g.n >= MIN_GROUP_ROWS for g in report.groups)


def test_a_uniform_model_flags_nothing():
    rng = np.random.default_rng(8)
    n = 900
    frame = pd.DataFrame({"region": rng.choice(["A", "B", "C"], n)})
    actual = rng.normal(0, 1, n)
    report = discover_subgroups(frame, actual, actual + rng.normal(0, 1, n),
                                TaskType.REGRESSION)
    assert not report.notable
    assert "No group stands out" in report.verdict


def test_the_verdict_never_says_bias_or_unfair():
    rng = np.random.default_rng(9)
    n = 900
    frame = pd.DataFrame({"region": rng.choice(["A", "B"], n)})
    actual = rng.normal(0, 1, n)
    predicted = actual + rng.normal(0, 1, n)
    predicted[(frame.region == "A").to_numpy()] += rng.normal(0, 6, int((frame.region == "A").sum()))
    report = discover_subgroups(frame, actual, predicted, TaskType.REGRESSION)
    # The words may appear only inside the disclaimer that says it is not that.
    assert "not a finding of bias or unfairness" in report.verdict
    readings = " ".join(g.reading for g in report.groups).lower()
    assert "bias" not in readings and "unfair" not in readings
    assert "investigate" in report.verdict


def test_subgroup_discovery_needs_enough_rows():
    frame = pd.DataFrame({"a": ["x"] * 20})
    report = discover_subgroups(frame, np.zeros(20), np.zeros(20), TaskType.REGRESSION)
    assert not report.usable
    assert "at least" in report.note


# --------------------------------------------------------------------------
# sensitivity
# --------------------------------------------------------------------------

def test_specifications_are_built_from_what_the_run_did(regression_run):
    run, _ = regression_run
    specs = build_specifications(run)
    assert len(specs) >= 8
    assert specs[0].is_baseline
    dimensions = {s.dimension for s in specs}
    assert {"model family", "preprocessing depth", "random seed", "fold count"} <= dimensions
    assert all(s.rationale for s in specs if not s.is_baseline)


def test_the_baseline_model_is_not_an_alternative_specification(regression_run):
    """A model that ignores every predictor is not a defensible alternative."""
    from dsai.registry.base import REGISTRY

    run, _ = regression_run
    for spec in build_specifications(run):
        if spec.model_key and not spec.is_baseline:
            assert REGISTRY.get(spec.model_key).family != "baseline"


def test_encoded_and_unencoded_names_are_the_same_variable():
    columns = ["service_tier", "region", "spend"]
    assert _base_feature("service_tier_premium", columns) == "service_tier"
    assert _base_feature("service_tier", columns) == "service_tier"
    assert _base_feature("spend", columns) == "spend"


# --------------------------------------------------------------------------
# dataset versions
# --------------------------------------------------------------------------

def test_the_same_data_gives_the_same_fingerprint():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert fingerprint_frame(frame) == fingerprint_frame(frame.copy())


def test_changed_values_give_a_different_fingerprint():
    """Same shape, same columns, different contents is a different dataset."""
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"x": [1, 2, 4]})
    assert fingerprint_frame(a) != fingerprint_frame(b)


def test_identical_versions_are_reported_as_identical():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    version = snapshot(frame)
    difference = diff_versions(version, version)
    assert difference.identical
    assert "same data" in difference.summary


def test_the_diff_names_what_changed():
    rng = np.random.default_rng(10)
    before = pd.DataFrame({"region": ["A"] * 200, "spend": rng.normal(100, 10, 200),
                           "gone": [1] * 200})
    after = pd.DataFrame({"region": ["A"] * 150 + ["B"] * 100,
                          "spend": np.concatenate([rng.normal(100, 10, 150),
                                                   rng.normal(160, 10, 100)]),
                          "added": [1] * 250})
    difference = diff_versions(snapshot(before), snapshot(after))
    assert not difference.identical
    assert difference.rows_added == 50
    assert difference.columns_added == ["added"]
    assert difference.columns_removed == ["gone"]
    assert "B" in difference.categories_added["region"]
    assert difference.distributions_moved
    assert "cannot score this version" in difference.summary


def test_a_version_never_stores_the_rows():
    frame = pd.DataFrame({"secret": ["a", "b", "c"] * 100})
    version = snapshot(frame)
    payload = version.to_json()
    # Category *names* are kept deliberately, so the contract can check them.
    # The rows themselves are not.
    assert '"n_rows": 300' in payload
    assert payload.count('"a"') <= 2


# --------------------------------------------------------------------------
# reviewer mode
# --------------------------------------------------------------------------

def test_the_review_covers_every_area(regression_run):
    from dsai.reporting.review import build_review

    run, _ = regression_run
    package = build_review(run)
    names = {a.name for a in package.areas}
    assert names == {"Question", "Data", "Preprocessing", "Validation", "Model",
                     "Evidence", "Recommendations"}
    assert all(a.summary for a in package.areas)


def test_the_review_refuses_to_give_a_verdict(regression_run):
    from dsai.reporting.review import build_review

    run, _ = regression_run
    markdown = build_review(run).to_markdown()
    assert "not a verdict" in markdown
    assert "reviewer's judgement" in markdown


def test_the_review_always_raises_the_causal_assumption(regression_run):
    from dsai.reporting.review import build_review

    run, _ = regression_run
    package = build_review(run)
    recommendations = next(a for a in package.areas if a.name == "Recommendations")
    assert any("associational" in c for c in recommendations.checks)


def test_leakage_makes_the_data_area_a_concern():
    from dsai.app.samples import build_sample
    from dsai.core.schema import BusinessContext
    from dsai.engines.orchestrator import AIDataScientist, RunSettings
    from dsai.reporting.review import build_review

    frame, source = build_sample("messy_survey")
    run = AIDataScientist().analyse(
        frame, "messy", BusinessContext(),
        RunSettings(max_models=2, time_budget="fast"), source,
    )
    package = build_review(run)
    data = next(a for a in package.areas if a.name == "Data")
    assert data.status == "concern"
    assert any("leakage" in c.lower() for c in data.checks)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_calibration_detects_over_confidence():
    from dsai.explain.diagnostics import calibration_check

    rng = np.random.default_rng(11)
    p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < p).astype(int)
    honest = calibration_check(y, np.column_stack([1 - p, p]), [0, 1])

    inflated = np.clip(p * 1.6, 0, 1)
    over = calibration_check(y, np.column_stack([1 - inflated, inflated]), [0, 1])

    assert honest["expected_calibration_error"] < over["expected_calibration_error"]
    assert "over-confident" in over["interpretation"]
    assert honest["brier_score"] < over["brier_score"]


def test_calibration_reports_skill_against_the_base_rate():
    from dsai.explain.diagnostics import calibration_check

    rng = np.random.default_rng(12)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    out = calibration_check(y, np.column_stack([1 - p, p]), [0, 1])
    assert out["brier_skill_score"] > 0        # better than predicting the base rate
    assert "base rate" in out["scoring_note"]
    assert "log loss" in out["scoring_note"].lower()
