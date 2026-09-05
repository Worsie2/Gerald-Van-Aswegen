"""Statistics must separate significance from importance, and check assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.statistics import tests as T
from dsai.statistics.descriptive import (
    correlation_pairs, describe_numeric, interpret_vif, variance_inflation_factors,
)
from dsai.statistics.timeseries import analyse_series


def test_t_test_detects_a_real_difference():
    rng = np.random.default_rng(1)
    result = T.t_test(rng.normal(100, 15, 80), rng.normal(112, 15, 80))
    assert result.significant
    assert abs(result.effect_size) > 0.5
    assert result.confidence_interval is not None
    assert "assumption" in result.summarise().lower() or result.assumptions


def test_large_sample_significance_is_called_out_as_trivial():
    """The core discipline: a significant p-value on a huge sample is not news."""
    rng = np.random.default_rng(2)
    result = T.t_test(rng.normal(100, 15, 60_000), rng.normal(100.4, 15, 60_000))
    assert result.significant
    assert abs(result.effect_size) < 0.2
    assert "practically small" in result.practical_note


def test_small_sample_non_significance_is_not_called_no_effect():
    rng = np.random.default_rng(3)
    result = T.t_test(rng.normal(100, 15, 12), rng.normal(108, 15, 12))
    if not result.significant:
        assert "not established" in result.practical_note


def test_anova_reports_effect_size_and_checks_variance():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "value": np.concatenate([rng.normal(m, 10, 60) for m in (10, 16, 22)]),
        "group": np.repeat(["a", "b", "c"], 60),
    })
    result = T.anova(frame, "value", "group")
    assert result.significant
    assert result.effect_size > 0.14  # a large eta-squared
    assert "equal_variance" in result.assumptions
    assert "does not say which pairs differ" in result.conclusion


def test_chi_square_falls_back_to_fisher_on_sparse_tables():
    frame = pd.DataFrame({
        "exposure": ["yes"] * 12 + ["no"] * 12,
        "outcome": ["event"] * 2 + ["none"] * 10 + ["event"] * 1 + ["none"] * 11,
    })
    result = T.chi_square(frame, "exposure", "outcome")
    assert "expected count below 5" in " ".join(result.assumption_warnings)
    assert "fisher" in result.test.lower()


def test_correlation_test_states_that_it_is_not_causation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    result = T.correlation_test(x, x * 2 + rng.normal(0, 0.5, 200))
    assert result.significant
    assert "not causation" in result.conclusion.lower()


def test_multiple_comparison_correction_quantifies_the_risk():
    result = T.multiple_comparison_correction([0.01, 0.02, 0.03, 0.04, 0.05] * 4, method="holm")
    assert result["n_tests"] == 20
    assert "false positive" in result["note"]
    assert sum(result["significant"]) < 20


def test_vif_identifies_redundant_predictors():
    rng = np.random.default_rng(6)
    base = rng.normal(size=300)
    frame = pd.DataFrame({"a": base, "b": base + rng.normal(0, 0.01, 300), "c": rng.normal(size=300)})
    vif = variance_inflation_factors(frame)
    worst = vif.iloc[0]
    assert worst.vif > 10
    assert "unreliable" in interpret_vif(worst.vif) or "duplicate" in interpret_vif(worst.vif)


def test_descriptive_summary_includes_shape_and_uncertainty():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"skewed": rng.lognormal(3, 1, 500)})
    summary = describe_numeric(frame).iloc[0]
    assert summary.skewness > 1
    assert summary.ci95_lower < summary["mean"] < summary.ci95_upper


def test_series_analysis_finds_planted_structure(series_frame):
    series = series_frame.set_index("month")["sales"]
    analysis = analyse_series(series)
    assert analysis["trend"]["significant"]
    assert analysis["trend"]["direction"] == "increasing"
    assert analysis["seasonal_period"] == 12
    assert analysis["seasonality"]["detected"]
    assert analysis["stationarity"]["verdict"] in {"non-stationary", "difference-stationary", "trend-stationary"}


def test_structural_break_is_located():
    rng = np.random.default_rng(8)
    values = np.concatenate([rng.normal(100, 5, 80), rng.normal(160, 5, 80)])
    series = pd.Series(values, index=pd.date_range("2015-01-31", periods=160, freq="ME"))
    breaks = analyse_series(series)["structural_breaks"]
    assert breaks["detected"]
    assert 70 <= breaks["position"] <= 90
