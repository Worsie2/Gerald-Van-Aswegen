"""The profiler must measure what is there, and say so accurately."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.core.profiler import class_imbalance, override_semantic_type, profile_dataset
from dsai.core.schema import SemanticType


def test_detects_semantic_types():
    frame = pd.DataFrame({
        "user_id": range(1, 201),
        "signed_up": pd.date_range("2024-01-01", periods=200, freq="D"),
        "tier": np.random.default_rng(0).choice(["low", "medium", "high"], 200),
        "country": np.random.default_rng(1).choice(["ZA", "NA", "BW"], 200),
        "is_active": np.random.default_rng(2).choice([0, 1], 200),
        "revenue": np.random.default_rng(3).lognormal(5, 1, 200),
        "note": ["a reasonably long free text comment about this account"] * 199 + ["different text here entirely"],
        "always_one": 1,
    })
    profile, _ = profile_dataset(frame)
    types = {name: column.semantic_type for name, column in profile.columns.items()}

    assert types["user_id"] is SemanticType.IDENTIFIER
    assert types["signed_up"] is SemanticType.DATETIME
    assert types["tier"] is SemanticType.CATEGORICAL_ORDINAL
    assert types["country"] is SemanticType.CATEGORICAL_NOMINAL
    assert types["is_active"] is SemanticType.BINARY
    assert types["revenue"] is SemanticType.NUMERIC_CONTINUOUS
    assert types["always_one"] is SemanticType.CONSTANT


def test_parses_dates_and_currency_stored_as_text():
    frame = pd.DataFrame({
        "when": ["2024-01-05", "2024-02-11", "2024-03-19", "2024-04-22"] * 10,
        "amount": ["R1,200.50", "R980.00", "R15,300.75", "R450.20"] * 10,
    })
    profile, typed = profile_dataset(frame)
    assert profile.columns["when"].semantic_type is SemanticType.DATETIME
    assert profile.columns["amount"].is_numeric
    assert pd.api.types.is_numeric_dtype(typed["amount"])


def test_flags_quality_problems(messy_frame):
    profile, _ = profile_dataset(messy_frame, target="value")
    codes = {issue.code for issue in profile.quality_issues}
    assert "duplicate_rows" in codes
    assert "severe_missingness" in codes
    assert "constant_columns" in codes
    assert "missing_values" in codes
    assert profile.quality_score < 100


def test_duplicated_rows_stop_a_column_being_an_identifier(messy_frame):
    """row_id repeats once rows are duplicated, so it is no longer unique.

    Reporting it as an identifier anyway would be wrong — the profiler measures
    what is in the data, not what the column name implies.
    """
    profile, _ = profile_dataset(messy_frame)
    assert "row_id" not in profile.identifier_columns
    clean, _ = profile_dataset(messy_frame.drop_duplicates(subset=["row_id"]))
    assert "row_id" in clean.identifier_columns


def test_detects_leakage_against_the_target(messy_frame):
    profile, _ = profile_dataset(messy_frame, target="value")
    leaking = {suspect["column"] for suspect in profile.leakage_suspects}
    assert "value_leak" in leaking, "a column correlating >0.98 with the target must be flagged"


def test_ranks_plausible_targets(regression_frame):
    profile, _ = profile_dataset(regression_frame)
    top = [candidate["column"] for candidate in profile.target_candidates[:2]]
    assert "annual_spend" in top


def test_computes_multicollinearity():
    rng = np.random.default_rng(7)
    base = rng.normal(size=300)
    frame = pd.DataFrame({
        "a": base,
        "b": base * 2 + rng.normal(0, 0.01, 300),   # near-duplicate of a
        "c": rng.normal(size=300),
    })
    profile, _ = profile_dataset(frame)
    assert profile.multicollinearity["a"] > 10
    assert profile.multicollinearity["c"] < 5


def test_user_can_correct_the_inferred_type(regression_frame):
    profile, _ = profile_dataset(regression_frame)
    override_semantic_type(profile, "meters_installed", SemanticType.CATEGORICAL_ORDINAL)
    column = profile.columns["meters_installed"]
    assert column.semantic_type is SemanticType.CATEGORICAL_ORDINAL
    assert column.semantic_type_overridden
    assert any("changed by user" in note.lower() for note in column.notes)


def test_class_imbalance_is_reported_honestly():
    series = pd.Series([0] * 970 + [1] * 30)
    balance = class_imbalance(series)
    assert balance["imbalanced"] and balance["severely_imbalanced"]
    assert balance["minority_share"] == pytest.approx(0.03)


def test_empty_frame_raises_a_useful_error():
    with pytest.raises(ValueError, match="empty DataFrame"):
        profile_dataset(pd.DataFrame())
