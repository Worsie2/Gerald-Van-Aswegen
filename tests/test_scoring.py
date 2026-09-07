"""Using a model on rows it has never seen — and refusing to when that is wrong.

The prediction is the easy part. What is tested here is everything that has to
be true before a prediction means anything: that the data carries what the model
needs, that a column changing kind is caught, and that data no longer resembling
what the model learned on is reported rather than quietly scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.engines.scoring import (
    MIN_ROWS_FOR_DRIFT, check_schema, drift_report, score_new_data,
)


@pytest.fixture
def training():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": [f"C{i:04d}" for i in range(200)],
        "region": rng.choice(["north", "south"], 200),
        "size": rng.normal(50, 10, 200),
        "spend": rng.normal(1000, 100, 200),
    })


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def test_a_missing_required_column_stops_the_scoring(training):
    check = check_schema(training.drop(columns=["size"]), training,
                         ["region", "size", "spend"], target="spend")
    assert not check.ok
    assert check.missing == ["size"]
    assert "cannot be scored" in check.summary()


def test_extra_columns_are_ignored_not_fatal(training):
    new = training.assign(irrelevant=1)
    check = check_schema(new, training, ["region", "size", "spend"], target="spend")
    assert check.ok
    assert "irrelevant" in check.extra
    assert any("ignored" in note for note in check.notes)


def test_a_column_changing_kind_is_reported(training):
    new = training.copy()
    new["size"] = new["size"].astype(str)
    check = check_schema(new, training, ["region", "size"], target=None)
    assert check.type_changed
    assert check.type_changed[0]["column"] == "size"
    assert any("changed kind" in note for note in check.notes)


def test_unseen_categories_are_reported(training):
    new = training.copy()
    new.loc[new.index[:5], "region"] = "east"
    check = check_schema(new, training, ["region", "size"], target=None)
    assert check.unseen_categories["region"] == ["east"]


def test_identifier_columns_do_not_raise_an_unseen_warning(training):
    """Every ID in a new file is new. Warning about it trains the reader to ignore warnings."""
    new = training.copy()
    new["id"] = [f"D{i:04d}" for i in range(len(new))]
    check = check_schema(new, training, ["id", "region", "size"], target=None)
    assert "id" not in check.unseen_categories


def test_a_jump_in_missingness_is_reported(training):
    new = training.copy()
    new.loc[new.index[:100], "size"] = np.nan
    check = check_schema(new, training, ["size"], target=None)
    assert "size" in check.new_missingness
    assert any("Missingness" in note for note in check.notes)


def test_schema_check_works_without_a_training_frame(training):
    check = check_schema(training, None, ["region", "size"], target=None)
    assert check.ok


# --------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------

def test_drift_flags_a_shifted_numeric_column(training):
    new = training.copy()
    new["size"] = new["size"] + 30           # three standard deviations
    frame = drift_report(new, training, ["size"])
    assert frame.iloc[0]["Column"] == "size"
    assert "suspicion" in frame.iloc[0]["Reading"]


def test_drift_is_quiet_when_nothing_moved(training):
    frame = drift_report(training.copy(), training, ["size", "region"])
    assert all("unchanged" in r for r in frame["Reading"])


def test_drift_skips_identifiers_and_dates(training):
    dated = training.assign(when=pd.date_range("2024-01-01", periods=len(training), freq="D"))
    frame = drift_report(dated, dated, ["id", "when", "size"])
    assert set(frame["Column"]) == {"size"}


def test_drift_says_so_when_there_are_too_few_rows(training):
    tiny = training.head(MIN_ROWS_FOR_DRIFT - 1)
    frame = drift_report(tiny, training, ["size", "region"])
    assert all(r == "too few new rows to tell" for r in frame["Reading"])


# --------------------------------------------------------------------------
# scoring against a real fitted model
# --------------------------------------------------------------------------

def test_scoring_produces_one_prediction_per_row(regression_run, water_customers):
    run, typed = regression_run
    engine = _engine_for(run)
    if engine is None:
        pytest.skip("the fitted model is not reachable from this fixture")
    new = water_customers.drop(columns=["annual_spend"]).head(40)
    result = score_new_data(engine, new, run.best, training_frame=typed,
                            keep_columns=["customer_id"])
    assert result.schema.ok
    assert result.n_scored == 40
    assert f"predicted_{run.best.target}" in result.predictions.columns
    assert "customer_id" in result.predictions.columns
    assert result.predictions[f"predicted_{run.best.target}"].notna().all()


def test_scoring_refuses_data_it_cannot_use(regression_run, water_customers):
    run, typed = regression_run
    engine = _engine_for(run)
    if engine is None:
        pytest.skip("the fitted model is not reachable from this fixture")
    broken = water_customers.drop(columns=["annual_spend", "meters_installed"]).head(20)
    result = score_new_data(engine, broken, run.best, training_frame=typed)
    assert not result.schema.ok
    assert result.predictions.empty
    assert "meters_installed" in result.schema.missing


def test_empty_rows_are_skipped_not_guessed(regression_run, water_customers):
    run, typed = regression_run
    engine = _engine_for(run)
    if engine is None:
        pytest.skip("the fitted model is not reachable from this fixture")
    new = water_customers.drop(columns=["annual_spend"]).head(20).copy()
    features = [c for c in run.best.features if c != run.best.target]
    new.loc[new.index[:4], features] = np.nan
    result = score_new_data(engine, new, run.best, training_frame=typed)
    assert result.n_skipped == 4
    assert result.n_scored == 16
    assert any("were empty" in c for c in result.caveats)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _engine_for(run):
    """The fitted model belongs to the engine that trained it."""
    engine = getattr(run, "_engine", None)
    return engine.fitted_model(run) if engine is not None else None


@pytest.fixture(scope="session")
def water_customers():
    from dsai.app.samples import build_sample

    frame, _ = build_sample("water_customers")
    return frame
