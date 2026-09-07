"""The workings a reader needs in order to check the analysis rather than trust it.

Two things are under test: that the report can show its own arithmetic, and that
imputation can be steered per column — including within a group, which is the
case a single column-wide statistic gets wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.app.imputation import STRATEGIES, Choice, preview_fill, steps_for_choices, strategies_for
from dsai.core.schema import SemanticType
from dsai.preprocessing.transformers import GroupImputer
from dsai.reporting.builder import build_report


# --------------------------------------------------------------------------
# group-wise imputation
# --------------------------------------------------------------------------

@pytest.fixture
def grouped_frame():
    return pd.DataFrame({
        "region": ["A", "A", "A", "B", "B", "B"],
        "spend": [10.0, 12.0, np.nan, 100.0, 104.0, np.nan],
    })


def test_group_imputer_uses_the_group_not_the_column(grouped_frame):
    filled = GroupImputer(columns=["spend"], group_by="region", strategy="mean") \
        .fit(grouped_frame).transform(grouped_frame)["spend"]
    # The column mean is 56.5. Neither gap should get anywhere near it.
    assert filled.iloc[2] == pytest.approx(11.0)
    assert filled.iloc[5] == pytest.approx(102.0)


def test_group_imputer_learns_only_from_the_rows_it_was_fitted_on(grouped_frame):
    """The whole point of fitting per fold: a test row must not inform itself."""
    train = grouped_frame.iloc[:3]          # region A only, one gap
    imputer = GroupImputer(columns=["spend"], group_by="region", strategy="mean").fit(train)
    # Region B never appeared in training, so it can only fall back.
    out = imputer.transform(grouped_frame)
    assert out["spend"].iloc[2] == pytest.approx(11.0)      # A, learned
    assert out["spend"].iloc[5] == pytest.approx(11.0)      # B, unseen -> training fallback
    assert not out["spend"].isna().any()


def test_group_imputer_survives_a_missing_group_column(grouped_frame):
    imputer = GroupImputer(columns=["spend"], group_by="not_here", strategy="median").fit(grouped_frame)
    out = imputer.transform(grouped_frame)
    assert not out["spend"].isna().any()


def test_group_imputer_never_imputes_the_grouping_column(grouped_frame):
    imputer = GroupImputer(columns=None, group_by="region", strategy="median").fit(grouped_frame)
    assert "region" not in imputer.columns_


def test_group_imputer_reports_what_it_learned(grouped_frame):
    imputer = GroupImputer(columns=["spend"], group_by="region", strategy="mean").fit(grouped_frame)
    learned = imputer.learned_values()["spend"]
    assert learned["A"] == pytest.approx(11.0)
    assert learned["B"] == pytest.approx(102.0)
    assert "(no group / unseen group)" in learned


def test_group_imputer_is_registered_as_a_step():
    from dsai.preprocessing.steps import STEPS

    spec = STEPS.get("impute_by_group")
    assert spec.leakage_safe is False          # it learns, so it must be fold-fitted
    assert {p.name for p in spec.params} == {"group_by", "strategy"}


# --------------------------------------------------------------------------
# per-column choices
# --------------------------------------------------------------------------

def test_numeric_and_categorical_get_different_strategies():
    numeric = strategies_for(SemanticType.NUMERIC_CONTINUOUS)
    categorical = strategies_for(SemanticType.CATEGORICAL_NOMINAL)
    assert "Mean of the column" in numeric
    assert "Mean of the column" not in categorical
    assert "Most frequent value" in categorical


def test_every_strategy_explains_itself():
    for name, spec in STRATEGIES.items():
        assert spec["note"].strip(), f"{name} has no note"
        assert len(spec["note"]) > 30


def test_preview_shows_the_actual_number(grouped_frame):
    summary, table = preview_fill(grouped_frame, Choice("spend", "Median of the column"))
    assert "56.5" in summary or "56" in summary
    assert table is None


def test_preview_of_a_group_choice_lists_the_groups(grouped_frame):
    summary, table = preview_fill(
        grouped_frame, Choice("spend", "Mean within a group", group_by="region"))
    assert table is not None
    assert set(table["region"]) == {"A", "B"}
    assert "rows it is based on" in table.columns


def test_preview_warns_when_a_group_is_too_thin(grouped_frame):
    summary, _ = preview_fill(
        grouped_frame, Choice("spend", "Mean within a group", group_by="region"))
    assert "fewer than 5 rows" in summary


def test_identical_choices_merge_into_one_step():
    steps = steps_for_choices([
        Choice("a", "Median of the column"),
        Choice("b", "Median of the column"),
        Choice("c", "Mean of the column"),
    ])
    median = [s for s in steps if s["step_key"] == "impute_median"]
    assert len(median) == 1
    assert median[0]["columns"] == ["a", "b"]
    assert len(steps) == 2


def test_leave_as_is_produces_no_step():
    assert steps_for_choices([Choice("a", "Leave as is")]) == []


def test_group_choices_with_different_groups_stay_separate():
    steps = steps_for_choices([
        Choice("a", "Mean within a group", group_by="region"),
        Choice("b", "Mean within a group", group_by="tier"),
    ])
    assert len(steps) == 2


# --------------------------------------------------------------------------
# methodology in the report
# --------------------------------------------------------------------------

def test_methodology_is_off_by_default(regression_run):
    run, frame = regression_run
    headings = {h for h, _ in build_report(run, frame=frame).sections}
    assert "Methodology" not in headings


def test_methodology_adds_three_sections(regression_run):
    run, frame = regression_run
    headings = [h for h, _ in build_report(run, frame=frame, include_methodology=True).sections]
    for wanted in ("Methodology", "The calculations", "Did it run cleanly"):
        assert wanted in headings


def test_methodology_states_the_validation_design(regression_run):
    run, frame = regression_run
    body = dict(build_report(run, frame=frame, include_methodology=True).sections)["Methodology"]
    assert "Validation design" in body
    assert "How the rows were divided" in body
    assert f"Random seed: {run.best.random_seed}" in body


def test_methodology_defines_every_metric_it_reports(regression_run):
    from dsai.engines import metrics as M

    run, frame = regression_run
    body = dict(build_report(run, frame=frame, include_methodology=True).sections)["Methodology"]
    # Pipes are escaped so they do not break the markdown table; undo that first.
    plain = body.replace("\\|", "|")
    for metric in M.default_metrics(run.best.task_type):
        assert M.METRIC_LABELS.get(metric, metric) in body
        formula = M.metric_formula(metric)
        if formula:
            assert formula in plain, f"{metric} formula missing"


def test_calculations_show_every_fold(regression_run):
    run, frame = regression_run
    body = dict(build_report(run, frame=frame,
                             include_methodology=True).sections)["The calculations"]
    folds = run.best.validation.n_splits
    assert f"Fold {folds}" in body
    assert "Fold-by-fold scores" in body
    # The train/test gap has to distinguish raw units from the relative measure.
    assert "Raw difference" in body and "Relative gap" in body


def test_integrity_section_lists_failures_rather_than_hiding_them(regression_run):
    run, frame = regression_run
    body = dict(build_report(run, frame=frame,
                             include_methodology=True).sections)["Did it run cleanly"]
    assert "Self-check" in body
    assert f"`{run.status}`" in body
    assert str(len([r for r in run.results if r.status == "success"])) in body


def test_methodology_reaches_the_html_export(regression_run):
    from dsai.reporting.exporters import to_html

    run, frame = regression_run
    plain = to_html(run, frame=frame)
    full = to_html(run, "technical", frame=frame, methodology=True)
    assert "Fold-by-fold scores" not in plain
    assert "Fold-by-fold scores" in full


# --------------------------------------------------------------------------
# what the preprocessing learned
# --------------------------------------------------------------------------

def test_learned_parameters_skip_columns_a_previous_step_dropped():
    from dsai.app.quality import quality_breakdown  # noqa: F401  (import sanity)
    from dsai.app.samples import build_sample
    from dsai.app.workings import learned_parameters
    from dsai.core.profiler import profile_dataset
    from dsai.core.schema import BusinessContext, TaskType
    from dsai.preprocessing.recommender import recommend_pipeline

    frame, _ = build_sample("messy_survey")
    profile, typed = profile_dataset(frame, name="messy")
    pipeline, _ = recommend_pipeline(profile, TaskType.REGRESSION, None, None, BusinessContext())
    tables = learned_parameters(pipeline, typed)
    assert tables

    dropped = {c for step in pipeline.active_steps if step.step_key == "drop_columns"
               for c in (step.params.get("columns") or [])}
    for title, table in tables:
        if "Drop columns" in title or "Column" not in table.columns:
            continue
        assert not (set(table["Column"]) & dropped), f"{title} names a dropped column"


def test_learned_parameters_are_empty_without_a_pipeline():
    from dsai.app.workings import learned_parameters

    assert learned_parameters(None, pd.DataFrame({"a": [1]})) == []
    assert learned_parameters(object(), None) == []
