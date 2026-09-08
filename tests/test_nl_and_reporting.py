"""Natural-language parsing, reporting, and project persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsai.core.profiler import profile_dataset
from dsai.core.schema import BusinessContext, TaskType
from dsai.engines.nl import answer_question, apply_filters, intent_to_objective, parse_command


@pytest.fixture
def question_frame() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame({
        "customer_id": range(n),
        "signup_date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "region": rng.choice(["Gauteng", "Western Cape"], n),
        "income": rng.lognormal(10, 0.5, n),
        "visits": rng.poisson(5, n),
        "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })
    frame["annual_spend"] = 0.2 * frame.income + 800 * frame.visits + rng.normal(0, 3_000, n)
    return frame


@pytest.mark.parametrize("question,action,task", [
    ("Create five customer segments.", "segment", TaskType.CLUSTERING),
    ("Find unusual customers.", "anomaly", TaskType.ANOMALY_DETECTION),
    ("Forecast annual spend for the next six months.", "forecast", TaskType.TIME_SERIES_FORECAST),
    ("Predict which customers are likely to churn.", "model", TaskType.BINARY_CLASSIFICATION),
    ("Which customers are most valuable?", "rank", TaskType.EXPLORATORY),
    ("Is annual spend significantly different between regions?", "test", TaskType.HYPOTHESIS_TESTING),
    ("Analyse this dataset.", "analyse", None),
])
def test_questions_map_to_the_right_workflow(question, action, task, question_frame):
    profile, _ = profile_dataset(question_frame)
    intent = parse_command(question, profile)
    assert intent.action == action
    if task is not None:
        assert intent.task_type is task


def test_parameters_are_extracted(question_frame):
    profile, _ = profile_dataset(question_frame)
    assert parse_command("create five customer segments", profile).n_clusters == 5
    assert parse_command("forecast for the next six months", profile).horizon == 6
    assert parse_command("show the top 10 customers by spend", profile).parameters["top_n"] == 10
    assert parse_command("try every suitable model", profile).parameters["time_budget"] == "thorough"


def test_column_names_are_matched_by_stem(question_frame):
    """'churn' must find the 'churned' column."""
    profile, _ = profile_dataset(question_frame)
    intent = parse_command("predict which customers will churn", profile)
    assert intent.target == "churned"


def test_naming_the_task_picks_a_target_that_actually_is_one(question_frame):
    """No column is named, so the target is guessed from the profiler's ranked
    candidates. Without the fix this ignores the word 'regression' entirely and
    can hand back a classification objective (if a binary column such as
    'churned' ranks first) for a question that explicitly asked for regression,
    or the reverse."""
    profile, _ = profile_dataset(question_frame)

    regression = parse_command("try every suitable regression model", profile)
    assert regression.task_type is TaskType.REGRESSION
    assert profile.columns[regression.target].is_numeric

    classification = parse_command("try every suitable classification model", profile)
    assert classification.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)


def test_named_models_are_filtered_to_the_task(question_frame):
    profile, _ = profile_dataset(question_frame)
    intent = parse_command("predict annual spend with random forest", profile)
    assert intent.model_keys == ["random_forest_regressor"], \
        "naming a family must not also offer the classifier on a regression problem"


def test_expensive_work_requires_confirmation(question_frame):
    profile, _ = profile_dataset(question_frame)
    expensive = parse_command("forecast annual spend for the next six months", profile)
    assert expensive.requires_confirmation
    assert expensive.plan_preview, "the user must be able to see the plan before approving it"

    cheap = parse_command("describe the data", profile)
    assert not cheap.requires_confirmation


def test_unclear_questions_ask_rather_than_guess(question_frame):
    profile, _ = profile_dataset(question_frame)
    intent = parse_command("asdf qwerty zxcv", profile)
    assert intent.action == "unknown"
    assert intent.clarification
    assert intent.task_type is None


def test_inline_filters_are_applied(question_frame):
    profile, _ = profile_dataset(question_frame)
    intent = parse_command("show income where income > 40000", profile)
    assert "income" in intent.filters
    filtered, log = apply_filters(question_frame, intent.filters)
    assert len(filtered) < len(question_frame)
    assert all(filtered.income > 40_000)
    assert log


def test_statistical_questions_are_answered_without_a_model(question_frame):
    profile, typed = profile_dataset(question_frame)
    result = answer_question("is annual spend different between regions?", typed, profile)
    assert result["answer"]
    assert "p =" in result["answer"] or "significant" in result["answer"].lower()


def test_correlation_questions_are_actually_answered(question_frame):
    """The Ask page only computes describe/correlate/test/rank answers when
    `not intent.clarification` -- so a planner that always fills in
    `clarification` as a permanent caveat (rather than a real, blocking
    ambiguity) silently suppresses its own answer. _plan_correlate did
    exactly that: 'correlation is not causation' is true of every answer,
    not a reason to withhold it, and _correlation_answer() already says so
    in the text the page actually shows."""
    profile, typed = profile_dataset(question_frame)
    intent = parse_command("what is the relationship between income and annual_spend?", profile)
    assert intent.action == "correlate"
    assert not intent.clarification, (
        "a permanent disclaimer here blocks the Ask page from ever computing an answer"
    )

    result = answer_question("what is the relationship between income and annual_spend?",
                             typed, profile)
    assert result.get("answer"), "no answer was computed"
    assert "correlat" in result["answer"].lower() or "r =" in result["answer"]


def test_intent_converts_into_a_runnable_objective(question_frame):
    profile, _ = profile_dataset(question_frame)
    intent = parse_command("create four customer segments", profile)
    objective = intent_to_objective(intent, profile)
    assert objective.task_type is TaskType.CLUSTERING
    assert objective.n_clusters == 4
    assert objective.source == "user_supplied"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def test_reports_differ_by_audience(regression_frame):
    from dsai.engines.orchestrator import AIDataScientist, RunSettings
    from dsai.reporting.exporters import to_html, to_markdown

    run = AIDataScientist().analyse(
        regression_frame, "water", BusinessContext(target_variable="annual_spend"),
        RunSettings(max_models=3),
    )
    business = to_markdown(run, "business")
    technical = to_markdown(run, "technical")

    assert "What this means for the business" in business
    assert "Models evaluated" not in business
    assert "Models evaluated" in technical
    assert "Reproducibility" in technical
    assert len(technical) > len(business)

    html = to_html(run)
    assert html.startswith("<!doctype html>")
    assert "prefers-color-scheme" in html, "the report must work in dark mode"
    assert "<table>" in html


def test_excel_export_has_the_expected_sheets(regression_frame, tmp_path):
    from dsai.engines.orchestrator import AIDataScientist, RunSettings
    from dsai.reporting.exporters import to_excel

    run = AIDataScientist().analyse(
        regression_frame, "water", BusinessContext(target_variable="annual_spend"),
        RunSettings(max_models=3),
    )
    path = to_excel(run, tmp_path / "report.xlsx")
    sheets = pd.ExcelFile(path).sheet_names
    for expected in ["Summary", "Model comparison", "Findings", "Recommendations", "Decision log"]:
        assert expected in sheets


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------

def test_project_round_trips(regression_frame, tmp_path):
    from dsai.engines.orchestrator import AIDataScientist, RunSettings
    from dsai.preprocessing.pipeline import PreprocessingPipeline
    from dsai.repro.project import Project, list_projects

    context = BusinessContext(description="test data", currency="ZAR", target_variable="annual_spend")
    project = Project.create(tmp_path / "water", regression_frame, "water", "a test", context)

    pipeline = PreprocessingPipeline("standard")
    pipeline.add("impute_median")
    project.save_pipeline(pipeline)

    run = AIDataScientist().analyse(regression_frame, "water", context, RunSettings(max_models=2))
    project.save_run(run)

    reopened = Project.open(tmp_path / "water")
    assert reopened.meta.name == "water"
    assert len(reopened.data()) == len(regression_frame)
    assert reopened.context().currency == "ZAR"
    assert "standard" in reopened.list_pipelines()
    assert len(reopened.load_pipeline("standard").steps) == 1

    runs = reopened.list_runs()
    assert len(runs) == 1 and runs[0]["target"] == "annual_spend"
    assert reopened.load_run(runs[0]["run_id"])["findings"]
    assert list_projects(tmp_path)
