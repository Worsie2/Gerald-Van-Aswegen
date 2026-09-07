"""Loading a second dataset, and telling two runs apart.

The lifecycle tests exist because of a bug that made the app unusable: loading a
new file left the *old* profile in place, so every page downstream described
columns that were no longer there and the only way out was restarting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dsai.app.state import Workspace
from dsai.core.schema import SemanticType
from dsai.engines.compare import comparable, compare_runs


@pytest.fixture
def loaded_workspace():
    space = Workspace()
    space.frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    space.dataset_name = "first"
    space.profile = object()
    space.typed_frame = space.frame
    space.objectives = ["something"]
    space.objective = "something"
    space.pipelines = {"default": object()}
    space.run = object()
    space.runs = [object(), object()]
    space.chat = [{"role": "user", "content": "hi"}]
    space.scientist = object()
    space.type_overrides = {"a": SemanticType.CATEGORICAL_NOMINAL}
    return space


# --------------------------------------------------------------------------
# a new dataset
# --------------------------------------------------------------------------

def test_a_new_dataset_clears_everything_derived_from_the_old_one(loaded_workspace):
    new = pd.DataFrame({"p": [1], "q": [2]})
    loaded_workspace.set_dataset(new, None, "second")

    assert loaded_workspace.dataset_name == "second"
    assert loaded_workspace.frame is new
    # None of this can survive: it all describes a dataset that is gone.
    assert loaded_workspace.profile is None
    assert loaded_workspace.typed_frame is None
    assert loaded_workspace.objectives == []
    assert loaded_workspace.objective is None
    assert loaded_workspace.pipelines == {}
    assert loaded_workspace.run is None
    assert loaded_workspace.runs == []
    assert loaded_workspace.chat == []
    assert loaded_workspace.type_overrides == {}
    # The engine caches the typed frame and every fitted model from the old data.
    assert loaded_workspace.scientist is None


def test_a_new_dataset_keeps_the_users_own_settings(loaded_workspace):
    loaded_workspace.access.high_contrast = True
    loaded_workspace.theme = "light"
    loaded_workspace.set_dataset(pd.DataFrame({"p": [1]}), None, "second")
    assert loaded_workspace.access.high_contrast is True
    assert loaded_workspace.theme == "light"


# --------------------------------------------------------------------------
# stale results, same data
# --------------------------------------------------------------------------

def test_a_context_change_keeps_the_profile(loaded_workspace):
    """The measurements still stand; only what they were taken to mean changed."""
    profile = loaded_workspace.profile
    loaded_workspace.reset_analysis()
    assert loaded_workspace.profile is profile
    assert loaded_workspace.typed_frame is not None
    assert loaded_workspace.objectives == []
    assert loaded_workspace.run is None
    assert loaded_workspace.pipelines == {}


def test_a_type_correction_forces_a_reprofile(loaded_workspace):
    loaded_workspace.reset_analysis(reprofile=True)
    assert loaded_workspace.profile is None
    assert loaded_workspace.typed_frame is None


def test_a_type_correction_is_not_lost_by_reprofiling(loaded_workspace):
    """The user is the authority on meaning; re-profiling must not overrule them."""
    loaded_workspace.reset_analysis(reprofile=True)
    assert loaded_workspace.type_overrides == {"a": SemanticType.CATEGORICAL_NOMINAL}


def test_neither_reset_touches_the_data(loaded_workspace):
    frame = loaded_workspace.frame
    loaded_workspace.reset_analysis()
    loaded_workspace.reset_analysis(reprofile=True)
    assert loaded_workspace.frame is frame
    assert loaded_workspace.dataset_name == "first"


def test_type_overrides_are_reapplied_after_a_reprofile():
    """End to end: correct a type, re-profile, and the correction survives."""
    from dsai.app.samples import build_sample

    space = Workspace()
    frame, source = build_sample("water_customers")
    space.set_dataset(frame, source, "wc")
    space.profile_dataset()
    original = space.profile.columns["support_tickets"].semantic_type

    space.type_overrides["support_tickets"] = SemanticType.CATEGORICAL_ORDINAL
    space.reset_analysis(reprofile=True)
    space.profile_dataset()

    corrected = space.profile.columns["support_tickets"]
    assert corrected.semantic_type is SemanticType.CATEGORICAL_ORDINAL
    assert corrected.semantic_type is not original
    assert corrected.semantic_type_overridden


def test_an_override_for_a_column_that_no_longer_exists_is_skipped():
    from dsai.app.samples import build_sample

    space = Workspace()
    frame, source = build_sample("water_customers")
    space.set_dataset(frame, source, "wc")
    space.type_overrides = {"not_a_column": SemanticType.BINARY}
    space.profile_dataset()          # must not raise
    assert space.profile is not None


# --------------------------------------------------------------------------
# comparing runs
# --------------------------------------------------------------------------

def test_runs_answering_different_questions_are_not_comparable(regression_run):
    import copy

    from dsai.core.schema import TaskType

    run_a, _ = regression_run
    run_b = copy.copy(run_a)
    run_b.objective = copy.copy(run_a.objective)
    run_b.objective.task_type = TaskType.BINARY_CLASSIFICATION

    ok, reason = comparable(run_a, run_b)
    assert not ok
    assert "different kinds of question" in reason


def test_runs_predicting_different_targets_are_not_comparable(regression_run):
    import copy

    run_a, _ = regression_run
    run_b = copy.copy(run_a)
    run_b.objective = copy.copy(run_a.objective)
    run_b.objective.target = "something_else"

    ok, reason = comparable(run_a, run_b)
    assert not ok
    assert "different things" in reason


def test_comparing_a_run_with_itself_reports_no_difference(regression_run):
    run, _ = regression_run
    diff = compare_runs(run, run)
    assert diff.comparable
    assert "No real difference" in diff.verdict or "unchanged" in diff.verdict.lower()
    assert not diff.setup.empty
    # Nothing changed, so nothing is marked changed.
    assert (diff.setup["Changed"] == "").all()


def test_a_difference_inside_the_fold_spread_is_called_a_tie(regression_run):
    import copy

    run_a, _ = regression_run
    metric = run_a.plan.primary_metric
    spread = run_a.best.validation.std_scores.get(metric, 0.0)
    if spread <= 0:
        pytest.skip("this run has no fold-to-fold spread to test against")

    run_b = _with_score(run_a, metric, run_a.best.primary(metric) + spread * 0.4)

    diff = compare_runs(run_a, run_b)
    assert "No real difference" in diff.verdict
    assert any("inside the noise" in c or "fold-to-fold spread" in c for c in diff.caveats)


def test_a_real_improvement_is_reported_as_one(regression_run):
    import copy

    run_a, _ = regression_run
    metric = run_a.plan.primary_metric
    # RMSE: lower is better, so halve it.
    run_b = _with_score(run_a, metric, run_a.best.primary(metric) * 0.5)

    diff = compare_runs(run_a, run_b)
    assert "Run B is better" in diff.verdict
    assert not diff.scores.empty


def _with_score(run, metric: str, value: float):
    """A copy of *run* whose selected model scores *value* on *metric*.

    Written against ``validation.mean_scores`` because that is what ``primary()``
    prefers — the cross-validated mean, not the single hold-out figure.
    """
    import copy

    clone = copy.copy(run)
    clone.best = copy.copy(run.best)
    clone.best.validation = copy.deepcopy(run.best.validation)
    clone.best.validation.mean_scores = dict(run.best.validation.mean_scores)
    clone.best.validation.mean_scores[metric] = value
    return clone


def test_the_setup_table_marks_what_changed(regression_run):
    import copy

    run_a, _ = regression_run
    run_b = copy.copy(run_a)
    run_b.dataset_name = "a different file"

    diff = compare_runs(run_a, run_b)
    changed = diff.setup[diff.setup["Changed"] == "yes"]
    assert "Dataset" in set(changed[""])


# --------------------------------------------------------------------------
# the paired test's direction
# --------------------------------------------------------------------------

def test_paired_test_reports_a_rise_as_a_rise():
    """The p-value is symmetric, so a reversed sign is invisible except in words."""
    from dsai.statistics.tests import wilcoxon_signed_rank

    before = pd.Series([10, 11, 12, 13, 14, 15, 16, 17])
    after = before + 5
    result = wilcoxon_signed_rank(before, after)
    assert result.detail["median_difference"] == pytest.approx(5.0)
    assert result.detail["direction"] == "increase"
    assert "rise" in result.conclusion


def test_paired_test_reports_a_fall_as_a_fall():
    from dsai.statistics.tests import wilcoxon_signed_rank

    before = pd.Series([10, 11, 12, 13, 14, 15, 16, 17])
    after = before - 5
    result = wilcoxon_signed_rank(before, after)
    assert result.detail["median_difference"] == pytest.approx(-5.0)
    assert result.detail["direction"] == "decrease"
    assert "fall" in result.conclusion


def test_paired_test_needs_enough_pairs():
    from dsai.statistics.tests import wilcoxon_signed_rank

    result = wilcoxon_signed_rank(pd.Series([1, 2]), pd.Series([2, 3]))
    assert "five complete pairs" in result.conclusion


def test_environment_snapshot_is_the_only_version_list():
    """Two lists of 'the versions that matter' that disagree is not a record."""
    from dsai.engines.experiment import ExperimentEngine
    from dsai.repro.provenance import environment_snapshot

    assert ExperimentEngine.environment(None) == environment_snapshot()
